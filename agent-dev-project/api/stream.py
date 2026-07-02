import json
import os
import time
import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ================================================================
# 0. 基础设置（和 Day 11 相同）
# ================================================================
app = FastAPI(
    title="汽车RAG（Day 12 流式升级版）",
    description="Day 11 同步 /chat + Day 12 SSE 流式 /chat，同一个接口",
    version="2.0",
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("car_api_stream")

START_TIME = time.time()

# 模拟知识库（和 Day 11 共用同一份数据）
CAR_PRICE_DB = {
    "小米 SU7":      "21.59-29.99 万",
    "比亚迪 海豚":    "9.98-13.98 万",
    "比亚迪 海豹":    "17.98-24.98 万",
    "特斯拉 Model Y": "24.99-35.49 万",
    "特斯拉 Model 3": "23.19-33.59 万",
    "理想 L6":       "24.98-27.98 万",
    "理想 L7":       "30.98-37.98 万",
    "问界 M7":       "24.98-32.98 万",
    "小鹏 G6":       "20.99-27.69 万",
    "蔚来 ET5":      "29.80-35.60 万",
    "极氪 001":      "26.90-32.90 万",
    "零跑 C11":      "15.58-19.98 万",
}

CAR_SPEC_DB = {
    "小米 SU7":      "CLTC续航 700-830km，零百加速 2.78-5.28s，Orin-X智驾芯片，激光雷达，800V高压平台",
    "比亚迪 海豚":    "CLTC续航 301-405km，零百加速 ~10s，L2基础智驾，无激光雷达，价格亲民",
    "比亚迪 海豹":    "CLTC续航 550-700km，零百加速 3.8s，DiPilot智驾，CTB电池车身一体化",
    "特斯拉 Model Y": "CLTC续航 545-688km，零百加速 3.7-5.0s，FSD智驾，超充网络覆盖广",
    "特斯拉 Model 3": "CLTC续航 556-713km，零百加速 3.3-6.1s，FSD智驾，操控精准",
    "理想 L6":       "增程，CLTC综合续航 1390km，零百加速 5.4s，AD Max智驾，家用大五座",
    "理想 L7":       "增程，CLTC综合续航 1315km，零百加速 5.3s，AD Max智驾，空气悬架",
    "问界 M7":       "增程/纯电，CLTC续航 1200km，零百加速 4.8s，ADS 2.0，鸿蒙座舱",
    "小鹏 G6":       "CLTC续航 580-755km，零百加速 3.9-6.2s，XNGP智驾，双激光雷达，800V",
    "蔚来 ET5":      "CLTC续航 560-710km，零百加速 4.0s，NAD智驾，换电服务",
    "极氪 001":      "CLTC续航 546-741km，零百加速 3.8s，NZP智驾，猎装造型",
    "零跑 C11":      "CLTC续航 502-650km，零百加速 4.5-7.9s，L2智驾，性价比高",
}


# ================================================================
# 辅助函数：模拟 RAG 检索（和 Day 11 相同）
# ================================================================
def simulate_rag_query(query: str, top_k: int = 3) -> dict:
    """关键词匹配模拟 RAG 检索，返回 contexts + 预生成回答"""
    keywords = query.replace(",", " ").replace("?", " ").split()

    matched = []
    for name, price in CAR_PRICE_DB.items():
        spec = CAR_SPEC_DB.get(name, "暂无规格")
        hits = 0
        for kw in keywords:
            if kw in name or kw in spec or kw in price:
                hits += 1
        if hits > 0:
            matched.append((hits, name, spec, price))

    matched.sort(key=lambda x: x[0], reverse=True)
    top_results = matched[:top_k]

    if not top_results:
        answer = f"抱歉，我在知识库中没有找到与「{query}」相关的车型信息。"
        contexts = []
    else:
        lines = [f"关于「{query}」，为您找到以下相关信息：\n"]
        contexts = []
        for i, (score, name, spec, price) in enumerate(top_results, 1):
            lines.append(f"{i}. **{name}** | 价格: {price}")
            lines.append(f"   参数: {spec}\n")
            contexts.append({
                "rank": i,
                "source": name,
                "content": f"{spec} | 价格: {price}",
                "score": round(score / max(1, len(keywords)), 2),
            })
        answer = "\n".join(lines)

    return {"answer": answer, "contexts": contexts, "rewritten": query}


# ================================================================
# Pydantic Schema（Day 11 ChatRequest + stream 字段）
# ================================================================
class SourceDoc(BaseModel):
    rank: int = Field(description="排名 1-N")
    source: str = Field(description="来源文档名/车型名")
    content: str = Field(description="内容摘要")
    score: float = Field(description="相关性分数 0-1")


class ChatRequest(BaseModel):
    """POST /chat 请求体 —— 比 Day 11 多了 stream 字段"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题",
        examples=["小米SU7多少钱"],
    )
    session_id: str = Field(default="default", min_length=1, max_length=64)
    top_k: int = Field(default=3, ge=1, le=10)
    stream: bool = Field(
        default=False,
        description="是否流式输出。true=SSE逐字推送，false=一次性返回JSON",
    )


class ChatResponse(BaseModel):
    """非流式模式响应体（和 Day 11 相同）"""
    answer: str
    sources: list[SourceDoc] = Field(default_factory=list)
    latency_ms: float
    session_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    uptime_seconds: float


async def demo_number_generator():
    for i in range(1, 11):
        chunk = json.dumps({"num": i, "text": f"第 {i} 条消息"}, ensure_ascii=False)
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.5)


@app.get("/stream-demo", tags=["练习1"])
async def stream_demo():
    return StreamingResponse(
        demo_number_generator(),
        media_type="text/event-stream",   # ← 关键：SSE 的 MIME 类型
        headers={
            "Cache-Control": "no-cache",           # 防止中间代理缓存
            "Connection": "keep-alive",            # 连接保持
            "X-Accel-Buffering": "no",             # 禁用 Nginx 缓冲（生产必备）
        },
    )


# ================================================================
# 浏览器测试页面：打开就能测流式效果
# ================================================================
@app.get("/chat-ui", tags=["测试"])
async def chat_ui():
    """返回一个自包含的 HTML 页面，浏览器直接测试 SSE 流式对话"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Day 12 SSE 流式对话测试</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; display: flex; justify-content: center; padding: 20px; }
  .container { width: 100%; max-width: 700px; display: flex; flex-direction: column; }
  h2 { text-align: center; margin-bottom: 16px; color: #e94560; }
  .chat-box { flex: 1; background: #16213e; border-radius: 8px; padding: 16px; overflow-y: auto; min-height: 300px; max-height: 60vh; margin-bottom: 12px; }
  .msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 8px; animation: fadeIn .3s; }
  .msg.user { background: #0f3460; text-align: right; }
  .msg.assistant { background: #1a1a3e; border-left: 3px solid #e94560; }
  .msg.assistant .token { transition: opacity .1s; }
  .msg.assistant .meta { font-size: 12px; color: #888; margin-top: 8px; border-top: 1px solid #333; padding-top: 6px; }
  .source-tag { display: inline-block; background: #0f3460; color: #58a6ff; padding: 2px 8px; margin: 2px 4px; border-radius: 4px; font-size: 12px; cursor: pointer; }
  .input-row { display: flex; gap: 8px; }
  .input-row input { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #333; background: #16213e; color: #eee; font-size: 15px; outline: none; }
  .input-row input:focus { border-color: #e94560; }
  .input-row button { padding: 12px 20px; border-radius: 8px; border: none; background: #e94560; color: #fff; font-size: 15px; cursor: pointer; }
  .input-row button:disabled { opacity: .5; cursor: not-allowed; }
  .quick-asks { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
  .quick-asks button { padding: 6px 12px; border-radius: 14px; border: 1px solid #444; background: transparent; color: #aaa; font-size: 13px; cursor: pointer; transition: .2s; }
  .quick-asks button:hover { border-color: #e94560; color: #e94560; }
  .status { font-size: 12px; color: #666; text-align: center; margin-bottom: 8px; min-height: 18px; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>
<div class="container">
  <h2>🚗 汽车导购助手 · SSE 流式测试</h2>
  <div class="status" id="status">输入问题开始测试 ↓</div>
  <div class="quick-asks" id="quickAsks">
    <button onclick="ask('小米SU7多少钱')">小米SU7多少钱</button>
    <button onclick="ask('20万左右推荐一款纯电SUV')">20万推荐纯电SUV</button>
    <button onclick="ask('特斯拉Model Y和理想L6对比')">Model Y vs L6</button>
    <button onclick="ask('问界M7续航多少')">问界M7续航</button>
  </div>
  <div class="chat-box" id="chatBox"></div>
  <div class="input-row">
    <input id="queryInput" placeholder="输入问题，回车发送..." onkeydown="if(event.key==='Enter')send()">
    <button id="sendBtn" onclick="send()">发送</button>
  </div>
</div>
<script>
  let es = null;

  function ask(text) {
    document.getElementById('queryInput').value = text;
    send();
  }

  function send() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    if (!query) return;

    // 关闭上一个连接
    if (es) { es.close(); es = null; }

    const box = document.getElementById('chatBox');
    const btn = document.getElementById('sendBtn');
    const status = document.getElementById('status');

    // 显示用户消息
    const userDiv = document.createElement('div');
    userDiv.className = 'msg user';
    userDiv.textContent = query;
    box.appendChild(userDiv);
    box.scrollTop = box.scrollHeight;

    // 创建助手消息区
    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'msg assistant';
    assistantDiv.id = 'currentMsg';
    box.appendChild(assistantDiv);

    input.value = '';
    btn.disabled = true;
    status.textContent = '⏳ 连接中...';

    // 用 fetch + ReadableStream 读 SSE（比 EventSource 灵活，支持 POST）
    fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, stream: true }),
    }).then(response => {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      status.textContent = '🟢 接收中...';

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let tokenCount = 0;

      function pump() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            status.textContent = `✅ 完成 · ${tokenCount} tokens`;
            btn.disabled = false;
            return;
          }

          buffer += decoder.decode(value, { stream: true });
          // 按 \\n\\n 分割 SSE 事件
          const parts = buffer.split('\\n\\n');
          buffer = parts.pop(); // 保留不完整的最后一个

          parts.forEach(part => {
            if (!part.trim() || !part.startsWith('data: ')) return;
            const data = part.replace(/^data: /, '');
            if (data === '[DONE]') return;
            try {
              const msg = JSON.parse(data.trim());
              if (msg.type === 'source') {
                // 显示来源标签
                const tagRow = document.createElement('div');
                tagRow.style.cssText = 'margin-bottom: 6px;';
                msg.sources.forEach(s => {
                  const tag = document.createElement('span');
                  tag.className = 'source-tag';
                  tag.textContent = `📎 ${s.source}`;
                  tag.title = s.content;
                  tagRow.appendChild(tag);
                });
                assistantDiv.appendChild(tagRow);
              } else if (msg.type === 'token') {
                assistantDiv.appendChild(document.createTextNode(msg.token));
                tokenCount++;
              } else if (msg.type === 'done') {
                const meta = document.createElement('div');
                meta.className = 'meta';
                meta.textContent = `▸ ${msg.total_tokens || tokenCount} tokens`;
                assistantDiv.appendChild(meta);
              }
            } catch(e) { /* 忽略解析失败 */ }
          });

          box.scrollTop = box.scrollHeight;
          return pump();
        });
      }
      return pump();
    }).catch(err => {
      status.textContent = '❌ 连接断开：' + err.message;
      btn.disabled = false;
    });
  }
</script>
</body>
</html>""")



async def llm_stream_generator(query: str, contexts: list):
    if contexts:
        yield f"data: {json.dumps({'type': 'source', 'sources': contexts}, ensure_ascii=False)}\n\n"

    answer = simulate_rag_query(query, top_k=len(contexts))["answer"]
    tokens = list(answer)
    full_answer = ""

    for i, token in enumerate(tokens):
        full_answer += token
        yield f"data: {json.dumps({'type': 'token', 'token': token, 'index': i}, ensure_ascii=False)}\n\n"

        if i % 3 == 0:
            await asyncio.sleep(0.05)

    yield f"data: {json.dumps({'type': 'done', 'full_length': len(full_answer)}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def llm_stream_generator_safe(query: str, contexts: list):
    # ⚠️ generated_count 必须在 try 外初始化，
    # 否则 CancelledError 在赋值前触发会 NameError
    generated_count = 0
    try:
        # ① 先推来源（type=source）
        if contexts:
            yield f"data: {json.dumps({'type': 'source', 'sources': contexts}, ensure_ascii=False)}\n\n"

        answer = simulate_rag_query(query, top_k=len(contexts))["answer"]

        # ② 逐 token 推送（type=token）
        for token in answer:  # 字符串可以直接迭代，不用 list() 转
            generated_count += 1
            yield f"data: {json.dumps({'type': 'token', 'token': token, 'index': generated_count - 1}, ensure_ascii=False)}\n\n"
            # 模拟 LLM 逐字生成延迟，真实接 LLM API 时不需要
            await asyncio.sleep(0.03)

        # ③ 结束信号
        yield f"data: {json.dumps({'type': 'done', 'total_tokens': generated_count}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        logger.warning(
            f"[客户端断开] query={query[:50]}... | "
            f"已生成 {generated_count} tokens | "
            f"已推送 {len(contexts)} 条来源（停止生成，不再消耗 LLM token）"
        )


@app.post("/chat", tags=["对话"])
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    练习3: 双模式 /chat 接口。
    
    - stream=false（默认）: 和 Day 11 一样，一次性返回 JSON
    - stream=true: 返回 SSE 流，逐字推送
    
    同一个接口，同一个 ChatRequest Schema，只是返回类型不同。
    """
    t0 = time.time()

    # 检索（流式和非流式都要做）
    result = simulate_rag_query(req.query, top_k=req.top_k)
    contexts = result.get("contexts", [])

    # ── 分支：流式 vs 非流式 ──
    if req.stream:
        # ============= 流式模式 =============
        logger.info(f"→ SSE stream | query={req.query[:50]}... | top_k={req.top_k}")

        # 后台写日志（流式模式下用安全版生成器需注意：BackgroundTasks 在响应发出后执行）
        background_tasks.add_task(
            _save_chat_log,
            query=req.query,
            answer_preview=f"[stream] {req.query[:50]}...",
            latency_ms=round((time.time() - t0) * 1000, 1),
            session_id=req.session_id,
        )

        return StreamingResponse(
            llm_stream_generator_safe(req.query, contexts),  # 练习4: 带中断保护
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Response-Time-ms": f"{(time.time() - t0) * 1000:.1f}",
            },
        )
    else:
        # ============= 非流式模式（和 Day 11 完全一样）============
        answer = result["answer"]
        sources = [
            SourceDoc(
                rank=c["rank"],
                source=c["source"],
                content=c["content"][:200],
                score=c["score"],
            )
            for c in contexts
        ]
        latency_ms = round((time.time() - t0) * 1000, 1)

        response = ChatResponse(
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            session_id=req.session_id,
        )

        background_tasks.add_task(
            _save_chat_log,
            query=req.query,
            answer_preview=answer[:100],
            latency_ms=latency_ms,
            session_id=req.session_id,
        )

        return response


def _save_chat_log(query: str, answer_preview: str, latency_ms: float, session_id: str):
    """后台异步写日志，不阻塞 HTTP 响应"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "query": query,
        "answer_preview": answer_preview,
        "latency_ms": latency_ms,
    }
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "chat_history.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")



@app.middleware("http")
async def log_and_time_middleware(request: Request, call_next):
    t0 = time.time()
    logger.info(f"→ {request.method} {request.url.path}")

    response = await call_next(request)

    latency = (time.time() - t0) * 1000
    log_level = logging.WARNING if latency > 3000 else logging.INFO
    logger.log(log_level, f"← {request.method} {request.url.path} | status={response.status_code} | latency={latency:.1f}ms")

    response.headers["X-Response-Time-ms"] = f"{latency:.1f}"
    return response


# ================================================================
# 健康检查 & 全局异常处理
# ================================================================
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health():
    return HealthResponse(
        status="ok",
        version="2.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未捕获异常 | path={request.url.path} | error={exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "detail": str(exc) if os.getenv("DEBUG") else "请联系管理员"},
    )


if __name__ == "__main__":
    print("=" * 60)
    print("流式 RAG 服务启动中...")
    print("=" * 60)
    print()
    print("浏览器测试（推荐）:")
    print("  http://127.0.0.1:8000/chat-ui           ← 可视化流式对话")
    print()
    print("命令行测试:")
    print("  练习1: curl -N http://127.0.0.1:8000/stream-demo")
    print('  练习2/3: curl -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d \'{"query":"小米SU7多少钱","stream":true}\'')
    print('  非流式: curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d \'{"query":"小米SU7多少钱"}\'')
    print()

    uvicorn.run(
        "stream:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )