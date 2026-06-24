import json
import os
import time
import logging
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 实例化app
app = FastAPI(
    title="汽车RAG",
    description="FastAPI包装agent",
    version="1.0"
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("car_api")

# 模拟知识库 & RAG 管线

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

def simulate_rag_query(query: str, top_k: int = 3) -> dict:
    """
    模拟 RAG 检索 + LLM 生成
    """
    # ✅ 修复1: 提取关键词（放在循环外）
    keywords = query.replace(",", " ").replace("?", " ").split()
    
    # 检索：关键词匹配
    matched = []
    for name, price in CAR_PRICE_DB.items():  # ✅ 修复2: 加括号
        # 获取该车型的规格信息
        spec = CAR_SPEC_DB.get(name, "暂无规格")
        
        # 计算匹配分数：车型名和规格中匹配的关键词数量
        hits = 0
        for kw in keywords:
            if kw in name or kw in spec or kw in price:
                hits += 1
        
        if hits > 0:
            matched.append((hits, name, spec, price))
    
    # 按匹配度排序
    matched.sort(key=lambda x: x[0], reverse=True)
    top_results = matched[:top_k]
    
    # 构造LLM prompt
    if not top_results:
        answer = (
            f"抱歉，我在知识库中没有找到与「{query}」相关的车型信息。\n"
            f"建议您换个关键词试试，或者告诉我您的预算范围，我来为您推荐。"
        )
        sources = []
    else:
        lines = [f"关于「{query}」，为您找到以下相关信息：\n"]
        sources = []
        for i, (score, name, spec, price) in enumerate(top_results, 1):
            lines.append(f"{i}. **{name}** | 价格: {price}")
            lines.append(f"   参数: {spec}")
            lines.append("")
            sources.append({
                "rank": i,
                "source": name,
                "content": f"{spec} | 价格: {price}",
                "score": round(score / max(1, len(keywords)), 2),
            })
        answer = "\n".join(lines)
    
    return {"answer": answer, "sources": sources, "rewritten": query}


# Pydantic Request/Response Schema
class SourceDoc(BaseModel):
    """单条检索来源"""
    rank: int = Field(description="排名 1-N")
    source: str = Field(description="来源文档名/车型名")
    content: str = Field(description="内容摘要，截断到 200 字")
    score: float = Field(description="相关性分数 0-1")


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题，如'小米SU7多少钱'",
        examples=["20-25万推荐一款纯电SUV"],
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        description="会话 ID，同一会话内可保持多轮对话上下文",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="检索返回文档数，越大越全但越慢",
    )


class ChatResponse(BaseModel):
    """POST /chat 响应体"""
    answer: str = Field(description="LLM 生成的回答")
    sources: list[SourceDoc] = Field(default_factory=list, description="检索到的参考来源")
    latency_ms: float = Field(description="端到端延迟（毫秒）")
    session_id: str = Field(description="回显会话 ID")


class HealthResponse(BaseModel):
    """GET /health 响应体"""
    status: str
    version: str
    timestamp: str
    uptime_seconds: float


# 记录服务启动时间
START_TIME = time.time()

@app.get("/health", response_model=HealthResponse, tags=["系统"])  # ✅ 修复3: response_model
async def health():
    return HealthResponse(  # ✅ 使用Pydantic模型
        status="ok",
        version="1.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# /chat 同步接口
@app.post("/chat", response_model=ChatResponse, tags=["对话"])  # ✅ 修复3: response_model
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    一次完整的 RAG 问答。
    流程: 参数校验(Pydantic自动) → RAG检索 → LLM生成 → 组装响应 → 后台写日志
    """
    t0 = time.time()
    result = simulate_rag_query(req.query, top_k=req.top_k)

    sources = [
        SourceDoc(
            rank=s["rank"],
            source=s["source"],
            content=s["content"][:200],
            score=s["score"],
        )
        for s in result["sources"]
    ]
    latency_ms = round((time.time() - t0) * 1000, 1)

    # 构造响应
    response = ChatResponse(
        answer=result["answer"],
        sources=sources,
        latency_ms=latency_ms,
        session_id=req.session_id,
    )

    # 后台异步写日志，不阻塞 HTTP 响应
    background_tasks.add_task(
        _save_chat_log,
        query=req.query,
        answer=result["answer"][:100],
        latency_ms=latency_ms,
        session_id=req.session_id,
    )

    return response


# BackgroundTasks —— 异步写日志，不阻塞响应
def _save_chat_log(query: str, answer: str, latency_ms: float, session_id: str):
    """
    后台任务：把对话记录写入 JSONL 日志文件。
    关键：这个函数在 HTTP 响应返回**之后**才执行，
    用户不会感知文件写入的延迟。
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "query": query,
        "answer_preview": answer,
        "latency_ms": latency_ms,
    }
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "chat_history.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# Middleware —— 请求日志 + 耗时统计（AOP 横切关注点）
@app.middleware("http")
async def log_and_time_middleware(request: Request, call_next):
    """
    全局中间件：记录每个 HTTP 请求的方法、路径、状态码、耗时。
    
    执行顺序：
      before: 记录 t0，打印"请求进入"日志
      await call_next(request): 执行真正的 handler（/chat 或 /health）
      after:  计算耗时，打印"请求完成"结构化日志
    """
    t0 = time.time()
    
    # ---- before: 请求进入 ----
    logger.info(f"→ {request.method} {request.url.path} | client={request.client.host if request.client else 'N/A'}")
    
    # ---- 执行真正的 handler ----
    response = await call_next(request)
    
    # ---- after: 请求完成 ----
    latency = (time.time() - t0) * 1000
    log_level = logging.WARNING if latency > 3000 else logging.INFO
    
    logger.log(
        log_level,
        f"← {request.method} {request.url.path} | "
        f"status={response.status_code} | "
        f"latency={latency:.1f}ms"
    )
    
    # 注入自定义响应头（调试用）
    response.headers["X-Response-Time-ms"] = f"{latency:.1f}"
    response.headers["X-Server-Version"] = "1.0.0"
    
    return response


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理：避免 500 时暴露内部细节给客户端"""
    logger.error(f"未捕获异常 | path={request.url.path} | error={exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "detail": str(exc) if os.getenv("DEBUG") else "请联系管理员"},
    )


if __name__ == "__main__":
    print("=" * 60)
    print("汽车 RAG 服务启动中...")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",          
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )