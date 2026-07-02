# Day 12 面试题：SSE 流式输出 —— 从一次性返回到逐字推送

> 对应文件：`api/stream.py`
> 核心能力：SSE 协议原理、StreamingResponse、LLM `stream=True` 逐 token yield、流式/非流式双模式、客户端断开保护、TTFT 首 token 延迟

---

## 为什么需要流式输出？

Day 11 的 `/chat` 是同步模式：用户问一个问题，LLM 生成 500 字回答需要 8 秒，8 秒后一次性返回全部文字。

```
同步模式（Day 11）：
  用户发送 query ────────────── 8秒后 ──→ 一次性显示全部回答
  用户心理：😐 "卡了吗？" → 😤 "刷新"

流式模式（Day 12）：
  用户发送 query ─→ 0.8秒 首字出现 ─→ 逐字蹦出 ─→ 8秒后完成
  用户心理：😊 "在回复了" → 边看边思考
```

**TTFT（Time To First Token）是流式的核心指标**——用户不关心总耗时，只关心"多久能看到第一个字"。生产环境要求 < 1.5 秒。

---

## Day 12 代码架构全景

```
请求进入 POST /chat {"query": "...", "stream": true}
  │
  ▼
┌─ Middleware ────────────────────────────────────────────┐
│  记录 t0 + 方法/路径 + 客户端 IP                        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
          ┌─ Pydantic 校验 ──────────────┐
          │ ChatRequest.stream == true   │
          └──────────────┬───────────────┘
                         ▼
          ┌─ /chat 路由分支 ─────────────┐
          │ if req.stream:               │
          │   → StreamingResponse(...)    │
          │ else:                        │
          │   → ChatResponse(...)        │
          └──────────────┬───────────────┘
                         ▼ (流式)
          ┌─ llm_stream_generator_safe ──┐
          │ ① yield type=source          │
          │ ② yield type=token × N       │
          │ ③ yield type=done            │
          │ ④ yield [DONE]               │
          │                              │
          │ except CancelledError:       │
          │   → 停止生成，记录日志         │
          └──────────────────────────────┘
                         │
                         ▼
          SSE 事件流逐条推送到客户端
```

---

## 五个练习对照表

| 练习 | 代码位置 | 关键知识点 | 验证方式 |
|------|---------|-----------|---------|
| 1. StreamingResponse | `/stream-demo` GET | `StreamingResponse(gen(), media_type="text/event-stream")` | 浏览器打开或 `curl -N` |
| 2. LLM stream=True | `llm_stream_generator()` | `.stream(messages)` 迭代器 → `yield f"data: ..."` | `curl -N POST /chat stream=true` |
| 3. 双模式 | `/chat` 内 `if req.stream` 分支 | 同一接口、同一 Schema，返回类型不同 | 同一 `/chat` 传不同 stream 值 |
| 4. 断开保护 | `llm_stream_generator_safe()` | `except asyncio.CancelledError` 捕获断连 | 浏览器刷新页面，日志打 `[客户端断开]` |
| 5. source 事件 | generator 开头 `yield type=source` | `source → token → done` 三阶段事件协议 | 前端同时展示引用 + 生成内容 |

---

## 练习 1：`StreamingResponse` + `text/event-stream` — 最简流式

```python
async def demo_number_generator():
    for i in range(1, 11):
        yield f"data: {json.dumps({'num': i})}\n\n"
        await asyncio.sleep(0.5)

@app.get("/stream-demo")
async def stream_demo():
    return StreamingResponse(
        demo_number_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

**SSE 消息格式**：每条消息 `data: <内容>\n\n`，`\n\n` 是消息分隔符。

**三个必需的响应头**：

| 响应头 | 作用 |
|--------|------|
| `Cache-Control: no-cache` | 防止中间代理缓存流内容 |
| `Connection: keep-alive` | 保持 TCP 连接不断开 |
| `X-Accel-Buffering: no` | 禁用 Nginx 缓冲（否则 Nginx 会收集所有小块再转发，流式效果归零） |

---

## 练习 2：LLM `stream=True` → 逐 token yield

```python
# 真实接 LLM 的写法（当前代码用 simulate_rag_query 模拟）
llm = ChatOpenAI(model="deepseek-chat", streaming=True, ...)

for chunk in llm.stream(messages):
    token = chunk.content
    if token:
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
```

**`llm.stream()` 内部原理**：

```
LLM API 请求（stream=true, Transfer-Encoding: chunked）
  ├─ chunk 1: {"choices":[{"delta":{"content":"小"}}]}
  ├─ chunk 2: {"choices":[{"delta":{"content":"米"}}]}
  ├─ chunk 3: {"choices":[{"delta":{"content":"S"}}]}
  └─ chunk N: {"choices":[{"finish_reason":"stop"}]}
```

每个 HTTP chunk 到达后立即 yield，不需要等全部生成完——这就是 TTFT 能做到 < 2s 的原因。

---

## 练习 3：流式/非流式双模式

```python
class ChatRequest(BaseModel):
    stream: bool = Field(default=False)  # ← 只加一个字段

@app.post("/chat")
async def chat(req: ChatRequest):
    if req.stream:
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        return ChatResponse(answer=..., sources=...)
```

**一个接口，两种模式，同一个 Schema**——前端传 `"stream": true` 走流式，不传走同步。API 契约不分裂，只是返回类型不同。

---

## 练习 4：客户端断开 → 停止生成

```python
async def llm_stream_generator_safe(query, contexts):
    generated_count = 0  # ⚠️ 必须在 try 外初始化
    try:
        for token in answer:
            generated_count += 1
            yield f"data: ..."
    except asyncio.CancelledError:
        logger.warning(f"[客户端断开] 已生成 {generated_count} tokens（停止生成）")
```

**为什么重要**：
- LLM API 按 token 计费——用户关了标签页，服务继续生成 = 白白烧钱
- 服务端协程被占用 → 其他请求排队 → 雪崩
- `asyncio.CancelledError` 是 ASGI 框架在客户端断开时抛给 async generator 的异常

---

## 练习 5：流式中传输检索来源（`type: "source"` 事件）

```python
# 事件协议：source → token → done
async def llm_stream_generator_safe(query, contexts):
    # ① 先推来源
    yield f"data: {json.dumps({'type': 'source', 'sources': contexts})}\n\n"
    # ② 再推 token
    for token in answer:
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
    # ③ 结束
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
    yield "data: [DONE]\n\n"
```

前端用 `fetch` + `ReadableStream` 逐条解析：`type=source` → 渲染引用卡片，`type=token` → 逐字追加气泡，`type=done` → 关闭连接。

---

## SSE vs WebSocket vs 轮询

| 协议 | 方向 | 复杂度 | LLM 场景适用性 |
|------|------|--------|---------------|
| SSE | 服务器 → 客户端 单向 | 低（标准 HTTP） | ✅ 最佳：LLM 生成是单向数据流 |
| WebSocket | 双向 | 中（需升级协议） | ❌ 过度设计：不需要客户端推消息 |
| 轮询 | 客户端反复请求 | 低但浪费 | ❌ 最差：频繁建连不可控 |

---

## Q1：SSE 的 `text/event-stream` 和普通 HTTP `application/json` 有什么本质区别？

**一句话**：`application/json` 是一次性返回的"包裹"，`text/event-stream` 是持续推送的"水管"——一个连接关闭才结束，一个连接保持直到服务端发 `[DONE]`。

**对比**：

| | `application/json` | `text/event-stream` |
|---|---|---|
| 数据方向 | 一收一发 | 持续单向推送 |
| 连接生命周期 | 响应结束即关闭 | 保持直到服务端主动结束 |
| 客户端读取 | `response.json()` 一次拿完 | `ReadableStream.getReader().read()` 循环读 |
| HTTP 状态码 | 200/400/500 | 始终 200（错误通过事件内 type=error 传递） |

**面试话术**："SSE 跑在标准 HTTP/1.1 长连接上，没有协议升级握手，不需要像 WebSocket 那样在反向代理层做特殊配置。`text/event-stream` 的 MIME 类型告诉浏览器这是流——浏览器的 `EventSource` API 只认这个类型。"

---

## Q2：为什么要用 `fetch` + `ReadableStream` 而不是 `EventSource`？

**一句话**：`EventSource` 只支持 GET 请求且不支持自定义请求头，`fetch` + `ReadableStream` 支持 POST + `Authorization` 头——生产级流式对话必须用后者。

**对比**：

| | EventSource | fetch + ReadableStream |
|---|---|---|
| HTTP 方法 | 仅 GET | GET/POST/PUT 任意 |
| 自定义请求头 | ❌ | ✅ Authorization / X-API-Key |
| POST Body | ❌ | ✅ JSON `{"query":"...","stream":true}` |
| 自动重连 | ✅ 内置 | ❌ 需手动实现 |
| SSE 格式解析 | ✅ 自动 | ❌ 需手动按 `\n\n` 分割 |

**面试话术**："Day 12 的 `/chat-ui` 测试页面用 `fetch` 而不是 `EventSource`，因为 `/chat` 是个 POST 接口，需要传 JSON Body。`EventSource` 的问题在它的名字里——它是给 Server-Sent Events 设计的标准 API，但它只能发 GET。而 LLM 对话显然不应该把 query 拼在 URL 里。"

---

## Q3：`asyncio.CancelledError` 是什么？为什么流式必须处理它？

**一句话**：当客户端断开 TCP 连接时，ASGI 框架会向正在执行的 async generator 抛出 `CancelledError`——不捕获它，服务端不知道客户端已经走了，继续生成直到结束。

**时序图**：

```
正常流程：
  Client ─req─→ Server ─token 1─→ ─token 2─→ ... ─[DONE]─→

断开流程（无保护）：
  Client ─req─→ Server ─token 1─→ ─token 2─→ [浏览器关闭]
    ↑ 已关闭                                      Server 还在生成 token 3..500 → 白烧钱

断开流程（有 CancelledError 保护）：
  Client ─req─→ Server ─token 1─→ ─token 2─→ [浏览器关闭]
    ↑ 已关闭                                      Server 收到 CancelledError → 立即停止
```

**面试话术**："`CancelledError` 在流式场景里本质是成本控制器。生产环境 LLM API 调一次可能要几分钱，1000 个用户里有 200 个中途关页面，不处理断连等于白烧 20% 成本。注意 `generated_count` 必须在 `try` 块外初始化——如果 `CancelledError` 在赋值前触发，except 里引用它会 `NameError`。"

---

## Q4：流式模式下 BackgroundTasks 怎么写日志？和同步模式有什么不同？

**一句话**：流式模式的 `StreamingResponse` 返回后 generator 还在后台跑，`background_tasks.add_task` 在响应首部发出后立即执行——所以流式日志只能写摘要（query + 时间），不能写完整回答（回答还没生成完）。

**对比**：

```python
# 非流式：响应已全部生成，日志可以写完整 answer
resp = ChatResponse(answer=result["answer"], ...)
background_tasks.add_task(save_log, query=query, answer=result["answer"][:100])
return resp

# 流式：响应只发了首部，answer 还在逐字生成
background_tasks.add_task(save_log, query=query, answer_preview=f"[stream] {query[:50]}...")
return StreamingResponse(generator(...))
```

**面试话术**："流式日志的 challenge 是——你写日志那一刻回答还没生成完。生产方案有两种：一是在 generator 内部 `yield` 的同时把 token 缓存到列表，finally 或 [DONE] 后写入完整日志；二是异步监听 Redis pub/sub，generator 每 yield 一个 token 就 publish，日志消费者 subscribe 收集。Day 12 目前的方案是最简单的——记摘要。"

---

## Q5：`StreamingResponse` 不走 `response_model` 校验，如何保证流式输出格式正确？

**一句话**：`response_model` 只能校验"一次性返回值"（`BaseModel` 或 `dict`），`StreamingResponse` 返回的是一个 async generator——FastAPI 无法校验 generator 里 yield 出来的每一块数据。正确性靠 generator 内部的事件协议保证。

**如何保证格式**：

```python
# 定义内部协议（非 Pydantic，但团队约定即契约）
# type=source → 包含 sources 数组
# type=token  → 包含 token 字符串
# type=done   → 包含 total_tokens 数字
# [DONE]      → 流结束信号

# 每个 yield 都用 json.dumps 统一序列化，前端按 type 字段 dispatch
yield f"data: {json.dumps({'type': 'token', 'token': token, 'index': i})}\n\n"
```

**面试话术**："Pydantic 校验的是'一次性契约'，SSE 流是'流式契约'——前者靠类型系统保证，后者靠协议约定 + 代码 review 保证。如果要在流式上也做 Schema 校验，可以在 generator 内部对每个 yield 做一次 Pydantic 校验，但一般没必要——token 就是一个字符串，出错的概率极低。"

---

## Q6：Day 12 的 `/chat-ui` 页面和 `/docs` Swagger 各自解决什么问题？

**一句话**：`/docs` 是给开发者调 API 的（POST /chat 填 JSON → 看 JSON 响应），`/chat-ui` 是给产品/测试演示流式效果的（输入框 → 逐字蹦出）——两者互补，缺一个都说不清"流式"是什么。

**面试话术**："FastAPI 的 `/docs` 有先天局限——它是基于 OpenAPI JSON Schema 生成的，而 `StreamingResponse` 没法用 Schema 描述（它返回的不是 JSON 对象，是事件流）。所以 Day 12 加了一个自包含的 HTML 测试页面——不仅验证功能，也是给非技术人员演示'我们的对话不是一次性出的，是像 ChatGPT 一样逐字蹦出来的'。"

---

## Q7：TTFT（首 token 延迟）vs 端到端延迟，哪个对用户体验影响更大？

**一句话**：TTFT 决定"用户觉得卡不卡"，端到端延迟决定"用户等多久看完"——前者是体验感知，后者是效率指标。两个都重要，但 TTFT < 2s 是及格线，超了用户就想刷新。

**面试话术**："TTFT 的行业标准是 < 1.5 秒——这是 Google 和 OpenAI 的基准线。影响 TTFT 的因素有三个：① RAG 检索耗时（向量检索 + Reranker）；② LLM API 首 chunk 延迟（网络 + 推理冷启动）；③ 服务端 `yield` 第一个 token 前的处理逻辑。优化方向依次是：缩小 top_k 减少检索量、用更快的 embedding 模型、LLM API 选离用户最近的 region。"

---

## Q8：`llm_stream_generator_safe` 中线程安全的考虑？

**一句话**：async generator 本身是协程级别的并发安全（每个请求一个独立 generator 实例），但 `generated_count` 是栈局部变量——不同请求之间不会共享，单请求内异常处理需保证变量已初始化。

**面试话术**："Python 的 async generator 天然是协程安全的——`async for` 每次迭代都是一个 `await`，事件循环在 await 点切换。多个请求同时到达时，每个请求有自己的 generator 实例和局部变量，不存在竞态。唯一需要注意的是 uvicorn 多 worker 模式下的共享状态——像 Redis 连接池这种需要按 worker 初始化。"

---

## Q9：Day 11 同步 `/chat` vs Day 12 流式 `/chat`，代码 diff 的核心变化？

**一句话**：`return ChatResponse(...)` 只执行一次 → `StreamingResponse(generator)` 连接保持持续推送。双模式通过 `if req.stream:` 一个分支实现，API 契约不需要分拆。

| Day 11 | Day 12 |
|--------|--------|
| `return ChatResponse(...)` 一次返回 | `return StreamingResponse(generator)` 持续推送 |
| `response_model=ChatResponse` 自动校验 | StreamingResponse 不走 Schema，靠协议约定 |
| 用户等 8 秒看完整回答 | 用户 0.8 秒看到首字 |
| BackgroundTasks 正常记完整日志 | 只能记摘要 |
| `def chat():` 同步 | `async def chat():` 加入 if/else 分支 |

**面试话术**："Day 12 不是取代 Day 11——是在 Day 11 的 `/chat` 上加一个分支。同步模式仍然是后台任务、批量评测、API 对接的首选（简单、可缓存、易测试）。流式模式是 C 端用户体验的标配。一个接口走两种模式，比拆成两个接口更干净。"

---

## Q10：如果 LLM API 支持 `stream=True` 但网络丢包，SSE 怎么处理？

**一句话**：SSE 基于 TCP，TCP 层保证数据按序到达；如果 TCP 连接断开，`CancelledError` 会被触发。SSE 协议本身定义了 `id:` 字段和 `retry:` 字段用于断线重连，但 LLM 场景不建议重连——丢失的 token 无法补发。

**解决方案**：

```
方案 1（推荐）：不重连，前端提示用户刷新
  → LLM 生成是不可重复的（temperature > 0），重连后 LLM 给出的回答会不同

方案 2（高级）：服务端缓存生成结果
  → generator 每 yield 一个 token 同时写入 Redis
  → 客户端重连时从缓存恢复（需带 last_event_id）
```

**面试话术**："SSE 过 CDN/代理层可能被缓冲，生产上线前必须确认：① 反向代理是否配置了 `proxy_buffering off`（Nginx）或 `X-Accel-Buffering: no`；② CDN 是否支持 `text/event-stream` 透传（大部分 CDN 默认会缓冲，需要开 Stream 模式）；③ 超时设置（`proxy_read_timeout` 和 `keepalive_timeout` 需要大于 LLM 最长生成时间）。"

---

## Day 11 → Day 12 的升级路径

| # | 变化 | 说明 |
|---|------|------|
| 1 | `return` → `yield` | 一次性返回变成多次推送 |
| 2 | `ChatResponse` → `StreamingResponse` | JSON Schema 校验变成事件流协议 |
| 3 | 无断连感知 → `CancelledError` | 成本控制的关键一步 |
| 4 | 单模式 → 双模式 | 通过 `stream: bool` 一个字段切换 |
| 5 | 前后端一体 | `/chat-ui` 自测页面加入，流式效果可视化 |

---

### Day 12 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | SSE 协议的消息格式是什么？`\n\n` 的作用？ | □ |
| 2 | `StreamingResponse` 的三个必需响应头及其作用？ | □ |
| 3 | `llm.stream(messages)` 内部是怎么工作的？TTFT 为什么能做到 < 2s？ | □ |
| 4 | 双模式 `/chat` 怎么用一个接口支持流式和非流式？ | □ |
| 5 | `asyncio.CancelledError` 在什么时机抛出？为什么 `generated_count` 必须在 try 外初始化？ | □ |
| 6 | 流式模式下 BackgroundTasks 怎么调整？为什么不能写完整回答？ | □ |
| 7 | `fetch` + `ReadableStream` vs `EventSource`，各自的优劣？ | □ |
| 8 | SSE 经过 Nginx/CDN 需要注意什么？`X-Accel-Buffering: no` 解决什么问题？ | □ |
| 9 | SSE vs WebSocket 在 LLM 生成场景的选型理由？ | □ |
| 10 | 客户端断开后继续生成的影响是什么？Day 12 怎么处理的？ | □ |
