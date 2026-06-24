# Day 11 面试题：FastAPI 包装 RAG 服务 —— 从命令行到 HTTP API

> 对应文件：`api/main.py`
> 核心能力：FastAPI 路由与 Schema 设计、Pydantic 请求/响应校验、Middleware 横切关注点、BackgroundTasks 异步任务、从 CLI 脚本到 HTTP 服务的架构转变

---

## 为什么需要 FastAPI 包装？

Day 1-10 的所有代码都是命令行脚本：

```bash
python car_advisor_v2.py    # 终端里看输出
python full_rag_agent.py    # 输入问题 → 终端打印回答
```

但真实用户不会 SSH 到你服务器上敲命令。他们通过浏览器、App、第三方系统调用你的 Agent。**FastAPI 把"能跑通的代码"变成"能调用的服务"**——这是 Week 3（生产化部署）的起点。

---

## Day 11 代码架构全景

```
请求进来
  │
  ▼
┌─ Middleware（练习 4）─────────────────────────┐
│  before: 记录 t0 + 请求方法/路径/客户端IP      │
│          await call_next(request)  ──────────┐│
│  after:  计算耗时 + 状态码 + 注入响应头        ││
└──────────────────────────────────────────────┘│
                                               ▼
                              ┌─ Pydantic 校验（练习 2）──┐
                              │ ChatRequest 自动校验:      │
                              │ query 非空, top_k 1-10    │
                              │ → 不合法直接返回 422       │
                              └──────────┬───────────────┘
                                         ▼
                              ┌─ /chat Handler（练习 3）──┐
                              │ ① simulate_rag_query()    │
                              │ ② 组装 SourceDoc 列表     │
                              │ ③ 构造 ChatResponse       │
                              └──────────┬───────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                          ▼
        ┌─ BackgroundTasks（练习 5）─┐          ┌─ Pydantic 校验（练习 2）──┐
        │ _save_chat_log()           │          │ response_model 自动校验:   │
        │ 响应先返回，日志后写入       │          │ 字段缺失 → 500            │
        └────────────────────────────┘          └──────────────────────────┘
                                                                │
                                                                ▼
                                                          响应返回
```

---

## 五个练习对照表

| 练习 | 代码位置 | 关键知识点 | 验证方式 |
|------|---------|-----------|---------|
| 1. `/health` | `@app.get("/health")` | GET 无参数，`response_model=HealthResponse` | `curl http://127.0.0.1:8000/health` |
| 2. Schema | `ChatRequest` / `ChatResponse` / `SourceDoc` | `Field(ge/le/min_length)` 自动校验，Swagger 自动生成 | `/docs` 页面查看 |
| 3. `/chat` | `@app.post("/chat")` | POST + Pydantic Body → RAG → 组装响应 | `curl -X POST ... -d '{"query":"..."}'` |
| 4. Middleware | `@app.middleware("http")` | `await call_next(request)` 前后钩子，AOP 思想 | 日志输出 + 响应头 `X-Response-Time-ms` |
| 5. BackgroundTasks | `background_tasks.add_task(...)` | 响应返回后才执行，不阻塞用户 | `logs/chat_history.jsonl` 文件生成 |

---

## 练习 1：`/health` 健康检查 — 最简单的接口

```python
START_TIME = time.time()   # 全局，记录服务启动时间

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health():
    return HealthResponse(
        status="ok",
        version="1.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(time.time() - START_TIME, 1),
    )
```

**设计要点**：

| 要素 | 说明 |
|------|------|
| GET 无参数 | 健康检查不需要任何输入 |
| `response_model` | FastAPI 自动校验返回结构，字段缺失会报错 |
| `uptime_seconds` | 服务存活时长，运维排查"什么时候重启的" |
| 生产扩展 | 检查 LLM API 连通性、向量库状态、Redis 连接 |

**为什么不用 POST**：健康检查通常由 K8s liveness probe 或负载均衡器周期性调用，GET 语义更准确（只读，无副作用），且可以被 CDN 缓存。

---

## 练习 2：Pydantic Request/Response Schema — 定义 API 契约

```python
# ── 请求体 ──
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="用户问题", examples=["20-25万推荐一款纯电SUV"])
    session_id: str = Field(default="default", min_length=1, max_length=64,
                            description="会话ID")
    top_k: int = Field(default=3, ge=1, le=10,
                       description="检索返回文档数")

# ── 响应体 ──
class SourceDoc(BaseModel):
    rank: int = Field(description="排名")
    source: str = Field(description="来源文档名")
    content: str = Field(description="内容摘要，截断到200字")
    score: float = Field(description="相关性分数")

class ChatResponse(BaseModel):
    answer: str = Field(description="LLM生成的回答")
    sources: list[SourceDoc] = Field(default_factory=list)
    latency_ms: float = Field(description="端到端延迟")
    session_id: str = Field(description="回显会话ID")
```

**Schema 即文档——三重价值**：

| 维度 | 效果 |
|------|------|
| 自动校验 | `query=""` → 422，`top_k=20` → 422，不需要手写 if |
| 自动文档 | 打开 `/docs`，所有字段、类型、约束、示例一目了然 |
| 类型安全 | IDE 自动补全 `req.query`、`req.session_id`，不会拼错字段名 |

**与 Day 8 的区别**：Day 8 用 Pydantic 约束 LLM 的 JSON 输出，Day 11 用 Pydantic 约束 API 的 JSON 输入/输出。同一个工具，从"约束模型"延伸到"约束系统边界"。

---

## 练习 3：`/chat` 同步接口 — 核心业务

```python
@app.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    t0 = time.time()

    # ① 调 RAG 管线（生产环境换成 RAGAgent.chat()）
    result = simulate_rag_query(req.query, top_k=req.top_k)

    # ② 组装 SourceDoc 列表（dict → Pydantic 对象）
    sources = [
        SourceDoc(rank=s["rank"], source=s["source"],
                  content=s["content"][:200], score=s["score"])
        for s in result["sources"]
    ]

    # ③ 后台写日志（不阻塞响应）
    background_tasks.add_task(_save_chat_log, ...)

    # ④ 返回 Pydantic 对象，FastAPI 自动序列化成 JSON
    return ChatResponse(answer=result["answer"], sources=sources,
                        latency_ms=round((time.time()-t0)*1000, 1),
                        session_id=req.session_id)
```

**从 CLI 到 API 的核心变化**：

| 维度 | Day 1-10 CLI 脚本 | Day 11 FastAPI |
|------|------------------|----------------|
| 调用方式 | `python script.py` | `curl -X POST /chat` |
| 输入校验 | `if not query: return` | `Field(min_length=1)` 自动 422 |
| 输出格式 | `print()` | JSON + Schema 校验 |
| 并发请求 | 不支持（单进程阻塞） | uvicorn 异步，天然支持并发 |
| 文档 | 读源码注释 | `/docs` Swagger 自动生成 |
| 监控 | 看控制台 | 结构化日志 + 响应头 + 可接入 ELK |

---

## 练习 4：Middleware — 请求日志 + 耗时统计

```python
@app.middleware("http")
async def log_and_time_middleware(request: Request, call_next):
    t0 = time.time()

    # ── before: 请求进入 ──
    logger.info(f"→ {request.method} {request.url.path} | client={...}")

    # ── 执行真正的 handler ──
    response = await call_next(request)

    # ── after: 请求完成 ──
    latency = (time.time() - t0) * 1000
    log_level = logging.WARNING if latency > 3000 else logging.INFO
    logger.log(log_level,
        f"← {request.method} {request.url.path} | "
        f"status={response.status_code} | latency={latency:.1f}ms")

    # 注入自定义响应头
    response.headers["X-Response-Time-ms"] = f"{latency:.1f}"
    return response
```

**Middleware 的核心思想：AOP（面向切面编程）**

```
没有 Middleware 时：
  /chat 函数里写日志
  /health 函数里写日志
  /admin 函数里写日志
  → 每个 handler 重复同样的日志代码

有了 Middleware 之后：
  Middleware 包在所有 handler 外面
  → 日志逻辑写一次，所有请求自动生效
  → handler 只关心业务逻辑
```

**`await call_next(request)` 是关键**——它把控制权交给下一个中间件（或最终的 handler），等它执行完再回来。这让你能在 handler 执行前后各插一段逻辑。

**扩展方向**：

| 扩展 | 做法 |
|------|------|
| 统计 QPS | 按 path 分组计数，每分钟打印 |
| 慢请求告警 | `latency > 5000ms` 标 ERROR，触发钉钉/飞书通知 |
| P95 延迟面板 | 把每次 latency 写入 Prometheus histogram |
| 限流 | 在 before 阶段检查 Redis 计数器，超限直接返回 429 |

---

## 练习 5：BackgroundTasks — 异步写日志，不阻塞响应

```python
@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    # ... RAG 逻辑 ...
    latency_ms = ...

    # 日志写入放入后台任务 → HTTP 响应先返回
    background_tasks.add_task(
        _save_chat_log,
        query=req.query,
        answer=result["answer"][:100],
        latency_ms=latency_ms,
        session_id=req.session_id,
    )
    return response   # ← 用户收到响应
                      #    此时 _save_chat_log 还未开始执行
                      #    它在响应返回后才运行


def _save_chat_log(query, answer, latency_ms, session_id):
    """响应返回后执行——用户不感知文件 I/O 延迟"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "chat_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({...}, ensure_ascii=False) + "\n")
```

**执行时序**：

```
客户端                        服务端
  │                             │
  ├─ POST /chat ──────────────→ │
  │                             ├─ t0: 开始处理
  │                             ├─ RAG 检索
  │                             ├─ 组装响应
  │                             ├─ background_tasks.add_task(_save_chat_log)
  │                             │    ↑ 只是把任务放入队列，不执行
  │                             ├─ return response ──→ HTTP 响应发送中
  │←── 200 OK + JSON ──────────┤
  │                             ├─ _save_chat_log() 开始执行
  │                             ├─ 写 JSONL 文件
  │                             └─ 完成
  │
  └─ 用户看到回答（不感知日志写入的延迟）
```

**适用场景 vs 不适用场景**：

| 适用（BackgroundTasks） | 不适用（用消息队列） |
|------------------------|---------------------|
| 写日志 / 审计记录 | 订单支付回调 |
| 更新统计计数 | 发送验证码短信 |
| 清理临时文件 | 重要数据写入数据库 |
| 发送非关键通知 | 需要保证一定执行成功的操作 |

> **关键限制**：BackgroundTasks 不保证执行成功——如果服务在响应返回后、后台任务执行前崩溃，任务就丢了。重要操作用 Celery / Kafka / Redis Queue。

---

## Q1：FastAPI 包装 Agent 和直接跑 Python 脚本有什么区别？为什么必须做这个转换？

**一句话**：CLI 脚本是"给自己看的"，HTTP API 是"给别人用的"——FastAPI 加上 Schema 校验、错误处理、并发支持、自动文档，让 Agent 从开发工具变成服务产品。

**核心区别**：

| 维度 | CLI 脚本 | FastAPI 服务 |
|------|---------|-------------|
| 调用方 | 开发者本人 | 前端/App/第三方系统 |
| 并发 | 单进程，一次一个 | uvicorn 异步，天然并发 |
| 输入校验 | 手写 if/else | Pydantic Field 自动 422 |
| 错误处理 | traceback 打印到控制台 | JSON 错误响应 + 日志持久化 |
| 文档 | 读源码注释 | `/docs` Swagger 交互式文档 |
| 部署 | `python x.py` | Docker + K8s + 负载均衡 |
| 监控 | 看终端 | 结构化日志 + Prometheus + ELK |

**面试话术**："Day 11 不是'学 FastAPI 语法'——语法在 `s01_basic.py` 里已经学过了。Day 11 的核心是把 Agent 从'能跑通'升级到'能部署'：Schema 约束让前后端对接有契约，Middleware 让横切关注点从业务代码中剥离，日志结构化让出问题时可追溯。面试官问'你的 Agent 怎么上线'，这就是答案的第一段。"

---

## Q2：Pydantic 的 `Field(ge=1, le=10)` 校验和手写 `if top_k > 10: return 400` 有什么区别？

**一句话**：手写校验散落在每个函数里，改一个规则要翻遍所有代码；Pydantic Field 把校验规则和字段定义绑定在一起，改一处全局生效，且自动生成文档。

**对比**：

```python
# ── 手写方式（Day 1-10 风格）──
async def chat(query: str, top_k: int):
    if not query:
        return {"error": "query 不能为空"}   # 返回格式不统一
    if top_k < 1 or top_k > 10:
        return {"error": "top_k 超出范围"}    # 与上面格式不一致
    # ... 业务逻辑 ...

# ── Pydantic 方式（Day 11 风格）──
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

async def chat(req: ChatRequest):
    # query 和 top_k 已经被 FastAPI 自动校验过了
    # 非法参数进不了这个函数，直接返回 422 + 结构化错误信息
    # ... 业务逻辑 ...
```

**Pydantic 的优势**：

| 手写校验 | Pydantic Field |
|---------|---------------|
| 校验逻辑散落在每个函数 | 校验规则和字段定义在一起 |
| 返回格式不统一 | 422 格式统一，FastAPI 内置 |
| 改 top_k 上限要全局搜索 | 改 Field(le=20) 一处生效 |
| 文档需要手写 | `/docs` 自动展示约束 |
| IDE 不知道参数类型 | `req.top_k: int` 有类型提示 |

**面试话术**："校验逻辑放在哪里，决定了系统的可维护性。Pydantic 把校验规则从业务代码中剥离到 Schema 定义层——这是 Day 8 '结构化输出'思想在 API 层的延伸。`Field(ge=1, le=10)` 同时承担三个角色：运行时校验、IDE 类型提示、Swagger 文档生成。手写 if 只能做到第一个。"

---

## Q3：`response_model=ChatResponse` 做了什么？不加会怎样？

**一句话**：`response_model` 让 FastAPI 在返回 JSON 之前自动校验响应结构——字段缺失、类型错误都会触发 500，防止"悄悄返回了错误格式的数据"。

**三种写法对比**：

```python
# ① 不加 response_model —— 无校验，返回什么都行
@app.post("/chat")
async def chat(req: ChatRequest):
    return {"answer": "你好", "latency": 100}  # 字段名拼错也不会报错

# ② response_model=ChatResponse —— 自动校验 + 过滤
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return ChatResponse(answer="你好", sources=[], latency_ms=100, session_id="x")
    # → 返回的 JSON 中只包含 ChatResponse 定义的字段

# ③ 返回 dict + response_model —— 自动校验 + 转换
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return {"answer": "你好", "sources": [], "latency_ms": 100, "session_id": "x"}
    # → FastAPI 自动把 dict 转成 ChatResponse，校验通过后返回 JSON
    # → 多了一个字段 "extra_field" 会被静默丢弃（生产隐患！）
```

**面试话术**："`response_model` 是 API 的'类型安全网'——你改了 ChatResponse 的字段名但忘了改 return 语句，不加 `response_model` 就是静默 bug，加了就当场报错。FastAPI 的设计哲学是 fail fast：与其让前端拿到格式错误的数据，不如在服务端就报 500 让你立刻发现。"

---

## Q4：Middleware 的 `await call_next(request)` 为什么是核心？它的执行模型是怎样的？

**一句话**：`await call_next(request)` 把控制权交给下一个中间件（或最终的路由 handler），等它执行完再回来——这让每个中间件都能在请求前后各插一段逻辑。

**执行模型（洋葱模型）**：

```
请求进入 ──────────────────────────────────────────→ 响应返回
  │                                                    ↑
  ▼                                                    │
┌──────────────────────────────────────────────────────┴─┐
│ Middleware A                                            │
│   before: t0 = time.time()                             │
│   response = await call_next(request)  ───────┐        │
│   after:  latency = time.time() - t0  ←───────┘        │
│   response.headers["X-Time"] = str(latency)             │
└─────────────────────────────────────────────────────────┘
  │                                                    ↑
  ▼                                                    │
┌──────────────────────────────────────────────────────┴─┐
│ Middleware B（如果有）                                   │
│   before: 检查 API Key                                 │
│   response = await call_next(request)  ───────┐        │
│   after:  无操作                          ←────┘        │
└─────────────────────────────────────────────────────────┘
  │                                                    ↑
  ▼                                                    │
┌──────────────────────────────────────────────────────┴─┐
│ 路由 Handler（/chat 或 /health）                         │
│   执行业务逻辑，返回 Response                            │
└─────────────────────────────────────────────────────────┘
```

**没有 Middleware 的替代方案 vs 有 Middleware**：

| 方案 | 代码重复 | 遗漏风险 |
|------|---------|---------|
| 每个 handler 手写日志 | 每个函数重复 10 行代码 | 新增路由忘了加 → 静默缺失 |
| 装饰器 `@log_request` | 每个函数加一行 | 新增路由忘了加 → 静默缺失 |
| Middleware | 写一次，全局生效 | 不可能遗漏 |

**面试话术**："Middleware 的洋葱模型是 Web 框架的经典设计——FastAPI 的 Middleware 基于 Starlette 的 ASGI 中间件协议。`await call_next(request)` 实际上是在调 ASGI app 的 `__call__` 方法，它接收 request 返回 response。理解了这个，你就理解了一切 Web 中间件的本质。"

---

## Q5：BackgroundTasks vs `asyncio.create_task()` vs Celery，什么场景用什么？

**一句话**：BackgroundTasks 用于"不重要但想异步"的轻量任务，`asyncio.create_task()` 用于"需要等待结果"的并发，Celery 用于"必须保证执行"的重任务。

**三者的本质区别**：

| 维度 | BackgroundTasks | `asyncio.create_task()` | Celery |
|------|----------------|------------------------|--------|
| 执行时机 | 响应返回**后** | 立即执行（并发） | 独立 Worker 进程 |
| 等待结果 | 不能 | 可以 `await` | 用 callback / 轮询 |
| 崩溃安全 | ❌ 服务崩了就丢了 | ❌ 服务崩了就丢了 | ✅ 消息持久化，重试 |
| 适用场景 | 写日志、更新计数 | 并行调 3 个 LLM | 订单处理、发邮件 |
| 复杂度 | 零（FastAPI 内置） | 低（标准库） | 高（需额外部署） |

**时序对比**：

```
BackgroundTasks:
  Client ─req─→ Handler ─resp─→ Client
                                  └→ _save_chat_log()

asyncio.create_task():
  Client ─req─→ Handler ──┬── resp ──→ Client
                          └── task 同时跑

Celery:
  Client ─req─→ Handler ─resp─→ Client
                 │
                 └→ Redis Queue ──→ Worker 进程 ──→ 执行
```

**面试话术**："BackgroundTasks 是 FastAPI 对 Starlette Background tasks 的封装，本质上就是 `asyncio.create_task()` 延迟到响应返回后执行。它的定位是'不重要的异步工作'——日志丢了无所谓，但订单不能丢。所以 Day 11 用 BackgroundTasks 写日志是对的，但不要用它处理支付回调。什么时候换 Celery？当你说'这个操作必须成功'的那一刻。"

---

## Q6：`simulate_rag_query()` 是模拟实现，生产环境需要换什么？怎么换？

**一句话**：把 `simulate_rag_query()` 替换为 `full_rag_agent.py` 的 `RAGAgent` 或 `car_advisor_agent.py` 的 Agent——而且 Agent 实例要在模块级别初始化一次（全局单例），不能每次请求都重建索引。

**替换步骤**：

```python
# ── 当前代码（Day 11 模拟）──
def simulate_rag_query(query: str, top_k: int = 3) -> dict:
    """关键词匹配 + 拼字符串 → 模拟 RAG"""
    ...

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    result = simulate_rag_query(req.query, top_k=req.top_k)


# ── 生产替换（一行改）──
from full_rag_agent import RAGAgent

# 模块级别初始化一次（全局单例）
rag_agent = RAGAgent(DATA_DIR)   # 离线建库：加载 → Embedding → FAISS → BM25 → Reranker
# ↑ 放在模块顶层，FastAPI 启动时执行一次
#   uvicorn 多 worker 模式下每个 worker 执行一次（可接受）

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    result = rag_agent.chat(req.query, top_k=req.top_k)
    # ↑ 一行换，API 层不需要改任何其他代码
```

**为什么不能在 `chat()` 函数里初始化 Agent**：

```
@app.post("/chat")
async def chat(req: ChatRequest):
    agent = RAGAgent(DATA_DIR)  # ← 每次都重新加载数据、建索引！
    return agent.chat(req.query)
```

每次请求都重建 RAGAgent → 加载 JSON → 跑 Embedding → 建 FAISS 索引 → 建 BM25 倒排 → 加载 Reranker 模型。一个 `/chat` 请求光初始化就要 5-10 秒。

**面试话术**："API 层和业务逻辑层的分离在这里体现得很清晰——API 层不关心 RAGAgent 内部怎么实现的，只关心入参（query, top_k）和出参（answer, sources）。这种分离让'替换实现'只需改一行 import，不用动路由逻辑。这就是 Pydantic Schema 作为'数据契约'的价值——契约不变，两边可以独立演进。"

---

## Q7：Day 11 的 Middleware + BackgroundTasks + Schema 校验，各自属于什么架构层次？

**一句话**：Schema 校验是"边界守卫层"（输入输出门禁），Middleware 是"横切关注层"（所有请求共享的逻辑），BackgroundTasks 是"副作用隔离层"（非关键操作异步化）——三层各司其职，让 handler 只关心业务逻辑。

**架构分层图**：

```
                    ┌─────────────────────────┐
  请求进来 ───────→ │  边界守卫层（Schema）      │
                    │  ChatRequest 校验        │
                    │  ChatResponse 校验        │
                    │  → 422 / 500 快速失败    │
                    └───────────┬─────────────┘
                                │ 合法数据
                    ┌───────────▼─────────────┐
                    │  横切关注层（Middleware）   │
                    │  日志、耗时、限流、认证    │
                    │  每个请求自动经过          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  业务逻辑层（Handler）      │
                    │  POST /chat              │
                    │  只关心：查RAG → 组响应   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  副作用隔离层（Background） │
                    │  写日志、发通知、更新计数  │
                    │  响应返回后才执行          │
                    └─────────────────────────┘
```

**面试话术**："Day 11 的四层架构和 Day 10 的三层 Prompt 架构是同一种设计思想——关注点分离。不同层次有不同的稳定性和变更频率：Schema 变更频率低（接口契约），Middleware 变更频率中等（加限流/加认证），Handler 变更频率高（业务迭代），Background 变更频率最低（日志格式很少改）。这种分层让每层的修改不影响其他层，是工程化的核心原则。"

---

## Q8：FastAPI 的 `tags=["系统"]` 和 `tags=["对话"]` 有什么用？

**一句话**：`tags` 是 FastAPI 提供的路由分组机制——在 `/docs` Swagger 页面中按 tag 分组展示接口，让 API 文档有结构层次。

```python
@app.get("/health", tags=["系统"])     # → Swagger 中归入"系统"组
@app.post("/chat", tags=["对话"])      # → Swagger 中归入"对话"组
```

**Swagger 页面效果**：

```
系统
  GET  /health          健康检查接口

对话
  POST /chat            一次完整的 RAG 问答
```

没有 tags 时所有接口混在 "default" 组下，20+ 个接口时前端找不到该调哪个。生产项目通常按模块分组：`["系统"]`、`["对话"]`、`["会话管理"]`、`["管理后台"]`、`["评测"]`。

---

## Q9：Day 11 的日志从 `print()` 换成 `logging` + JSONL 文件，解决了什么问题？

**一句话**：`print()` 只有人眼能消费，`logging` + JSONL 让日志可搜索、可统计、可接入日志平台——这是从"自己看"到"运维看"的升级。

**对比**：

```python
# Day 1-10 风格
print(f"[INFO] {request.method} {request.url.path} | latency={latency}ms")
# → 控制台刷屏，无法搜索"今天下午3点哪个请求超过3秒"

# Day 11 风格
logger.info(f"← POST /chat | status=200 | latency=123.4ms")
# + 响应头注入: X-Response-Time-ms: 123.4
# + JSONL 文件: {"timestamp":"2026-06-24T15:00:00","query":"...","latency_ms":123.4}
```

| 需求 | print | logging + JSONL |
|------|-------|-----------------|
| 统计今日调用量 | 人肉数 | `wc -l logs/chat_history.jsonl` |
| 找慢查询（>3s） | 不可能 | `grep` 或 `jq 'select(.latency_ms>3000)'` |
| 接入 ELK/Grafana | 解析文本 → 噩梦 | JSON → 原生支持 |
| 按级别过滤 | 不支持 | `logger.setLevel(WARNING)` |
| 多文件输出 | 不支持 | `FileHandler` + `StreamHandler` |

**面试话术**："`print()` 到 `logging` 的升级看起来小，实际上反映了工程思维的转变——Day 1-10 是给自己开发用的，Day 11 是给运维团队用的。生产系统的日志需要三个特性：结构化（可被工具解析）、分级（ERROR 触发告警，DEBUG 开发用）、持久化（出问题时有据可查）。JSONL 格式一行一条 JSON，既可以 `tail -f`，也可以 `jq` 过滤，是生产和开发兼顾的最简方案。"

---

## Q10：Day 11 的输出从"终端打印"变成了"HTTP JSON 响应"，这对调试和测试有什么影响？

**一句话**：终端打印只能人眼看，HTTP 响应可以自动化测试（pytest + httpx）、可以抓包分析（Charles/Wireshark）、可以被 CI 自动化验证——可测试性是工程化的分水岭。

**自动测试示例**：

```python
# 不需要启动服务，直接用 ASGI 客户端测试
from httpx import AsyncClient, ASGITransport
from main import app
import pytest

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data

@pytest.mark.asyncio
async def test_chat_empty_query():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/chat", json={"query": "", "top_k": 3})
        assert resp.status_code == 422   # Pydantic 校验自动拦截

@pytest.mark.asyncio
async def test_chat_valid():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/chat", json={"query": "小米SU7多少钱", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["sources"]) > 0
```

**对比 Day 1-10 无法自动测试的困境**：

```python
# Day 5 的测试方式：手动观察终端输出
agent.invoke({"messages": [HumanMessage(content="小米SU7多少钱")]})
# → "嗯，输出看起来对" → 没有 pass/fail，没法放进 CI
```

**面试话术**："Day 11 把 Agent 包装成 HTTP API 后，最大的隐性收益不是'能上网了'，而是'能自动测试了'。`httpx.ASGITransport` 可以在不启动真实服务器的情况下测试 FastAPI app——这意味着你可以把 Agent 的回归测试放进 CI pipeline，每次改 Prompt 后自动跑 50 条评测 case，成功率低于 85% 就阻断合并。这是 Day 10 评测体系从'手动跑'到'自动化'的关键一步。"

---

## Day 11 从 CLI 到 API 的五个思维转变

| # | 转变 | 说明 |
|---|------|------|
| 1 | 用户变了 | 从"开发者自己"变成"前端/App/第三方"——需要 Schema 契约 |
| 2 | 错误处理变了 | 从"抛异常看 traceback"变成"返回结构化 JSON 错误" |
| 3 | 并发要求变了 | 从"一次一个"变成"同时多人调用"——uvicorn 异步处理 |
| 4 | 可观测性变了 | 从"看 print"变成"结构化日志 + 响应头 + JSONL 持久化" |
| 5 | 测试方式变了 | 从"人眼看"变成"pytest + httpx 自动化" |

---

### Day 11 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | FastAPI 包装 Agent 和直接跑 Python 脚本的五个核心区别是什么？ | □ |
| 2 | Pydantic `Field(ge=1, le=10)` 校验 vs 手写 if 的三个优势？ | □ |
| 3 | `response_model=ChatResponse` 的作用？不加会有什么隐患？ | □ |
| 4 | Middleware 的洋葱模型是怎么执行的？`await call_next(request)` 的本质是什么？ | □ |
| 5 | BackgroundTasks vs asyncio.create_task() vs Celery，各自的适用场景？ | □ |
| 6 | 生产环境里 `simulate_rag_query()` 怎么替换？为什么 Agent 要全局初始化？ | □ |
| 7 | Day 11 的四层架构（Schema / Middleware / Handler / Background）各解决什么问题？ | □ |
| 8 | `tags=["系统"]` 和 `tags=["对话"]` 的作用？Swagger 里怎么呈现？ | □ |
| 9 | 日志从 `print()` 换成 `logging` + JSONL 解决了什么工程问题？ | □ |
| 10 | Python 脚本的测试方式 vs FastAPI 的测试方式有什么区别？`httpx.ASGITransport` 的价值？ | □ |
