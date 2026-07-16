# Day 3 — FastAPI 路由注册与前后端打通

> **今日目标**：用 FastAPI 的 Depends 依赖注入系统把 Day 1/Day 2 的模块组装起来，实现 8 个可调用的 HTTP 端点。这是 Phase 2 的收尾——写完就能用 curl 或浏览器验证整套系统。

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [FastAPI 依赖注入原理](#2-fastapi-依赖注入原理)
3. [deps.py：依赖组装工厂](#3-depspy依赖组装工厂)
4. [sessions.py：6 个会话端点](#4-sessionspy6-个会话端点)
5. [chat.py：流式与非流式双模式](#5-chatpy流式与非流式双模式)
6. [main.py：路由注册](#6-mainpy路由注册)
7. [完整请求链路推演](#7-完整请求链路推演)
8. [初学者常见疑问](#8-初学者常见疑问)
9. [面试模拟问答](#9-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `api/deps.py` | ~45 | 定义 4 个 Depends 依赖函数，组装认证+限流+会话管理 |
| `api/routes/sessions.py` | ~85 | 6 个 RESTful 端点：增删改查+历史 |
| `api/routes/chat.py` | ~90 | 1 个 POST 端点：stream=true→SSE / false→JSON |
| `main.py`（改） | +6 | 注册两个路由模块 |

---

## 2. FastAPI 依赖注入原理

### 2.1 没有依赖注入的代码长什么样

```python
# 每个路由函数都要重复这些步骤
@router.post("/chat")
async def chat(body: ChatReq, request: Request):
    # 1. 鉴权
    cred = request.headers.get("Authorization")
    user_id = verify_token(cred)
    if not user_id:
        raise HTTPException(401)

    # 2. 取 Redis
    redis = request.app.state.redis
    if not redis:
        raise HTTPException(503)

    # 3. 限流
    limiter = SlidingWindowRateLimiter(redis)
    if not await limiter.is_allowed(user_id):
        raise HTTPException(429)

    # 4. 会话管理
    session_mgr = SessionManager(redis)

    # 5. 真正的业务逻辑...
```

7 个端点，每个端点重复 1-4 步。代码膨胀 4 倍，改一个步骤要改 7 处。

### 2.2 FastAPI Depends 让代码变成这样

```python
@router.post("/chat")
async def chat(
    body: ChatReq,
    user_id: str = Depends(check_rate_limit),          # 自动执行认证+限流
    session_mgr: SessionManager = Depends(get_session_manager),  # 自动构造
):
    # 直接拿 user_id 和 session_mgr 用，前面所有检查全自动完成
    await session_mgr.add_message(body.session_id, "user", body.query)
```

**Depends 做了什么？**

```
1. 路由函数定义了 Depends(check_rate_limit) → user_id
2. FastAPI 检查 check_rate_limit 的函数签名
3. 发现 check_rate_limit 又依赖了 Depends(get_current_user_auto) 和 Depends(get_rate_limiter)
4. 又发现 get_rate_limiter 依赖了 Depends(get_redis_client)
5. FastAPI 构建依赖图，按拓扑序执行
6. 每次执行结果自动匹配下一层参数名
```

这就像工厂流水线：上一道工序的输出自动流到下一道。

### 2.3 核心概念：洋葱模型

```
请求进入 FastAPI
  ┌──────────────────────────────────────┐
  │ get_redis_client (最外层)             │  ← 从 app.state 取连接
  │  ┌─────────────────────────────────┐ │
  │  │ get_rate_limiter                │ │  ← 构造限流器
  │  │  ┌────────────────────────────┐ │ │
  │  │  │ get_current_user_auto      │ │ │  ← 提取 API Key/JWT
  │  │  │  ┌───────────────────────┐ │ │ │
  │  │  │  │ check_rate_limit     │ │ │ │  ← 执行限流检查
  │  │  │  │  ┌─────────────────┐ │ │ │ │
  │  │  │  │  │ 路由函数体       │ │ │ │ │  ← 业务逻辑
  │  │  │  │  └─────────────────┘ │ │ │ │
  │  │  │  └───────────────────────┘ │ │ │
  │  │  └────────────────────────────┘ │ │
  │  └─────────────────────────────────┘ │
  └──────────────────────────────────────┘
```

外层先执行，内层后执行。每层可以访问外面所有层的结果。

**同一个请求中，相同依赖只执行一次。** 如果一个端点的两个参数都 `Depends(get_redis_client)`，FastAPI 只会从 `app.state` 取一次，第二次直接从缓存拿结果。

---

## 3. deps.py：依赖组装工厂

### 3.1 四个依赖函数

```python
# 1. 基础：取 Redis 连接
async def get_redis_client(request: Request) -> Redis:
    redis_client = request.app.state.redis
    if redis_client is None:
        raise HTTPException(status_code=503, detail="服务暂不可用")
    return redis_client

# 2. 业务：构造 SessionManager
async def get_session_manager(
    redis_client: Redis = Depends(get_redis_client),
) -> SessionManager:
    return SessionManager(redis_client)

# 3. 业务：构造限流器
async def get_rate_limiter(
    redis_client: Redis = Depends(get_redis_client),
) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(redis_client)

# 4. 组合：认证 + 限流
async def check_rate_limit(
    user_id: str = Depends(get_current_user_auto),   # 先认证
    rate_limiter = Depends(get_rate_limiter),         # 再取限流器
) -> str:
    allowed = await rate_limiter.is_allowed(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail="请求过于频繁")
    return user_id
```

### 3.2 为什么"先认证后限流"？

```
如果先限流后认证：
  攻击者用随机 API Key 狂刷 → 每个随机 Key 都被 ZADD 记录
  → Redis 里堆积数百万个 rate_limit:fakeKey1, rate_limit:fakeKey2...
  → 内存被耗尽 ← 这就是"内存消耗型 DoS"

如果先认证后限流：
  随机 Key → 认证失败 → 401 → 根本没走到限流逻辑
  → Redis 干干净净
```

未认证的请求不应该消耗任何系统资源。

### 3.3 get_redis_client 为什么返回 503 而不是让请求继续

如果 Redis 在 `lifespan` 中连接失败，`app.state.redis = None`。后续请求如果不检查直接 `await redis.hgetall(key)`，会抛出 `AttributeError` 或连接错误，最终被全局异常处理捕获变成 500。

500 = "服务器内部错误" = 不知道发生了什么。503 = "服务暂不可用" = 运维人员一看就知道是下游服务挂了。**精确的状态码让排查问题快 10 倍。**

---

## 4. sessions.py：6 个会话端点

### 4.1 端点总览

```
POST   /sessions              201 Created      创建会话
GET    /sessions              200 OK           会话列表
GET    /sessions/{id}         200 OK           会话详情
PATCH  /sessions/{id}         200 OK           重命名
DELETE /sessions/{id}         204 No Content   删除
GET    /sessions/{id}/history 200 OK           消息历史
```

### 4.2 归属校验模式

```python
@router.get("/{session_id}")
async def get_session(session_id, user_id, session_mgr):
    session = await session_mgr.get_session(session_id)
    if session is None or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
```

**为什么返回 404 而不是 403？**

```
用户 A 的 session_id = "abc123"
用户 B 试图访问 GET /sessions/abc123

返回 403  → "你不能看这个会话" → B 知道这个 session_id 确实存在
返回 404  → "会话不存在"       → B 不知道是真的不存在还是别人的

这是安全最佳实践：不让攻击者探测哪些资源 ID 是有效的。
```

**对拥有者也是 404**——如果用户误输入了自己的其他 session_id，看到的是 "不存在" 而不是 "别人的"。

### 4.3 为什么每个端点都注入 check_rate_limit

```
POST /sessions  → Depends(check_rate_limit) → 认证 + 限流
GET  /sessions  → Depends(check_rate_limit) → 认证 + 限流
GET  /{id}      → Depends(check_rate_limit) → 认证 + 限流
...
```

每个端点都有保护。不加的话——比如 GET 列表没有限流——攻击者虽然不能发消息，但可以反复请求列表把 Redis 的 SMEMBERS 打爆。

### 4.4 204 响应 —— 删除成功不返回体

```python
@router.delete("/{session_id}", status_code=204)
async def delete_session(...):
    return JSONResponse(status_code=204, content=None)
```

HTTP 204 No Content：操作成功，但响应体为空。比返回 `{"success": true}` 更精确——删除操作本来就没有"结果数据"可展示。

---

## 5. chat.py：流式与非流式双模式

### 5.1 占位回答的作用

```python
async def _placeholder_generator(query: str):
    yield {"type": "source", "documents": []}
    placeholder = f"收到您的问題：「{query}」。（这是占位回答...）"
    for char in placeholder:
        yield {"type": "token", "content": char}
        await asyncio.sleep(0.03)
```

这不是偷懒——这是在验证 SSE 全链路是否正常工作：

```
占位生成器 → sse_generator 包装 → StreamingResponse
    ↓             ↓                    ↓
 产生字典     格式化 SSE 字符串       HTTP chunked 发送

如果 curl -N 能逐字看到"收→到→您→的→问→题→..."
说明整条链路没有阻塞、缓冲、断掉。
Phase 3 换上真 Agent，这个验证的价值就体现出来了。
```

### 5.2 流式模式的 _stream_and_save 内部函数

这是 chat.py 中最复杂的一段：

```python
async def _stream_and_save():
    full_parts = []
    async for sse_str in sse_generator(_placeholder_generator(body.query)):
        # 从 SSE 字符串中提取每个 token
        if '"type":"token"' in sse_str:
            prefix = '"content":"'
            idx = sse_str.find(prefix)
            if idx != -1:
                end = sse_str.find('"', idx + len(prefix))
                if end != -1:
                    full_parts.append(sse_str[idx + len(prefix):end])
        yield sse_str  # 先发给客户端

    # 流结束后，拼接完整回答并保存
    full_answer = "".join(full_parts)
    if full_answer:
        await session_mgr.add_message(session_id, "assistant", full_answer)
```

**为什么必须用内部函数？**

`StreamingResponse` 一旦返回，FastAPI 就接管了控制权。这个请求的生命周期变成：FastAPI → 生成器 yield → 客户端 → 生成器 yield → ... → 生成器结束 → 请求结束。

如果你想在 `return StreamingResponse(...)` 之后写代码：

```python
return StreamingResponse(_stream_and_save(), ...)
# ↓ 这行永远不会执行！因为生成器还在生成，函数已经 return 了
await session_mgr.add_message(...)  # 永远等不到
```

所以必须把"保存到 Redis"的逻辑**嵌入到生成器内部**，在 done 事件之前执行。

### 5.3 字符串解析 vs JSON 解析

```python
# 当前实现：字符串查找
if '"type":"token"' in sse_str:
    # 手动提取 content 字段

# 如果改用 JSON 解析：
data = json.loads(sse_str[6:])  # 去掉 "data: " 前缀，解析 JSON
if data.get("type") == "token":
    full_parts.append(data["content"])
```

**当前实现更快**：每个 token（约 40 字节）都做一次 `json.loads` 是浪费——字符串查找只需要扫描几个字符，而 JSON 解析需要完整的词法分析。40 个 token × 30ms = 1.2s 的流式响应中，每个 token 少做一次 JSON 解析，累积节省可观。**但可维护性差。** Phase 3 接入真 Agent 时建议改成 JSON 解析。

### 5.4 非流式模式的逻辑

```python
start = time.time()
full_answer = ""
async for event in _placeholder_generator(body.query):
    if event.get("type") == SSE_TOKEN:
        full_answer += event.get("content", "")

latency_ms = (time.time() - start) * 1000
await session_mgr.add_message(body.session_id, "assistant", full_answer)

return ChatResp(answer=full_answer, sources=[], latency_ms=..., session_id=...)
```

非流式模式下代码直接消费生成器（不通过 sse_generator），收集全部 token 后一次性返回。保存到 Redis 在 return 之前——没有流式模式的那种控制权问题。`latency_ms` 记录了从接收到返回的耗时。

---

## 6. main.py：路由注册

只改了 2 处：

```python
from backend.api.routes import chat, sessions  # 新增 import

# ... 中间件、全局异常处理、基础路由不变 ...

app.include_router(chat.router)       # 新增
app.include_router(sessions.router)   # 新增
```

`include_router` 的行为：
- `/chat` 端点（挂载在 chat.router 上，prefix="/chat"）→ `POST /chat`
- `/sessions` 端点族（挂载在 sessions.router 上，prefix="/sessions"）→ `POST/GET/PATCH/DELETE /sessions/...`

两个 router 是独立的对象，各自定义自己的 prefix、tags、路由。main.py 只负责"挂载"。

---

## 7. 完整请求链路推演

```
用户: curl -N -X POST http://localhost:8000/chat \
        -H "Authorization: Bearer sk-dev-user-001" \
        -d '{"query":"推荐SUV","session_id":"abc123","stream":true}'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TCP 连接建立 → HTTP 请求到达 FastAPI
│
├─ CORS 中间件处理 → Origin检查 → 放行
│
├─ 依赖解析链启动:
│   │
│   ├─ get_redis_client(request)
│   │   └─ request.app.state.redis → 已连接的 Redis 客户端
│   │
│   ├─ get_rate_limiter(redis_client)
│   │   └─ 返回 SlidingWindowRateLimiter 实例
│   │
│   ├─ get_current_user_auto(credentials)
│   │   ├─ Authorization header → "Bearer sk-dev-user-001"
│   │   ├─ HTTPBearer 提取 "sk-dev-user-001"
│   │   ├─ parse_api_keys() 匹配 → "user_001"
│   │   └─ 返回 "user_001"
│   │
│   ├─ check_rate_limit(user_id="user_001", rate_limiter)
│   │   ├─ rate_limiter.is_allowed("user_001")
│   │   │   └─ Redis EVALSHA <sha> 1 rate_limit:user_001 <now> <window> <max> <ttl>
│   │   │       └─ Lua: ZREMRANGEBYSCORE → ZCARD=3 → 未超限 → ZADD → return 1
│   │   ├─ True → 放行
│   │   └─ 返回 "user_001"
│   │
│   ├─ get_session_manager(redis_client)
│   │   └─ 返回 SessionManager 实例
│   │
│   └─ chat(body=ChatReq, user_id="user_001", session_mgr)
│       │
│       ├─ session_mgr.add_message("abc123", "user", "推荐SUV")
│       │   └─ Pipeline: RPUSH + HINCRBY + EXPIRE * 3 → 1 次网络往返
│       │
│       └─ body.stream == True → StreamingResponse(_stream_and_save())
│           │
│           ├─ _placeholder_generator("推荐SUV")
│           │   ├─ yield {"type":"source","documents":[]}
│           │   └─ for char in placeholder: yield {"type":"token","content":char}
│           │
│           ├─ sse_generator(generator)
│           │   ├─ format_sse(event) → 'data: {...}\n\n'
│           │   └─ yield sse_str (逐帧)
│           │
│           └─ _stream_and_save 内部:
│               ├─ 每个 token 拼接到 full_parts
│               ├─ done 后拼接完整回答
│               └─ session_mgr.add_message("abc123","assistant",full_answer)
│
├─ FastAPI 发送 HTTP 响应:
│   HTTP/1.1 200 OK
│   Content-Type: text/event-stream
│   Transfer-Encoding: chunked
│   Cache-Control: no-cache
│
│   chunk1: data: {"type":"source","documents":[]}\n\n
│   chunk2: data: {"type":"token","content":"收"}\n\n
│   chunk3: data: {"type":"token","content":"到"}\n\n
│   ...
│   chunkN: data: {"type":"done","total_tokens":40}\n\n
│   chunk0: (length=0, 流结束)
│
└─ curl -N 逐行显示:
    data: {"type":"source","documents":[]}
    data: {"type":"token","content":"收"}
    data: {"type":"token","content":"到"}
    data: {"type":"token","content":"您"}
    ...
    data: {"type":"done","total_tokens":40}
```

---

## 8. 初学者常见疑问

**Q: `Depends` 是怎么识别该调用哪个函数的？**

FastAPI 在应用启动时检查路由函数的签名。每个 `Depends(xxx)` 是 `Depends` 类的实例，包含了被依赖函数 `xxx` 的引用。FastAPI 通过 `inspect.signature()` 分析 `xxx` 的参数，继续递归找子依赖，一层层构建依赖图。

**Q: 同一个依赖被多次注入时，执行几次？**

执行一次，结果缓存，后续直接用缓存值。缓存 key 是依赖函数的标识 + 参数组合。但这是**请求级缓存**——不同请求之间不共享，隔离是安全的。

**Q: 为什么 check_rate_limit 返回 user_id，而不是只返回 True？**

因为路由函数需要 user_id。如果 `check_rate_limit` 只返回 True，路由函数就要再调用一次 `get_current_user_auto`——这个依赖已经被执行过了，但路由函数拿不到结果。直接返回 user_id，让路由同时拿到"检查通过"和"用户是谁"两个信息。

**Q: `StreamingResponse` 的 `media_type="text/event-stream"` 如果不写会怎样？**

FastAPI 默认 `media_type` 是根据返回内容自动检测的（通常为 `text/plain` 或 `application/json`）。浏览器 JavaScript 的 `new EventSource(url)` 要求 `Content-Type: text/event-stream`，不对的话 EventSource 不会触发 `onmessage` 事件。中间代理也可能因为类型不对而缓冲。

**Q: 归属校验为什么只在获取单个会话时才加，而创建时不加？**

创建会话时 `user_id` 是 `user_id: str = Depends(check_rate_limit)` 注入的——**只能用自己的 ID 创建**。列表接口用同样的机制，只查自己所属的会话。单个会话详情加额外校验是因为 `session_id` 来自 URL 路径——用户可能手动改 URL 尝试访问别人的会话。

**Q: Phase 2 占位回答中 latency_ms 有什么用？**

占位模式测出的 latency 是基准线（~1-2ms 字符串拼接延迟）。Phase 3 接入 LLM 后延迟会跳到 2000-5000ms。前后对比能清晰看到延迟增加来自哪里。`latency_ms` 字段贯穿 Phase 2 → Phase 5，最终可被 Prometheus 采集用于 SLO 监控。

---

## 9. 面试模拟问答

> **Q: 说一下这个项目的 API 设计思路。**

8 个端点分为三组：基础端点（`/`、`/health`）给运维用，会话端点（`/sessions` CRUD）给前端管理对话列表，对话端点（`POST /chat`）是核心——支持流式和非流式双模式。所有端点都通过 FastAPI Depends 统一注入认证和限流，避免每处重复。流式模式用 SSE 协议推 token，非流式一次性返回 JSON，前端可以按需选择。

> **Q: 为什么要同时支持流式和非流式？**

流式（`stream=true`）适合前端界面逐字显示，TTFT（首字延迟）通常 500-2000ms，用户体验好——用户能立即看到回应在生成。非流式（`stream=false`）适合 API 调用、脚本批量处理、或者前端不支持 SSE 的场景（如 SSR 首屏渲染、curl 测试）。两个模式走同一个端点、同一个参数区分，API 设计简洁。

> **Q: 怎么防止用户访问别人的会话？**

两层：第一层——列表和创建通过 Depends 注入 user_id，用户只能拿到自己的数据。第二层——单个会话的详情/改名/删除/历史，先查到会话后比对其 user_id 和当前用户是否一致，不一致返回 404（不是 403）。返回 404 是因为不让攻击者探测"别人的 session_id 是否有效"。

> **Q: 如果 Redis 中途挂了（服务启动后、处理请求中），会怎样？**

限流器有降级策略：Redis 异常 → 返回 True（放行）。SessionManager 没有降级——`redis.RedisError` 会向上传播，最终被 `global_exception_handler` 捕获返回 500。正确的改进：在 `get_redis_client` 中加健康检查（每次请求 ping），或者在 SessionManager 的每个方法中加 try/except + 降级逻辑。这属于 Phase 4 的容错增强。

> **Q: 依赖注入最大的好处是什么？说一个具体的例子。**

可测试性。如果全局单例 `redis = Redis()` 散落在各模块，测试时要 monkey-patch 全局变量，A 测试改了会影响 B 测试。Depends 让每个测试可以直接向路由函数传入 mock 对象：

```python
# 测试代码 (Phase 4 实现)
async def test_chat(mock_redis):
    response = await client.post("/chat", json={"query":"test"})
    # mock_redis 自动注入，不碰真实的 Docker Redis
```

依赖注入 = 把"创建依赖"和"使用依赖"解耦。创建由 Depends 负责，使用由路由函数负责，测试时可以替换创建逻辑。

---

## 附：Phase 2 完成后的文件清单

```
backend/
├── config.py              ✅ 配置中心
├── main.py                ✅ FastAPI 骨架 + 路由注册
├── core/
│   ├── security.py        ✅ 认证 (API Key + JWT)
│   ├── session_manager.py ✅ 会话管理 (Redis CRUD)
│   ├── stream.py          ✅ SSE 协议处理
│   └── resilience.py      ✅ 限流 + 熔断
├── schemas/
│   ├── chat.py            ✅ 对话模型 + SSE 常量
│   └── session.py         ✅ 会话模型
└── api/
    ├── deps.py            ✅ 依赖注入
    └── routes/
        ├── chat.py        ✅ 对话端点 (流式+非流式)
        └── sessions.py    ✅ 会话端点 (CRUD)
```

**11 个文件，8 个端点，完整的 HTTP API 已就绪。** 接下来 Phase 3 用真实 AI 替换占位回答。
