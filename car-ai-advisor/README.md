# 智能汽车导购助手 (Smart Car Advisor)

> **全链路 AI Agent 系统** — 从 RAG 知识库 → ReAct Agent → FastAPI 流式服务 → Redis 会话管理 → Streamlit 前端，一站式垂直落地。
>
> 对标岗位：**AI 应用开发工程师 / 大模型应用开发 / AI Agent 开发**

---

## 项目定位

**一句话**：用 FastAPI + LangGraph + Redis + SSE 构建的智能汽车导购系统，支持流式多轮对话、RAG 混合检索、会话管理和 Docker 一键部署。

**与学习项目的区别**：

| 维度 | agent-dev-project（学习项目） | car-ai-advisor（全链路项目） |
|------|------|------|
| 目的 | 每日一个知识点，独立练习 | 整合所有技能，全链路打通 |
| 架构 | 各文件独立运行 | 模块化分层，统一入口 |
| 数据 | 12条车型+5篇行业报告 | 16款车型+10篇深度文档+术语词典 |
| 前端 | curl / HTML测试页 | Streamlit 全功能对话框 |
| 部署 | 手动 python xxx.py | Docker Compose 一键启动 |
| 测试 | 手动验证 | 单元/集成/E2E 三层测试 |

---

## 目录结构（含完成状态）

```
car-ai-advisor/
│
├── README.md                          # ✅ 本文件：全链路架构与开发计划
├── .env.example                       # ✅ 环境变量模板
├── .gitignore                         # ✅
├── Makefile                           # ✅ 常用命令快捷方式
├── docker-compose.yml                 # ✅ Phase 1 完成
├── Dockerfile                         # ✅ Phase 1 多阶段构建
├── requirements.txt                   # ⚠️ 需追加 PyJWT
│
├── backend/                           # === FastAPI 后端 ===
│   ├── main.py                        # ✅ Phase 1 骨架完成，待注册路由
│   ├── config.py                      # ✅ Phase 1 20+配置项完成
│   ├── api/                           # API 层
│   │   ├── deps.py                    # ❌ Day 3 依赖注入
│   │   └── routes/
│   │       ├── chat.py                # ❌ Day 3 POST /chat 双模式
│   │       └── sessions.py            # ❌ Day 3 会话 CRUD
│   ├── core/                          # 核心基础设施
│   │   ├── session_manager.py         # ❌ Day 1 Redis 会话管理
│   │   ├── stream.py                  # ❌ Day 1 SSE 流式生成器
│   │   ├── resilience.py              # ❌ Day 2 熔断+限流
│   │   └── security.py                # ✅ 双模式认证已就绪
│   ├── agent/                         # Agent 决策层
│   │   ├── advisor.py                 # ❌ Day 6 ReAct Agent
│   │   ├── tools.py                   # ❌ Day 5 5个Tool函数
│   │   └── prompts.py                 # ❌ Day 5 三层Prompt
│   ├── rag/                           # RAG 检索管线
│   │   ├── retriever.py               # ❌ Day 4 混合检索+RRF融合
│   │   ├── embeddings.py              # ❌ Day 4 BGE模型加载
│   │   ├── chunker.py                 # ❌ Day 4 文档分块
│   │   └── reranker.py                # ❌ Day 5 CrossEncoder重排序
│   └── schemas/                       # Pydantic 数据模型
│       ├── chat.py                    # ✅ ChatReq/ChatResp+SSE常量
│       └── session.py                 # ✅ Session CRUD schemas
│
├── frontend/                          # === Streamlit 前端 ===
│   ├── app.py                         # ❌ Day 7 主界面
│   ├── api_client.py                  # ❌ Day 7 API调用封装
│   └── components/
│       ├── chat.py                    # ❌ Day 8 对话渲染
│       ├── sidebar.py                 # ❌ Day 8 侧边栏
│       └── tools.py                   # ❌ Day 8 工具可视化
│
├── knowledge_base/                    # === 企业级知识库 ===
│   ├── README.md                      # ✅ 知识库说明
│   ├── raw/                           # ✅ 原始数据 ~36500字（10个文件）
│   ├── processed/                     # 向量索引存放目录
│   └── scripts/                       # 数据处理脚本
│       ├── build_index.py             # ❌ Day 6 构建向量索引
│       ├── data_validator.py          # ❌ Day 11 数据校验
│       └── load_data.py               # ❌ Day 6 数据加载器
│
├── models/                            # === 本地模型 ===
│   └── bge-base-zh-v1.5/              # ✅ Embedding 模型已下载
│
├── tests/                             # === 测试 ===
│   ├── conftest.py                    # ❌ Day 9 fixtures
│   ├── unit/                          # 单元测试
│   │   ├── test_session.py            # ❌ Day 9
│   │   ├── test_rag.py                # ❌ Day 9
│   │   └── test_agent.py              # ❌ Day 9
│   ├── integration/                   # 集成测试
│   │   └── test_api.py                # ❌ Day 9
│   └── e2e/                           # 端到端测试
│       └── test_full_flow.py          # ❌ Day 9
│
├── deploy/                            # === 部署配置 ===
│   ├── nginx/default.conf             # ❌ Day 10
│   ├── prometheus/prometheus.yml      # ❌ Day 10
│   └── scripts/start.sh               # ❌ Day 10
│
└── docs/                              # === 文档 ===
    ├── ARCHITECTURE.md                # ❌ Day 10
    ├── API.md                         # ❌ Day 11
    ├── RAG_DESIGN.md                  # ❌ Day 11
    └── DEPLOY.md                      # ❌ Day 11
```

> **进度**：7/42 代码文件已完成（✅），35 个待实现（❌）。知识库数据 10/10 已完成，BGE 模型已下载。

---

## 系统架构

```
                         ┌──────────────────────────────┐
                         │       Streamlit 前端          │
                         │  · 对话界面 + 会话侧边栏       │
                         │  · 工具调用可视化              │
                         │  · fetch + ReadableStream    │
                         └─────────────┬────────────────┘
                                       │ HTTP POST /chat
                                       │ Authorization: Bearer sk-xxx
                                       ▼
                         ┌──────────────────────────────┐
                         │      FastAPI 网关层            │
                         │                              │
                         │  Auth (HTTPBearer + API Key) │
                         │    ↓                          │
                         │  Rate Limiter (滑动窗口)      │
                         │    ↓                          │
                         │  Circuit Breaker (熔断器)     │
                         │    ↓                          │
                         │  Concurrency Slot (Lua原子)  │
                         │    ↓                          │
                         │  Session Manager (Redis)     │
                         │    ↓                          │
                         │  /chat Handler               │
                         │    ├─ stream=true  → SSE     │
                         │    └─ stream=false → JSON    │
                         └─────────────┬────────────────┘
                                       │
                         ┌─────────────▼────────────────┐
                         │        Agent 决策层            │
                         │  ReAct 循环:                  │
                         │    Thought → Action → Obs → … │
                         │    ┌──────────┐              │
                         │    │ LLM      │              │
                         │    │ DeepSeek │  streaming   │
                         │    └──────────┘              │
                         └──────┬──────────┬────────────┘
                                │          │
                   ┌────────────┼──────────┼────────────┐
                   ▼            ▼          ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ 检索工具  │ │ 价格工具  │ │ 对比工具  │ │ 推荐工具  │
            └────┬─────┘ └──────────┘ └──────────┘ └──────────┘
                 │
                 ▼
            ┌──────────────────────────────────────────┐
            │              RAG 检索管线                  │
            │  ① Query 改写 (LLM)                       │
            │  ② FAISS 向量检索 (BGE Embedding)        │
            │  ③ BM25 关键词检索 (jieba分词)            │
            │  ④ RRF 融合排序 (k=60)                    │
            │  ⑤ BGE Reranker 精排 (CrossEncoder)       │
            │  ⑥ Top-K 文档拼接 + 引用标注               │
            └──────────────┬───────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────────────┐
            │            知识库 (~36500字)               │
            │  16款车型 | 10条评价 | 4篇指南 | 14个FAQ  │
            └──────────────────────────────────────────┘

    ───────────── 基础设施 ─────────────
    Redis    ← 会话存储 + 并发控制 + 限流计数器
    Docker   ← docker-compose up 一键启动
```

---

## API 设计

| 方法 | 路径 | 认证 | 说明 |
|------|------|:---:|------|
| `GET` | `/health` | 无 | 健康检查：Redis 连通性 + 服务状态 |
| `POST` | `/chat` | Bearer | 对话接口：stream=true(SSE) / false(JSON) |
| `POST` | `/sessions` | Bearer | 创建新会话 |
| `GET` | `/sessions` | Bearer | 列出用户所有会话 |
| `PATCH` | `/sessions/rename` | Bearer | 重命名会话 |
| `DELETE` | `/sessions/{id}` | Bearer | 删除会话 |
| `GET` | `/sessions/{id}/history` | Bearer | 获取对话历史 |

### SSE 事件类型

| event type | 含义 | payload 示例 |
|------|------|------|
| `source` | 检索到的参考文档 | `{"type":"source","sources":[...]}` |
| `token` | LLM 生成的单个 token | `{"type":"token","token":"理","index":42}` |
| `done` | 生成完成 | `{"type":"done","total_tokens":342}` |
| `error` | 发生错误 | `{"type":"error","message":"并发请求过多"}` |

---

## 11 天开发计划（每日任务 + 技术原理详解）

---

### 📅 Day 1 — Redis 会话管理 + SSE 流式生成器

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `backend/core/session_manager.py` | ~150 | SessionManager 类，9 个方法：add_message / get_history / touch_session / acquire_slot / release_slot / create_session / list_sessions / rename_session / delete_session |
| `backend/core/stream.py` | ~60 | `format_sse()` 格式化函数 + `sse_generator()` 异步生成器（source → token × N → done → [DONE]），含 CancelledError 保护 |

#### 技术原理详解

##### 1.1 Redis List：为什么用它存对话历史？

**原理**：
Redis List 底层是双向链表（quicklist），支持两端 O(1) 操作。存对话历史时：
- `RPUSH key msg` — 从右侧追加消息，时间复杂度 O(1)
- `LRANGE key -20 -1` — 取最后 20 条消息，用负索引从尾部倒数，不需要遍历全量
- `LLEN key` — 获取消息总数，Redis 维护了链表长度计数器，O(1)
- `EXPIRE key 1800` — 30 分钟自动过期，不活跃会话自动清理

**为什么不用 String（JSON 字符串）？**

如果把整个对话历史存成一个 JSON 字符串：
```
SET session:abc:messages '{"messages":[...100条消息...]}'
```
- 每次追加消息需要：GET → 反序列化 → append → 序列化 → SET（读-改-写，4步操作）
- 存在竞态条件：两个并发请求同时读到旧版本，最后一个 SET 会覆盖前面的更新（Lost Update）
- 网络传输全量数据，100 条消息时每次操作传输几十 KB

**List 方案的优势**：追加只需 1 次 RPUSH，取最近 N 条用 LRANGE 负索引，不传输不需要的数据。

##### 1.2 Redis Pipeline：如何减少网络往返？

**原理**：
TCP 通信有 RTT（Round-Trip Time，往返时间），每次 Redis 命令需要：客户端发送 → 网络传输 → Redis 处理 → 网络返回 → 客户端接收。Pipeline 把多个命令打包成一个 TCP 包发送，Redis 按顺序执行后一次性返回所有结果。

`add_message` 方法中一次 Pipeline 包含 5 个命令：
```python
pipe.rpush(messages_key, msg)    # 追加消息
pipe.expire(messages_key, ttl)   # 消息列表 TTL
pipe.expire(meta_key, ttl)       # 元信息 TTL
pipe.sadd(user_sessions_key, sid) # 维护用户→会话映射
await pipe.execute()              # 全部命令一次网络往返
```

**效果**：5 次独立命令 = 5×RTT。1 次 Pipeline = 1×RTT。RTT 通常 0.1-1ms，Pipeline 节省 80% 网络时间。

##### 1.3 Lua 脚本原子操作：并发控制的核心

**原理**：
Redis 执行 Lua 脚本时是**单线程原子执行** —— 整个脚本执行期间，不会有其他命令插入执行。这保证了 INCR→判断→DECR 三个步骤不会被并发请求打断。

`acquire_slot` 的 Lua 脚本：
```lua
local current = redis.call('INCR', KEYS[1])   -- ① 原子递增
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2])) -- ② 设置兜底 TTL
if current > tonumber(ARGV[1]) then            -- ③ 判断是否超限
    redis.call('DECR', KEYS[1])                -- ④ 超限则回滚
    return 0                                    -- ⑤ 返回失败
end
return 1                                         -- ⑥ 返回成功
```

**面试追问：为什么不能先 GET 再 SET？**
```
# 错误做法（存在竞态条件）：
current = redis.get("concurrency:user_001")  # T1: 读到 2
# ... 另一个请求也读到 2 ...
if current < 3:
    redis.incr("concurrency:user_001")        # T2: 两个请求都 INCR，变成 4
```
在 T1 和 T2 之间有时间窗口，两个并发请求都认为没超限，最终并发数变成 4，突破限制。

**Lua 脚本方案**：整个 INCR→判断→DECR 在 Redis 服务端一气呵成，**没有时间窗口**。

**兜底 TTL 60s 的作用**：如果 `release_slot` 因异常未被调用（如进程崩溃），60 秒后计数器自动过期清零，不会造成"槽位永久泄漏"。

##### 1.4 SSE（Server-Sent Events）协议：单向流式推送

**原理**：
SSE 是 HTTP 长连接上的单向推送协议（服务器→客户端），比 WebSocket 更轻量。

**协议格式**：
```
data: {"type":"source","sources":[...]}\n\n
data: {"type":"token","token":"理","index":0}\n\n
data: {"type":"done","total_tokens":342}\n\n
data: [DONE]\n\n
```
- 每条消息以 `data: ` 开头
- 以 `\n\n`（两个换行）结尾作为消息分隔符
- `[DONE]` 是约定俗成的结束信号（借鉴 OpenAI 的 SSE 风格）

**HTTP 响应头要求**：
```
Content-Type: text/event-stream   ← 告诉浏览器这是 SSE 流
Cache-Control: no-cache           ← 禁止代理缓冲
Connection: keep-alive            ← 保持连接
X-Accel-Buffering: no             ← 禁用 Nginx 缓冲（生产环境关键！）
```

**为什么用 SSE 而不是 WebSocket？**
- AI 对话场景是单向推送为主（用户发一条，AI 回一条），不需要双向实时通信
- SSE 基于 HTTP 协议，浏览器原生支持 `EventSource` API，自动重连
- 更轻量：不需要 WebSocket 的握手升级和帧协议
- 但 SSE 不支持 POST 请求体，所以我们用 `fetch + ReadableStream` 手动消费（Day 7 详解）

##### 1.5 asyncio.CancelledError：客户端断开后优雅退出

**原理**：
FastAPI 的 `StreamingResponse` 底层是 ASGI 协议的异步生成器。当客户端关闭连接（关闭页面/取消请求）时，ASGI 服务器（uvicorn）会向正在 yield 的协程发送 `CancelledError` 信号。

关键陷阱：`generated_count` 必须定义在 `try` **外面**：
```python
async def sse_generator(full_answer):
    generated_count = 0          # ← 必须在 try 外面！
    try:
        for char in full_answer:
            generated_count += 1
            yield format_sse("token", {"token": char})
    except asyncio.CancelledError:
        logger.warning(f"客户端断开，已生成 {generated_count} tokens")
        # 不 re-raise，优雅退出
```

**如果在 try 里面定义**：`CancelledError` 可能在 `generated_count = 0` 之前触发，此时变量未定义，`except` 块里的 `logger.warning(f"...{generated_count}...")` 会抛 `NameError`，覆盖原本的异常信息。

#### 参考代码

`agent-dev-project/api/session_manager.py` L18-138（SessionManager 全部方法实现）
`agent-dev-project/api/stream.py` L328-375（`llm_stream_generator_safe` 函数）

#### 验证

```bash
python -c "from backend.core.session_manager import SessionManager; print('OK')"
python -c "from backend.core.stream import sse_generator, format_sse; print('OK')"
```

---

### 📅 Day 2 — 韧性层：限流 + 熔断

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `backend/core/resilience.py` | ~120 | `SlidingWindowRateLimiter`（Redis ZSet 滑动窗口限流）+ `CircuitBreaker`（三态熔断器） |

#### 技术原理详解

##### 2.1 Redis Sorted Set（ZSet）：滑动窗口的底层数据结构

**原理**：
ZSet 是有序集合，每个元素有一个 score（分数），元素按 score 自动排序。底层是**跳表（Skip List）**，范围查询 O(log N + M)。

**为什么用 ZSet 做限流？**
- score 存时间戳（`time.time()`），member 可以是任意唯一标识
- `ZREMRANGEBYSCORE key min max` — 按分数范围删除元素，O(log N)
- `ZCARD key` — 获取集合大小，O(1)

**限流操作（一次 Pipeline）**：
```python
pipe.zadd(rk, {str(now): now})                    # ① 插入当前请求时间戳
pipe.zremrangebyscore(rk, 0, now - window)         # ② 删除窗口外的旧记录
pipe.zcard(rk)                                      # ③ 统计窗口内请求数
pipe.expire(rk, window + 10)                        # ④ TTL 兜底防内存泄漏
_, _, count, _ = await pipe.execute()
return count <= max_requests
```

##### 2.2 滑动窗口 vs 固定窗口：为什么滑动窗口更好？

**固定窗口的问题**（面试高频考点）：

固定窗口按整秒/整分钟计数，如"每分钟 60 次"：
```
时间线:  | 12:00:00 — 12:00:59 | 12:01:00 — 12:01:59 |
窗口 1:  60 次请求              窗口 2: 60 次请求
```

攻击者可以在窗口边界发起"加倍攻击"：
```
12:00:59 发 60 次请求（窗口 1 的配额用满）
12:01:00 发 60 次请求（窗口 2 的配额用满）
→ 实际 2 秒内发出 120 次请求，但没有一个窗口超限！
```

**滑动窗口方案**：不是按整点切分，而是"过去 60 秒内最多 60 次"。每个请求到来时，清理 60 秒前的旧记录，统计当前窗口内的实际数量。时间窗口随当前时间滑动，没有固定边界可利用。

##### 2.3 熔断器（Circuit Breaker）：保护下游服务的经典模式

**原理**：
熔断器源自电气工程，Michael Nygard 在《Release It!》中引入软件领域。核心思想：当下游服务（如 LLM API）连续失败时，快速失败比反复重试更好 —— 避免雪崩效应和资源浪费。

**三态状态机**：
```
        连续失败 ≥5 次
CLOSED ─────────────────► OPEN
  (正常)                    (拒绝)
    ▲                        │
    │      超时 30s 后        │
    │   HALF_OPEN ◄──────────┘
    │   (试探)
    │     │
    ├─────┘ 成功 ≥3 次 → CLOSED（恢复）
    └─────── 失败 → OPEN（继续熔断）
```

**为什么需要 HALF_OPEN 状态？**

如果直接从 OPEN 跳回 CLOSED，所有请求立即涌入，而此时下游可能还没真正恢复 → 再次触发熔断 → 震荡。HALF_OPEN 只放行少量请求"试探"，成功确认恢复后才全量放开。

**`call()` 方法的降级逻辑**：
```python
async def call(self, coro, fallback=None):
    if self.state == OPEN:
        if timeout_elapsed:
            self.state = HALF_OPEN   # 试探恢复
        else:
            return fallback()        # 降级兜底
    try:
        result = await coro          # 实际执行
        self._on_success()
        return result
    except Exception:
        self._on_failure()
        if fallback:
            return fallback()        # 失败降级
        raise
```

**面试话术**：在 LLM API 调用场景，熔断器防止 API 不可用时大量请求堆积在连接池，避免线程/协程耗尽。fallback 可返回缓存的通用回答或"系统繁忙"提示。

#### 参考代码

学习项目此文件也是空壳，需从零手写。Day 15 知识点，面试高频考点。

#### 验证

```bash
python -c "from backend.core.resilience import CircuitBreaker, SlidingWindowRateLimiter; print('OK')"
```

---

### 📅 Day 3 — 依赖注入 + API 路由（Phase 2 完成）

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `backend/api/deps.py` | ~50 | 3 个 FastAPI 依赖：get_session_manager / check_rate_limit / 复用 security 的 get_current_user_auto |
| `backend/api/routes/chat.py` | ~70 | POST /chat 双模式端点，并发控制，占位回答 |
| `backend/api/routes/sessions.py` | ~55 | 5 个会话 CRUD 端点 |
| `backend/main.py` | +3 行 | 注册路由模块 |
| `requirements.txt` | +1 行 | 追加 `PyJWT==2.9.0` |

#### 技术原理详解

##### 3.1 FastAPI 依赖注入（Depends）：洋葱模型

**原理**：
FastAPI 的 `Depends` 实现了类似中间件的洋葱模型。每个 `Depends` 函数按声明顺序层层嵌套执行：

```python
@router.post("/chat")
async def chat(
    user_id: str = Depends(get_current_user_auto),     # 第 1 层
    session_mgr: SessionManager = Depends(get_session_manager), # 第 2 层
):
```

执行顺序：
```
请求进入
  → get_current_user_auto(credentials) → 校验 API Key/JWT → 返回 user_id
    → get_session_manager(request) → 从 app.state.redis 创建实例 → 返回 session_mgr
      → chat() 函数体执行
    ← 返回响应
  ← 清理资源
```

**实际依赖链（多层洋葱）**：
```python
user_id = Depends(get_current_user_auto)        # ① 认证
user_id = Depends(check_rate_limit)             # ② 限流（依赖 ① 的 user_id）
session_mgr = Depends(get_session_manager)      # ③ 会话管理
```

当一个 Depends 依赖前一个 Depends 的结果时，FastAPI 自动解析参数名匹配。例如 `check_rate_limit` 的参数名 `user_id` 与 `get_current_user_auto` 的返回值名一致，就会自动传递。

**为什么用依赖注入而不是全局变量？**
- 可测试：测试时可以注入 mock Redis
- 解耦：session_manager 不需要知道 Redis 从哪来
- 显式：每个端点的依赖关系一目了然

##### 3.2 StreamingResponse：SSE 流式响应的实现

**原理**：
`StreamingResponse` 接收一个异步生成器，uvicorn 每次 `yield` 就推送一块数据给客户端：

```python
return StreamingResponse(
    sse_generator(contexts, answer),       # 异步生成器
    media_type="text/event-stream",        # SSE 的 MIME 类型
    headers={
        "Cache-Control": "no-cache",       # 禁止缓冲
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",         # 禁用 Nginx 缓冲
    },
)
```

**与 JSONResponse 的核心区别**：
- JSONResponse：等数据全部构建完，一次性 `json.dumps()` 后发送
- StreamingResponse：边生成边发送，客户端不需要等到生成完毕就能看到内容

**TTFT（Time To First Token）的意义**：
- 非流式：用户等 2-6 秒才看到完整回答
- 流式：用户 0.5-1 秒就看到第一个 token，体验更好

##### 3.3 FastAPI Router：模块化路由组织

**原理**：
`APIRouter` 类似 Flask 的 Blueprint 或 Django 的 URLconf。每个功能模块定义自己的 router，在主应用统一注册：

```python
# routes/chat.py
router = APIRouter(tags=["对话"])
@router.post("/chat")
async def chat(...): ...

# routes/sessions.py
router = APIRouter(prefix="/sessions", tags=["会话"])
@router.post("")     # 实际路径 = /sessions
async def create_session(...): ...

# main.py
app.include_router(chat.router)
app.include_router(sessions.router)
```

`tags` 参数会让 Swagger 文档按分组展示，`prefix` 参数为整个 router 统一加路径前缀。

#### 验证（Docker 集成测试）

```bash
docker compose up -d --build
docker compose ps                          # backend + redis 均为 healthy

# 会话 CRUD
curl -X POST http://localhost:8000/sessions \
  -H "Authorization: Bearer sk-dev-user-001" \
  -H "Content-Type: application/json" -d '{"title":"测试"}'

# 流式对话
curl -N -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer sk-dev-user-001" \
  -H "Content-Type: application/json" \
  -d '{"query":"你好","stream":true}'

# 无认证 → 401
curl -X POST http://localhost:8000/chat -d '{"query":"test"}'

# Swagger 文档
浏览器 http://localhost:8000/docs → Authorize → Try it out
```

**里程碑 🎯**：Phase 2 完成。全部 API 端点可用，认证、限流、并发控制、SSE 流式全部打通。

---

### 📅 Day 4 — RAG 管线（上）：文档分块 + Embedding

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `backend/rag/chunker.py` | ~80 | `load_vehicles()` / `load_guides()` / `chunk_sections()` — JSON 按条目切块、Markdown 按标题切块、为每块标注元数据 |
| `backend/rag/embeddings.py` | ~60 | `EmbeddingModel` 类：加载 BGE 模型、`embed_documents()` 批量编码、`embed_query()` 单条编码 |

#### 技术原理详解

##### 4.1 文档分块策略：切多大？怎么切？

**原理**：
大模型一次能处理的文本长度有限（上下文窗口），且检索精度与 chunk 大小直接相关。

**chunk 大小的权衡**：

| 块大小 | 优点 | 缺点 |
|--------|------|------|
| 太小（100-200字） | 检索精确，匹配到的内容密度高 | 丢失上下文，看半句话不知道在说什么 |
| 适中（500-800字） | 信息完整，检索召回率高 | 匹配精度略降 |
| 太大（2000+字） | 上下文完整 | 检索稀释，不相关内容占比高，降低 LLM 回答质量 |

**本项目分块策略**：
- **车型 JSON（16 款）**：每款车为一个独立 chunk（约 500 字），`car_to_text()` 将嵌套 JSON 展平为可读段落
- **Markdown 文档（10 篇）**：按 H2 标题边界切分，每块 500-800 字，保留标题层级路径作为元数据
- **FAQ（14 个）**：每个 Q&A 为一个 chunk
- **术语词典（20+ 条）**：每个 term 为一个 chunk

**语义边界切分的重要性**：
按字符数等距硬切（如每 500 字一刀）会割裂语义。按文档结构（章节标题、段落边界、JSON 条目）切分，每个 chunk 保持语义完整。

**元数据标注**：
每个 chunk 携带元数据，用于检索后过滤和来源标注：
```json
{
  "chunk_id": "v_001",
  "source_type": "vehicle",       // vehicle / review / guide / industry / faq / glossary
  "brand": "比亚迪",
  "category": "中大型纯电轿车",
  "price_range": "25-32万元",
  "content": "比亚迪海豹08 中大型智能运动旗舰轿车..."
}
```

##### 4.2 Embedding 向量：文本如何变成数字？

**原理**：
Embedding 模型（如 BGE）将文本映射为一个固定长度的浮点数向量（如 1024 维）。过程分两步：

**① Tokenization（分词）**：把文本切分成模型认识的 token ID 序列。BGE 用 WordPiece 分词，中文约 1.5 个字 = 1 个 token。
```
"25万预算推荐什么车？"
→ [CLS] [25] [万] [预] [算] [推] [荐] [什] [么] [车] [？] [SEP]
→ [101, 1234, 5678, ...]
```

**② 模型前向传播**：将 token IDs 输入 Transformer 模型，取最后一层的池化输出作为向量。

**池化策略（Pooling）**：
- **CLS Pooling**：取 `[CLS]` token 的输出向量（BERT 风格）
- **Mean Pooling**：取所有 token 输出的平均值（Sentence-BERT 风格，BGE 使用此法）

Mean Pooling 对句子级任务更稳定，因为 `[CLS]` token 在预训练时未专门为语义相似度优化。

**余弦相似度（Cosine Similarity）**：
检索时，用查询向量与所有文档向量计算余弦相似度：
```
cos(A, B) = A·B / (|A| × |B|)
```
余弦值范围 [-1, 1]，越接近 1 表示语义越相似。因为只关心方向不关心长度，对文本长度不敏感。

**为什么 BGE 选择 1024 维？**
- 维度太低（256）：信息压缩太狠，精度不够
- 维度太高（4096）：存储和计算成本大，精度提升边际递减
- 1024 维是性价比甜点：足够表达语义，FAISS 检索也很快

**Bi-Encoder（双塔）架构**：
BGE 是 Bi-Encoder：query 和 document 分别独立编码为向量，检索时只需计算向量间的点积/余弦相似度。离线把所有文档编码好存到 FAISS，在线只需编码查询向量 → 检索速度极快。

#### 参考代码

`agent-dev-project/chapters/naive_rag.py`（分块逻辑）
`agent-dev-project/chapters/embedding_test.py`（Embedding 模型加载）

#### 验证

```bash
python -c "from backend.rag.chunker import load_vehicles, chunk_sections; print('OK')"
python -c "from backend.rag.embeddings import EmbeddingModel; print('OK')"
```

---

### 📅 Day 5 — RAG 管线（下）：混合检索 + 精排 + Agent Prompt + Tools

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `backend/rag/retriever.py` | ~150 | `VectorIndex`（FAISS）+ `BM25`（jieba IDF）+ `hybrid_rrf()` 融合 + `HybridRetriever.search()` 统一入口 |
| `backend/rag/reranker.py` | ~80 | `Reranker` 类：加载 BGE CrossEncoder，`rerank()` 精排 |
| `backend/agent/prompts.py` | ~100 | 三层 Prompt 模板：外层角色 + 中层工具描述 + 内层 CoT 推理链 |
| `backend/agent/tools.py` | ~150 | 5 个 @tool 函数：search_knowledge / get_car_price / compare_cars / recommend_cars / calculate_ownership_cost |

#### 技术原理详解

##### 5.1 FAISS（Facebook AI Similarity Search）：向量检索库

**原理**：
FAISS 是 Meta 开源的高效向量相似度检索库。核心思想：向量检索等价于在高维空间中找最近邻。

**IndexFlatIP（内积索引）**：
暴力检索，计算查询向量与所有文档向量的内积。文档数少（<10万）时最快且 100% 精确。内积在向量归一化后等价于余弦相似度。

```python
import faiss
index = faiss.IndexFlatIP(1024)     # 1024 维内积索引
index.add(embeddings)               # 添加 N 个文档向量，O(N)
D, I = index.search(query_vec, k=5) # 检索 Top-5，O(N)
```

**近似最近邻（ANN）简介**（扩展知识，本项目用 IndexFlatIP 即可）：
- IVF_FLAT：先聚类（K-means），查询时只搜最近几个聚类中心，牺牲 ~5% 精度换 10-100× 加速
- HNSW：基于图的最邻近搜索，构建多层可跳转的导航图

##### 5.2 BM25：经典关键词检索算法

**原理**：
BM25 是 TF-IDF 的进化版，是目前最有效的关键词检索算法之一。

**BM25 公式**（简化理解）：
```
score(D, Q) = Σ IDF(qi) × (tf(qi, D) × (k1 + 1)) / (tf(qi, D) + k1 × (1 - b + b × |D|/avgDL))
              qi∈Q
```
- `IDF(qi)`：逆文档频率，稀有词权重高（如"兆瓦闪充"比"汽车"重要）
- `tf(qi, D)`：词频，但上限被 k1 参数饱和（出现 5 次和出现 50 次差别不大）
- `|D|/avgDL`：文档长度归一化，b 参数控制程度（长文档不被惩罚过重）

**BM25 相比 TF-IDF 的改进**：
- **词频饱和**：TF-IDF 线性假设（词出现 10 次相关性是 1 次的 10 倍）不符合实际。BM25 通过 k1 参数使词频贡献非线性增长
- **文档长度归一化**：TF-IDF 默认长文档和短文档同等对待。BM25 考虑了平均文档长度

**用 jieba 分词实现 BM25**：
```python
import jieba
class BM25:
    def __init__(self, documents):
        self.docs = [list(jieba.cut(doc)) for doc in documents]  # 分词
        self.idf = self._compute_idf()                            # 计算 IDF
    def search(self, query, k=5):
        tokens = list(jieba.cut(query))
        scores = [self._score(tokens, doc) for doc in self.docs]  # BM25 打分
        return top_k(scores, k)
```

##### 5.3 RRF（Reciprocal Rank Fusion）：无参数融合方法

**原理**：
向量检索返回 Top-50（用分数排序），BM25 返回 Top-50（用分数排序），但两类分数的量级不同（向量相似度 0.8-1.0，BM25 0-50），直接加权需要调参。RRF 只关心排名，不关心分数绝对值：

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
                i∈{vector, bm25}
```

其中 k=60 是经验常数（源自论文）。文档在两个列表中都排名靠前 → RRF 得分高。

**例子**：
| 文档 | 向量检索排名 | BM25 排名 | RRF 得分 |
|------|:--:|:--:|------|
| 宋L（完美匹配） | 1 | 2 | 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 |
| Model Y（偏差匹配） | 15 | 1 | 1/(60+15) + 1/(60+1) = 0.0133 + 0.0164 = 0.0297 |

RRF 不依赖分数分布假设，对异常值鲁棒。

##### 5.4 Cross-Encoder vs Bi-Encoder：精排与召回的分工

**原理**：

| | Bi-Encoder（双塔） | Cross-Encoder（交叉编码器） |
|------|------|------|
| **工作方式** | Query 和 Doc 分别独立编码 | Query 和 Doc 拼接后一起编码 |
| **速度** | 极快（Doc 向量可预计算） | 慢（每对 query-doc 都要过模型） |
| **精度** | 较低（Query 和 Doc 没有交互） | 高（Query 和 Doc 在每层 Attention 中交互） |
| **用途** | 召回（从 N 个候选中找 Top-K） | 精排（对 Top-K 重新精确排序） |

**RAG 管线的分工**：
1. FAISS + BM25 → 从 100+ 个 chunk 中召回 Top-30（Bi-Encoder，快但粗）
2. RRF 融合 → 合并去重 → 候选取交集
3. CrossEncoder → 对 30 个候选精排取 Top-5（慢但精）
4. Top-5 文档拼接 → 送入 LLM 生成回答

**为什么需要精排？**
Bi-Encoder 的向量在编码时就"凝固"了，无法根据查询的细微差异调整。CrossEncoder 让 query 和 document 的全层 Attention 交互，能捕捉到更精细的语义匹配。例如"家用 SUV"和"宋L" —— 向量检索可能因"家用"和宋L的描述不完全重合而排低，但 CrossEncoder 读完整句能识别出隐含关联。

##### 5.5 Agent 三层 Prompt 架构

**原理**：
Prompt 的质量直接决定 Agent 的可靠性。三层架构将 System Prompt 按功能拆分：

**外层 — 角色设定**：
```
你是专业汽车导购助手。你只能基于知识库中的真实数据回答。
如果信息不足，必须诚实告知。绝对不编造不存在的数据。
```

**中层 — 工具描述**：
```
你有以下工具可用：
1. search_knowledge(query) — 在汽车知识库中检索信息，返回相关文档
2. get_car_price(model_name) — 获取指定车型的价格区间
3. compare_cars(model_list) — 对比多款车型的核心参数
...
调用工具时，一次只调用一个，获得结果后再决定下一步。
```

**内层 — CoT 推理链**：
```
当用户问推荐类问题时，请按以下步骤思考：
1. 先用 search_knowledge 检索符合用户预算和需求的车型
2. 对候选车型调用 get_car_price 获取价格
3. 如果用户要比较，调用 compare_cars
4. 最后综合所有信息给出推荐，并注明每款车的适用场景

不要在未检索的情况下凭记忆推荐车型。
```

**为什么三层分开？**
- 修改工具只需改中层，不影响角色和推理链
- 修改推理逻辑只需改内层
- A/B 测试不同版本的 Prompt 很容易

**@tool 装饰器原理**：
LangChain/LangGraph 的 `@tool` 装饰器将 Python 函数转换为 LLM 可理解的工具定义。装饰器提取函数的 docstring 作为工具描述、参数类型作为 JSON Schema，注入到 LLM 的 tool_choice 中：

```python
@tool
def get_car_price(model_name: str) -> str:
    """获取指定车型的价格区间。
    
    Args:
        model_name: 车型名称，如"宋L"或"理想L6"
    """
    # 从 vehicles.json 查找价格
    ...
```

LLM 看到的格式（function calling API）：
```json
{
  "name": "get_car_price",
  "description": "获取指定车型的价格区间。",
  "parameters": {
    "type": "object",
    "properties": {
      "model_name": {"type": "string", "description": "车型名称，如'宋L'或'理想L6'"}
    },
    "required": ["model_name"]
  }
}
```

#### 参考代码

`agent-dev-project/chapters/retrieval_test.py`（FAISS + BM25 + RRF）
`agent-dev-project/chapters/full_rag_agent.py` L38-117（CrossEncoder Reranker）
`agent-dev-project/agent/car_advisor_agent.py` L56-177（5 个工具函数）
`agent-dev-project/agent/car_advisor_agent.py` L211-243（System Prompt）

#### 验证

```bash
python -c "from backend.rag.retriever import HybridRetriever; print('OK')"
python -c "from backend.rag.reranker import Reranker; print('OK')"
python -c "from backend.agent.tools import search_knowledge, get_car_price; print('OK')"
python -c "from backend.agent.prompts import SYSTEM_PROMPT; print('OK')"
```

---

### 📅 Day 6 — Agent 决策层 + 知识库索引构建

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `backend/agent/advisor.py` | ~200 | `create_agent()` + `stream_agent_execution()`（ReAct 循环 + streaming 输出） |
| `knowledge_base/scripts/load_data.py` | ~60 | 读取 vehicles.json + reviews.json + guides/*.md → 统一文档列表 |
| `knowledge_base/scripts/build_index.py` | ~80 | 编排：load_data → chunker → embeddings → FAISS.save_index() |

#### 技术原理详解

##### 6.1 ReAct 循环：Agent 的决策引擎

**原理**：
ReAct（Reasoning + Acting）是 LLM Agent 的核心范式。每次循环包含三个步骤：

```
Step 1:
  Thought: 需要先检索25万价位的SUV车型
  Action: search_knowledge("25万 家用 SUV")
  Observation: 找到5篇相关文档：[宋L, 理想L6, 问界M7, ...]

Step 2:
  Thought: 已经拿到候选车型，需要查询价格
  Action: get_car_price("宋L")
  Observation: 宋L价格区间 18.98-24.98万元

Step 3:
  Thought: 还需要查询理想L6的价格
  Action: get_car_price("理想L6")
  Observation: 理想L6价格区间 24.98-28.98万元

Step 4:
  Thought: 信息足够，可以给出推荐了
  Final Answer: 根据您的25万预算和家用需求...
```

**LangGraph StateGraph 实现**：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[list, "消息列表（用户+AI+Tool）"]
    iteration: int                           # 当前步数

def call_model(state: AgentState) -> AgentState:
    """调用 LLM，可能返回 tool_calls"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """判断继续调用工具还是结束"""
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"                        # → 执行工具
    if state["iteration"] >= 10:             # 安全上限
        return END
    return END                                # → 结束

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_edge("tools", "agent")             # 工具结果 → 回到 LLM
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END,
})
graph.set_entry_point("agent")
```

**防止无限循环的三种机制**：

1. **最大步数限制**：`iteration >= 10 → END`。防止 Agent 反复调用工具找不到答案
2. **重复调用检测**：如果连续两次调用相同工具+相同参数，强制结束
3. **Token 预算**：累计 token 超过 4096 时截断

##### 6.2 Agent 流式执行：让用户看到推理过程

**原理**：
`stream_agent_execution()` 不是等 Agent 完全执行完再一起返回，而是每一步都通过 SSE 推送：

```python
async def stream_agent_execution(agent, query: str):
    yield format_sse("status", {"msg": "开始分析..."})
    
    async for event in agent.astream_events({"messages": [query]}):
        if event["event"] == "on_tool_start":
            yield format_sse("tool_start", {"tool": event["name"]})
        elif event["event"] == "on_tool_end":
            yield format_sse("tool_result", {"output": event["data"]["output"]})
        elif event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                yield format_sse("token", {"token": token})
    yield format_sse("done", {})
    yield "data: [DONE]\n\n"
```

前端收到这些事件后，可以在"工具调用折叠面板"中实时展示推理过程。

##### 6.3 知识库索引构建：从原始数据到 FAISS 索引

**执行流程**：

```
① load_data()
   读取 vehicles.json → 16 个文档对象
   读取 reviews.json → 10 个文档对象
   读取 guides/*.md → 4 个文档对象
   读取 industry/*.md → 2 个文档对象
   读取 faq/*.md → 14 个文档对象
   读取 glossary.json → 20+ 个文档对象
   合计: ~66 个文档对象

② chunk_sections(docs)
   JSON 按条目 → ~66 个 chunk
   Markdown 按 H2 标题 → ~40 个 chunk
   合计: ~106 个 chunk

③ embed_documents(chunks)
   批量编码 → [106 × 1024 维向量]
   保存 numpy 数组到 processed/embeddings.npy

④ faiss.write_index(index, "processed/faiss.index")
   序列化 FAISS 索引到磁盘
   启动时 faiss.read_index() 加载，不需要重新编码
```

#### 参考代码

`agent-dev-project/agent/car_advisor_agent.py`（ReAct Agent 完整实现）
`agent-dev-project/chapters/naive_rag.py` load_data()

#### 验证

```bash
make build-index    # 构建知识库索引
# POST /chat 返回基于真实知识库的智能回答
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer sk-dev-user-001" \
  -H "Content-Type: application/json" \
  -d '{"query":"25万预算推荐什么车？","stream":true}'
# → source 事件（真实检索结果）+ token 事件（LLM 实时生成）
```

**里程碑 🎯**：Phase 3 完成。Agent 可自主调用工具检索知识库，SSE 流式返回带引用的智能回答。

---

### 📅 Day 7 — Streamlit 前端（上）：主界面 + API 客户端

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `frontend/api_client.py` | ~80 | `APIClient` 类：`chat_stream()` 用 httpx 消费 SSE、`chat_sync()` 非流式调用、会话 CRUD 方法 |
| `frontend/app.py` | ~80 | Streamlit 主入口：session_state 管理 + 侧边栏 + 对话区 + `st.chat_input()` |

#### 技术原理详解

##### 7.1 Streamlit 响应式执行模型

**原理**：
Streamlit 与传统前端框架（React/Vue）完全不同。它不是"事件驱动"而是"脚本重跑"：

```
用户点击按钮
  → Streamlit 从上到下重新执行 app.py 全部代码
  → 只有被修改的 st 组件重新渲染
  → 未修改的组件用缓存结果跳过
```

**session_state 的作用**：
因为每次交互后代码重跑，所以普通 Python 变量会丢失。需要 `st.session_state` 来持久化：
```python
if "messages" not in st.session_state:
    st.session_state.messages = []       # 首次初始化

st.session_state.messages.append(msg)     # 后续交互保留
```

**缓存装饰器 `@st.cache_resource`**：
只加载一次的重量级资源（如 APIClient）：
```python
@st.cache_resource
def get_api_client():
    return APIClient(base_url="http://localhost:8000")
```
Streamlit 用 content hash 检测函数是否变化，未变化则直接返回缓存结果。

##### 7.2 消费 SSE 流：为什么不能用 EventSource？

**EventSource 的限制**：
浏览器原生 `EventSource` 只支持 GET 请求，不能传 POST body，也不能自定义 Header（如 `Authorization: Bearer xxx`）。AI 对话场景需要 POST + JSON body + auth header，所以需要 `fetch + ReadableStream`。

**httpx 消费 SSE 流（Python 端）**：
```python
async def chat_stream(self, query: str, session_id: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", f"{self.base_url}/chat",
            json={"query": query, "session_id": session_id, "stream": True},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])   # 去掉 "data: " 前缀
                    yield data
                    if data["type"] == "done":
                        break
```

**HTTP 流式响应的底层原理**：
`httpx.stream()` 不等待完整响应体到达，而是建立连接后立即开始逐块读取。底层 TCP 连接持续接收数据，`aiter_lines()` 每次 `yield` 一个以 `\n` 结尾的完整行，内存占用恒定（不随响应大小增长）。

#### 参考代码

`agent-dev-project/api/stream.py` chat-ui 内嵌 HTML 的 JS 部分（fetch + ReadableStream 模式）

#### 验证

```bash
cd frontend && streamlit run app.py --server.port 8501
# 浏览器 http://localhost:8501 → 能看到对话界面
```

---

### 📅 Day 8 — Streamlit 前端（下）：对话组件 + 侧边栏 + 工具可视化

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `frontend/components/chat.py` | ~60 | 对话气泡渲染（用户/AI 消息样式区分）+ 流式更新 + 来源引用标签 |
| `frontend/components/sidebar.py` | ~60 | 侧边栏：会话列表 + 新建按钮 + 点击切换 + 删除确认 |
| `frontend/components/tools.py` | ~60 | 工具调用折叠面板（`st.expander`），展示 Thought → Action → Observation |

#### 技术原理详解

##### 8.1 Streamlit 动态渲染流式文本

**原理**：
流式更新的关键是 `st.empty()` 占位符。创建一个空容器，每收到一个 token 就更新其内容：

```python
placeholder = st.empty()        # 创建空白占位符
full_text = ""
for chunk in api_client.chat_stream(query, session_id):
    if chunk["type"] == "token":
        full_text += chunk["token"]
        placeholder.markdown(full_text + "▌")  # 更新占位符内容
    elif chunk["type"] == "done":
        placeholder.markdown(full_text)         # 去掉光标，完成显示
```

`st.empty()` 占位符类似于 React 的 "ref" 或 Vue 的 "template ref" —— 是一个可以被多次更新但位置不变的 DOM 节点。

##### 8.2 工具调用可视化

**原理**：
Agent 在执行过程中会产生中间事件（`tool_start` / `tool_result`），前端用 `st.expander` 折叠面板展示这些中间步骤。每个工具调用展开后可以看到入参和输出，增加系统可解释性（面试加分项）。

**SSE 事件流**：
```
data: {"type":"status","msg":"正在检索知识库..."}
data: {"type":"tool_start","tool":"search_knowledge","args":{"query":"25万 家用 SUV"}}
data: {"type":"tool_result","output":"找到 5 篇文档"}
data: {"type":"status","msg":"正在生成回答..."}
data: {"type":"token","token":"根"}
data: {"type":"token","token":"据"}
...
```

前端将这些事件分组渲染：
- `tool_start` → 创建新的 `st.expander("🔧 search_knowledge")`
- `tool_result` → 在 expander 内部填充输出内容
- `status` → 顶部进度提示

#### 验证

创建新会话 → 输入问题 → 看流式打字效果 → 展开工具调用面板 → 切到另一会话 → 删除会话，全流程无报错。

**里程碑 🎯**：Phase 4 前半完成。前端 + 后端全链路打通。

---

### 📅 Day 9 — 三层测试（单元 + 集成 + E2E）

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `tests/conftest.py` | ~40 | pytest fixtures：`AsyncClient(app)` + `fakeredis` mock + 测试 API Key |
| `tests/unit/test_session.py` | ~80 | SessionManager 9 个方法单元测试（用 fakeredis） |
| `tests/unit/test_rag.py` | ~80 | chunker → embeddings → retriever → reranker 各模块独立测试 |
| `tests/unit/test_agent.py` | ~80 | 5 个 Tool 单独测试 + ReAct 循环 mock 测试 |
| `tests/integration/test_api.py` | ~100 | 所有端点 HTTP 请求/响应验证 + 认证 + 限流 |
| `tests/e2e/test_full_flow.py` | ~80 | 完整对话流程：创建 → 对话 → 查历史 → 删除 |

#### 技术原理详解

##### 9.1 测试金字塔：三层测试的分工

```
         ┌─────┐
         │ E2E │  ← 1 个文件：完整用户流程
         └──┬──┘
       ┌────┴────┐
       │ 集成测试  │  ← 1 个文件：API 端点 HTTP 测试
       └────┬────┘
   ┌────────┴────────┐
   │    单元测试       │  ← 3 个文件：模块级测试
   └─────────────────┘
```

| 层级 | 速度 | 范围 | 数量 | 信噪比 |
|------|:--:|------|:--:|------|
| 单元 | 极快(ms) | 单个函数/类 | 最多 | 定位问题精确 |
| 集成 | 中等(s) | 多模块交互 | 中等 | 验证契约和边界 |
| E2E | 慢(min) | 完整用户流程 | 最少 | 验证业务价值 |

##### 9.2 fakeredis：在测试中模拟 Redis

**原理**：
`fakeredis` 是纯 Python 实现的 Redis 替代品，与 `redis-py` 使用完全相同的 API。与 mock 的区别：mock 需要手动指定每次调用的返回值，fakeredis 真的模拟了 Redis 的全部数据结构行为。

```python
import fakeredis.aioredis

@pytest.fixture
async def redis_client():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield redis
    await redis.close()

@pytest.fixture
async def session_mgr(redis_client):
    return SessionManager(redis_client)

async def test_add_message(session_mgr):
    await session_mgr.add_message("sid1", "user", "你好", "user_001")
    history = await session_mgr.get_history("sid1")
    assert len(history) == 1
    assert history[0]["role"] == "user"
```

不使用 fakeredis 的话，需要启动 Redis Docker 容器，测试速度慢且依赖外部环境。fakeredis 让单元测试在内存中秒级完成。

##### 9.3 pytest-asyncio + httpx.AsyncClient：测试 FastAPI 端点

**原理**：
`httpx.AsyncClient` 配合 `asgi_transport` 可以直接测试 FastAPI 应用，不需要启动服务器：

```python
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),  # ← 直接测试 ASGI 应用
        base_url="http://test"
    ) as ac:
        yield ac

async def test_chat_stream(client):
    async with client.stream(
        "POST", "/chat",
        json={"query": "你好", "stream": True},
        headers={"Authorization": "Bearer sk-dev-user-001"}
    ) as response:
        assert response.status_code == 200
        # 逐行读取 SSE 事件
        async for line in response.aiter_lines():
            if line.startswith("data: [DONE]"):
                break
```

`ASGITransport` 绕过网络层，直接调用 ASGI 应用。测试速度比真实 HTTP 快 5-10 倍，且不需要端口。

##### 9.4 Mock LLM 调用：不花 API 费用的 Agent 测试

**原理**：
Agent 测试的最大挑战：每次测试都调用真实 LLM API 既慢又花钱。通过 `unittest.mock.AsyncMock` 模拟 LLM 响应：

```python
from unittest.mock import AsyncMock, patch

async def test_agent_search():
    mock_llm = AsyncMock()
    # 第一次调用：让 LLM 返回 tool_call
    mock_llm.ainvoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge", "args": {"query": "SUV"}}]
    )

    with patch("backend.agent.advisor.llm", mock_llm):
        # Agent 会调用 search_knowledge，得到工具结果后再次调用 LLM
        result = await agent.ainvoke({"messages": ["推荐SUV"]})
        ...
```

**mock 的陷阱**：过度 mock 会让测试变成"测自己的假设"。mock 只应该用于外部依赖（LLM API、数据库），内部逻辑（chunker、reranker）应该用真实数据测试。

#### 验证

```bash
make test
# pytest tests/ -v --cov=backend --cov-report=term-missing
# 目标：覆盖率 >80%，全部通过
```

**里程碑 🎯**：Phase 4 完成。有测试覆盖，可放心重构和扩展。

---

### 📅 Day 10 — 部署配置 + 架构文档

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `deploy/nginx/default.conf` | ~30 | 反向代理：`/api/*` → FastAPI:8000，`/` → Streamlit:8501，SSE 缓冲禁用 |
| `deploy/prometheus/prometheus.yml` | ~20 | 指标采集：scrape FastAPI metrics endpoint |
| `deploy/scripts/start.sh` | ~30 | 一键启动：docker compose up + 健康检查等待 + 构建索引 |
| `docs/ARCHITECTURE.md` | ~200 | 架构决策记录（ADR）：每个技术选型都有原因和备选方案对比 |
| `docs/API.md` | ~100 | API 接口文档：端点列表 + 请求/响应示例 + 错误码 |

#### 技术原理详解

##### 10.1 Nginx 反向代理：为什么需要它？

**原理**：
反向代理是位于客户端和后端服务器之间的中间层，客户端只和 Nginx 通信，Nginx 负责转发请求到后端。

**三个核心功能**：
1. **路由分发**：`/api/*` → FastAPI，`/` → Streamlit，同一域名访问
2. **静态文件服务**：Nginx 直接响应静态文件（比 Python 快 10-100 倍）
3. **SSE 缓冲禁用**：`proxy_buffering off` — Nginx 默认会缓冲后端响应，等积累到一定大小再一起发给客户端。这对 SSE 是致命的（用户会在几秒后才突然收到一大块数据）。必须显式关闭

```nginx
location /api/ {
    proxy_pass http://backend:8000;
    proxy_buffering off;              # ← SSE 关键配置！
    proxy_cache off;
    proxy_set_header X-Accel-Buffering no;
}
```

##### 10.2 ADR（Architecture Decision Record）：架构决策记录

**原理**：
ADR 是记录技术决策及其理由的轻量文档。每条记录包含：**背景**（为什么要做这个决策）、**决策**（选了哪种方案）、**后果**（选了这个方案的利弊）。

**本项目的关键 ADR**：
| 决策 | 方案 | 备选 | 理由 |
|------|------|------|------|
| 流式协议 | SSE | WebSocket | 单向推送够用，协议更轻，自动重连 |
| 会话存储 | Redis List | Redis String(JSON) | O(1) 追加，无竞态条件 |
| 向量检索 | FAISS | Milvus | 本地开发零配置，生产可切换 |
| 精排 | CrossEncoder | 只用 Bi-Encoder | 精度优先，Top-30 候选量小可接受 |
| LLM 流式 | 逐 token SSE | 等完全生成后一次返回 | TTFT 体验更好 |
| 前端 | Streamlit | React | Python 技术栈统一，快速原型 |

#### 验证

```bash
bash deploy/scripts/start.sh
# http://localhost → Streamlit
# http://localhost/api/docs → Swagger
```

---

### 📅 Day 11 — 文档完善 + 最终验收

#### 要做什么

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `docs/RAG_DESIGN.md` | ~150 | RAG 管线设计文档：分块策略、检索参数调优、评估指标、消融实验 |
| `docs/DEPLOY.md` | ~100 | 部署手册：环境要求、配置项说明、日志查看、故障排查 |
| `knowledge_base/scripts/data_validator.py` | ~40 | 数据校验：JSON 字段完整性、类型一致性、枚举值规范 |

#### 最终验收清单

```bash
# 1. 一键启动
make docker-up && docker compose ps   # 全部 healthy

# 2. 知识库索引
make build-index                      # 成功构建

# 3. API 端到端
# /health → 200
# POST /sessions → 201
# POST /chat stream=false → JSON 回答
# POST /chat stream=true → SSE 逐字流式
# 无 Token → 401
# 超频率 → 429

# 4. 前端
# http://localhost:8501 → 流式对话正常

# 5. 测试
make test                             # 全部通过，覆盖率 >80%

# 6. 文档
# docs/ 四个文档齐全
```

**里程碑 🎯**：全部 5 个 Phase 完成。全链路 AI Agent 系统可演示、可面试、可部署。

---

## 快速开始（目标状态）

```bash
# 1. 配置环境
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY

# 2. 构建知识库索引
make build-index

# 3. 一键启动
make docker-up

# 4. 访问
# 前端: http://localhost:8501
# API文档: http://localhost:8000/docs
```

---

## 简历话术

### 项目描述（约150字）

> **智能汽车导购助手** — 全链路 AI Agent 系统
>
> 技术栈：FastAPI · LangGraph · Redis · SSE · FAISS · Streamlit · Docker
>
> - 设计并实现基于 ReAct 范式的汽车导购 Agent，集成检索/查价/对比/推荐/成本计算 5 个工具，三层 Prompt 架构使工具调用准确率 >85%
> - 基于 SSE 协议实现流式对话，TTFT < 1.5s，`CancelledError` 保护客户端断开后自动停止生成
> - Redis 管理会话：List 存历史 + TTL 滑动过期 + Lua 原子并发控制（≤3 并发/用户）
> - RAG 管线：BM25 + FAISS 混合检索 RRF 融合 → BGE Reranker 重排序，知识库覆盖 16 款车型 + 10 篇深度文档（3.6 万字）
> - 滑动窗口限流（Redis ZSet）+ 三态熔断器保护下游 LLM API
> - Streamlit 全功能对话界面 + pytest 三层测试 + Docker Compose 一键部署

### 面试追问准备

| 追问 | 回答要点 |
|------|------|
| "为什么用 SSE 而不是 WebSocket？" | 单向推送够用、协议更轻、浏览器原生支持自动重连 |
| "ReAct 循环怎么防止无限调用？" | 最大步数限制(10) + token 预算 + 重复调用检测 |
| "Redis 为什么用 List 而不是 String？" | RPUSH O(1)追加、LRANGE 负索引取尾、无需读-改-写 |
| "Lua 并发控制脚本怎么写的？" | INCR→判上限→超限则 DECR 回滚→60s TTL 兜底防槽位泄漏 |
| "RRF 为什么不用简单的分数加权？" | 向量分数和 BM25 分数不在同一量级，直接加权需调参；RRF 无参数且对分数分布鲁棒 |
| "滑动窗口和固定窗口的区别？" | 固定窗口边界可被利用（加倍攻击），滑动窗口连续统计无边界效应 |
| "熔断器三态切换逻辑？" | CLOSED→连续失败≥5→OPEN→30s后→HALF_OPEN→成功≥3→CLOSED |
| "Bi-Encoder 和 Cross-Encoder 的区别？" | Bi-Encoder 独立编码速度飞快（召回用），Cross-Encoder 联合编码精度更高（精排用） |

---

## 技术决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 前端框架 | Streamlit | 快速原型、Python 技术栈统一、适合数据应用 |
| 流式协议 | SSE | 单向推送够用、协议更轻、比 WebSocket 更适合 AI 流式场景 |
| 会话存储 | Redis List + Hash + Set | O(1)追加、Pipeline 批量操作、TTL 自动过期、无竞态条件 |
| 并发控制 | Lua 原子脚本 | 避免 INCR→检查→DECR 的 READ-MODIFY-WRITE 竞态条件 |
| 向量库 | FAISS | 开发友好零配置、10 万级文档秒级检索、生产可切换 Milvus |
| LLM | DeepSeek Chat | 中文效果好、API 兼容 OpenAI、streaming 稳定 |
| Embedding | BGE-base-zh-v1.5 | 中文 SOTA、本地部署、1024 维性价比高 |
| Reranker | BGE-Reranker-base | 与 Embedding 同厂商、中文精排效果好 |
| 容器化 | Docker Compose | 本地开发+演示够用，生产再上 K8s |

---

## 技术栈与技能映射

| 技术层 | 技术 | 面试考点 |
|--------|------|------|
| **前端** | Streamlit, httpx SSE 消费 | 响应式模型、session_state 持久化 |
| **API网关** | FastAPI, Depends, APIRouter | 依赖注入洋葱模型、模块化路由设计 |
| **流式输出** | SSE, StreamingResponse, asyncio | SSE 协议帧格式、TTFT 优化、CancelledError |
| **会话管理** | Redis List/Hash/Set, Pipeline, Lua | 数据结构选型、Pipeline 减少 RTT、Lua 原子性 |
| **容错** | Circuit Breaker, Sliding Window | 三态切换、ZSet 滑动窗口、防雪崩 |
| **Agent** | ReAct, LangGraph StateGraph, @tool | Agent 循环、工具调用协议、防无限循环 |
| **Prompt** | 三层架构, CoT | 模块化 Prompt、Chain of Thought |
| **RAG** | FAISS, BM25, RRF, CrossEncoder | 混合检索、RRF 无参融合、双塔 vs 交叉编码器 |
| **Embedding** | BGE, SentenceTransformer, Mean Pooling | 向量语义、池化策略、余弦相似度 |
| **部署** | Docker, docker-compose, Nginx | 多阶段构建、反向代理、SSE 缓冲禁用 |
| **测试** | pytest, fakeredis, httpx.AsyncClient | 测试金字塔、mock 策略、ASGI 直接测试 |

---

> **当前进度**：Phase 1 ✅ | Phase 2 ████░░░░ (schemas✅ security✅ 剩余 6 文件) | Phase 3-5 待开始
>
> **下一步**：Day 1 — 实现 `session_manager.py` + `stream.py`
