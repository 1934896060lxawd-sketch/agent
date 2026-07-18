# Day 7 — Streamlit 前端（上）：主界面 + API 客户端

> **今日目标**：用 Streamlit 搭建智能导购对话界面，实现 httpx SSE 流式消费 + 会话状态管理。Day 7 是 Phase 4 的前端上半场——建立前后端通信桥梁，让用户在浏览器里体验流式打字效果。

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [前后端通信全景](#2-前后端通信全景)
3. [Streamlit 响应式执行模型](#3-streamlit-响应式执行模型)
4. [SSE 流式消费：httpx 异步逐行解析](#4-sse-流式消费httpx-异步逐行解析)
5. [APIClient 设计](#5-apiclient-设计)
6. [主入口 app.py 设计](#6-主入口-apppy-设计)
7. [核心技术原理](#7-核心技术原理)
8. [初学者常见疑问](#8-初学者常见疑问)
9. [面试模拟问答](#9-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `frontend/api_client.py` | 174 | `APIClient` 类：`chat_stream()` httpx SSE 消费 + `chat_sync()` 非流式 + 5 个会话 CRUD 方法 |
| `frontend/app.py` | 161 | Streamlit 主入口：`session_state` 持久化 + `@st.cache_resource` 缓存 + `asyncio.run()` 桥接 |

---

## 2. 前后端通信全景

```
用户浏览器 http://localhost:8501
        │
        ▼
┌─────────────────────────────────────────┐
│  app.py (Streamlit 主入口)               │
│  · st.chat_input() → 获取用户输入       │
│  · asyncio.run() → 桥接异步 API 调用    │
│  · st.empty() → 流式打字占位符          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  api_client.py (APIClient)              │
│  · chat_stream()                        │
│    → httpx.stream("POST", /chat)        │
│    → aiter_lines() 逐行消费 SSE         │
│    → json.loads() 解析事件              │
│    → AsyncGenerator yield event dict    │
│  · chat_sync()                          │
│    → httpx.post() → response.json()     │
│  · create/list/rename/delete session    │
│    → httpx.*() CRUD                     │
└──────────────┬──────────────────────────┘
               │ HTTP POST /chat (stream=true)
               │ Authorization: Bearer sk-dev-user-001
               ▼
┌─────────────────────────────────────────┐
│  FastAPI 后端 (localhost:8000)           │
│  → Auth → Rate Limit → Circuit Breaker  │
│  → Agent ReAct → RAG → SSE Streaming    │
└─────────────────────────────────────────┘
```

**为什么前后端分离？** 后端 FastAPI 是纯 API 服务，可以被任何客户端调用（curl、Postman、手机 App）。Streamlit 只是前端选项之一，后续可以替换为 React/Vue 而不影响后端。

---

## 3. Streamlit 响应式执行模型

### 3.1 脚本重跑 vs 事件驱动

传统前端框架（React/Vue）是**事件驱动**：点击按钮 → 触发特定回调函数 → 更新局部 DOM。Streamlit 是**脚本重跑**：

```
用户点击按钮
  → Streamlit 从上到下重新执行 app.py 全部代码
  → 只有被修改的 st.* 组件重新渲染
  → 未修改的组件用缓存结果跳过
```

**为什么这样设计？** Streamlit 定位是"数据应用的 Python 替代品"——用 Python 写前端，不需要 JS/HTML。脚本重跑模型意味着你只需要关心 Python 变量的变化，Streamlit 自动 diff 并更新 DOM。

### 3.2 session_state 持久化

因为每次交互后代码重跑，普通 Python 变量会丢失。`st.session_state` 是唯一的持久化手段：

```python
# ❌ 错误：每次重跑 messages 都会被重置为空列表
messages = []

# ✅ 正确：用 session_state 跨重跑持久化
if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.messages.append({"role": "user", "content": "你好"})
```

**session_state 的底层实现**：Streamlit 在 Tornado 服务器端维护了一个 dict，key 是 session ID（WebSocket 连接标识），value 是用户的 session_state dict。每次脚本重跑时，Streamlit 把对应的 dict 注入到 `st.session_state` 对象中。

### 3.3 @st.cache_resource 缓存

重量级资源（如 APIClient、数据库连接）不应该每次重跑都重新创建：

```python
@st.cache_resource
def get_api_client() -> APIClient:
    return APIClient(base_url="http://localhost:8000")

# 首次调用: 创建 APIClient，存入缓存
# 后续调用: 直接返回缓存的实例
# 除非函数体变化（content hash 改变），否则永不过期
```

与 `@st.cache_data` 的区别：
| | @st.cache_data | @st.cache_resource |
|------|------|------|
| 用途 | 缓存数据（DataFrame、list、dict） | 缓存资源（连接、模型、客户端） |
| TTL | 可配置过期时间 | 永不过期（除非代码改变） |
| 序列化 | pickle 序列化存储 | 不序列化，直接返回对象引用 |
| 线程安全 | 是 | 是 |

APIClient 不需要序列化（它只是持有 base_url 和 api_key），也不需要过期，所以用 `@st.cache_resource`。

---

## 4. SSE 流式消费：httpx 异步逐行解析

### 4.1 为什么不能用浏览器 EventSource？

浏览器原生的 `EventSource` API 只支持 GET 请求，有两个致命限制：

1. **不能传 POST body**：AI 对话的 query 太长（几百字），放在 URL query string 里有长度限制和编码问题
2. **不能自定义 Header**：无法设置 `Authorization: Bearer xxx`，认证无法通过

所以后端虽然是标准 SSE 协议，前端必须手动消费。浏览器端用 `fetch + ReadableStream`，Python/Streamlit 端用 `httpx + aiter_lines()`。

### 4.2 httpx.stream() 底层原理

```python
async with httpx.AsyncClient() as client:
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])  # 去掉 "data: " 前缀
                yield event
```

`httpx.stream()` 与 `httpx.post()` 的核心区别：

| | httpx.post() | httpx.stream() |
|------|------|------|
| 响应读取 | 等待全部响应体到达，一次性返回 | 建立连接后逐块读取 |
| 内存 | 完整响应体大小 | 每行大小（恒定） |
| 首字节时间 | 等全部生成完 | 第一个 token 到达即可读取 |
| 适用场景 | JSON API（数据量小） | SSE 流（数据量大、实时推送） |

底层 TCP 连接持续接收数据分片，HTTP 解析器逐行分割。`aiter_lines()` 每次 yield 一个以 `\n` 结尾的完整行，内存占用恒定。

### 4.3 SSE 事件解析

后端发送的 SSE 格式：
```
data: {"type":"source","documents":[...]}\n\n
data: {"type":"token","content":"根"}\n\n
data: {"type":"token","content":"据"}\n\n
data: {"type":"done","total_tokens":342}\n\n
```

前端解析流程：
```
aiter_lines()
  → "data: {"type":"token","content":"根"}"
  → line.startswith("data: ") ?  Yes
  → data_str = line[6:]  # 去掉 "data: " 前缀
  → data_str == "[DONE]" ?  No
  → json.loads(data_str) → {"type":"token","content":"根"}
  → yield event
  → event["type"] == "done" ?  No → 继续
  → ...
  → event["type"] == "done" ?  Yes → break
```

**为什么用 AsyncGenerator 返回？** `chat_stream()` 返回 `AsyncGenerator[dict, None]`，调用方用 `async for event in client.chat_stream()` 逐事件消费。这种模式让前端可以边接收边渲染，实现打字机效果。

---

## 5. APIClient 设计

### 5.1 类结构

```python
class APIClient:
    def __init__(self, base_url, api_key, timeout=120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
```

**为什么用 @property？** `_headers` 每次访问都重新构建 dict，保证 api_key 变化时自动反映。如果写成 `self._headers = {...}`，修改 api_key 后需要手动重建。虽然当前场景下 api_key 不变，但这是防御性设计。

### 5.2 方法设计

| 方法 | HTTP | 返回 | 用途 |
|------|------|------|------|
| `chat_stream(query, session_id)` | POST /chat (stream) | AsyncGenerator[dict] | 流式对话，逐事件 yield |
| `chat_sync(query, session_id)` | POST /chat (非 stream) | dict | 非流式对话，返回完整 JSON |
| `create_session(title)` | POST /sessions | dict | 创建新会话 |
| `list_sessions()` | GET /sessions | list[dict] | 获取会话列表 |
| `rename_session(sid, title)` | PATCH /sessions/{sid} | dict | 重命名 |
| `delete_session(sid)` | DELETE /sessions/{sid} | bool | 删除 |
| `get_history(sid)` | GET /sessions/{sid}/history | dict | 加载历史消息 |

**导航类方法为什么不是流式的？** 会话创建、列表、删除等操作数据量小（<1KB），不需要流式传输。只有 `/chat` 的 AI 回答需要流式（因为生成耗时、内容长）。

### 5.3 错误处理策略

```python
if response.status_code != 200:
    error_text = await response.aread()
    yield {"type": "error", "message": f"HTTP {response.status_code}: ..."}
    return
```

**为什么不在 APIClient 中 raise？** APIClient 将 HTTP 错误统一转换为 `{"type": "error", ...}` 事件，和正常的 `token`/`done` 事件一样通过 AsyncGenerator 返回。调用方不需要区分"网络错误"和"后端业务错误"——统一处理 `event["type"] == "error"` 即可。

---

## 6. 主入口 app.py 设计

### 6.1 页面结构

```
┌────────────────────────────────────────────────┐
│  🚗 智能汽车导购助手                            │
│  基于 DeepSeek + RAG 知识库的新能源汽车导购     │
├──────────────┬─────────────────────────────────┤
│  💬 会话列表  │                                 │
│              │  🤖 根据您的25万预算...           │
│  ➕ 新建会话  │  (流式打字中...)                  │
│              │                                 │
│  🔵 对话 1   │  📚 参考来源 (3)                 │
│  · 对话 2    │    #1 [vehicles.json] 0.856     │
│  · 对话 3    │    #2 [guide-xxx.md] 0.723     │
│              │    #3 [faq-xxx.md] 0.691       │
│              │                                 │
│              ├─────────────────────────────────┤
│              │  请输入您的问题...      📤       │
└──────────────┴─────────────────────────────────┘
```

### 6.2 初始化流程

```python
def init_session_state():
    defaults = {
        "messages": [],          # 对话历史
        "session_id": "default", # 当前会话
        "tool_events": [],       # 工具调用事件
        "pending_sources": [],   # 检索来源
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
```

**关键设计**：`if key not in st.session_state` 保证只在首次加载时初始化。如果改成 `st.session_state[key] = default`，每次重跑都会重置，丢失之前的对话。

### 6.3 对话流程

```
1. 用户输入 "25万预算推荐什么SUV？"
2. st.chat_input() 返回 prompt
3. 添加 {"role":"user", "content":prompt} 到 messages
4. 渲染用户消息气泡
5. 创建 assistant 的 st.empty() 占位符
6. asyncio.run(_stream()) 调用 APIClient.chat_stream()
7. 收到 token → placeholder.markdown(full_text + "▌")  更新打字效果
8. 收到 source → 更新工具调用面板
9. 收到 done → placeholder.markdown(full_text)  去掉光标
10. 保存 assistant 消息到 messages
```

### 6.4 asyncio.run() 桥接

Streamlit 是同步框架，APIClient 是异步的。桥接方式：

```python
async def _stream():
    async for event in client.chat_stream(prompt, session_id):
        # 处理事件...

asyncio.run(_stream())
```

`asyncio.run()` 创建一个新的事件循环，运行协程直到完成，然后关闭循环。每次对话调用一次 `asyncio.run()`，意味着每次对话有独立的事件循环。

**为什么不用 `asyncio.get_event_loop().run_until_complete()`？** 因为 Streamlit 的 Tornado 服务器已经有一个事件循环在运行，`run_until_complete()` 会冲突。`asyncio.run()` 创建独立循环，互不干扰。

---

## 7. 核心技术原理

### 7.1 Streamlit 的渲染 diff 算法

Streamlit 不是每次重跑都重新渲染整个页面。它使用类似 React 的虚拟 DOM diff：

1. 记录本次脚本执行产生的所有 `st.*` 调用及其参数
2. 与上次执行的结果做 diff
3. 只向前端发送变化的组件

例如：
```python
st.markdown("Hello")         # 上次也是 "Hello" → 不渲染
st.button("Click me")        # 上次也是 "Click me" → 不渲染
st.markdown(full_text)       # 内容变了 → 只更新这个组件！
```

这就是为什么 `st.empty()` + 占位符循环更新只有占位符区域在刷新，页面其他部分不动。

### 7.2 HTTP 流式响应的 TCP 层面

当后端 `StreamingResponse` 开始生成时：

```
客户端                         服务端
  │                              │
  │──── TCP SYN ────────────────→│
  │←─── TCP SYN-ACK ─────────────│  ← 三次握手
  │──── TCP ACK ────────────────→│
  │──── HTTP POST /chat ────────→│
  │←─── HTTP 200 + Headers ──────│  ← 响应头先到
  │←─── data: {"type":"token"...}│  ← 第一个 chunk
  │←─── data: {"type":"token"...}│  ← 第二个 chunk
  │     ...（持续接收）...        │
  │←─── data: [DONE]             │
  │←─── TCP FIN ─────────────────│  ← 四次挥手
```

关键点：**响应头和第一个数据块是分开到达的**。`httpx.stream()` 在收到响应头后就返回 `response` 对象，`aiter_lines()` 逐行消费后续的数据块。这保证了 TTFT（首 token 时间）最优。

### 7.3 AsyncGenerator 的背压机制

`chat_stream()` 返回 `AsyncGenerator`，不是一次性返回所有事件。这意味着：

- **生产端**（httpx）：收到一行就 yield 一行
- **消费端**（app.py）：处理完一个事件才请求下一个
- **自动背压**：消费端处理慢时，TCP 接收缓冲区满 → 服务端发送变慢 → 形成自然流控

如果一次性返回所有事件（如 `return [event1, event2, ...]`），大响应会撑爆内存。

---

## 8. 初学者常见疑问

**Q: Streamlit 和 React/Vue 有什么区别？什么场景用哪个？**

Streamlit 是"数据应用框架"——Python 开发者不需要写 HTML/CSS/JS，几行代码就能搭建数据仪表盘、AI 对话界面。适合内部工具、原型验证、数据科学展示。React/Vue 是通用前端框架，适合面向 C 端用户的产品、复杂交互、自定义 UI。本项目用 Streamlit 是因为：①全 Python 技术栈，②快速原型，③AI 对话场景不需要复杂前端交互。

**Q: 为什么 chat_stream() 用 AsyncGenerator 而不是返回一个列表？**

AsyncGenerator 是"惰性求值"——收到一行 yield 一行，调用方不用等全部数据到齐就能开始渲染。返回 `list[dict]` 需要等所有 SSE 行都接收完，用户在此期间看白屏。这是流式体验的核心：减少 TTFT（首 token 时间）。

**Q: asyncio.run() 每次对话都创建新事件循环，不浪费吗？**

Python 事件循环的创建开销在微秒级（<10μs），可以忽略不计。真正耗时的是等待网络 I/O（百毫秒级）。`asyncio.run()` 的简单性（每次独立循环、自动清理）远大于微秒级的创建开销。

**Q: @st.cache_resource 和 @st.cache_data 的 content hash 是什么？**

Streamlit 对函数体源码计算 hash 值。当源码变化时 hash 改变 → 缓存失效 → 重新执行函数。这就是为什么修改 `get_api_client()` 函数体后会自动重建 APIClient——不需要手动清除缓存。

**Q: 如果后端挂了，前端会怎样？**

`httpx.stream()` 连接失败会抛 `httpx.ConnectError`，在 `_stream()` 的 try/except 中被捕获，显示 `st.error()` 红色提示。不会导致 Streamlit 崩溃。chat_stream() 中的 HTTP 非 200 响应被包装为 `{"type":"error"}` 事件，正常走 SSE 管道。

---

## 9. 面试模拟问答

> **Q: 你们的前端是怎么消费 SSE 流的？为什么不用 EventSource？**

我们用 `httpx.stream()` + `aiter_lines()` 手动消费 SSE。浏览器原生 `EventSource` 只支持 GET 请求，无法传 POST body 和自定义 Header（如 `Authorization`）。AI 对话的 query 可能很长（几百字），放 URL query string 有编码和长度问题。所以我们手动解析 SSE 协议——逐行读取 `data: {...}` 前缀，`json.loads()` 解析事件 dict，用 AsyncGenerator 逐事件 yield 给上层。

> **Q: Streamlit 的 session_state 和 Redis 的 session 是什么关系？**

它们是完全不同层级的概念。Streamlit 的 `session_state` 是前端 WebSocket 连接级别的状态（用户在浏览器里看到的对话历史），存的是 Python 对象，浏览器关闭就没了。Redis 的 session 是后端业务级别的持久化存储（用户的历史对话数据），存的是 JSON 字符串，有 TTL 过期机制。前端切回一个旧会话时，需要从 Redis 加载历史消息到 `session_state.messages`——这是前后端状态同步。

> **Q: 流式打字效果是怎么实现的？**

用 `st.empty()` 创建一个空白占位符，每收到一个 token 就调用 `placeholder.markdown(full_text + "▌")` 更新占位符内容。`full_text` 是累计的完整文本，`▌` 是闪烁光标符号。收到 `done` 事件后去掉光标：`placeholder.markdown(full_text)`。`st.empty()` 是 Streamlit 的"可变占位符"，类似 React 的 ref——同一个 DOM 节点被多次更新但位置不变。

> **Q: 你们的 API 客户端怎么处理网络错误？**

APIClient 区分两类错误：① HTTP 状态码非 200（如 429 限流、503 索引未就绪）→ 包装为 `{"type":"error","message":"..."}` 事件，通过 AsyncGenerator 正常返回，前端显示红色错误提示；② 网络连接失败（ConnectError、Timeout）→ 在 app.py 的 try/except 中捕获，同样显示错误提示。两种错误都不会导致前端崩溃。这个设计原则是"错误也是事件"——让异常走正常管道，调用方不需要区分"正常结果"和"错误结果"。

> **Q: 为什么 Streamlit 适合 AI 应用的前端？**

三个原因：① Python 技术栈统一——后端用 FastAPI，前端用 Streamlit，全团队只需要 Python 技能；② 快速原型——不需要 HTML/CSS/JS，几行代码就能搭出对话界面，MVP 开发速度快 5-10 倍；③ 流式支持好——`st.empty()` 天然支持逐 token 更新，不需要手动管理 WebSocket 或 DOM 操作。局限是自定义样式有限，面向 C 端的高要求产品还是用 React。

> **Q: asyncio.run() 和直接 await 有什么区别？**

`await` 只能在 `async def` 函数内部使用，需要一个外层事件循环。Streamlit 的脚本是同步的（没有事件循环），所以不能直接 `await client.chat_stream()`。`asyncio.run()` 做了三件事：① 创建一个新的事件循环，② 在里面运行协程直到完成，③ 关闭循环并返回结果。缺点是循环创建和销毁有微不足道的开销，优点是简单安全（每次独立循环，不会泄漏资源）。

---

## 附：今日文件依赖关系

```
app.py (Streamlit 主入口)
  ├── api_client.py (APIClient)
  │     └── httpx → FastAPI /chat + /sessions
  ├── components/chat.py (渲染消息、来源)
  ├── components/sidebar.py (会话列表管理)
  └── components/tools.py (工具调用可视化)

后端:
  FastAPI:8000
    ├── POST /chat (stream=true → SSE)
    ├── POST /sessions (CRUD)
    └── GET /sessions (list)
```
