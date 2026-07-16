# Day 1 — 会话管理与 SSE 流式传输

> **今日目标**：实现两个核心基础设施模块 —— 基于 Redis 的会话管理器 + Server-Sent Events 流式协议层。这是整个对话系统的数据底座和传输通道。

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [Redis 数据结构全景图](#2-redis-数据结构全景图)
3. [SessionManager 逐方法详解](#3-sessionmanager-逐方法详解)
4. [SSE 协议与流式输出](#4-sse-协议与流式输出)
5. [核心技术原理](#5-核心技术原理)
6. [初学者常见疑问](#6-初学者常见疑问)
7. [面试模拟问答](#7-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `core/session_manager.py` | ~170 | 会话 CRUD + 消息历史存储，所有操作基于 Redis |
| `core/stream.py` | ~65 | SSE 协议格式化 + 生成器安全包装 |

---

## 2. Redis 数据结构全景图

这是理解 Day 1 代码的关键——先看清 Redis 里存了什么，再读代码就一目了然。

```
                                Redis 内存
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. 会话元数据 — Hash 类型                                        │
│     Key: sessions:{session_id}                                   │
│     ┌──────────────────────────────┐                             │
│     │ title:         "选SUV"       │                             │
│     │ user_id:       "user_001"    │                             │
│     │ created_at:    "1765874400"  │                             │
│     │ updated_at:    "1765878000"  │                             │
│     │ message_count: "4"           │  ← 注意：Redis Hash 的 value │
│     └──────────────────────────────┘     全部是字符串             │
│     TTL: 1800s (每次读/写时续期)                                   │
│                                                                  │
│  2. 消息历史 — List 类型                                          │
│     Key: sessions:{session_id}:messages                          │
│     ┌─────────────────────────────────────────┐                  │
│     │ [0] '{"role":"user","content":"推荐SUV"}'│  ← RPUSH 追加    │
│     │ [1] '{"role":"assistant","content":"...'}'│                │
│     │ [2] '{"role":"user","content":"预算25万"}'│                │
│     │ [3] ...                                  │                  │
│     └─────────────────────────────────────────┘                  │
│     TTL: 同会话元数据                                              │
│                                                                  │
│  3. 用户会话索引 — Set 类型                                        │
│     Key: user:{user_id}:sessions                                 │
│     ┌───────────────────────┐                                     │
│     │ "a1b2c3d4e5f6"        │                                     │
│     │ "e5f6g7h8i9j0"        │  ← SADD 添加、SMEMBERS 全取         │
│     │ "k1l2m3n4o5p6"        │  ← SREM 移除、SISMEMBER 判断归属     │
│     └───────────────────────┘                                     │
│     TTL: 1800s                                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 为什么这样设计？

| 问题 | 答案 |
|------|------|
| **为什么 Hash 存元数据？** | 可以单字段读写——`HSET sessions:abc title "新标题"` 只改标题不动其他。比整体 JSON 序列化省操作 |
| **为什么 List 存消息？** | 消息是时序数据，`RPUSH`（尾部追加）+ `LRANGE`（范围读取）天然对应 List 语义 |
| **为什么 Set 存会话索引？** | 一个用户多个会话，Set 自动去重、O(1) 判断归属、不需要关心顺序 |
| **为什么 TTL 是 1800 秒？** | 30 分钟无操作自动清理。每次读写续期 = 僵尸会话 30 分钟后自动消失，活跃会话永远不过期 |

---

## 3. SessionManager 逐方法详解

### 3.1 构造函数与 Key 生成

```python
class SessionManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client              # 外部注入的 Redis 连接
        self.ttl = settings.session_ttl_seconds  # 默认 1800
        self.max_concurrent = settings.max_concurrent_per_user  # 默认 3
```

**为什么不自己创建 Redis 连接？**

如果写成 `self.redis = redis.Redis(...)`，每个 SessionManager 都开新连接。100 个请求 = 100 个 TCP 连接，Redis 默认 maxclients 才 10000。依赖注入让你复用 `app.state` 中已有的连接池。

```python
@staticmethod
def _session_key(session_id: str) -> str:
    return f"sessions:{session_id}"          # → "sessions:a1b2c3d4e5f6"

@staticmethod
def _messages_key(session_id: str) -> str:
    return f"sessions:{session_id}:messages" # → "sessions:a1b2c3d4e5f6:messages"

@staticmethod
def _user_sessions_key(user_id: str) -> str:
    return f"user:{user_id}:sessions"        # → "user:user_001:sessions"
```

静态方法不依赖实例状态，`@staticmethod` 比 `@classmethod` 更合适。集中管理 key 命名，万一要改前缀，只改一处。

---

### 3.2 create_session — 创建会话

```python
session_id = uuid.uuid4().hex[:12]  # 12位十六进制 → 16^12 种组合
```

**为什么用 uuid4？** `uuid4` 是随机 UUID（122 位熵），`uuid1` 是时间+MAC 地址组合（会把服务器 MAC 地址暴露出去）。取前 12 位 hex，碰撞概率在十万个会话级别可忽略。

```python
pipe = self.redis.pipeline()
pipe.hset(self._session_key(session_id), mapping=session_data)  # 元数据
pipe.expire(self._session_key(session_id), self.ttl)            # TTL
pipe.sadd(self._user_sessions_key(user_id), session_id)         # 用户索引
pipe.expire(self._user_sessions_key(user_id), self.ttl)
pipe.expire(self._messages_key(session_id), self.ttl)           # 消息列表预设
await pipe.execute()
```

**5 条命令 → 1 次网络往返**。不用 Pipeline 的话，5 × 局域网 RTT(1ms) = 5ms。Pipeline 后约 1ms。高频接口场景下差异巨大。

---

### 3.3 get_session — 读取 + 自动续期

```python
data = await self.redis.hgetall(key)
if not data:
    return None

# 滑动过期：每次读都重置 TTL
await self.redis.expire(key, self.ttl)
await self.redis.expire(self._messages_key(session_id), self.ttl)
```

这叫**滑动过期**（Sliding Expiry）。固定过期是："会话创建后 30 分钟必死"。滑动过期是："最后活跃后 30 分钟后才死"。用户在聊天就不掉线，不活动才自动清理。

---

### 3.4 list_sessions — 批量查询

```python
session_ids = await self.redis.smembers(self._user_sessions_key(user_id))

pipe = self.redis.pipeline()
for sid in session_ids:
    pipe.hgetall(self._session_key(sid))  # N 个 HGETALL 打包
results = await pipe.execute()
```

**N 个会话只发 2 次网络请求**（SMEMBERS 1 次 + Pipeline 1 次），而不是 N+1 次。对 10 个会话 = 2ms vs 11ms。

---

### 3.5 add_message — 追加消息 + 更新计数

```python
pipe = self.redis.pipeline()
pipe.rpush(self._messages_key(session_id), msg)
pipe.expire(self._messages_key(session_id), self.ttl)
pipe.hincrby(self._session_key(session_id), "message_count", 1)  # 原子 ±1
pipe.hset(self._session_key(session_id), "updated_at", now)
pipe.expire(self._session_key(session_id), self.ttl)
await pipe.execute()
```

**HINCRBY 是原子操作**，不需要先 HGET→+1→HSET（两步之间有竞态）。`EXPIRE` 在每次写入时都调用，保证活跃会话的 TTL 不断被推后。

---

### 3.6 get_history — 取最近 N 条

```python
raw_messages = await self.redis.lrange(key, -limit, -1)
```

**`-limit` 负数索引**直接定位到倒数第 N 条。等价于 `LLEN → start = LLEN - limit → LRANGE start -1`，但一步完成。Redis 的 LRANGE 取少量元素是 O(N)（N=返回数量），不是 O(全列表长度)，所以即使列表有 1000 条，取最后 50 条也很快。

---

## 4. SSE 协议与流式输出

### 4.1 先理解 SSE 协议本身

SSE 是纯 HTTP 协议，底层用 `Transfer-Encoding: chunked`，消息格式极其简单：

```
data: {"type":"token","content":"你"}\n\n
data: {"type":"token","content":"好"}\n\n
data: {"type":"done","total_tokens":2}\n\n
```

每条消息 = `"data: "` + JSON + `"\n\n"`。两个连续换行符是协议的消息分隔符。

**SSE vs WebSocket：**

| 维度 | SSE | WebSocket |
|------|-----|-----------|
| 通信方向 | 服务器 → 客户端（单向） | 双向 |
| 协议 | 纯 HTTP，无升级握手 | HTTP → WebSocket 升级 |
| 穿透性 | 天然过代理/防火墙/Nginx | 部分代理不支持 |
| 浏览器 API | `new EventSource(url)` | `new WebSocket(url)` |
| 自动重连 | 内置 | 需手动实现 |
| LLM 流式场景 | ✅ 完美匹配 | ❌ 功能过剩 |

**对于 LLM 逐字输出这种场景，SSE 几乎是专为它设计的。**

---

### 4.2 stream.py 的职责

```
LLM API (外部)  → {"type":"token","content":"字"}  (dict 生成器)
                    │
                    ▼  sse_generator() 包装
    StreamingResponse  → 'data: {"type":"token","content":"字"}\n\n'  (str 生成器)
```

它只是格式转换器，不关心数据从哪来。Phase 2 用占位生成器测链路，Phase 3 换成 Agent 流，stream.py 一行不用改。

---

### 4.3 format_sse

```python
def format_sse(data: dict) -> str:
    json_str = json.dumps(data, ensure_ascii=False)
    return f"data: {json_str}\n\n"
```

**`ensure_ascii=False`** 知识点：

```python
# 默认 (ensure_ascii=True)
json.dumps({"content":"你好"})  # → '{"content":"\\u4f60\\u597d"}'
# ensure_ascii=False
json.dumps({"content":"你好"}, ensure_ascii=False)  # → '{"content":"你好"}'
```

前者把中文字符转义为 `\uXXXX`，增加传输体积且不可读。

---

### 4.4 sse_generator — 生成器包装 + 异常保护

```python
generated_count = 0  # ← 注意：在 try 外面

try:
    async for event in generator:
        yield format_sse(event)
        generated_count += 1

    yield format_sse({"type": "done", "total_tokens": generated_count})

except Exception as exc:
    yield format_sse({"type": "error", "message": str(exc)})
```

**两个关键点：**

1. **`generated_count` 在 try 外面**：如果客户端断开连接触发 `CancelledError`，Python 解释器在进入 `except` 时 `try` 块内变量可能未定义。放在外面保证访问安全。

2. **异常不往上抛**：SSE 流已经开始，HTTP 200 已经发出。如果中途抛异常，客户端只会看到截断的半条流——不知道是正常结束还是出错了。改为发送 `{"type":"error"}` 事件，客户端明确知道出错了。

---

## 5. 核心技术原理

### 5.1 Redis Pipeline：为什么快？

```
没有 Pipeline:
  客户端 ──HSET──→ Redis ──OK──→ 客户端  (RTT #1, ~1ms)
  客户端 ──EXPIRE─→ Redis ──OK──→ 客户端  (RTT #2, ~1ms)
  客户端 ──SADD──→ Redis ──OK──→ 客户端  (RTT #3, ~1ms)
  客户端 ──EXPIRE─→ Redis ──OK──→ 客户端  (RTT #4, ~1ms)
  总耗时: ~4ms (99% 是网络等待)

有 Pipeline:
  客户端 ──[HSET, EXPIRE, SADD, EXPIRE]──→ Redis
  客户端 ←──────── [OK, OK, OK, OK] ────── Redis
  总耗时: ~1ms
```

**Pipeline ≠ 事务**：Pipeline 打包的是**网络传输**，命令还是逐个执行的。如果第三条 SADD 失败，前两条 HSET+EXPIRE 已经执行了，不会回滚。要原子性需要 `MULTI/EXEC` 或 Lua 脚本。

---

### 5.2 滑动过期 vs 固定过期

```
固定过期:
  会话创建时间: 14:00:00  →  TTL 1800s  → 过期时间: 14:30:00
  用户在 14:28:00 还在聊天，2 分钟后会话被删 — 体验差

滑动过期:
  会话创建时间: 14:00:00  →  TTL 1800s  → 过期: 14:30:00
  用户 14:05 发消息 → EXPIRE 重置 → 过期: 14:35:00
  用户 14:20 发消息 → EXPIRE 重置 → 过期: 14:50:00
  ...活跃用户永远不会掉线
```

---

### 5.3 Redis List 做消息队列的原理

Redis List 底层是 `quicklist`（3.2+ 版本），结合了 `linkedlist` 和 `ziplist` 的优点：

- **LinkedList** 的优势：头尾插入 O(1)（LPUSH/RPOP）
- **Ziplist** 的优势：连续内存，省内存，缓友好
- **Quicklist**：一个大链表，每个节点是小 ziplist。兼顾时间和空间

对消息历史场景：RPUSH（尾部追加）和 LRANGE（范围读）都是高效操作。

---

### 5.4 SSE 与 HTTP 的底层关系

SSE 跑在标准 HTTP/1.1 之上，底层用 `Transfer-Encoding: chunked`：

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Transfer-Encoding: chunked

1E\r\n                    ← 第一个 chunk 长度 = 30 字节
data: {"type":"token","content":"你"}\n\n
\r\n
1E\r\n                    ← 第二个 chunk 长度 = 30 字节
data: {"type":"token","content":"好"}\n\n
\r\n
0\r\n                     ← 长度为 0 表示流结束
```

**Nginx 关键配置**：`proxy_buffering off;` —— 如果不开，Nginx 会等所有 chunk 到齐了再转发，SSE 变成普通 HTTP 长请求。

---

### 5.5 asyncio 单线程并发模型

Python 的 `async/await` 不是多线程！是**单线程协作式并发**。

```
单线程事件循环:
  请求1: 等待Redis → 让出控制权 ──────────────→ 收到结果 → 恢复执行
  请求2: 等待Redis → 让出控制权 → 收到结果 → 恢复执行
  请求3: 等待Redis → 让出控制权 → 收到结果 → 恢复执行

时间线上的"同时"是因为它们都在 I/O 等待时让出了控制权。
事件循环 = 一个无限循环，不断捡起"可执行"的任务。
```

**I/O 密集 = async 最擅长的场景**。CPU 密集不适合——计算的时候不让出控制权，其他协程就被饿死了。

---

## 6. 初学者常见疑问

**Q: `session_manager.py` 中为什么 Hash 的 value 全是字符串？**

Redis Hash 的 value 只能是字符串（或数字会被转成字符串）。`'message_count': '0'` 不能写成 `'message_count': 0`。取出时需要 `int(data['message_count'])`。

**Q: `uuid4().hex[:12]` 安全吗？为什么不直接用 `uuid4()`？**

`uuid4()` 返回 `a1b2c3d4-e5f6-7890-abcd-ef1234567890`（36 字符），URL 中太长。取前 12 位 hex 仍有 `16^12 ≈ 2.8×10^14` 种组合，商用场景撞车概率 ≈ 0。如果你要绝对安全（比如支付系统），用完整 UUID 或 `secrets.token_hex(16)`。

**Q: `json.dumps(msg, ensure_ascii=False)` 在 Redis 中存了什么？**

```
# ensure_ascii=True（默认）
'{"role":"user","content":"推荐\\u597dSUV"}'

# ensure_ascii=False
'{"role":"user","content":"推荐好SUV"}'
```

Redis 中的中文直接可读，对调试友好。

**Q: 流式响应和非流式响应都调用同一个 `/chat` 端点，FastAPI 怎么区分？**

StreamingResponse 和 JSONResponse/Pydantic Response 是互斥的——如果你 return 了一个 StreamingResponse，FastAPI 就不做 JSON 序列化。同一个端点可以返回不同类型。用 `body.stream` 参数决定走哪个分支。

**Q: 为什么 `get_history` 用 `LRANGE key -50 -1` 而不是 `LRANGE key 0 -1`？**

`LRANGE key 0 -1` 取全部消息。如果单次会话 1000 条消息，全部取出浪费内存和网络。取最近的 50 条作为 LLM 上下文足够了（约 2000-4000 tokens）。

---

## 7. 面试模拟问答

> **Q: 说一下这个项目中会话是怎么持久化的？**

用 Redis 的三种数据结构：Hash 存会话元数据（标题、创建时间等），List 存消息历史（RPUSH 追加），Set 存用户→会话的映射。TTL 设 30 分钟，每次读写续期（滑动过期），所以活跃会话不会丢失，不活跃会话自动清理。所有写操作用 Pipeline 合并网络往返。

> **Q: Pipeline 和事务的区别是什么？**

Pipeline 打包网络传输，命令顺序执行但不保证原子性——中间有命令失败，其他命令照样执行。MULTI/EXEC 事务保证原子性——要么全执行，要么全不执行。Pipeline 快在减少 RTT，事务强在原子性。它们可以组合使用（在 MULTI/EXEC 中 Pipeline 发送）。

> **Q: SSE 和 WebSocket 怎么选？**

LLM 流式输出场景选 SSE：单向推送 + 基于 HTTP（天然穿透代理）+ 浏览器原生 EventSource（自动重连）。WebSocket 适合双向实时通信（如在线协作、游戏），但对 LLM 流式来说功能过剩且需要额外处理代理问题。

> **Q: 如果客户端在流式传输中途断开，服务器怎么知道？**

asyncio 检测客户端断开会向生成器抛 `CancelledError`。`sse_generator` 的 `except Exception` 会捕获它，发送 error 事件后优雅退出。`generated_count` 放在 try 外部就是为了在断开时能安全记录"已经发了多少个 token"。

> **Q: 为什么存消息用 Redis 而不是数据库？**

对话消息的特征：写频繁（每轮对话两次写入）、读频繁（每次对话都要读历史）、有自然过期（会话结束 TTL 清理）、结构简单（JSON 键值对）。Redis 的内存存储 + List 结构 + TTL 机制天然匹配这个场景。如果用 PostgreSQL，磁盘 I/O + 定期清理的定时任务，开发成本和运维成本都高很多。

---

## 附：今日文件依赖关系

```
config.py (settings) ──→ session_manager.py ──→ deps.py (Day 3)
                     ──→ stream.py         ──→ routes/chat.py (Day 3)
```

明日的 `resilience.py` 不依赖今天的内容，独立开发。后天 Day 3 会把三者组装起来。
