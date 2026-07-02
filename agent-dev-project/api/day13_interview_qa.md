# Day 13 面试题：Redis 会话管理 —— 从"每次失忆"到"持久记忆"

> 对应文件：`api/session_manager.py`
> 核心能力：Redis 对话历史持久化、TTL 滑动过期、API Key 身份校验、Lua 脚本原子并发控制、会话 CRUD 完整 API

---

## 为什么需要 Redis 会话管理？

Day 11/12 的 `/chat` 每次请求都带 `session_id`，但只是个字符串——对话历史不存在任何地方：

```
Day 12 现状：
  Client ─"小米SU7续航多少？"─→ /chat ─→ 回答
  Client ─"那充电快吗？"─────→ /chat ─→ ❌ 不知道"那"指什么

Day 13 Redis 改造：
  Client ─"小米SU7续航多少？"─→ /chat ─→ 回答 ─→ 存入 Redis
  Client ─"那充电快吗？"─────→ /chat ─→ 从 Redis 读 history ─→ 理解"那"=小米SU7
```

---

## 为什么选 Redis 而不是 dict/SQLite？

| 方案 | 持久化 | 多 Worker 共享 | TTL 自动过期 | 性能 |
|------|--------|--------------|-------------|------|
| `dict` | ❌ 重启丢失 | ❌ 每个 worker 独立 | ❌ 手动清理 | 高 |
| JSONL 文件 | ✅ | ❌ 并发写冲突 | ❌ 手动清理 | 低 |
| SQLite | ✅ | ⚠️ 写锁竞争 | ❌ 手动清理 | 中 |
| **Redis** | ✅ RDB/AOF | ✅ 天然共享 | ✅ EXPIRE 原生 | 极高 |

**面试话术**："Redis 做会话存储的三个关键：TTL 自动清理过期会话（省内存）、Pipeline 批量读写（降低 RTT）、分布式锁防止同 session 并发写入导致的 history 混乱。"

---

## Day 13 代码架构全景

```
请求进入 POST /chat
  │
  ▼
┌─ HTTPBearer ────────────────────────┐
│  Authorization: Bearer sk-xxx      │
│  → get_current_user → user_id      │
│  失败 → 401                        │
└────────────────┬────────────────────┘
                 ▼
┌─ 并发控制 ──────────────────────────┐
│  acquire_slot(user_id, max=3)      │
│  → Lua 原子 INCR + 上限检查        │
│  超限 → 429 + Retry-After          │
└────────────────┬────────────────────┘
                 ▼
┌─ 会话管理 ──────────────────────────────┐
│  ① get_history(session_id)  ──取历史── │
│  ② add_message("user", query) ──存消息  │
│  ③ touch_session              ──刷 TTL  │
│  ④ RAG + LLM 生成 answer               │
│  ⑤ add_message("assistant", answer)    │
└────────────────┬────────────────────────┘
                 ▼
┌─ 并发释放 ─────────────────────────┐
│  release_slot(user_id)            │
│  finally 保证一定执行              │
└────────────────────────────────────┘
```

---

## 五个练习对照表

| 练习 | Redis 数据结构 | 关键命令 | 验证方式 |
|------|-------------|---------|---------|
| 1. 对话存取 | List | `RPUSH` + `LRANGE` + `Pipeline` | 发消息后查 `/sessions/{id}/history` |
| 2. TTL 过期 | Key-level TTL | `EXPIRE` 滑动刷新 | `redis-cli TTL session:xxx:messages` |
| 3. 身份校验 | — | `HTTPBearer` + `Depends` 依赖注入 | 不带 `Authorization` 头 → 401 |
| 4. 并发控制 | String (计数器) | Lua 脚本原子 `INCR` + 上限 | 同时发 4 请求，第 4 个 → 429 |
| 5. 会话 CRUD | Hash + Set | `HSET` / `HGETALL` / `SADD` / `SREM` | 创建 → 列表 → 改名 → 删除 全流程 |

---

## 练习 1：Redis 存储对话历史

**数据结构设计**：

```
session:{sid}:messages  →  List<JSON>    对话消息（RPUSH 追加，LRANGE 取尾部）
session:{sid}:meta      →  Hash          标题/用户/创建时间
user:{uid}:sessions     →  Set<sid>      用户拥有的会话 ID 集合
user:{uid}:concurrency  →  String        当前并发请求计数
```

**为什么用 List 而不是 String(JSON)**：

| 操作 | List | String(JSON) |
|------|------|-------------|
| 追加一条消息 | `RPUSH` O(1) | 读整个 JSON → 改 → 写整个 JSON |
| 取最近 N 条 | `LRANGE -N -1` | 读整个 JSON → 切片 |
| 取消息总数 | `LLEN` O(1) | 读整个 JSON → `len()` |
| 并发安全 | 天然追加 | 读改写 → 竞态条件 |

---

## 练习 2：TTL 滑动过期

**两种 TTL 策略**：

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| 固定过期 | 创建时设 30 分钟，到点删除 | 简单场景 |
| **滑动过期（Day 13）** | 每次活动刷新，活跃用户永不过期 | 对话类产品 |

```python
async def touch_session(self, session_id: str):
    # 每次请求调用，把 TTL 重新设为 30 分钟
    async with self.redis.pipeline() as pipe:
        pipe.expire(self._msg_key(session_id), self.ttl)
        pipe.expire(self._meta_key(session_id), self.ttl)
        await pipe.execute()
```

**面试话术**："TTL 的核心价值不是省 Redis 内存，而是自动清理僵尸会话。一个用户打开网页后关了浏览器，这个 session 永远不会再有新请求，数据就是垃圾。滑动过期比固定过期更合理——30 分钟不使用才判定为过期。"

---

## 练习 3：用户身份校验

```python
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    api_key = credentials.credentials
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return API_KEYS[api_key]

# 使用：在路由参数中注入
@app.post("/chat")
async def chat(req: ChatReq, uid: str = Depends(get_current_user)):
    ...
```

**`Depends` 依赖注入的执行顺序**：FastAPI 先解析 `Authorization: Bearer <key>` 请求头 → 调用 `get_current_user` → 返回值注入 `uid` 参数 → 然后才执行路由函数体。校验失败直接 401，不进入业务逻辑。

**API Key vs JWT**：

| | API Key | JWT |
|---|---|---|
| 复杂度 | 低（查表匹配） | 中（加密 + 过期 + 刷新） |
| 无状态 | ❌ 需查数据库 | ✅ Token 自包含 |
| 撤销 | 立即（删 key） | 需黑名单或等过期 |
| Day 13 建议 | ✅ 先用这个 | 多服务场景再升级 |

---

## 练习 4：并发限制（Lua 脚本原子操作）

**为什么要用 Lua 脚本**：

```
❌ 非原子操作（有竞态条件）：
  Worker A: GET counter → 2
  Worker B: GET counter → 2         ← 两个都读到 2！
  Worker A: INCR → 3 ✅ 通过了
  Worker B: INCR → 4 ✅ 也通过了  ← 实际 4 个并发，超标！

✅ Lua 脚本（Redis 单线程内执行）：
  Worker A: EVAL → INCR + 检查 + 退回（三步在一原子步骤）
  Worker B: 等 A 执行完 → 读到 counter=3 → 退回 → 返回 0
```

```lua
-- acquire_slot Lua 脚本
local current = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
if current > tonumber(ARGV[1]) then
    redis.call('DECR', KEYS[1])   -- 超限，退回
    return 0
end
return 1
```

**槽位 TTL 的作用**：设为 60 秒——防止进程崩溃后 `release_slot` 没执行，槽位永久不释放。60 秒足够一次 `/chat` 请求完成，超时后自动回收。

**`finally` 保证释放**：

```python
if not await session_mgr.acquire_slot(uid, max_concurrent=3):
    raise HTTPException(status_code=429)
try:
    # ... 业务逻辑 ...
finally:
    await session_mgr.release_slot(uid)  # 异常也要释放
```

---

## 练习 5：会话 CRUD API

| 端点 | 方法 | Redis 操作 | 说明 |
|------|------|-----------|------|
| `/sessions` | POST | `HSET` + `SADD` | 创建会话 |
| `/sessions` | GET | `SMEMBERS` + `HGETALL` | 列表（按时间倒序） |
| `/sessions/rename` | PATCH | `HSET` | 重命名 |
| `/sessions/{id}` | DELETE | `DEL` × 2 + `SREM` | 删除（消息 + 元信息 + 关联） |
| `/sessions/{id}/history` | GET | `LRANGE` + `TTL` | 历史消息 + 剩余存活时间 |

**Pipeline 的价值**：`add_message` 一次执行 `RPUSH` + `EXPIRE` × 2 + `SADD` = 四条命令。不用 Pipeline 需要 4 次网络往返（4 × RTT ≈ 4ms），Pipeline 只需要 1 次（1ms）。高并发下差距巨大。

---

## Q1：Redis List（RPUSH/LRANGE）vs String（GET/SET），为什么 Day 13 选 List？

**一句话**：对话消息是"只追加不修改"的序列，List 的 RPUSH 追加 O(1) + LRANGE 尾部读取 = 完美匹配；String 每次追加需要读→改→写整个 JSON，消息多了就是性能灾难。

**面试话术**："Redis 的数据结构选型体现了'数据访问模式决定存储结构'的原则。对话历史有三个特征：频繁追加、只读尾部、不定长。List 的 RPUSH + LRANGE 天然支持这种模式。如果选 String，每次追加都要 GET → 反序列化 → append → 序列化 → SET，消息日志 100 条时每次操作都要搬运 100 条。"

---

## Q2：Pipeline 把多条命令打包，具体优化了多少？

**一句话**：不用 Pipeline = N 条命令 N 次网络往返；用 Pipeline = N 条命令 1 次网络往返。在 RTT = 1ms 的场景下，`add_message` 的 4 条命令从 4ms 降到 1ms。

**面试话术**："Pipeline 不是批处理——Redis 还是逐条执行你的命令，只是把请求打包在一个 TCP 帧里发出。它的优化点在网络层，不在 Redis 执行层。生产建议：Pipeline 里放 3-10 条命令最佳，超过 100 条分批发，避免单个 Pipeline 太大阻塞其他客户端。"

---

## Q3：并发控制为什么必须用 Lua 脚本？`INCR` 本身就是原子的啊？

**一句话**：`INCR` 是原子操作，但"检查是否超限"不在 `INCR` 里面——`INCR` + 检查是两个 Redis 命令，中间存在时间窗口。Lua 把三步（INCR + 检查 + 退回）打包成一个不可分割的操作。

**面试话术**："面试官很可能追问'那 SET NX 呢？'。`SET NX` 只能做互斥锁（0 或 1），不能做计数上限（0~N）。并发计数器上限的正确方案只有 Lua 脚本或 Redis 7.0 的 `INCR` + 客户端检查（后者的竞态窗口极小但在极端并发下仍存在）。Day 13 选 Lua 是'一次写对'的策略。"

---

## Q4：`acquire_slot` 的 Lua 脚本中为什么要 `EXPIRE` 防槽位泄漏？

**一句话**：如果 `acquire_slot` 成功后进程崩溃了，`release_slot` 就不会执行，槽位计数器永远不归零——用户永远无法再发请求。`EXPIRE` 设 60 秒兜底，60 秒后自动回收。

**面试话术**："`EXPIRE` 在这里是'故障安全'设计，不是正常流程的一部分。正常流程下 `finally` 保证 `release_slot` 一定执行，槽位在请求结束后 1 秒内归还。`EXPIRE` 只处理极端情况——进程被 kill -9、OOM、K8s 强制驱逐。生产环境经验：任何加锁/获取资源操作都必须有超时自动释放机制。"

---

## Q5：`Depends(get_current_user)` 的依赖注入是怎么工作的？

**一句话**：FastAPI 在调用路由函数前，先解析函数签名中的 `Depends` 参数，递归执行依赖函数，把返回值注入对应参数——中间任何一环失败（如 401），直接短路，不进入路由函数。

**执行链路**：

```
请求 → 解析 Authorization 头 → HTTPBearer() 提取 credentials
    → credentials.credentials 传入 get_current_user
    → get_current_user 查表校验 → 返回 user_id（或 401）
    → user_id 注入路由函数的 uid 参数
    → 路由函数执行
```

**面试话术**："`Depends` 是 FastAPI 的核心机制，支持嵌套依赖、缓存（`use_cache=True`）、类依赖。Day 13 只用了一层，但生产场景下是 `Depends(get_current_user) → Depends(get_rate_limit_for_user) → Depends(get_db_session)` 的依赖链。改了认证方式只改 `get_current_user` 函数，所有路由自动生效。"

---

## Q6：Day 13 的 `/chat` 和 Day 12 的 `/chat` 怎么整合？

**一句话**：Day 13 的 `/chat` 在 Day 12 基础上加了三个中间步骤（身份校验 → 并发控制 → 会话读写），核心的 RAG + LLM + 流式逻辑不变。

**整合后的完整链**：

```python
@app.post("/chat")
async def chat(req: ChatReq, uid: str = Depends(get_current_user)):
    # Day 13 新增: 并发控制
    if not await session_mgr.acquire_slot(uid):
        raise HTTPException(status_code=429)

    try:
        # Day 13 新增: 会话记忆
        history = await session_mgr.get_history(req.session_id)
        await session_mgr.add_message(req.session_id, "user", req.query, uid)
        await session_mgr.touch_session(req.session_id)

        # Day 12 原有: RAG + LLM + 流式/非流式
        if req.stream:
            return StreamingResponse(generator(req.query, contexts), ...)
        else:
            answer = rag.chat(req.query)
            await session_mgr.add_message(req.session_id, "assistant", answer, uid)
            return ChatResponse(answer=answer, ...)
    finally:
        await session_mgr.release_slot(uid)
```

**面试话术**："Day 13 的升级是非侵入式的——身份校验和并发控制在路由入口处拦截，会话读写在业务逻辑前后包裹，核心的 RAG + LLM 调用不需要任何修改。这就是分层架构的价值：每一层只关心自己的事，改一层不影响其他层。"

---

## Q7：JWT 比 API Key 好在哪？什么时候必须换？

**一句话**：API Key 是一张"身份证"（查表验证），JWT 是一张"自带信息的签证"（Token 内含 user_id + 过期时间 + 权限）——多服务共享认证场景必须用 JWT。

**必须换 JWT 的场景**：

| 场景 | 为什么 API Key 不够 |
|------|-------------------|
| 微服务架构 | API Key 验证需要查同一个数据库，服务间互相依赖 |
| 第三方授权 | 给外部开发者临时访问权，需内置过期 + 权限范围 |
| SSO 单点登录 | 一个 Token 在 A/B/C 三个系统都能用，无需重新登录 |
| 移动端 | 刷新 Token 机制（access token 短 + refresh token 长） |

**面试话术**："API Key 和 JWT 不是好坏之分，是适用场景不同。内部工具/MVP 用 API Key 足够了，实现成本低。但面试官问到为什么不直接用 JWT，你要能说清楚 JWT 的三大价值：无状态（不用查数据库）、自带过期、可扩展（payload 里放权限/角色/租户 ID）。"

---

## Q8：`SMEMBERS + HGETALL` 取会话列表，会话多了会不会慢？

**一句话**：`SMEMBERS` 返回所有 member，用户有 1000 个会话时一次返 1000 个 ID → 再逐条 `HGETALL` → O(N) 次 Redis 查询。当前没问题，量大了需要优化。

**优化方案**：

```
方案 1（中期）: Sorted Set 替代 Set
  → ZADD user:{uid}:sessions <timestamp> <sid>
  → ZREVRANGE 取最近 50 个（分页）
  → 不需要拉全部

方案 2（长期）: 冗余存储
  → 在创建会话时额外写一份 summary 到 Sorted Set
  → 列表直接读 Sorted Set，不需要逐条 HGETALL
```

**面试话术**："Day 13 的会话列表用的是最简单的实现，适合学习阶段。但面试官问'如果用户有 1000 个会话怎么办'，你要能说出 Sorted Set 分页方案。这说明你考虑过扩展性，不是只会写 Demo。"

---

## Q9：`finally` 里的 `release_slot` 如果也抛异常怎么办？

**一句话**：`release_slot` 的 Lua 脚本有 `if v and tonumber(v) > 0` 判断，不会减到负数——即使被重复调用也是安全的（幂等）。如果 Redis 连接断开导致 release 失败，槽位的 60 秒 TTL 会兜底回收。

**面试话术**："`finally` 的异常处理是门学问——`finally` 里抛异常会覆盖 `try` 里的原始异常。Day 13 的 `release_slot` 设计为'不会抛异常'：Lua 脚本内部判空、Redis 连接超时由客户端配置控制。生产环境还会在 `finally` 外再套一层 `try/except` 确保日志记录不被吞掉。"

---

## Q10：Day 13 的代码和 Day 12 的 `stream.py` 是什么关系？能合到一起吗？

**一句话**：`session_manager.py` 是功能模块（提供 `session_mgr` 全局实例和认证依赖），可以被 `stream.py` 导入复用——最终目标是整合成一个完整的 `main.py`。

**整合建议**：

```
api/
├── session_manager.py   # Day 13: SessionManager 类 + 认证 + 会话 CRUD API
├── stream.py            # Day 12: 流式 /chat + /chat-ui
├── main.py              # 整合版: 导入 session_mgr + 升级 /chat
└── resilience.py        # Day 15 预留: 熔断限流
```

```python
# stream.py 导入 Day 13 的能力
from session_manager import session_mgr, get_current_user

@app.post("/chat")
async def chat(req: ChatReq, uid: str = Depends(get_current_user)):
    history = await session_mgr.get_history(req.session_id)
    # ... 流式生成 ...
```

**面试话术**："模块拆分反映架构设计。`SessionManager` 是独立的功能模块——它只依赖 Redis 和 FastAPI 的 `Depends`，不依赖任何 RAG/LLM 代码。这种低耦合意味着：你可以把 `SessionManager` 单独拿到另一个项目用，也可以换成 MongoDB 实现而不影响上层 API。"

---

## 从 Day 12 到 Day 13 的升级路径

| # | Day 12 | Day 13 | 说明 |
|---|--------|--------|------|
| 1 | `session_id="default"` | 真实创建/管理 session | session_id 不再是摆设 |
| 2 | 无身份校验 | `Depends(get_current_user)` | 任意请求 → 必须认证 |
| 3 | 无并发控制 | `acquire_slot` / `release_slot` | 过量请求 → 429 |
| 4 | 无对话记忆 | `get_history` + `add_message` | 跨轮指代消解 |
| 5 | 无会话管理 | 完整 CRUD API | 我的对话列表 |

---

### Day 13 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | Redis 做会话存储 vs dict/文件/SQLite 的四个优势？ | □ |
| 2 | 对话历史为什么用 List 而不是 String？RPUSH + LRANGE 的复杂度？ | □ |
| 3 | TTL 滑动过期和固定过期的区别？为什么滑动的更合理？ | □ |
| 4 | Pipeline 的关键价值是什么？`add_message` 没有 Pipeline 会多慢？ | □ |
| 5 | API Key vs JWT，各自的通用/不适用场景？什么时候必须换 JWT？ | □ |
| 6 | `Depends(get_current_user)` 的执行链路？依赖注入的优势？ | □ |
| 7 | 并发控制为什么必须用 Lua？INCR 单独不够吗？画出竞态时序图 | □ |
| 8 | `acquire_slot` 的 Lua 脚本中 EXPIRE 解决什么问题？槽位 TTL 设多久合适？ | □ |
| 9 | `finally` 里 `release_slot` 的设计考量？为什么它是幂等的？ | □ |
| 10 | Day 13 和 Day 12 怎么整合？`session_manager` 和 `stream` 的导入关系？ | □ |
