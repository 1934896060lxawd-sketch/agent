# Day 2 — 滑动窗口限流与熔断器

> **今日目标**：实现两个企业级韧性组件——Reids ZSet 滑动窗口限流器 + 三态熔断器。这是面试中最容易被深挖的模块，每个设计决策都要能说清楚为什么。

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [限流器：固定窗口 vs 滑动窗口](#2-限流器固定窗口-vs-滑动窗口)
3. [Redis ZSet 滑动窗口实现](#3-redis-zset-滑动窗口实现)
4. [Lua 脚本原子性原理](#4-lua-脚本原子性原理)
5. [熔断器三态状态机](#5-熔断器三态状态机)
6. [两个组件的配合关系](#6-两个组件的配合关系)
7. [初学者常见疑问](#7-初学者常见疑问)
8. [面试模拟问答](#8-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `core/resilience.py` | ~210 | SlidingWindowRateLimiter + CircuitBreaker |

只写一个文件，但**信息密度是 Day 1 的两倍**。

---

## 2. 限流器：固定窗口 vs 滑动窗口

### 2.1 固定窗口为什么有致命漏洞

限流规则："每分钟最多 60 次请求"。

```
固定窗口 —— 按自然时间划分桶:
  [12:00:00 ─────── 12:00:59]    桶1
  [12:01:00 ─────── 12:01:59]    桶2

  攻击者: 12:00:59 的一秒内发 60 次 → 桶1 刚好 60 次 ✓ 放行
          12:01:00 的一秒内再发 60 次 → 桶2 刚好 60 次 ✓ 放行

  结果:  2 秒内发了 120 次！但没触发限流！
  原因:  桶1 和桶2 是独立计数的，攻击发生在桶的边界
```

这叫**边界突发漏洞**（Boundary Burst Bug）。固定窗口相当于给系统装了两个"60 升的水桶"，但水可以跨桶流——只要在交界处倒就行。

### 2.2 滑动窗口怎么堵住漏洞

```
滑动窗口 —— 窗口是"过去 60 秒"，随当前时间平滑移动:

  时间点: 12:01:00
  窗口范围: [12:00:00 ← 12:01:00]    (过去 60 秒)
  窗口内: 60(12:00:59发的) + 60(12:01:00发的) = 120 次
  → 超限！第 61 次就被拒绝！

  无论什么时间点发请求，窗口都精确覆盖过去 60 秒。
  没有"桶边界"可以钻。
```

**本质区别**：

| | 固定窗口 | 滑动窗口 |
|------|------|------|
| 边界 | 自然的分钟线 | 当前时刻 - N 秒 |
| 跨边界攻击 | ✅ 可绕过 | ❌ 不可能 |
| 实现复杂度 | 计数器 + 定时重置 | 有序数据结构 |
| 精确度 | 差 | 精确 |

---

## 3. Redis ZSet 滑动窗口实现

### 3.1 为什么是 ZSet？

ZSet（有序集合）是 Redis 中唯一一个能同时满足两个需求的数据结构：

1. **按时间排序**（score = 时间戳，O(log N) 范围查询）
2. **快速计数**（ZCARD 是 O(1)）

具体到限流场景：

```
Key: rate_limit:user_001
Type: ZSet

member (唯一ID)         score (时间戳, 毫秒)
─────────────────────   ──────────────────
1765874400123:45678     1765874400123     ← 最早记录
1765874400234:12345     1765874400234
1765874400345:98765     1765874400345
1765874400456:34567     1765874400456     ← 最新记录
                     ↑
              score 自动排序，跳表维护
```

操作流程：
1. `ZREMRANGEBYSCORE key 0 (now - window)` → 删除窗口外的旧记录
2. `ZCARD key` → 统计窗口内剩余记录数
3. 判断是否超限
4. 未超限 → `ZADD key now_ms member` → 记录本次请求

### 3.2 member 为什么要拼随机数

**这是最容易栽跟头的细节，面试官最爱问。**

```lua
-- ❌ 错误写法
redis.call('ZADD', key, 1700000000123, 1700000000123)  -- 请求A
redis.call('ZADD', key, 1700000000123, 1700000000123)  -- 请求B (同毫秒)
-- ZCARD → 1 ← 两次请求只记了一次！

-- ✅ 正确写法
redis.call('ZADD', key, 1700000000123, "1700000000123:523")  -- 请求A
redis.call('ZADD', key, 1700000000123, "1700000000123:871")  -- 请求B
-- ZCARD → 2 ← 正确！
```

原因：ZSet 的 member 是唯一键。第二个 ZADD 如果 member 已存在，只是**更新 score**而不是新增元素。拼上 `math.random(1000000)` 后每个 member 都不同，同毫秒的请求都能正确计数。

### 3.3 SlidingWindowRateLimiter 完整代码分析

```python
class SlidingWindowRateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.window_ms = 60 * 1000           # 60 秒窗口
        self.max_requests = 60                # 窗口内最多 60 次
        self._script_sha: Optional[str] = None  # 懒加载的 SHA1
```

**`_script_sha` 的设计意图**：Redis 支持两种执行 Lua 的方式：

| 方式 | 每次传输 | 首次 | 后续 |
|------|------|------|------|
| `EVAL script` | 完整脚本文本 | 都一样 | 每次都传完整脚本 |
| `SCRIPT LOAD` → `EVALSHA sha` | 40 字节 SHA1 | 需加载 | 只需传 SHA1 |

500 字节脚本 × 每秒 1000 次 EVAL = 500KB/s 带宽浪费。EVALSHA 降到 40KB/s。

```python
async def _get_sha(self) -> str:
    if self._script_sha is None:
        self._script_sha = await self.redis.script_load(_RATE_LIMIT_LUA)
    return self._script_sha
```

懒加载模式：第一次调用 `is_allowed` 时才 SCRIPT LOAD，后续直接用 SHA。

```python
except redis.exceptions.NoScriptError:
    self._script_sha = None   # 清空缓存
    sha = await self._get_sha()  # 重新加载
    result = await self.redis.evalsha(sha, ...)  # 重试
```

**NoScriptError 的防御性处理**：Redis 重启或执行 `SCRIPT FLUSH` 后缓存丢失。捕获后重新加载再执行，保证健壮性。

```python
except redis.RedisError as exc:
    logger.critical(f"限流器 Redis 异常，临时放行: {exc}")
    return True  # 降级策略：保可用性
```

**降级策略是最难的抉择**：

| 策略 | 好处 | 风险 |
|------|------|------|
| Redis 挂了 → 拒绝所有 | 安全，不会超限 | 服务完全不可用 |
| Redis 挂了 → 放行所有 | 服务可用 | 可能被攻击者利用 |

对于面向用户的 AI 导购，"可用"优先于"严格限流"。生产环境需同步触发 PagerDuty 告警。

---

## 4. Lua 脚本原子性原理

### 4.1 TOCTOU 问题

如果没有 Lua 脚本，滑动窗口需要 4 条 Redis 命令：

```
协程A: ZREMRANGEBYSCORE → ZADD → ZCARD → 判断
协程B: ZREMRANGEBYSCORE → ZADD → ZCARD → 判断
```

两个协程交错执行时：

```
T1: A 执行 ZREMRANGEBYSCORE — 窗口内剩 59 条
T2: B 执行 ZREMRANGEBYSCORE — 窗口内剩 59 条 (B 不知道 A 马上要 ZADD)
T3: A 执行 ZADD — 窗口内变成 60 条
T4: B 执行 ZADD — 窗口内变成 61 条 ← 超限！
T5: A 执行 ZCARD → 60 → 放行 ✓
T6: B 执行 ZCARD → 61 → 但 B 以为自己在 T2 时是安全的 ✗
```

这就是 **TOCTOU** (Time-of-Check to Time-of-Use)：检查时（T2）状态是安全的，使用时（T4）状态已变。

### 4.2 Lua 为什么能解决

**Redis 是单线程执行命令的**（核心事件循环）。Lua 脚本一旦开始执行，就会**独占这个单线程**直到脚本结束。

```
Lua 脚本执行期间:
  事件循环暂停 → 其他任何客户端的任何命令都在队列里等待
  就像一个不可打断的"原子操作"
```

所以 ZREMRANGEBYSCORE、ZCARD、ZADD 三个操作变成了一个不可分割的整体。中间不会有其他命令插入。

### 4.3 Lua 脚本详解

```lua
local key = KEYS[1]                    -- "rate_limit:user_001"
local now_ms = tonumber(ARGV[1])       -- 当前毫秒时间戳
local window_ms = tonumber(ARGV[2])    -- 窗口大小(毫秒)
local max_requests = tonumber(ARGV[3]) -- 最大请求数
local ttl_sec = tonumber(ARGV[4])      -- TTL(秒)

-- 步骤1: 清理窗口外过期记录
-- ZREMRANGEBYSCORE 通过跳表定位到 now_ms - window_ms 位置
-- 删除该位置之前的所有节点，复杂度 O(log N + M)
redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)

-- 步骤2: 统计当前窗口内请求数 (O(1))
local current_count = redis.call('ZCARD', key)

-- 步骤3: 超限判断
if current_count >= max_requests then
    return 0  -- 被限流
end

-- 步骤4: 记录本次请求
-- member = 时间戳:随机数，同毫秒不覆盖
redis.call('ZADD', key, now_ms, now_ms .. ':' .. math.random(1000000))
redis.call('EXPIRE', key, ttl_sec)  -- 续期 TTL
return 1  -- 放行
```

**6 行 Lua 约 0.5ms 执行**，对 Redis 的事件循环影响极小。脚本中不要有循环遍历大集合——那会长时间阻塞。我们的脚本只有两个 O(log N) 操作 + 一个 O(1) + 两个 O(1)，非常安全。

---

## 5. 熔断器三态状态机

### 5.1 先理解"雪崩效应"

没有熔断器的场景：

```
LLM API 开始变慢（或挂了）
  → 每个请求都要等 30 秒超时
    → 100 个并发协程全部阻塞 30 秒
      → 用户看到一直转圈，不断刷新页面
        → 更多请求涌入！
          → 线程池/连接池耗尽
            → 整个服务无响应 ← 雪崩！
```

熔断器的干预：

```
LLM API 连续失败 5 次
  → 熔断器 OPEN
    → 所有后续请求直接返回 "服务暂不可用"（1ms）
      → 用户立即看到明确提示，停止刷新
        → 30 秒冷却后，发一个探测请求
          → 成功 → 恢复正常
          → 失败 → 继续熔断
```

### 5.2 三态状态机

```
    连续失败 >= 5 次
CLOSED ────────────────→ OPEN
  ↑                       │
  │ 探测成功               │ 冷却 30 秒到
  │                       ↓
  └─────────────────── HALF_OPEN
          探测失败 → 回到 OPEN ──→
```

| 状态 | 含义 | 行为 |
|------|------|------|
| **CLOSED** | 一切正常 | 请求正常通过，记录失败次数 |
| **OPEN** | 熔断中 | 直接抛 CircuitBreakerOpenError，毫秒级失败 |
| **HALF_OPEN** | 试探性恢复 | 只放行**一个**请求来探测下游状态 |

### 5.3 为什么 HALF_OPEN 只放一个请求

这是熔断器模式最精妙的设计。去掉 HALF_OPEN 会怎样？

```
CLOSED → (连续5次失败) → OPEN → (冷却30秒) → 直接回 CLOSED
                                                    ↓
                                         100 个积压请求瞬间涌入
                                                    ↓
                                        下游还没完全恢复，再次打挂
                                                    ↓
                                    → OPEN → 冷却 → CLOSED → 打挂 ...
                                            ↑                        │
                                            └────────────────────────┘
                                                无限震荡 (flapping)
```

HALF_OPEN 的保守策略：用最小的代价（一个探测请求）验证下游是否真的恢复了。成功 → 逐步恢复流量；失败 → 继续等待。

### 5.4 代码关键细节

```python
async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    # ── OPEN 状态判断 ──
    if self._state == CircuitState.OPEN:
        elapsed = time.time() - self._last_failure_time
        if elapsed >= self.timeout:
            async with self._lock:         # asyncio.Lock 保护
                if self._state == CircuitState.OPEN:  # 双重检查
                    self._transition_to(CircuitState.HALF_OPEN)
        else:
            raise CircuitBreakerOpenError(...)
```

**为什么用 `asyncio.Lock` 而不是 `threading.Lock`？**

asyncio 是单线程事件循环。`threading.Lock` 会阻塞整个线程（让事件循环停转），`asyncio.Lock` 只是让出协程控制权（其他协程照常执行）。

**为什么有双重检查（Double Check）？**

多个协程可能同时看到 `self._state == OPEN` 且 `elapsed >= timeout`，同时进入 `if` 分支。但 `asyncio.Lock` 保证同时只有一个协程能执行临界区代码。第一个进去的把状态改为 HALF_OPEN，后面的再检查时 `self._state != OPEN` 就不重复改了。

```python
try:
    result = await fn(*args, **kwargs)
    self._failure_count = 0  # 成功 → 重置失败计数
    self._transition_to(CircuitState.CLOSED)
    return result
except Exception as exc:
    self._failure_count += 1
    if self._state == CircuitState.CLOSED and self._failure_count >= threshold:
        self._transition_to(CircuitState.OPEN)   # 触发熔断
    elif self._state == CircuitState.HALF_OPEN:
        self._transition_to(CircuitState.OPEN)   # 探测失败，退回熔断
    raise
```

**成功时重置失败计数**：哪怕是 CLOSED 状态下偶尔失败了一两次，只要后续有成功的，计数器归零。只有**连续**失败才触发熔断。偶尔的网络抖动不会误触发。

---

## 6. 两个组件的配合关系

```
请求进入 FastAPI 路由
  │
  ├─ 1. check_rate_limit
  │     ├─ get_current_user_auto → user_id (认证)
  │     └─ rate_limiter.is_allowed(user_id) → 超限返回 429
  │
  ├─ 2. 业务逻辑
  │     └─ circuit_breaker.call(llm_api_invoke, messages)
  │           ├─ CLOSED → 正常调用 → 异常则累计失败数
  │           ├─ OPEN → 直接抛异常 → 返回 503
  │           └─ HALF_OPEN → 只放行一个探测
  │
  └─ 3. 响应
```

**它们解决的问题完全不同**：

| 组件 | 保护对象 | 触发条件 | 响应 | 类比 |
|------|------|------|------|------|
| 限流器 | 系统不被某个用户滥用 | 单用户频率超阈值 | 429 | 收费站——每辆车都要过 |
| 熔断器 | 系统不被下游拖垮 | 下游连续失败 | 503 | 电闸——短路了跳闸 |

---

## 7. 初学者常见疑问

**Q: ZSet 底层是什么数据结构？为什么 ZREMRANGEBYSCORE 是 O(log N) 而不是 O(N)？**

ZSet 底层是**跳表**（Skip List）。普通链表查找是 O(N)；跳表在链表上加了多层"快速通道"——第 k 层每隔 2^k 个节点建一个指针。查找时从最高层开始，遇到比目标大的就降一层。查找、插入、删除都是平均 O(log N)。实现比红黑树简单得多，且天然支持范围查询（这是红黑树不擅长的）。

```
普通链表:  1→2→3→4→5→6→7→8→9→10  查找 9 要 9 步
跳表:
  第3层: 1═══════════5═══════════9    ← 大步跳过
  第2层: 1══════3══════5══════7══════9  ← 中步
  第1层: 1→2→3→4→5→6→7→8→9→10         ← 底层链表（完整）
  查找 9: 1→5→9  只需 3 步！
```

**Q: print('result') 在 Python 中写到哪了？和 logging 有什么区别？**

`print()` 输出到 stdout，无法控制级别、没有时间戳、无法输出到文件。`logging.getLogger(__name__)` 输出到 stderr/文件，带时间戳、模块名、日志级别。生产环境用 logging 才能在 ELK/Loki 中搜索和聚合。

**Q: `self._state` 存在内存里，多进程部署时怎么保证一致性？**

当前实现是"进程本地"熔断器。4 个 Gunicorn worker = 4 个独立的 CircuitBreaker 实例，worker-1 熔断了但 worker-2 还不知道。这是一个**工程权衡**——内存存储的优点是零网络开销、纳秒级状态切换。改进方案：用 Redis Hash + Lua 做全局状态（Day 2 检验题第 6 题）。

**Q: `circuit_breaker.call(llm_api_fn, prompt)` 中为什么传函数而不是直接传结果？**

`call(fn, *args)` 把"调用"的控制权交给了熔断器——熔断器决定调用还是不调用，以及何时重试。如果你写成：

```python
result = await llm_api_fn(prompt)         # 已经调用了，熔断器怎么阻止？
circuit_breaker.call(result)              # 太晚了
```

**Q: 为什么限流器降级策略是"放行"而不是"拒绝"？**

可用性 > 限流准确性（对 AI 导购而言）。如果是支付接口，策略就是"拒绝"。不同场景选择不同。原则是"保护自己"还是"保护用户"：限流器保护自己（拒绝合法用户也防不了攻击者）、支付接口保护用户（重复扣款比服务不可用更严重）。

---

## 8. 面试模拟问答

> **Q: 说一下滑动窗口和固定窗口的本质区别。**

固定窗口按自然时间分段（12:00-12:01），两个窗口的边界是固定的，攻击者可以在边界处发动"双倍流量攻击"。滑动窗口的边界随当前时间平滑移动，窗口 = [now-N, now]，任何时候窗口内记录的都是过去 N 秒的真实流量。不存在可被利用的边界。

> **Q: Lua 脚本为什么是原子的？Redis 6.0 不是多线程了吗？**

Redis 6.0 的多线程只用在网络 I/O 层面（读客户端数据、写客户端数据），**命令执行仍然是单线程**。一个 Lua 脚本一旦开始执行，就会独占事件循环，其他命令排队等待。所以脚本内部的多条命令之间不会插入其他客户端的命令。注意：Multi-IO 线程默认关闭，需要 `io-threads` 配置显式开启。

> **Q: ZSet 做限流，Key 会不会无限增长？**

不会。每次 Lua 脚本执行时，第一步就是 `ZREMRANGEBYSCORE` 删除窗口外的旧记录。另外 `EXPIRE` 设置了 TTL，如果某个用户长期不活跃，整个 Key 都会过期删除。最坏情况下，一个持续活跃的用户 Key 中最多保持 `max_requests` 左右的记录数（因为窗口内请求数触达上限后会被拒绝，不再新增记录）。

> **Q: 熔断器 HALF_OPEN 如果同时来了多个请求怎么办？**

当前代码中使用了 `asyncio.Lock` 保护状态迁移，但多个协程可能同时看到 HALF_OPEN 状态并都执行 `fn(*args, **kwargs)`。这是因为 HALF_OPEN 的"只放一个"没有在代码中严格限制。这是当前实现的简化——asyncio 同一时刻通常只有一个协程在 HALF_OPEN 临界区，实际影响很小。严格实现可以加 `_probe_lock` 标记。

> **Q: 如果让你给这个限流器加一个"分级限流"功能（不同用户级别不同限额），怎么改？**

在 `SlidingWindowRateLimiter.is_allowed` 中加一个 `max_requests` 参数覆盖默认值。或者在 Lua 脚本中再接收一个 `max_requests` 参数。调用时根据用户级别传入不同的限额：免费用户 30/min、付费用户 300/min。数据结构不变，只是判断阈值不同。

---

## 附：今日内容与前后关系

```
Day 2 独立模块 (不依赖 Day 1)
  resilience.py
    ├─ SlidingWindowRateLimiter ──→ deps.py (Day 3) 中的 check_rate_limit
    └─ CircuitBreaker ──→ routes/chat.py (Day 3) 中保护 LLM 调用

Day 3 会把 Day 1 + Day 2 的所有模块组装成可工作的 API。
```
