"""
系统韧性组件 — 滑动窗口限流器 + 熔断器。

面试高频考点:
  1. 滑动窗口 vs 固定窗口 — 边界漏洞
  2. Redis ZSet (跳表) 做滑动窗口原理
  3. Lua 脚本保证原子性 (TOCTOU 竞态)
  4. 熔断器三态机 — HALF_OPEN 单探测防冲击
  5. 429 (限流) vs 503 (熔断) 状态码
"""

import asyncio
import enum
import logging
import time
from typing import Awaitable, Callable, Optional, TypeVar

import redis.asyncio as redis
from backend.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ================================================================
# Lua 脚本 — 原子化滑动窗口检查
# ================================================================
# 为什么用 Lua？消除 TOCTOU 竞态条件
# 固定窗口缺陷: 0:59 和 1:00 边界各打满，实际 2 秒内双倍流量
# ZSet: score=时间戳，跳表 O(logN) 范围删除 + 计数

_RATE_LIMIT_LUA = r"""
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local ttl_sec = tonumber(ARGV[4])

-- 清理过期 + 计数 + 判断 + 新增 (原子执行)
redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)
local current_count = redis.call('ZCARD', key)

if current_count >= max_requests then
    return 0
end

-- member = 时间戳:随机数，同毫秒不覆盖
redis.call('ZADD', key, now_ms, now_ms .. ':' .. math.random(1000000))
redis.call('EXPIRE', key, ttl_sec)
return 1
"""


class SlidingWindowRateLimiter:
    """Redis ZSet 滑动窗口限流器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.window_ms = settings.rate_limit_window_seconds * 1000
        self.max_requests = settings.rate_limit_requests
        self._script_sha: Optional[str] = None

    async def _get_sha(self) -> str:
        """懒加载 SHA，后续 EVALSHA 省带宽"""
        if self._script_sha is None:
            self._script_sha = await self.redis.script_load(_RATE_LIMIT_LUA)
        return self._script_sha

    async def is_allowed(self, identifier: str) -> bool:
        """检查请求是否放行"""
        key = f"rate_limit:{identifier}"
        now_ms = int(time.time() * 1000)
        ttl_sec = settings.rate_limit_window_seconds + 1

        try:
            sha = await self._get_sha()
            result = await self.redis.evalsha(
                sha, 1, key, now_ms, self.window_ms, self.max_requests, ttl_sec
            )
            return result == 1
        except redis.exceptions.NoScriptError:
            # 脚本缓存丢失，重加载
            self._script_sha = None
            sha = await self._get_sha()
            result = await self.redis.evalsha(
                sha, 1, key, now_ms, self.window_ms, self.max_requests, ttl_sec
            )
            return result == 1
        except redis.RedisError as exc:
            # 降级: Redis 故障则放行，保可用性
            logger.critical(f"限流器 Redis 异常，临时放行: {exc}")
            return True

    async def get_remaining(self, identifier: str) -> int:
        """查询剩余配额 (X-RateLimit-Remaining)"""
        key = f"rate_limit:{identifier}"
        now_ms = int(time.time() * 1000)
        try:
            await self.redis.zremrangebyscore(key, 0, now_ms - self.window_ms)
            count = await self.redis.zcard(key)
            return max(0, self.max_requests - count)
        except redis.RedisError:
            return -1


# ================================================================
# 熔断器 (Circuit Breaker)
# ================================================================

class CircuitState(enum.Enum):
    CLOSED = "closed"        # 正常
    OPEN = "open"            # 熔断拦截
    HALF_OPEN = "half_open"  # 半开探测


class CircuitBreaker:
    """熔断器 — 防止雪崩
    
    状态流转:
      CLOSED → (连续失败N次) → OPEN → (冷却超时) → HALF_OPEN
      HALF_OPEN → (探测成功) → CLOSED
      HALF_OPEN → (探测失败) → OPEN
    
    为什么 HALF_OPEN 只放 1 个请求？避免下游刚恢复就被冲垮
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """通过熔断器执行异步函数"""
        
        # OPEN: 检查是否可进入 HALF_OPEN
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.timeout:
                async with self._lock:
                    if self._state == CircuitState.OPEN:
                        self._transition_to(CircuitState.HALF_OPEN)
            else:
                raise CircuitBreakerOpenError(
                    f"熔断器 [{self.name}] 保护中，{self.timeout - elapsed:.0f}s 后重试"
                )

        # 执行
        try:
            result = await fn(*args, **kwargs)
            self._failure_count = 0
            self._transition_to(CircuitState.CLOSED)
            return result
        except Exception as exc:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            raise

    def _transition_to(self, new_state: CircuitState) -> None:
        if self._state != new_state:
            logger.warning("熔断器 [%s] %s → %s", self.name, self._state.value, new_state.value)
            self._state = new_state


class CircuitBreakerOpenError(Exception):
    """熔断异常 → 返回 503"""
    pass