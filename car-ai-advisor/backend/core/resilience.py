"""
系统韧性组件 — 滑动窗口限流器 + 熔断器
作用：分布式服务高可用防护，流量控制 + 故障雪崩防护
面试高频考点:
  1. 滑动窗口 vs 固定窗口 — 固定窗口存在时间边界双倍流量漏洞
  2. Redis ZSet(跳表)实现滑动窗口原理：score存毫秒时间戳，O(logN)清理过期数据
  3. Lua脚本保证多操作原子性，消除TOCTOU（检查-使用）竞态条件
  4. 熔断器三态状态机：CLOSED/OPEN/HALF_OPEN，半开仅少量探测防下游冲击
  5. 状态码区分：429 Too Many Requests(限流) vs 503 Service Unavailable(熔断)
"""

import asyncio
import enum
import logging
import time
from typing import Awaitable, Callable, Optional, TypeVar

import redis.asyncio as redis
from backend.config import settings

# 全局日志对象
logger = logging.getLogger(__name__)
# 泛型：适配任意异步函数返回值类型
T = TypeVar("T")


# ================================================================
# Lua 脚本 — 原子化滑动窗口限流核心逻辑
# ================================================================
# 关键设计点：
# 1. 全部Redis操作封装在一段Lua，Redis单线程串行执行，原子不可分割
# 2. 解决分开执行ZREMRANGEBYSCORE + ZCARD + ZADD产生的并发超量问题(TOCTOU竞态)
# 3. ZSet存储每条请求记录，score=当前毫秒时间戳，用于滑动窗口过期清理
# 4. member=时间戳+随机数，避免同一毫秒多条请求覆盖记录
# 5. 自动给限流Key设置过期时间，冷用户不长期占用Redis内存
_RATE_LIMIT_LUA = r"""
-- KEYS[1]：限流唯一标识对应的Redis ZSet Key，格式 rate_limit:{identifier}
local key = KEYS[1]
-- ARGV[1]：当前系统毫秒时间戳
local now_ms = tonumber(ARGV[1])
-- ARGV[2]：滑动窗口总时长(毫秒)，配置文件窗口秒数*1000
local window_ms = tonumber(ARGV[2])
-- ARGV[3]：窗口内允许的最大请求次数
local max_requests = tonumber(ARGV[3])
-- ARGV[4]：ZSet Key的过期时间，窗口时长+1秒，避免提前删除
local ttl_sec = tonumber(ARGV[4])

-- 步骤1：清理窗口边界外的过期请求记录
-- 窗口左边界 = 当前时间 - 窗口时长，删除所有score小于等于边界的旧数据
redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)
-- 步骤2：统计当前窗口内有效请求总数量
local current_count = redis.call('ZCARD', key)

-- 步骤3：判断是否达到限流阈值，超限直接返回0(拒绝)
if current_count >= max_requests then
    return 0
end

-- 步骤4：未超限，写入本次请求记录
-- score=当前毫秒时间戳；member拼接随机数，同毫秒并发请求不会覆盖
redis.call('ZADD', key, now_ms, now_ms .. ':' .. math.random(1000000))
-- 步骤5：更新Key过期时间，窗口结束后自动清理整条ZSet
redis.call('EXPIRE', key, ttl_sec)
-- 返回1代表允许本次请求通过
return 1
"""


class SlidingWindowRateLimiter:
    """
    基于Redis ZSet实现分布式滑动窗口限流器
    优势：相比固定窗口无临界流量溢出，分布式多实例计数统一
    """
    def __init__(self, redis_client: redis.Redis):
        # 外部注入异步Redis客户端，复用全局连接池，不重复创建连接
        self.redis = redis_client
        # 窗口时长转换：配置秒 → 毫秒
        self.window_ms = settings.rate_limit_window_seconds * 1000
        # 单个窗口最大允许请求数，取自项目配置
        self.max_requests = settings.rate_limit_requests
        # 缓存Lua脚本SHA1哈希，用于evalsha减少网络传输
        self._script_sha: Optional[str] = None

    async def _get_sha(self) -> str:
        """
        懒加载Lua脚本SHA哈希值
        1. 首次调用时上传完整Lua脚本到Redis，返回SHA字符串并缓存
        2. 后续限流请求仅传递SHA短串，无需重复传输大段脚本文本，节省带宽
        """
        if self._script_sha is None:
            # script_load：上传脚本、Redis编译缓存、返回sha，不执行脚本逻辑
            self._script_sha = await self.redis.script_load(_RATE_LIMIT_LUA)
        return self._script_sha

    async def is_allowed(self, identifier: str) -> bool:
        """
        校验当前请求是否放行
        :param identifier: 限流标识，可为客户端IP/用户ID/接口路径，区分不同限流主体
        :return: True=允许请求，False=触发限流拦截
        """
        # 拼接限流Key前缀，隔离不同用户/IP的限流数据
        key = f"rate_limit:{identifier}"
        # 获取当前毫秒时间戳
        now_ms = int(time.time() * 1000)
        # Key过期时间比窗口多1秒，防止窗口结束后数据立刻被删
        ttl_sec = settings.rate_limit_window_seconds + 1

        try:
            # 获取缓存的脚本SHA哈希
            sha = await self._get_sha()
            # evalsha执行缓存脚本，参数说明：sha, KEY数量, KEYS..., ARGV...
            result = await self.redis.evalsha(
                sha, 1, key, now_ms, self.window_ms, self.max_requests, ttl_sec
            )
            # Lua返回1放行，0限流
            return result == 1
        except redis.exceptions.NoScriptError:
            # 异常场景：Redis重启/执行SCRIPT FLUSH，本地缓存的SHA失效
            # 清空本地sha缓存，重新上传脚本后重试一次
            self._script_sha = None
            sha = await self._get_sha()
            result = await self.redis.evalsha(
                sha, 1, key, now_ms, self.window_ms, self.max_requests, ttl_sec
            )
            return result == 1
        except redis.RedisError as exc:
            # Redis服务宕机/网络故障降级策略：临时放行，优先保障业务可用性
            logger.critical(f"限流器 Redis 异常，临时放行: {exc}")
            return True

    async def get_remaining(self, identifier: str) -> int:
        """
        查询当前限流窗口剩余可用请求配额
        用于填充HTTP响应头 X-RateLimit-Remaining，前端展示剩余可请求次数
        :param identifier: 限流唯一标识
        :return: >=0 剩余次数；-1 Redis异常无法查询
        """
        key = f"rate_limit:{identifier}"
        now_ms = int(time.time() * 1000)
        try:
            # 先清理过期记录，保证计数准确
            await self.redis.zremrangebyscore(key, 0, now_ms - self.window_ms)
            # 获取当前窗口有效请求总数
            count = await self.redis.zcard(key)
            # 剩余配额不小于0
            return max(0, self.max_requests - count)
        except redis.RedisError:
            # Redis故障返回-1，上层中间件识别后不展示限流数值
            return -1


# ================================================================
# 熔断器 (Circuit Breaker) — 防止下游故障引发服务雪崩
# ================================================================

class CircuitState(enum.Enum):
    """熔断器三状态枚举"""
    CLOSED = "closed"        # 正常闭合状态：所有请求正常放行，统计失败次数
    OPEN = "open"            # 熔断打开状态：直接拦截请求，快速失败，不调用下游
    HALF_OPEN = "half_open"  # 半开探测状态：冷却完成，少量请求探测下游是否恢复


class CircuitBreaker:
    """
    异步熔断器实现，适配FastAPI异步接口，保护下游依赖(Redis/LLM/数据库)
    完整状态流转：
      CLOSED → 连续失败达到阈值N次 → OPEN
      OPEN → 等待冷却超时 → HALF_OPEN
      HALF_OPEN → 探测请求成功 → CLOSED(恢复全量流量)
      HALF_OPEN → 探测请求失败 → OPEN(重新冷却)
    设计要点：HALF_OPEN仅放行少量探测请求，避免下游刚恢复瞬间被流量冲垮
    """
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 30.0,
    ):
        self.name = name  # 熔断器唯一名称，区分不同下游依赖（redis/llm/db）
        self.failure_threshold = failure_threshold  # CLOSED状态触发熔断的连续失败次数阈值
        self.timeout = timeout_seconds  # OPEN状态冷却恢复时长(秒)
        self._state = CircuitState.CLOSED  # 初始化默认正常闭合
        self._failure_count = 0  # CLOSED状态连续失败计数器，成功则清零
        self._last_failure_time: float = 0.0  # 最近一次请求失败的时间戳
        self._lock = asyncio.Lock()  # 异步锁，多协程并发切换状态时防止竞态错乱

    @property
    def state(self) -> CircuitState:
        """只读属性，外部仅允许查看状态，禁止直接修改内部_state"""
        return self._state

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        通过熔断器包装执行任意异步下游函数
        :param fn: 需要保护的异步函数（Redis查询、大模型调用、数据库操作）
        :param args: 函数可变位置参数
        :param kwargs: 函数可变关键字参数
        :return: 下游函数正常返回结果
        :raises CircuitBreakerOpenError: 熔断打开拦截请求时抛出
        """
        # ========== 分支1：当前状态为OPEN熔断拦截 ==========
        if self._state == CircuitState.OPEN:
            # 计算距离上次失败已过去的时长
            elapsed = time.time() - self._last_failure_time
            # 判断冷却时间是否到期，可以进入半开探测
            if elapsed >= self.timeout:
                # 加锁防止多协程同时切换状态，双重校验避免重复变更
                async with self._lock:
                    if self._state == CircuitState.OPEN:
                        self._transition_to(CircuitState.HALF_OPEN)
            else:
                # 冷却未完成，直接抛出熔断异常，上层捕获返回503
                raise CircuitBreakerOpenError(
                    f"熔断器 [{self.name}] 保护中，{self.timeout - elapsed:.0f}s 后重试"
                )

        # ========== 分支2：执行下游异步函数 ==========
        try:
            # 正常调用下游依赖
            result = await fn(*args, **kwargs)
            # 请求成功，清空连续失败计数器
            self._failure_count = 0
            # 无论当前是CLOSED还是HALF_OPEN，成功后切回正常状态
            self._transition_to(CircuitState.CLOSED)
            return result
        except Exception as exc:
            # 捕获下游所有异常，标记本次请求失败
            self._failure_count += 1
            # 更新最新失败时间戳，用于OPEN状态计算冷却时长
            self._last_failure_time = time.time()

            # 场景1：正常状态，连续失败达到阈值，切换为熔断OPEN
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
            # 场景2：半开探测请求失败，立刻重新熔断，重置冷却计时
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            # 向上抛出原始业务异常，由上层接口统一处理报错
            raise

    def _transition_to(self, new_state: CircuitState) -> None:
        """
        统一状态切换工具方法，所有状态变更统一入口
        :param new_state: 目标切换状态
        """
        # 新旧状态不一致才执行切换，避免重复打印日志
        if self._state != new_state:
            logger.warning("熔断器 [%s] %s → %s", self.name, self._state.value, new_state.value)
            self._state = new_state


class CircuitBreakerOpenError(Exception):
    """
    熔断器打开自定义异常
    上层FastAPI中间件捕获该异常，统一返回HTTP 503 服务不可用
    """
    pass