"""FastAPI 依赖注入 — 组装认证、限流、会话管理为可注入依赖。

依赖链（外层先执行）:
  get_redis_client → get_rate_limiter / get_session_manager
  check_rate_limit = get_current_user_auto (认证) + is_allowed (限流)
"""

from fastapi import Depends, HTTPException, Request
from redis.asyncio import Redis

from backend.core.security import get_current_user_auto
from backend.core.session_manager import SessionManager
from backend.core.resilience import SlidingWindowRateLimiter


async def get_redis_client(request: Request) -> Redis:
    """从 app.state 取 Redis 客户端，不可用时返回 503。"""
    redis_client = request.app.state.redis
    if redis_client is None:
        raise HTTPException(status_code=503, detail="服务暂不可用，缓存未就绪")
    return redis_client


async def get_session_manager(
    redis_client: Redis = Depends(get_redis_client),
) -> SessionManager:
    """每个请求创建新的 SessionManager 实例（轻量，无连接开销）。"""
    return SessionManager(redis_client)


async def get_rate_limiter(
    redis_client: Redis = Depends(get_redis_client),
) -> SlidingWindowRateLimiter:
    """每个请求创建新的限流器实例。"""
    return SlidingWindowRateLimiter(redis_client)


async def check_rate_limit(
    user_id: str = Depends(get_current_user_auto),
    rate_limiter: SlidingWindowRateLimiter = Depends(get_rate_limiter),
) -> str:
    """认证 + 限流组合检查。先认证（401）再限流（429），
    未认证请求不消耗限流配额。返回 user_id 供路由使用。"""
    allowed = await rate_limiter.is_allowed(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
    return user_id
