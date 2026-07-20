"""FastAPI 依赖注入 — 组装认证、限流、会话管理、Agent 为可注入依赖。

依赖链:
  get_redis_client → get_rate_limiter / get_session_manager
  check_rate_limit = get_current_user_auto (认证) + is_allowed (限流)
  get_agent = 知识库索引加载 + ToolExecutor + CarAdvisorAgent (单例)
"""

import logging
import pickle
import threading
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from redis.asyncio import Redis

from backend.config import settings
from backend.core.security import get_current_user_auto
from backend.core.session_manager import SessionManager
from backend.core.resilience import SlidingWindowRateLimiter
from backend.agent.tools import ToolExecutor
from backend.agent.advisor import CarAdvisorAgent

logger = logging.getLogger(__name__)

# Agent 单例 + 构建锁（预热任务与首个请求可能并发，锁保证只加载一次）
_agent: "CarAdvisorAgent | None" = None
_agent_lock = threading.Lock()


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
    """认证 + 限流组合检查。先认证（401）再限流（429）。"""
    allowed = await rate_limiter.is_allowed(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
    return user_id


def build_agent_singleton() -> CarAdvisorAgent:
    """构建（或返回）Agent 单例。线程安全，供 FastAPI 依赖与启动预热共用。

    需要先运行 build_index.py 构建索引文件：
      python knowledge_base/scripts/build_index.py
    """
    global _agent
    with _agent_lock:
        if _agent is not None:
            return _agent

        import faiss
        from backend.rag.retriever import VectorIndex, BM25

        project_root = Path(__file__).resolve().parent.parent.parent
        processed_dir = project_root / settings.knowledge_base_dir / "processed"
        index_path = str(processed_dir / "faiss_index.bin")
        docs_path = str(processed_dir / "documents.pkl")

        if not Path(index_path).exists() or not Path(docs_path).exists():
            raise HTTPException(
                status_code=503,
                detail="知识库索引未构建，请先运行: python knowledge_base/scripts/build_index.py",
            )

        logger.info("加载知识库索引...")
        vector_index = VectorIndex.load(index_path, docs_path)
        bm25 = BM25(vector_index.documents)

        executor = ToolExecutor()
        executor.set_retrievers(vector_index, bm25)

        _agent = CarAdvisorAgent(executor)
        logger.info("Agent 就绪")
        return _agent


async def get_agent() -> CarAdvisorAgent:
    """FastAPI 依赖：获取 Agent 单例（首次调用时构建）。"""
    if _agent is not None:
        return _agent
    return build_agent_singleton()
