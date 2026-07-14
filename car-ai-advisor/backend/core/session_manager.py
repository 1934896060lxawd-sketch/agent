"""
会话管理器 — 基于 Redis 的会话 CRUD 与消息历史存储。

Redis 数据结构:
  - sessions:{session_id}          (Hash)   会话元数据
  - sessions:{session_id}:messages (List)   消息历史，RPUSH 追加
  - user:{user_id}:sessions        (Set)    用户拥有的会话 ID 集合

设计要点:
  1. 构造函数接收 redis_client（依赖注入），不自己创建连接
  2. TTL 从 settings 读取，每次读取时续期（滑动过期）
  3. 不做全局单例 — 每个请求通过 Depends 拿到同一个实例
  4. 多次 Redis 操作用 Pipeline 合并，减少网络往返
"""

import json
import logging
import time
import uuid
from typing import Optional

import redis.asyncio as redis

from backend.config import settings

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器 — 封装所有会话相关的 Redis 操作。

    用法:
        session_mgr = SessionManager(app.state.redis)
        session = await session_mgr.create_session("user_001", "选SUV")
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Args:
            redis_client: Redis 异步客户端（由 lifespan 注入到 app.state.redis）
        """
        self.redis = redis_client
        self.ttl = settings.session_ttl_seconds          # 默认 1800 秒
        self.max_concurrent = settings.max_concurrent_per_user  # 默认 3

    # ================================================================
    # 内部工具方法 — Redis Key 生成（集中管理 key 命名，方便修改）
    # ================================================================
    @staticmethod
    def _session_key(session_id: str) -> str:
        """会话元数据 Hash 的 key"""
        return f"sessions:{session_id}"

    @staticmethod
    def _messages_key(session_id: str) -> str:
        """会话消息列表的 key"""
        return f"sessions:{session_id}:messages"

    @staticmethod
    def _user_sessions_key(user_id: str) -> str:
        """用户会话集合的 key"""
        return f"user:{user_id}:sessions"

    # ================================================================
    # 会话 CRUD
    # ================================================================
    async def create_session(self, user_id: str, title: str = "新对话") -> dict:
        """创建新会话。

        原子操作:
          1. 生成 12 位短 ID（uuid4 前 12 位，兼顾唯一性和可读性）
          2. HSET 会话元数据 Hash
          3. SADD 将 session_id 加入用户会话集合
          4. 设置 TTL（消息列表也预设置，防止孤儿数据）

        Args:
            user_id: 用户标识
            title: 会话标题，默认"新对话"

        Returns:
            {"session_id": "...", "title": "...", "message_count": 0, "created_at": "..."}
        """
        session_id = uuid.uuid4().hex[:12]  # 12 位 hex，碰撞概率 ≈ 1/16^12，可忽略
        now = time.time()

        session_data = {
            "title": title,
            "user_id": user_id,
            "created_at": str(now),
            "updated_at": str(now),
            "message_count": "0",         # Hash 值必须是字符串
        }

        # 【关键】Pipeline 把 4 条命令打包成一次网络往返
        # 原生的 4 次往返 ≈ 4ms（局域网），Pipeline 后 ≈ 1ms
        pipe = self.redis.pipeline()
        pipe.hset(self._session_key(session_id), mapping=session_data)
        pipe.expire(self._session_key(session_id), self.ttl)
        pipe.sadd(self._user_sessions_key(user_id), session_id)
        pipe.expire(self._user_sessions_key(user_id), self.ttl)
        pipe.expire(self._messages_key(session_id), self.ttl)   # 预设置，避免孤儿
        await pipe.execute()

        logger.info(f"会话创建成功: session_id={session_id}, user={user_id}")
        return {
            "session_id": session_id,
            "title": title,
            "message_count": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        }

    async def get_session(self, session_id: str) -> Optional[dict]:
        """获取单个会话元数据。

        读取时自动续期 TTL（滑动过期策略）— 活跃会话不会因为 30 分钟
        就到了而被清理。

        Returns:
            会话 dict，不存在则返回 None
        """
        key = self._session_key(session_id)
        data = await self.redis.hgetall(key)
        if not data:
            return None

        # 【滑动过期】每次读取都续期，活跃用户不会掉线
        await self.redis.expire(key, self.ttl)
        await self.redis.expire(self._messages_key(session_id), self.ttl)

        return {
            "session_id": session_id,
            "title": data.get("title", ""),
            "user_id": data.get("user_id", ""),
            "message_count": int(data.get("message_count", 0)),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
        }

    async def list_sessions(self, user_id: str) -> list[dict]:
        """列出用户的所有会话，按更新时间倒序排列。

        用 SMEMBERS 获取用户的所有 session_id，
        然后用 Pipeline 批量 HGETALL 每个会话的元数据。
        N 个会话只需 2 次网络往返（而非 1+N 次）。

        如果会话数据已过期（data 为空），自动过滤。
        """
        session_ids = await self.redis.smembers(self._user_sessions_key(user_id))
        if not session_ids:
            return []

        # Pipeline 批量获取：N 个 HGETALL 合并成 1 次网络往返
        pipe = self.redis.pipeline()
        for sid in session_ids:
            pipe.hgetall(self._session_key(sid))
        results = await pipe.execute()

        sessions = []
        for sid, data in zip(session_ids, results):
            if data:   # 过滤已过期（data 为空 {} 的）
                sessions.append({
                    "session_id": sid,
                    "title": data.get("title", ""),
                    "message_count": int(data.get("message_count", 0)),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })

        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    async def rename_session(self, session_id: str, new_title: str) -> bool:
        """重命名会话。

        Returns:
            True 表示成功，False 表示会话不存在
        """
        key = self._session_key(session_id)
        exists = await self.redis.exists(key)
        if not exists:
            return False

        pipe = self.redis.pipeline()
        pipe.hset(key, "title", new_title)
        pipe.hset(key, "updated_at", str(time.time()))
        pipe.expire(key, self.ttl)
        await pipe.execute()

        logger.info(f"会话重命名: {session_id} → '{new_title}'")
        return True

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除会话及其所有消息历史。

        原子操作:
          1. DEL 会话元数据 Hash
          2. DEL 会话消息列表
          3. SREM 从用户集合中移除

        Returns:
            True 表示删除成功，False 表示会话不存在
        """
        pipe = self.redis.pipeline()
        pipe.delete(self._session_key(session_id))
        pipe.delete(self._messages_key(session_id))
        pipe.srem(self._user_sessions_key(user_id), session_id)
        results = await pipe.execute()

        # results[0] 是 DELETE 返回值：1=删了，0=key不存在
        deleted = results[0] == 1
        if deleted:
            logger.info(f"会话删除: {session_id}, user={user_id}")
        return deleted

    # ================================================================
    # 消息历史
    # ================================================================
    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """向会话历史追加一条消息。

        每追加一条消息，同时:
          - 续期会话 TTL
          - 递增 message_count（HINCRBY 原子操作）
          - 更新 updated_at

        Args:
            session_id: 会话 ID
            role: "user" 或 "assistant"
            content: 消息文本
        """
        msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        now = str(time.time())

        pipe = self.redis.pipeline()
        pipe.rpush(self._messages_key(session_id), msg)            # 追加到 List 尾部
        pipe.expire(self._messages_key(session_id), self.ttl)      # 续期
        pipe.hincrby(self._session_key(session_id), "message_count", 1)  # 计数+1
        pipe.hset(self._session_key(session_id), "updated_at", now)
        pipe.expire(self._session_key(session_id), self.ttl)
        await pipe.execute()

    async def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        """获取会话的消息历史（最近 N 条）。

        用 LRANGE 的负数索引直接取尾部：
          -limit = -50  →  倒数第 50 条到最后一条
          不需要先 LLEN 再算 offset，一步到位。

        Args:
            session_id: 会话 ID
            limit: 最大返回条数，默认 50

        Returns:
            [{"role": "user", "content": "..."}, ...]，按时间正序
        """
        key = self._messages_key(session_id)
        raw_messages = await self.redis.lrange(key, -limit, -1)

        messages = []
        for raw in raw_messages:
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning(f"消息 JSON 解析失败，跳过: {raw[:80]}...")
        return messages

    # ================================================================
    # 并发控制
    # ================================================================
    async def check_concurrent_limit(self, user_id: str) -> bool:
        """检查用户并发会话数是否超限。

        Returns:
            True 表示未超限（可以创建新会话），False 表示已达上限
        """
        count = await self.redis.scard(self._user_sessions_key(user_id))
        return count < self.max_concurrent
