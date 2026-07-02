import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

import redis.asyncio as redis
import jwt
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger("session_manager")


# ================================================================
# 练习1+2+4+5: SessionManager 类
# ================================================================
class SessionManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 1800):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def _msg_key(self, sid: str) -> str:
        return f"session:{sid}:messages"

    def _meta_key(self, sid: str) -> str:
        return f"session:{sid}:meta"

    def _user_sessions_key(self, uid: str) -> str:
        return f"user:{uid}:sessions"

    def _concurrency_key(self, uid: str) -> str:
        return f"user:{uid}:concurrency"

    async def close(self):
        await self.redis.close()

    # ========== 练习1: 对话历史存取 ==========
    async def add_message(self, session_id: str, role: str, content: str, user_id: str = "anonymous") -> None:
        """RPUSH 追加到 List 尾部，Pipeline 一组命令一次网络往返"""
        msg = json.dumps({"role": role, "content": content, "timestamp": time.time()}, ensure_ascii=False)
        async with self.redis.pipeline() as pipe:
            pipe.rpush(self._msg_key(session_id), msg)
            pipe.expire(self._msg_key(session_id), self.ttl)
            pipe.expire(self._meta_key(session_id), self.ttl)
            pipe.sadd(self._user_sessions_key(user_id), session_id)
            await pipe.execute()

    async def get_history(self, session_id: str, max_messages: int = 20) -> list[dict]:
        """LRANGE 负索引取最近 N 条，不在内存做全量读取"""
        messages = await self.redis.lrange(self._msg_key(session_id), -max_messages, -1)
        return [json.loads(m) for m in messages]

    async def get_message_count(self, session_id: str) -> int:
        return await self.redis.llen(self._msg_key(session_id))

    # ========== 练习2: TTL 滑动过期 ==========
    async def touch_session(self, session_id: str):
        """每次活动刷新过期时间，活跃用户不会断线"""
        async with self.redis.pipeline() as pipe:
            pipe.expire(self._msg_key(session_id), self.ttl)
            pipe.expire(self._meta_key(session_id), self.ttl)
            await pipe.execute()

    async def get_ttl(self, session_id: str) -> int:
        return await self.redis.ttl(self._msg_key(session_id))

    async def is_alive(self, session_id: str) -> bool:
        return await self.redis.exists(self._msg_key(session_id)) > 0

    # ========== 练习4: 并发控制（Lua 原子操作） ==========
    async def acquire_slot(self, user_id: str, max_concurrent: int = 3) -> bool:
        """INCR + 上限检查打包为 Lua 脚本，避免先查后改的竞态条件"""
        return await self.redis.eval("""
            local current = redis.call('INCR', KEYS[1])
            redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
            if current > tonumber(ARGV[1]) then
                redis.call('DECR', KEYS[1])
                return 0
            end
            return 1
        """, 1, self._concurrency_key(user_id), max_concurrent, 60) == 1

    async def release_slot(self, user_id: str):
        """释放槽位，Lua 判断防减到负数"""
        await self.redis.eval("""
            local v = redis.call('GET', KEYS[1])
            if v and tonumber(v) > 0 then redis.call('DECR', KEYS[1]) end
            return 1
        """, 1, self._concurrency_key(user_id))

    # ========== 练习5: 会话 CRUD ==========
    async def create_session(self, user_id: str, title: str = "新对话") -> str:
        """HSET 存元信息，SMEMBERS 维护用户→会话映射"""
        sid = str(uuid.uuid4())[:8]
        async with self.redis.pipeline() as pipe:
            pipe.hset(self._meta_key(sid), mapping={
                "title": title, "user_id": user_id,
                "created_at": str(time.time()), "message_count": "0",
            })
            pipe.expire(self._meta_key(sid), self.ttl)
            pipe.sadd(self._user_sessions_key(user_id), sid)
            await pipe.execute()
        return sid

    async def list_sessions(self, user_id: str) -> list[dict]:
        sids = await self.redis.smembers(self._user_sessions_key(user_id))
        sessions = []
        for sid in sids:
            meta = await self.redis.hgetall(self._meta_key(sid))
            if meta:
                sessions.append({
                    "session_id": sid,
                    "title": meta.get("title", ""),
                    "message_count": await self.redis.llen(self._msg_key(sid)),
                    "created_at": meta.get("created_at", ""),
                })
        sessions.sort(key=lambda s: s["created_at"], reverse=True)
        return sessions

    async def rename_session(self, session_id: str, new_title: str) -> bool:
        if not await self.redis.exists(self._meta_key(session_id)):
            return False
        await self.redis.hset(self._meta_key(session_id), "title", new_title)
        return True

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """Pipeline 删除：消息 List + 元信息 Hash + 用户 Set 中的引用"""
        async with self.redis.pipeline() as pipe:
            pipe.delete(self._msg_key(session_id))
            pipe.delete(self._meta_key(session_id))
            pipe.srem(self._user_sessions_key(user_id), session_id)
            results = await pipe.execute()
        return results[0] == 1


# ================================================================
# 练习3: 身份校验
# ================================================================
security = HTTPBearer()

API_KEYS = {"sk-test-user-001": "user_001", "sk-test-admin-002": "user_admin"}

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """依赖注入: Authorization: Bearer <key> → user_id"""
    api_key = credentials.credentials
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return API_KEYS[api_key]


class JWTAuth:
    """JWT 令牌认证（可选升级方案）"""
    def __init__(self, secret: str):
        self.secret = secret

    def create_token(self, user_id: str, expire_minutes: int = 60) -> str:
        payload = {"sub": user_id, "iat": datetime.utcnow(),
                   "exp": datetime.utcnow() + timedelta(minutes=expire_minutes)}
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def verify_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="令牌已过期")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="无效令牌")


# ================================================================
# Pydantic Schema
# ================================================================
class CreateSessionReq(BaseModel):
    title: str = Field(default="新对话", max_length=100)

class RenameSessionReq(BaseModel):
    session_id: str = Field(..., min_length=1)
    new_title: str = Field(..., min_length=1, max_length=100)

class SessionItem(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str

class SessionListResp(BaseModel):
    sessions: list[SessionItem]
    total: int


# ================================================================
# 全局实例（Day 12 stream.py 可 import session_mgr 共用）
# ================================================================
session_mgr = SessionManager()


# ================================================================
# FastAPI 应用
# ================================================================
app = FastAPI(title="汽车RAG · Day13 会话管理", version="3.0")

# -- 练习5: 会话 CRUD 端点 --
@app.post("/sessions", tags=["会话"])
async def create_session(req: CreateSessionReq, uid: str = Depends(get_current_user)):
    sid = await session_mgr.create_session(uid, req.title)
    return {"session_id": sid, "title": req.title}

@app.get("/sessions", response_model=SessionListResp, tags=["会话"])
async def list_sessions(uid: str = Depends(get_current_user)):
    sessions = await session_mgr.list_sessions(uid)
    return SessionListResp(sessions=sessions, total=len(sessions))

@app.patch("/sessions/rename", tags=["会话"])
async def rename_session(req: RenameSessionReq, uid: str = Depends(get_current_user)):
    ok = await session_mgr.rename_session(req.session_id, req.new_title)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return {"status": "ok"}

@app.delete("/sessions/{session_id}", tags=["会话"])
async def delete_session(session_id: str, uid: str = Depends(get_current_user)):
    ok = await session_mgr.delete_session(session_id, uid)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return {"status": "deleted", "session_id": session_id}

@app.get("/sessions/{session_id}/history", tags=["会话"])
async def get_history(session_id: str, max_messages: int = 50, uid: str = Depends(get_current_user)):
    history = await session_mgr.get_history(session_id, max_messages)
    ttl = await session_mgr.get_ttl(session_id)
    return {"session_id": session_id, "message_count": len(history), "ttl_remaining_seconds": ttl, "messages": history}


# -- 练习1+2+4: 升级版 /chat --
class ChatReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    stream: bool = Field(default=False)

@app.post("/chat", tags=["对话"])
async def chat(req: ChatReq, uid: str = Depends(get_current_user)):
    """集成身份校验 + 并发控制 + 会话记忆的 /chat"""

    if not await session_mgr.acquire_slot(uid, max_concurrent=3):
        raise HTTPException(status_code=429, detail="并发请求过多，请稍后重试",
                            headers={"Retry-After": "5"})
    try:
        history = await session_mgr.get_history(req.session_id)
        await session_mgr.add_message(req.session_id, "user", req.query, uid)
        await session_mgr.touch_session(req.session_id)

        # TODO: 替换为真实 RAG + LLM
        answer = f"[Session {req.session_id}] 收到: {req.query}（历史 {len(history)} 条）"
        await session_mgr.add_message(req.session_id, "assistant", answer, uid)

        return {"answer": answer, "history_count": len(history) + 1, "session_id": req.session_id}
    finally:
        await session_mgr.release_slot(uid)


# ================================================================
# 启动
# ================================================================
if __name__ == "__main__":
    import sys, asyncio as _a

    # 检查 Redis
    try:
        r = redis.from_url("redis://localhost:6379/0")
        _a.new_event_loop().run_until_complete(r.ping())
        _a.new_event_loop().run_until_complete(r.close())
        print(" Redis 连接正常")
    except Exception:
        print(" Redis 未运行，请先启动:")
        print("   docker run -d --name redis -p 6379:6379 redis:7-alpine")
        sys.exit(1)

    print("=" * 50)
    print("Day 13 会话管理服务")
    print('  测试: curl -H "Authorization: Bearer sk-test-user-001" http://127.0.0.1:8000/sessions')
    print("=" * 50)
    uvicorn.run("session_manager:app", host="127.0.0.1", port=8000, reload=True)
