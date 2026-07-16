"""会话管理 REST API。

POST   /sessions              创建会话
GET    /sessions              会话列表
GET    /sessions/{id}         会话详情
PATCH  /sessions/{id}         重命名
DELETE /sessions/{id}         删除会话
GET    /sessions/{id}/history 消息历史
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from backend.schemas.session import CreateSessionReq, RenameSessionReq, SessionListResp
from backend.core.session_manager import SessionManager
from backend.api.deps import get_session_manager, check_rate_limit

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", status_code=201)
async def create_session(
    body: CreateSessionReq,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    can_create = await session_mgr.check_concurrent_limit(user_id)
    if not can_create:
        raise HTTPException(status_code=429, detail="会话数量已达上限")
    return await session_mgr.create_session(user_id, body.title)


@router.get("", response_model=SessionListResp)
async def list_sessions(
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    sessions = await session_mgr.list_sessions(user_id)
    return SessionListResp(sessions=sessions, total=len(sessions))


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    session = await session_mgr.get_session(session_id)
    if session is None or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.patch("/{session_id}")
async def rename_session(
    session_id: str,
    body: RenameSessionReq,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    session = await session_mgr.get_session(session_id)
    if session is None or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")

    success = await session_mgr.rename_session(session_id, body.new_title)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"success": True, "session_id": session_id, "title": body.new_title}


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    success = await session_mgr.delete_session(user_id, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(status_code=204, content=None)


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    session = await session_mgr.get_session(session_id)
    if session is None or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = await session_mgr.get_history(session_id)
    return {"session_id": session_id, "count": len(messages), "messages": messages}
