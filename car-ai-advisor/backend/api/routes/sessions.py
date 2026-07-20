"""
会话管理 REST API 路由模块
接口清单 REST 规范设计：
POST   /sessions              创建新对话会话
GET    /sessions             获取当前用户全部会话列表
GET    /sessions/{id}        根据会话ID查询单条会话详情
PATCH  /sessions/{id}        修改会话标题（重命名）
DELETE /sessions/{id}        删除指定会话，同时清理会话消息、用户关联
GET    /sessions/{id}/history 查询该会话下所有历史聊天消息
"""

from fastapi import APIRouter, Depends, HTTPException, Response

# Pydantic 请求/响应数据模型（入参校验、出参结构化）
from backend.schemas.session import CreateSessionReq, RenameSessionReq, SessionListResp
# 核心业务逻辑层：Redis会话管理器（封装hash/zset/set所有redis操作）
from backend.core.session_manager import SessionManager
# 全局依赖注入工厂函数
from backend.api.deps import get_session_manager, check_rate_limit

# 创建路由分组
# prefix：统一路由前缀 /sessions
# tags：接口文档分类标签，在Swagger文档中分栏展示
router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", status_code=201)
async def create_session(
    # 请求体：创建会话的入参（title会话标题），自动json校验
    body: CreateSessionReq,
    # 全局依赖1：限流鉴权依赖，返回当前操作用户user_id
    # 内部逻辑：校验IP/用户限流、token鉴权，未登录/限流直接抛401/429
    user_id: str = Depends(check_rate_limit),
    # 全局依赖2：注入Redis会话管理实例（依赖注入自动传入redis客户端）
    session_mgr: SessionManager = Depends(get_session_manager),
):
    # 业务校验：检查当前用户在线会话是否达到最大并发限制（配置max_concurrent_per_user）
    can_create = await session_mgr.check_concurrent_limit(user_id)
    if not can_create:
        # 超过设备/会话上限，返回429请求过多
        raise HTTPException(status_code=429, detail="会话数量已达上限")
    # 调用底层Redis逻辑创建会话，返回会话完整信息
    return await session_mgr.create_session(user_id, body.title)


@router.get("", response_model=SessionListResp)
async def list_sessions(
    # 鉴权限流依赖，拿到当前登录用户
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    # 查询该用户名下所有session_id集合（redis set），批量读取会话基础信息
    sessions = await session_mgr.list_sessions(user_id)
    # 按统一返回模型封装：会话列表 + 总条数
    return SessionListResp(sessions=sessions, total=len(sessions))


@router.get("/{session_id}")
async def get_session(
    # 路径参数：唯一会话ID
    session_id: str,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    # 从Redis Hash读取会话完整信息
    session = await session_mgr.get_session(session_id)
    # 双重校验：会话不存在 或 会话不属于当前登录用户（越权拦截）
    if session is None or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.patch("/{session_id}")
async def rename_session(
    session_id: str,
    # PATCH 请求体：仅需要新标题
    body: RenameSessionReq,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    # 先校验会话归属，防止越权修改他人会话
    session = await session_mgr.get_session(session_id)
    if session is None or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 更新redis hash中的title字段
    success = await session_mgr.rename_session(session_id, body.new_title)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    # 返回修改结果
    return {"success": True, "session_id": session_id, "title": body.new_title}


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    # 删除逻辑：1.删除会话hash 2.删除会话消息zset 3.从用户session集合移除sid
    success = await session_mgr.delete_session(user_id, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    # 204 必须返回空响应体。旧代码 return JSONResponse(204, None) 会产出
    # "null" 4 字节 body，与 204 无body语义冲突，uvicorn 抛
    # "Response content longer than Content-Length" 并掐断长连接
    return Response(status_code=204)


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    # 权限校验：防止读取别人的聊天记录
    session = await session_mgr.get_session(session_id)
    if session is None or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 读取该会话全部历史消息（zset时序消息）
    messages = await session_mgr.get_history(session_id)
    return {"session_id": session_id, "count": len(messages), "messages": messages}