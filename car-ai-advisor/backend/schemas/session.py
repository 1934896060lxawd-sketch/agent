"""会话管理相关 Pydantic 数据模型"""
from pydantic import BaseModel, Field


class CreateSessionReq(BaseModel):
    """创建会话请求"""
    title: str = Field(default="新对话", max_length=100)


class RenameSessionReq(BaseModel):
    """重命名会话请求"""
    session_id: str = Field(..., min_length=1)
    new_title: str = Field(..., min_length=1, max_length=100)


class SessionItem(BaseModel):
    """会话列表中的单条记录"""
    session_id: str
    title: str
    message_count: int
    created_at: str


class SessionListResp(BaseModel):
    """会话列表响应"""
    sessions: list[SessionItem]
    total: int
    