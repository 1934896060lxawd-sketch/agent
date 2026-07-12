"""对话相关 Pydantic 数据模型 & SSE 事件类型常量"""
from pydantic import BaseModel, Field


# SSE 事件类型
SSE_SOURCE = "source"   # 检索到的参考文档列表
SSE_TOKEN  = "token"    # LLM 生成的单个 token
SSE_DONE   = "done"     # 生成完成（含 token 总数）
SSE_ERROR  = "error"    # 发生错误


class SourceDoc(BaseModel):
    """单条检索来源"""
    rank: int = Field(description="排名 1-N")
    source: str = Field(description="来源文档名/车型名")
    content: str = Field(description="内容摘要")
    score: float = Field(description="相关性分数 0-1")


class ChatReq(BaseModel):
    """POST /chat 请求体"""
    query: str = Field(
        ..., min_length=1, max_length=2000,
        description="用户问题",
        examples=["25万预算推荐什么车？"],
    )
    session_id: str = Field(
        default="default", min_length=1, max_length=64,
        description="会话 ID",
    )
    stream: bool = Field(
        default=False,
        description="true=SSE 流式推送, false=一次性返回 JSON",
    )


class ChatResp(BaseModel):
    """非流式模式响应体"""
    answer: str
    sources: list[SourceDoc] = Field(default_factory=list)
    latency_ms: float
    session_id: str