"""对话 API — 流式(SSE) + 非流式(JSON) 双模式。

stream=false → JSON ChatResp
stream=true  → SSE StreamingResponse (text/event-stream)
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.schemas.chat import ChatReq, ChatResp, SourceDoc
from backend.core.session_manager import SessionManager
from backend.core.stream import sse_generator, format_sse
from backend.api.deps import get_session_manager, check_rate_limit, get_agent
from backend.agent.advisor import CarAdvisorAgent, _strip_all_xml

# ═══════════════════════════════════════════════════════════════
# XML 清理统一复用 advisor 的多层安全网实现（hy- 前缀 / DSML /
# 全角管道 / 截断标签），避免两套正则各自演进、互有盲区。
# ═══════════════════════════════════════════════════════════════
_strip_xml = _strip_all_xml


router = APIRouter(prefix="/chat", tags=["chat"])


async def _agent_generator(
    query: str,
    history: list[dict],
    agent: CarAdvisorAgent,
) -> AsyncGenerator[dict, None]:
    """Agent 流式生成器 — 桥接 Agent.stream_chat() 和 sse_generator()。"""
    async for event in agent.stream_chat(query, history):
        yield event


def _parse_sse_event(sse_str: str) -> dict | None:
    """从一条 SSE 消息中解析完整事件 dict，失败返回 None。"""
    if not sse_str.startswith("data: "):
        return None
    try:
        return json.loads(sse_str[len("data: "):])
    except json.JSONDecodeError:
        return None


def _parse_sse_content(sse_str: str) -> str | None:
    """从一条 SSE 消息中安全提取 token 内容。

    正确的做法：解析 JSON，而不是用字符串 find('content":"')。
    旧代码用字符串匹配，遇到转义字符或内嵌引号会截断或漏掉。
    """
    event = _parse_sse_event(sse_str)
    if event and event.get("type") == "token":
        return event.get("content", "")
    return None


@router.post("")
async def chat(
    body: ChatReq,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
    agent: CarAdvisorAgent = Depends(get_agent),
):
    # ── 保存用户消息 ──
    await session_mgr.add_message(body.session_id, "user", body.query)

    # ── 加载历史（最近 20 条）──
    raw_history = await session_mgr.get_history(body.session_id, limit=20)
    history: list[dict] = []
    for msg in raw_history[:-1]:
        try:
            data = json.loads(msg) if isinstance(msg, str) else msg
            history.append({"role": data["role"], "content": data["content"]})
        except (json.JSONDecodeError, KeyError):
            continue

    # ═══════════════════════════════════════
    # 流式模式
    # ═══════════════════════════════════════
    if body.stream:
        async def _stream_and_save() -> AsyncGenerator[str, None]:
            full_parts: list[str] = []
            async for sse_str in sse_generator(
                _agent_generator(body.query, history, agent)
            ):
                # 用 JSON 解析代替字符串匹配提取 token
                content = _parse_sse_content(sse_str)
                if content:
                    full_parts.append(content)

                # 流式模式同样在后端过滤内部工具调用事件（与非流式对齐），
                # 不再依赖前端过滤，避免 DevTools 中暴露内部工具调用序列
                event = _parse_sse_event(sse_str)
                if event and event.get("type") == "source":
                    docs = [
                        d for d in event.get("documents", [])
                        if not str(d.get("source", "")).startswith("tool:")
                    ]
                    if not docs:
                        continue
                    event["documents"] = docs
                    yield format_sse(event)
                    continue

                yield sse_str

            full_answer = _strip_xml("".join(full_parts))
            if full_answer:
                await session_mgr.add_message(
                    body.session_id, "assistant", full_answer,
                )

        return StreamingResponse(
            _stream_and_save(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ═══════════════════════════════════════
    # 非流式模式
    # ═══════════════════════════════════════
    start = time.time()
    full_answer = ""
    error_msg = ""
    retrieved_sources: list[SourceDoc] = []

    async for event in _agent_generator(body.query, history, agent):
        if event.get("type") == "token":
            full_answer += event.get("content", "")
        elif event.get("type") == "error":
            error_msg = event.get("message", "服务暂时不可用，请稍后重试")
        elif event.get("type") == "source":
            for i, doc in enumerate(event.get("documents", []), 1):
                source_name = doc.get("source", "未知")
                # 过滤掉内部工具调用源，不暴露给前端
                if source_name.startswith("tool:"):
                    continue
                retrieved_sources.append(SourceDoc(
                    rank=i,
                    source=source_name,
                    content=_strip_xml(doc.get("content", "")[:200]),
                    score=doc.get("score", 0.0),
                ))

    latency_ms = (time.time() - start) * 1000
    # Agent 出错时返回友好文案，而不是 HTTP 200 + 空回答
    if not full_answer and error_msg:
        full_answer = error_msg
    full_answer = _strip_xml(full_answer)
    if full_answer:
        await session_mgr.add_message(body.session_id, "assistant", full_answer)

    return ChatResp(
        answer=full_answer,
        sources=retrieved_sources[:5],
        latency_ms=round(latency_ms, 1),
        session_id=body.session_id,
    )
