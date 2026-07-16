"""对话 API — 流式(SSE) + 非流式(JSON) 双模式。

stream=false → JSON ChatResp
stream=true  → SSE StreamingResponse (text/event-stream)

Phase 2 用占位回答验证全链路。Phase 3 替换 _placeholder_generator 为 Agent。
"""

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.schemas.chat import ChatReq, ChatResp, SSE_SOURCE, SSE_TOKEN
from backend.core.session_manager import SessionManager
from backend.core.stream import sse_generator
from backend.api.deps import get_session_manager, check_rate_limit

router = APIRouter(prefix="/chat", tags=["chat"])


async def _placeholder_generator(query: str) -> AsyncGenerator[dict, None]:
    """占位生成器，Phase 3 替换为 Agent 流式执行。"""
    yield {"type": SSE_SOURCE, "documents": []}

    placeholder = (
        f"收到您的问题：「{query}」。"
        f"（这是占位回答，Phase 3 将接入真实 AI 导购能力。）"
    )
    for char in placeholder:
        yield {"type": SSE_TOKEN, "content": char}
        await asyncio.sleep(0.03)


@router.post("")
async def chat(
    body: ChatReq,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
):
    await session_mgr.add_message(body.session_id, "user", body.query)

    if body.stream:
        async def _stream_and_save() -> AsyncGenerator[str, None]:
            full_parts: list[str] = []
            async for sse_str in sse_generator(_placeholder_generator(body.query)):
                if '"type":"token"' in sse_str:
                    prefix = '"content":"'
                    idx = sse_str.find(prefix)
                    if idx != -1:
                        end = sse_str.find('"', idx + len(prefix))
                        if end != -1:
                            full_parts.append(sse_str[idx + len(prefix):end])
                yield sse_str

            full_answer = "".join(full_parts)
            if full_answer:
                await session_mgr.add_message(body.session_id, "assistant", full_answer)

        return StreamingResponse(
            _stream_and_save(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 非流式模式
    start = time.time()
    full_answer = ""
    async for event in _placeholder_generator(body.query):
        if event.get("type") == SSE_TOKEN:
            full_answer += event.get("content", "")

    latency_ms = (time.time() - start) * 1000
    await session_mgr.add_message(body.session_id, "assistant", full_answer)

    return ChatResp(
        answer=full_answer,
        sources=[],
        latency_ms=round(latency_ms, 1),
        session_id=body.session_id,
    )
