"""对话 API — 流式(SSE) + 非流式(JSON) 双模式。

stream=false → JSON ChatResp
stream=true  → SSE StreamingResponse (text/event-stream)

Phase 3: 接入真实 DeepSeek ReAct Agent，替换占位回答。
"""

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.schemas.chat import ChatReq, ChatResp, SourceDoc
from backend.core.session_manager import SessionManager
from backend.core.stream import sse_generator
from backend.api.deps import get_session_manager, check_rate_limit, get_agent
from backend.agent.advisor import CarAdvisorAgent

router = APIRouter(prefix="/chat", tags=["chat"])


async def _agent_generator(
    query: str,
    history: list[dict],
    agent: CarAdvisorAgent,
) -> AsyncGenerator[dict, None]:
    """Agent 流式生成器 — 桥接 Agent.stream_chat() 和 sse_generator()。"""
    async for event in agent.stream_chat(query, history):
        yield event


@router.post("")
async def chat(
    body: ChatReq,
    user_id: str = Depends(check_rate_limit),
    session_mgr: SessionManager = Depends(get_session_manager),
    agent: CarAdvisorAgent = Depends(get_agent),
):
    # 保存用户消息
    await session_mgr.add_message(body.session_id, "user", body.query)

    # 加载历史消息（最近 20 条，排除刚加入的当前消息）
    raw_history = await session_mgr.get_history(body.session_id, limit=20)
    history: list[dict] = []
    for msg in raw_history[:-1]:
        try:
            data = json.loads(msg) if isinstance(msg, str) else msg
            history.append({"role": data["role"], "content": data["content"]})
        except (json.JSONDecodeError, KeyError):
            continue

    if body.stream:
        async def _stream_and_save() -> AsyncGenerator[str, None]:
            full_parts: list[str] = []
            async for sse_str in sse_generator(_agent_generator(body.query, history, agent)):
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
    retrieved_sources: list[SourceDoc] = []

    async for event in _agent_generator(body.query, history, agent):
        if event.get("type") == "token":
            full_answer += event.get("content", "")
        elif event.get("type") == "source":
            for i, doc in enumerate(event.get("documents", []), 1):
                retrieved_sources.append(SourceDoc(
                    rank=i,
                    source=doc.get("source", "未知"),
                    content=doc.get("content", "")[:200],
                    score=doc.get("score", 0.0),
                ))

    latency_ms = (time.time() - start) * 1000
    if full_answer:
        await session_mgr.add_message(body.session_id, "assistant", full_answer)

    return ChatResp(
        answer=full_answer,
        sources=retrieved_sources[:5],
        latency_ms=round(latency_ms, 1),
        session_id=body.session_id,
    )
