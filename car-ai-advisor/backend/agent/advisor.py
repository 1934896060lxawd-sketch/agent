"""智能导购 Agent — DeepSeek ReAct Agent，手动实现 function-calling 循环。

与 LangGraph 的 create_agent() 功能等价，但不依赖 langchain/langgraph。
直接使用 OpenAI-compatible API，更轻量，可控性更强。

流式管线:
  用户 query → ReAct 循环 (Thought→Action→Observation)
    → 最终回答 (streaming tokens) → SSE 事件流
"""

import json
import logging
import time
from typing import AsyncGenerator

from openai import AsyncOpenAI

from backend.config import settings
from backend.agent.prompts import CAR_ADVISOR_SYSTEM_PROMPT
from backend.agent.tools import TOOL_SCHEMAS, ToolExecutor
from backend.schemas.chat import SSE_SOURCE, SSE_TOKEN, SSE_DONE, SSE_ERROR

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5  # 最大 ReAct 循环次数，防止无限循环


class CarAdvisorAgent:
    """汽车导购 ReAct Agent。

    使用 DeepSeek API 的原生 function calling 实现:
      LLM 决定调用工具 → 执行工具 → 工具结果注入对话 → LLM 继续推理
    """

    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.llm = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60.0,
        )
        self.model = settings.llm_model_id  # "deepseek-chat"
        logger.info(f"Agent 就绪: model={self.model}, tools={len(TOOL_SCHEMAS)}")

    async def stream_chat(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """流式对话 — 返回 SSE 事件 dict 的异步生成器。

        Args:
            query: 用户当前问题
            history: 历史消息 [{"role":"user","content":"..."}, ...]
                     不含 system prompt，本方法会自动拼接

        Yields:
            {"type": SSE_SOURCE, "documents": [...]}  工具调用/检索结果
            {"type": SSE_TOKEN, "content": "字"}      最终回答 token 流
            {"type": SSE_DONE, "total_tokens": N}     完成
            {"type": SSE_ERROR, "message": "..."}     错误
        """
        messages: list[dict] = [
            {"role": "system", "content": CAR_ADVISOR_SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        try:
            # ── ReAct 循环 ──
            for iteration in range(MAX_ITERATIONS):
                response = await self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.7,
                )
                msg = response.choices[0].message

                # 有工具调用 → 执行工具，注入结果，继续循环
                if msg.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    })

                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        logger.info(f"  ⚡ Action: {tool_name}({args})")
                        result = await self.executor.execute(tool_name, args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                        # 向客户端推送工具调用状态
                        yield {
                            "type": SSE_SOURCE,
                            "documents": [{
                                "rank": 0,
                                "source": f"tool:{tool_name}",
                                "content": str(args)[:200],
                                "score": 0.0,
                            }],
                        }

                    continue  # 继续 ReAct 循环

                # 无工具调用 → 最终回答，退出循环
                if msg.content:
                    messages.append({"role": "assistant", "content": msg.content})
                break

            # ── 流式输出最终回答 ──
            total_tokens = 0
            stream = await self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    total_tokens += 1
                    yield {"type": SSE_TOKEN, "content": delta.content}

            yield {"type": SSE_DONE, "total_tokens": total_tokens}

        except Exception as e:
            logger.error(f"Agent 流式执行失败: {e}", exc_info=True)
            yield {"type": SSE_ERROR, "message": str(e)}
