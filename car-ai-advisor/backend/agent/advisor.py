"""智能导购 Agent — ReAct 循环 + 流式输出 + XML 兼容 + 上下文注入。"""

import json
import logging
import re
from typing import AsyncGenerator

from openai import AsyncOpenAI

from backend.config import settings
from backend.agent.prompts import CAR_ADVISOR_SYSTEM_PROMPT
from backend.agent.tools import TOOL_SCHEMAS, ToolExecutor, CAR_PRICE_DB
from backend.schemas.chat import SSE_SOURCE, SSE_TOKEN, SSE_DONE, SSE_ERROR

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]

# ── XML 工具调用解析 ──
# 匹配 <invoke name="xxx"> ... </invoke>，用非贪婪方式
_XML_INVOKE_RE = re.compile(
    r'<invoke\s+name="(\w+)"\s*>(.*?)</invoke>',
    re.DOTALL,
)
# 匹配 <parameter name="xxx">value</parameter>
_XML_PARAM_RE = re.compile(
    r'<parameter\s+name="(\w+)"\s*>(.*?)</parameter>',
    re.DOTALL,
)
# 开放标签——流式检测时用
_XML_OPEN_RE = re.compile(r'<invoke\s+name="(\w+)"\s*>')

# ── 中文字符正则（用于拆分品牌+车型）──
_CJK_RE = re.compile(r'[一-鿿]+')


def _extract_xml_tool_calls(text: str) -> list[dict]:
    """从任意文本中提取所有 XML 格式的工具调用。更宽松的匹配。"""
    results = []
    for m in _XML_INVOKE_RE.finditer(text):
        tool_name = m.group(1)
        if tool_name not in TOOL_NAMES:
            continue
        params_str = m.group(2)
        args = {}
        for pm in _XML_PARAM_RE.finditer(params_str):
            name = pm.group(1)
            value = pm.group(2).strip()
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                pass
            args[name] = value
        results.append({"name": tool_name, "arguments": args, "id": f"xml_{tool_name}_{len(results)}"})
    return results


def _extract_car_names(text: str) -> list[str]:
    """从 AI 回复中提取提到的车型名（去空格模糊匹配）。"""
    found = []
    normalized_text = text.replace(" ", "")
    for name in CAR_PRICE_DB:
        # 去空格比较，容忍 "特斯拉 Model Y" vs "特斯拉Model Y" 的差异
        if name.replace(" ", "") in normalized_text:
            found.append(name)
    return found


def _build_context_hint(history: list[dict]) -> str:
    """从历史中提取上一个 assistant 消息的关键信息，构造上下文提示。

    如果上一条 AI 消息提到了车型或给出了推荐列表，提取出来作为提示注入到用户 query 前。
    """
    if not history:
        return ""
    # 找最近的 assistant 消息
    last_assistant = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break
    if not last_assistant:
        return ""

    cars = _extract_car_names(last_assistant)
    hint_parts = []
    if cars:
        hint_parts.append(f"上一轮提到了这些车型：{'、'.join(cars[:5])}")

    # 检测是否有推荐列表（"1. XX 2. YY"）
    numbered = re.split(r'\n?\d+\.\s*', last_assistant)
    if len(numbered) >= 2:
        items = [n.strip()[:30] for n in numbered[1:6] if n.strip()]
        if items:
            hint_parts.append(f"推荐列表：{' | '.join(items)}")

    if hint_parts:
        return "【对话上下文：" + "；".join(hint_parts) + "】\n"
    return ""


def _has_xml(content: str) -> bool:
    """快速检测文本是否包含 XML 工具调用标记（多种格式）。"""
    return '<invoke' in content and 'name=' in content


_XML_STRIP_RE = re.compile(
    r'<invoke\s+name="\w+"\s*>.*?</invoke>',
    re.DOTALL,
)


class CarAdvisorAgent:
    """汽车导购 ReAct Agent。兼容原生 tool_calls 和 XML 内联两种格式。

    流式输出阶段实时检测 XML：一旦发现立即截断、执行工具、重新生成。
    """

    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.llm = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60.0,
        )
        self.model = settings.llm_model_id

    # ── 工具调用执行（公共逻辑）──
    async def _execute_tool_calls(self, calls: list[dict], messages: list[dict],
                                  content: str = "") -> AsyncGenerator[dict, None]:
        """执行一系列工具调用，追加到 messages，并 yield source 事件。"""
        tc_list = []
        for tc in calls:
            tc_list.append({
                "id": tc.get("id", f"tc_{tc['name']}"),
                "type": "function",
                "function": {"name": tc["name"],
                             "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
            })
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tc_list,
        })
        for tc in calls:
            tool_name = tc["name"]
            args = tc["arguments"]
            logger.info(f"  Tool: {tool_name}({args})")
            result = await self.executor.execute(tool_name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"tc_{tool_name}"),
                "content": result,
            })
            yield {
                "type": SSE_SOURCE,
                "documents": [{"rank": 0, "source": f"tool:{tool_name}",
                               "content": str(args)[:200], "score": 0.0}],
            }

    # ── 带 XML 过滤的流式输出 ──
    async def _stream_with_xml_filter(
        self, messages: list[dict], iteration: int,
    ) -> AsyncGenerator[dict, None]:
        """流式输出，边产出边过滤 XML，检测到 XML 时截断并重新执行工具。"""
        stream = await self.llm.chat.completions.create(
            model=self.model, messages=messages,
            stream=True, temperature=0.7,
        )
        full_text = ""
        xml_found_at = -1
        total_tokens = 0
        # token 缓冲区 — 逐个 yield 但保留完整文本用于 XML 检测
        token_buffer: list[str] = []

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
            total_tokens += 1
            token = delta.content
            full_text += token
            token_buffer.append(token)

            # 每收到一个 token 就检查一次 XML
            if xml_found_at < 0 and _has_xml(full_text):
                xml_found_at = len(full_text) - len(token) - 50  # 回溯 50 字符找精确位置
                logger.info("  XML detected in stream, aborting")

        # ── 无 XML：正常 yield 所有 token ──
        if xml_found_at < 0:
            for t in token_buffer:
                yield {"type": SSE_TOKEN, "content": t}
            yield {"type": SSE_DONE, "total_tokens": total_tokens}
            return

        # ── 有 XML ──
        # 1) 找到 XML 开始前的干净文本
        invoke_idx = full_text.find('<invoke')
        clean = full_text[:invoke_idx].strip() if invoke_idx > 0 else ""

        # 2) yield 干净文本
        if clean:
            for ch in clean:
                yield {"type": SSE_TOKEN, "content": ch}

        # 3) 提取 XML 工具调用
        xml_calls = _extract_xml_tool_calls(full_text)
        logger.info(f"  Extracted {len(xml_calls)} XML calls: {[x['name'] for x in xml_calls]}")

        # 4) 执行工具并 yield source 事件
        if xml_calls and iteration + 1 < MAX_ITERATIONS:
            async for evt in self._execute_tool_calls(xml_calls, messages, full_text):
                yield evt

            # 5) 重新生成（不带 XML 的干净回答）
            stream2 = await self.llm.chat.completions.create(
                model=self.model, messages=messages,
                stream=True, temperature=0.7,
            )
            async for chunk in stream2:
                delta = chunk.choices[0].delta
                if delta.content:
                    # 逐 token 再次过滤，防止死循环
                    if not _has_xml(delta.content):
                        yield {"type": SSE_TOKEN, "content": delta.content}

        yield {"type": SSE_DONE, "total_tokens": total_tokens}

    # ── 主入口 ──
    async def stream_chat(
        self, query: str, history: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """流式对话入口。"""
        # 仅当用户 query 简短（<=10字）或明显是确认/追问词时，注入上下文
        short_query = len(query) <= 10
        confirm_words = any(w in query for w in ['需要', '好的', '可以', '是的', '对', '行',
                                                    '不用', '算了', '详细', '对比', '第一个',
                                                    '第二', '第三', '哪款', '哪个'])
        has_car_in_query = bool(_extract_car_names(query))
        needs_context = (short_query or confirm_words) and not has_car_in_query

        context_hint = _build_context_hint(history or []) if needs_context else ""
        effective_query = context_hint + query if context_hint else query

        messages: list[dict] = [
            {"role": "system", "content": CAR_ADVISOR_SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": effective_query})

        try:
            # ── ReAct 循环 ──
            for iteration in range(MAX_ITERATIONS):
                response = await self.llm.chat.completions.create(
                    model=self.model, messages=messages,
                    tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.7,
                )
                msg = response.choices[0].message

                # 检测工具调用
                tool_calls: list[dict] = []

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({
                            "name": tc.function.name,
                            "arguments": args,
                            "id": tc.id,
                        })

                if not tool_calls and msg.content and _has_xml(msg.content):
                    tool_calls = _extract_xml_tool_calls(msg.content)

                if tool_calls:
                    async for evt in self._execute_tool_calls(
                        tool_calls, messages, msg.content or "",
                    ):
                        yield evt
                    continue

                # 无工具调用 → 最终回答
                if msg.content:
                    messages.append({"role": "assistant", "content": msg.content})
                break

            # ── 流式输出（带 XML 实时过滤）──
            async for evt in self._stream_with_xml_filter(messages, iteration):
                yield evt

        except Exception as e:
            logger.error(f"Agent 异常: {e}", exc_info=True)
            yield {"type": SSE_ERROR, "message": str(e)}
