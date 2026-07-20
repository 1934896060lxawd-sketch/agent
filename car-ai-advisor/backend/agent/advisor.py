"""智能导购 Agent — ReAct 循环 + 流式输出。

架构：
  1. ReAct 循环（非流式 + tools）处理所有工具调用 —— 原生 function_calling 优先，XML 内联兜底
  2. 循环结束后的最终回答字符级流式输出给前端
  3. 绝不发起第二次流式 LLM 调用（那是 XML 泄漏的根本原因）

设计原则（针对 XML 泄漏 5 大根因）：
  (1) 工具调用拦截在 ReAct 循环内完成 —— 绝不会将含 XML 的 content 原样抛出
  (2) 前端只接收 SSE_TOKEN（纯文本），SSE_SOURCE 中的 tool: 前缀事件已被过滤
  (3) 错误消息不包含原始工具调用信息
  (4) logger 日志与 SSE 事件严格分离
  (5) 不用文本补全接口 + 正则解析这种脆弱模式；ReAct 循环用原生 tools 参数
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import AsyncGenerator

from openai import AsyncOpenAI

from backend.config import settings
from backend.agent.prompts import CAR_ADVISOR_SYSTEM_PROMPT
from backend.agent.tools import TOOL_SCHEMAS, ToolExecutor, CAR_PRICE_DB
from backend.schemas.chat import SSE_SOURCE, SSE_TOKEN, SSE_ERROR

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6
# 端到端时间预算：防止"慢 API × 多轮迭代"让前端干等数分钟（前端超时 120s）
LLM_BUDGET_SECONDS = 95
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]

# ═══════════════════════════════════════════════════════════════
# XML 检测 — 覆盖所有已知变体（hy- 前缀 / DSML 标记 / 全角管道 / 截断标签）
# ═══════════════════════════════════════════════════════════════

# 管道符变体：ASCII "|" 或全角 "｜"(U+FF5C)，单个或连续两个
# DeepSeek 实际输出过 <｜｜DSML｜｜invoke ...>（全角双管道），必须兼容
_PIPE = r'[|｜]{1,2}'
# 标签名前的可选 DSML 装饰：如 "|DSML|"、"｜｜DSML｜｜"、"｜DSML｜"
_DSML_DECOR = rf'(?:{_PIPE}(?:DSML{_PIPE})?)?'

# 核心关键词 —— 标签名可能有前缀（如 hy-invoke, hy-parameter, hy-tool_calls）
_XML_KEYWORDS = (
    r'(?:[\w-]*invoke|[\w-]*parameter|[\w-]*tool[\w-]*calls?|'
    r'[\w-]*function[\w-]*calls?)'
)
# 完整装饰标签名（可选 DSML 管道前缀 + 关键词）
_DECORATED = rf'{_DSML_DECOR}{_XML_KEYWORDS}'


def _has_any_xml_or_markup(content: str) -> bool:
    """检测任何 XML/DSML 工具调用标记。

    支持的格式：
      <invoke name="xxx">            — 标准格式
      <hy-invoke name="xxx">        — DeepSeek hy- 前缀格式
      <hy-tool_calls>               — DeepSeek 包装器
      <function_calls>              — 通用 function calling XML
      <|DSML|function_calls>        — DeepSeek DSML 标记（ASCII 管道）
      <｜｜DSML｜｜invoke …>          — DeepSeek DSML 标记（全角管道，线上实测变体）
      未闭合的截断标签（半个 <invoke …，无右尖括号）
    """
    if not content:
        return False
    # 任何装饰后的工具标签（开放或闭合，允许 < 后带空格）
    if re.search(rf'<\s*/?\s*{_DECORATED}\b', content, re.IGNORECASE):
        return True
    # 裸 DSML 字样 + 管道符（含全角），如 "<｜｜DSML" 或 "DSML｜｜"
    if 'DSML' in content and ('|' in content or '｜' in content):
        return True
    # 任何 <| 或 <｜ 起始的标记
    if '<|' in content or '<｜' in content:
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# XML 解析 — 提取工具调用参数
# ═══════════════════════════════════════════════════════════════

# 匹配 <invoke name="xxx">...</invoke>、<hy-invoke ...>、<|DSML|invoke ...>、<｜｜DSML｜｜invoke ...>
# name 属性兼容单/双引号与等号两侧空格；闭合标签允许 < 后带空格
_INVOKE_BLOCK_RE = re.compile(
    rf'<\s*({_DSML_DECOR}[\w-]*invoke)\s+name\s*=\s*["\'](\w+)["\'][^>]*>(.*?)</\s*\1\s*>',
    re.DOTALL,
)
# 匹配 <parameter name="xxx">value</parameter> 及各种装饰变体
# 注意：DeepSeek 的参数标签可能带额外属性如 string="true"
_PARAM_RE = re.compile(
    rf'<\s*({_DSML_DECOR}[\w-]*parameter)\s+name\s*=\s*["\'](\w+)["\'][^>]*>(.*?)</\s*\1\s*>',
    re.DOTALL,
)


def _extract_xml_tool_calls(text: str) -> list[dict]:
    """从文本中提取 XML 格式的工具调用（支持 hy- 前缀）。"""
    results = []
    for m in _INVOKE_BLOCK_RE.finditer(text):
        tool_name = m.group(2)
        if tool_name not in TOOL_NAMES:
            continue
        params_str = m.group(3)
        args = {}
        for pm in _PARAM_RE.finditer(params_str):
            name = pm.group(2)
            value = pm.group(3).strip()
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                pass
            args[name] = value
        results.append({
            "name": tool_name,
            "arguments": args,
            "id": f"xml_{tool_name}_{len(results)}",
        })
    return results


# ═══════════════════════════════════════════════════════════════
# XML 清理 — 多层安全网（覆盖 hy- 前缀 / DSML / 全角管道 / 截断标签）
# ═══════════════════════════════════════════════════════════════

# 完整 invoke 块（含全部装饰变体，name 属性兼容单/双引号）
_INVOKE_STRIP_RE = re.compile(
    rf'<\s*({_DSML_DECOR}[\w-]*invoke)\s+name\s*=\s*["\']\w+["\'][^>]*>.*?</\s*\1\s*>',
    re.DOTALL | re.IGNORECASE,
)
# 完整 tool_calls / function_calls 包装器块（含 DSML 装饰；内容允许嵌套标签）
_TOOL_CALLS_STRIP_RE = re.compile(
    rf'<\s*({_DSML_DECOR}(?:[\w-]*tool[\w-]*calls?|[\w-]*function[\w-]*calls?))\s*>.*?</\s*\1\s*>',
    re.DOTALL | re.IGNORECASE,
)
# 所有 XML/DSML 标签（开放/闭合，允许 < 后带空格，含全角管道装饰）
_TAG_STRIP_RE = re.compile(
    rf'<\s*/?\s*{_DECORATED}\b[^>]*>',
    re.IGNORECASE,
)
# 残留片段（开放标签但未闭合，吃到下一个 < 或文末）
_FRAGMENT_STRIP_RE = re.compile(
    rf'<\s*/?\s*{_DECORATED}\b[^>]*>.*?(?=<|$)',
    re.DOTALL | re.IGNORECASE,
)
# 被 max_tokens 截断的半个尾标签（无右尖括号）：
#   "为您对比 <invoke name=\"compare_cars\"" 或裸管道残片 "<｜｜DSML"
_TAIL_STRIP_RE = re.compile(
    rf'(?:<\s*/?\s*{_DECORATED}\b[^>]{{0,200}}|<\s*{_PIPE}[^>]{{0,60}})$',
    re.DOTALL | re.IGNORECASE,
)
# 裸 DSML 管道标记（如 <|DSML|function_calls> 单标签、｜｜DSML｜｜ 残片）
_BARE_DSML_RE = re.compile(
    rf'<\s*/?\s*{_PIPE}(?:DSML)?{_PIPE}?[^>|｜]{{0,40}}>',
    re.IGNORECASE,
)


def _strip_all_xml(text: str) -> str:
    """多层清理所有 XML/DSML 标记，确保输出纯净。

    清理顺序：
      1. 完整 invoke 块（带内容）
      2. tool_calls / function_calls 包装器块
      3. 所有独立标签（含全角管道装饰、杂散闭合标签）
      4. 未闭合片段
      5. 被截断的半个尾标签
      6. 裸 DSML 管道标记
      7. 压缩多余空行
    """
    if not text:
        return ""
    text = _INVOKE_STRIP_RE.sub("", text)
    text = _TOOL_CALLS_STRIP_RE.sub("", text)
    text = _TAG_STRIP_RE.sub("", text)
    text = _FRAGMENT_STRIP_RE.sub("", text)
    text = _TAIL_STRIP_RE.sub("", text)
    text = _BARE_DSML_RE.sub("", text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# 上下文注入 — 短追问理解
# ═══════════════════════════════════════════════════════════════

_CJK_RE = re.compile(r'[一-鿿]+')


def _build_car_alias_map() -> dict[str, str]:
    """车型别名 → 规范名映射。

    来源 1：CAR_PRICE_DB（全名 + 去空格 + 车型部分，如"海豚"→"比亚迪 海豚"）
    来源 2：知识库 vehicles.json（价格库没有但知识库有的车，如"宋L"→"比亚迪 宋L"）
    """
    alias: dict[str, str] = {}
    for full in CAR_PRICE_DB:
        alias[full] = full
        alias[full.replace(" ", "")] = full
        parts = full.split(" ", 1)
        if len(parts) == 2 and len(parts[1]) >= 2:
            alias.setdefault(parts[1], full)
    try:
        kb_file = Path(settings.knowledge_base_dir) / "raw" / "vehicles.json"
        if kb_file.exists():
            for v in json.loads(kb_file.read_text(encoding="utf-8")):
                brand = str(v.get("brand", "")).strip()
                model = str(v.get("model", "")).strip()
                if not model:
                    continue
                canonical = f"{brand} {model}".strip()
                alias[model] = canonical
                alias[model.replace(" ", "")] = canonical
                if brand:
                    alias[f"{brand}{model}"] = canonical
    except Exception as exc:  # 知识库缺失不影响主流程
        logger.warning(f"车型别名表加载失败（仅用价格库）: {exc}")
    return alias


_CAR_ALIAS_MAP = _build_car_alias_map()
# 长别名优先匹配，避免 "宋" 抢占 "宋L"、"P7" 抢占 "P7i"
_CAR_ALIAS_KEYS = sorted(_CAR_ALIAS_MAP, key=len, reverse=True)
_ASCII_ALNUM = frozenset(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)


def _alias_boundary_ok(normalized: str, start: int, end: int, alias: str) -> bool:
    """纯字母数字别名（L6/G6/M7/007…）要求两侧不是字母数字，
    防止 "L6" 误中 "宋L 662km" 里的 "L662"。
    含中文的别名（宋L/海豹08…）不需要边界检查。
    注意：不能用 str.isalnum()——中文字符也算 alnum。
    """
    if re.search(r'[一-鿿]', alias):
        return True
    if start > 0 and normalized[start - 1] in _ASCII_ALNUM:
        return False
    if end < len(normalized) and normalized[end] in _ASCII_ALNUM:
        return False
    return True


def _extract_car_names(text: str) -> list[str]:
    """从文本中提取提到的车型规范名，按出现顺序返回（价格库+知识库别名）。

    采用"长匹配优先 + 区间不重叠"：短别名（如"L6"）若落在更长匹配
    （如"宋L"后接"662km"形成的"L662"）的区间内则丢弃，防止数字串误伤。
    """
    normalized = text.replace(" ", "")
    matches: list[tuple[int, int, int, str]] = []
    for key in _CAR_ALIAS_KEYS:
        k = key.replace(" ", "")
        start = 0
        while True:
            idx = normalized.find(k, start)
            if idx < 0:
                break
            if _alias_boundary_ok(normalized, idx, idx + len(k), k):
                matches.append((idx, idx + len(k), len(k), _CAR_ALIAS_MAP[key]))
            start = idx + 1
    # 长度降序接受，跳过与已接受区间重叠的短匹配
    matches.sort(key=lambda m: -m[2])
    occupied: list[tuple[int, int]] = []
    accepted: list[tuple[int, str]] = []
    for s, e, _len, name in matches:
        if any(s < oe and e > os_ for os_, oe in occupied):
            continue
        occupied.append((s, e))
        accepted.append((s, name))
    seen: set[str] = set()
    ordered: list[str] = []
    for _, name in sorted(accepted):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _final_sentence(text: str) -> str:
    """取一条消息的结尾句（通常是向用户提出的问题/提议）。

    注意过滤纯表情/符号尾巴：消息以"……养车成本？😊"结尾时，
    按标点切分后最后一段是"😊"，必须跳过它取到真正的问句。
    """
    parts = [s.strip() for s in re.split(r'[。！？!?\n]', text) if s.strip()]
    meaningful = [p for p in parts if re.search(r'[一-鿿A-Za-z0-9]', p)]
    return meaningful[-1][:100] if meaningful else ""


def _detect_previous_intent(final_line: str, full_text: str) -> str:
    """识别上一条 assistant 消息的意图——以结尾问句为准，避免被正文干扰。

    历史教训：结尾问"要不要算宋L的养车成本"，正文对比表里出现小鹏G6，
    旧的宽泛匹配把意图误判成"对比小鹏G6"，导致用户说"需要"时答非所问。
    """
    if not final_line:
        return ""
    # 客套式收尾不算具体提议："如果您还有其他问题欢迎咨询""想了解其他信息
    # 可以告诉我"这类话没有可执行的事项，若误判为"查参数提议"，用户回
    # "需要"时模型会把参数原样复读一遍（答非所问）。
    if re.search(
        r'(如果|若|倘若)[^。！？!?\n]{0,20}(其他|其它)[^。！？!?\n]{0,4}(问题|信息|疑问)'
        r'|随时(告诉|联系|提问|咨询|找)'
        r'|欢迎(继续)?(咨询|提问|联系)'
        r'|有问题[^。！？!?\n]{0,8}(告诉|联系|问|咨询)',
        final_line,
    ):
        return ""
    # 结尾句里的车型优先；结尾没有再看全篇
    cars = _extract_car_names(final_line) or _extract_car_names(full_text)
    car_label = f"（{cars[0]}）" if cars else ""

    if re.search(r'落地|养车|用车成本|成本|费用', final_line):
        return (f"上轮提议帮用户算落地价/用车成本{car_label}——用户说'需要/好的'时，"
                f"立即对该车型调用 calculate_ownership_cost")
    if re.search(r'对比|哪个好|怎么选|选哪|PK|比一比', final_line):
        cars2 = _extract_car_names(final_line)
        label = f"（{' 和 '.join(cars2[:2])}）" if cars2 else car_label
        return (f"上轮提议对比车型{label}——用户说'需要/好的'时，"
                f"立即调用 compare_cars")
    # "详细了解参数"类提议必须是真问句（要不要/吗/？），陈述句不算
    if re.search(r'详细|参数|配置|介绍|说说|讲讲|了解', final_line) and \
            re.search(r'要不要|想不想|需不需要|吗|？|\?|呢', final_line):
        return (f"上轮提议深入了解某车型{car_label}——用户说'需要/好的'时，"
                f"立即查该车型详细参数")
    if re.search(r'推荐', final_line):
        return "上轮询问是否需要推荐——用户说'需要/好的'时，立即用之前讨论的预算调用 recommend_cars"
    if re.search(r'要不要|需不需要|是否|可以吗|行吗', final_line):
        return f"上轮结尾是一个待用户确认的提议：{final_line}"
    return ""


def _build_context_hint(history: list[dict]) -> str:
    """从历史中提取上下文提示，注入到短追问前面。

    原则：用户的短回复（"需要""好的""第一款"）几乎都是在回应上一轮结尾的
    提问/提议，因此结尾句权重最高，正文里顺带提到的车型不得喧宾夺主。
    """
    if not history:
        return ""
    last_assistant = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break
    if not last_assistant:
        return ""

    hint_parts: list[str] = []
    final_line = _final_sentence(last_assistant)

    # 1) 意图（以结尾问句为准）
    intent = _detect_previous_intent(final_line, last_assistant)
    if intent:
        hint_parts.append(intent)

    # 编号推荐列表（供"第一款/第二款"指代解析）
    #    (?!\d) 防止把 "18.98万" 这类小数当成列表编号切错
    numbered = re.split(r'\n?\d+\.\s*(?!\d)', last_assistant)
    has_numbered_list = len(numbered) >= 2

    # 上一轮既没有可识别的提议、也没有推荐列表 → 用户的短确认词
    # （"需要/好的"）没有明确指向。注入"澄清"提示，防止模型把上一轮
    # 内容原样复读一遍（答非所问的一种形态）。
    if not intent and not has_numbered_list:
        all_cars0 = _extract_car_names(last_assistant)
        cars_txt = f"，讨论过：{'、'.join(all_cars0[:3])}" if all_cars0 else ""
        return (
            f"【对话上下文：上一轮只是陈述信息{cars_txt}，并没有提出待用户确认的具体事项】\n"
            "用户的这条简短回复没有明确指向。请用一句话询问用户具体需要什么，"
            "并结合上轮讨论给出 2-3 个可选方向（如：算落地价/养车成本、对比车型、查详细参数），"
            "不要重复上一轮已经说过的内容。\n"
        )

    # 2) 结尾句提到的车型（主角）
    end_cars = _extract_car_names(final_line) if final_line else []
    if end_cars:
        hint_parts.append(f"上轮结尾提到的车型：{'、'.join(end_cars)}")

    # 3) 全篇讨论过的其他车型（配角，仅作背景）
    all_cars = _extract_car_names(last_assistant)
    other_cars = [c for c in all_cars if c not in end_cars]
    if other_cars:
        hint_parts.append(f"本轮还讨论过：{'、'.join(other_cars[:4])}")

    # 4) 编号推荐列表（复用上方已切分的结果）
    if has_numbered_list:
        items = [n.strip()[:30] for n in numbered[1:6] if n.strip()]
        if items:
            hint_parts.append(f"推荐列表：{' | '.join(items)}")

    # 5) 结尾原话（最高权重，放在最后强调）
    if final_line and any(kw in final_line for kw in [
        '要不要', '是否', '需要', '可以', '试试', '哪个', '哪款', '推荐', '算', '对比',
    ]):
        hint_parts.append(f"上轮结尾原话：{final_line}")

    if hint_parts:
        return (
            "【对话上下文：" + "；".join(hint_parts) + "】\n"
            "用户的这条回复很简短，是在回应上轮结尾的提议，"
            "请务必针对'上轮结尾'涉及的车型和事项作答，不要转移到其他车型。\n"
        )
    return ""


# ═══════════════════════════════════════════════════════════════
# Agent 主类
# ═══════════════════════════════════════════════════════════════

class CarAdvisorAgent:
    """汽车导购 ReAct Agent。

    流程：
      ReAct 循环（非流式 + tools）→ 拦截所有工具调用 → 执行 → 结果注入消息
      → 最终回答字符级流式 yield 给前端
    """

    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.llm = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60.0,
            max_retries=1,  # 默认重试2次会让慢API卡顿放大到3×60s，限制为1次
        )
        self.model = settings.llm_model_id

    # ── 工具调用执行 ──

    async def _execute_tool_calls(
        self, calls: list[dict], messages: list[dict],
        raw_content: str = "",
    ) -> AsyncGenerator[dict, None]:
        """执行工具调用，追加标准化消息到 messages，yield SSE_SOURCE 事件。

        关键：raw_content（可能含 XML）经过 _strip_all_xml 清理后再存入消息历史。
        这样 LLM 不会在后续轮次中看到 XML 格式并模仿它。
        """
        clean_content = _strip_all_xml(raw_content) if raw_content else ""
        tc_list = []
        for tc in calls:
            tc_list.append({
                "id": tc.get("id", f"tc_{tc['name']}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            })
        messages.append({
            "role": "assistant",
            "content": clean_content,
            "tool_calls": tc_list,
        })
        for tc in calls:
            tool_name = tc["name"]
            args = tc["arguments"]
            logger.info(f"  Tool: {tool_name}({args})")
            try:
                result = await self.executor.execute(tool_name, args)
            except Exception as exc:
                # 工具执行失败 → 返回纯文本错误（不包含 XML/调用信息）
                result = json.dumps(
                    {"error": f"工具 {tool_name} 执行失败，请稍后重试"},
                    ensure_ascii=False,
                )
                logger.error(f"  Tool {tool_name} failed: {exc}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"tc_{tool_name}"),
                "content": result,
            })
            yield {
                "type": SSE_SOURCE,
                "documents": [{
                    "rank": 0,
                    "source": f"tool:{tool_name}",
                    "content": str(args)[:200],
                    "score": 0.0,
                }],
            }

    # ── 主入口 ──

    async def stream_chat(
        self, query: str, history: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """流式对话 —— 只有最终回答以 token 级流式输出。

        关键设计决策：
        - ReAct 循环使用非流式调用（确保 tool_calls 被正确拦截）
        - 循环结束后只 yield 最终回答的每个字符
        - 不存在第二次流式 LLM 调用（根除 XML 泄漏源）
        """
        # ── 上下文注入 ──
        short_query = len(query) <= 15
        confirm_words = any(w in query for w in [
            '需要', '好的', '可以', '是的', '对', '行', '不用', '算了',
            '详细', '对比', '第一个', '第二', '第三', '哪款', '哪个',
            '看看', '说说', '讲讲', '如何', '怎么样',
        ])
        has_car_in_query = bool(_extract_car_names(query))
        needs_context = (short_query or confirm_words) and not has_car_in_query

        context_hint = _build_context_hint(history or []) if needs_context else ""
        effective_query = context_hint + query if context_hint else query

        if context_hint:
            logger.info(
                "Context injected for '%s': %s",
                query, context_hint.strip()[:120],
            )

        # 澄清型提示（"上一轮没有具体提议"）→ 本轮应先问清用户意图，
        # 不强制调工具；其余情况首轮强制 tool_choice=required，
        # 杜绝模型跳过检索、凭记忆编造车型参数（实测发生过：未调工具
        # 直接回答"宋L是三元锂71kWh"——知识库里是刀片电池87kWh）。
        is_clarify_hint = context_hint.startswith("【对话上下文：上一轮只是陈述信息")

        # ── 组装消息 ──
        messages: list[dict] = [
            {"role": "system", "content": CAR_ADVISOR_SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": effective_query})

        # ═══════════════════════════════════════
        # ReAct 循环 — 所有工具调用在此拦截
        # ═══════════════════════════════════════
        try:
            final_content = ""
            t0 = time.monotonic()

            for iteration in range(MAX_ITERATIONS):
                # 端到端时间预算：超时立即停止迭代，进入收尾流程
                if time.monotonic() - t0 > LLM_BUDGET_SECONDS:
                    logger.warning(
                        "LLM 时间预算(%ds)耗尽（第%d轮），提前收尾",
                        LLM_BUDGET_SECONDS, iteration,
                    )
                    break

                response = await self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice=(
                        "required"
                        if (iteration == 0 and not is_clarify_hint)
                        else "auto"
                    ),
                    temperature=0.7,
                )
                msg = response.choices[0].message

                # (A) 原生 function calling（OpenAI/DeepSeek 兼容）
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

                # (B) XML 内联兜底（DeepSeek 某些版本把工具调用嵌入 content）
                if not tool_calls and msg.content and _has_any_xml_or_markup(msg.content):
                    tool_calls = _extract_xml_tool_calls(msg.content)
                    if tool_calls:
                        logger.info(
                            "XML tool calls extracted: %s",
                            [tc["name"] for tc in tool_calls],
                        )

                # ── 有工具调用 → 拦截执行 → 继续循环 ──
                if tool_calls:
                    async for evt in self._execute_tool_calls(
                        tool_calls, messages, msg.content or "",
                    ):
                        yield evt
                    continue

                # ── 无工具调用 → 最终回答 ──
                # 这就是要给用户看的内容。做最后一次 XML 安全网清理。
                if msg.content:
                    final_content = _strip_all_xml(msg.content)
                    messages.append({
                        "role": "assistant",
                        "content": final_content,
                    })
                break

            # ═══════════════════════════════════════
            # 迭代/预算耗尽仍无最终回答 → 强制一次无工具收尾调用
            # （不传 tools，模型只能输出文本，必然给出面向用户的回答）
            # ═══════════════════════════════════════
            if not final_content:
                logger.warning("ReAct 循环未产生最终回答，发起无工具收尾调用")
                try:
                    wrap = await asyncio.wait_for(
                        self.llm.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=0.7,
                        ),
                        timeout=45.0,
                    )
                    final_content = _strip_all_xml(
                        wrap.choices[0].message.content or ""
                    )
                except Exception as wrap_exc:
                    logger.error(f"无工具收尾调用失败: {wrap_exc}")
                    final_content = ""

            # ── 兜底文案：保证用户永远看得到回应，不出现空气泡 ──
            if not final_content:
                final_content = (
                    "抱歉，这个问题我一时没处理好 😅 "
                    "请换个问法再试一次，或者直接告诉我你的预算和心仪车型，我马上帮你查！"
                )

            # ═══════════════════════════════════════
            # 流式输出最终回答（字符级，无二次 LLM 调用）
            # done 事件由 sse_generator 统一发送，这里不再重复 yield
            # ═══════════════════════════════════════
            if final_content:
                for ch in final_content:
                    yield {"type": SSE_TOKEN, "content": ch}

        except Exception as exc:
            # 错误信息不包含任何原始 LLM 输出，只报告通用错误
            safe_msg = f"服务暂时不可用，请稍后重试"
            logger.error(f"Agent error: {exc}", exc_info=True)
            yield {"type": SSE_ERROR, "message": safe_msg}
