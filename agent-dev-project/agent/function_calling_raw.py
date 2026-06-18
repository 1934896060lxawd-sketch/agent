"""
Day 3: 原生 SDK 实现 Function Calling —— Tool Calling 全流程手写

目的：不依赖任何框架封装，用原生 OpenAI SDK 手写 tool calling 循环，
      理解 LLM 如何"决定调用工具"、参数如何传递、结果如何回传。

核心概念：
  ① Tool 定义 — JSON Schema 描述函数签名，告诉 LLM"我能做什么"
  ② Tool Calling 循环 — LLM 返回 tool_calls → 执行 → 结果喂回 → 循环
  ③ 并行调用 — 一次返回多个 tool_calls，互不依赖时并行执行
  ④ tool_choice 三种模式 — auto / required / none 的行为差异
  ⑤ 错误处理 — 工具执行失败时，把错误信息回传给 LLM 让其调整

对比 LangChain：
  原生 SDK：你控制每一步（while True / 手动拼接 messages / 手动执行工具）
  LangChain @tool：装饰器 + AgentExecutor 自动循环，快但遮蔽了细节
  Day 3 先手写原生 → Day 4 用 LangGraph 重写，理解"框架帮你省了什么"

运行方式：
  cd agent-dev-project && python -m agent.function_calling_raw
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional

# ---- 路径 & 环境 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "chapters"))

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from openai import OpenAI

# ---- 复用第五章的检索能力作为 tool 的后端实现 ----
from naive_rag import load_data, search as keyword_search, ask_llm
from full_rag_agent import Reranker, extract_filters
from retrieval_test import VectorIndex, BM25, hybrid_rrf
from embedding_test import _embed_model, embed_documents

# ============================================================
# 0. 初始化：LLM 客户端 + 检索索引（一次性）
# ============================================================

_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
_model = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")

# 记忆存储：tool 共享的车型数据（模拟一个简单数据库）
CAR_PRICE_DB = {
    "小米 SU7":   "21.59-29.99 万",
    "比亚迪 海豚": "9.98-13.98 万",
    "比亚迪 海豹": "17.98-24.98 万",
    "特斯拉 Model Y": "24.99-35.49 万",
    "特斯拉 Model 3": "23.19-33.59 万",
    "理想 L6":    "24.98-27.98 万",
    "理想 L7":    "30.98-37.98 万",
    "问界 M7":    "24.98-32.98 万",
    "小鹏 G6":    "20.99-27.69 万",
    "蔚来 ET5":   "29.80-35.60 万",
    "极氪 001":  "26.90-32.90 万",
    "零跑 C11":  "15.58-19.98 万",
    "埃安 Y":    "11.98-18.98 万",
}

CAR_SPEC_DB = {
    "小米 SU7": "CLTC续航 700-830km，零百加速 2.78-5.28s，智驾芯片 Orin-X，激光雷达有",
    "比亚迪 海豚": "CLTC续航 301-405km，零百加速 ~10s，智驾 L2 基础，无激光雷达",
    "比亚迪 海豹": "CLTC续航 550-700km，零百加速 3.8s，DiPilot 智驾",
    "特斯拉 Model Y": "CLTC续航 545-688km，零百加速 3.7-5.0s，FSD 智驾",
    "特斯拉 Model 3": "CLTC续航 556-713km，零百加速 3.3-6.1s，FSD 智驾",
    "理想 L6": "增程，CLTC综合续航 1390km，零百加速 5.4s，AD Max 智驾",
    "理想 L7": "增程，CLTC综合续航 1315km，零百加速 5.3s，AD Max 智驾",
    "问界 M7": "增程/纯电，CLTC续航 1200km，零百加速 4.8s，ADS 2.0",
    "小鹏 G6": "CLTC续航 580-755km，零百加速 3.9-6.2s，XNGP 智驾，双激光雷达",
    "蔚来 ET5": "CLTC续航 560-710km，零百加速 4.0s，NAD 智驾，激光雷达",
    "极氪 001": "CLTC续航 546-741km，零百加速 3.8s，NZP 智驾",
    "零跑 C11": "CLTC续航 502-650km，零百加速 4.5-7.9s，L2 智驾",
    "埃安 Y": "CLTC续航 430-610km，零百加速 ~8s，ADiGO 智驾",
}


# ============================================================
# 1. 知识点 1：定义 Tool（JSON Schema）
# ============================================================
# 每个 tool 是一个字典，描述：
#   name        — 函数名（LLM 用它来指定要调哪个）
#   description — 何时该调这个函数（给 LLM 看的，不是给人看的）
#   parameters  — 入参的 JSON Schema（LLM 按这个结构填参数值）
#
# 关键设计原则：
#   ① description 要写清楚"何时调"：例如"当用户问价格有关的问题时"
#   ② properties 里每个字段也要写 description：LLM 用它判断填什么值
#   ③ required 标记必填字段：否则 LLM 可能漏参

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_car_knowledge",
            "description": (
                "搜索汽车知识库，获取车型的详细参数、续航、智驾、空间等信息。"
                "当用户询问车型的具体配置、性能参数、或是需要了解某款车时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询语句，应包含品牌和车型名，如'小米SU7续航里程'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_car_price",
            "description": (
                "查询指定车型的最新市场指导价。"
                "当用户询问价格、预算、多少钱、贵不贵时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brand": {
                        "type": "string",
                        "description": "品牌名称，如'小米'、'比亚迪'、'特斯拉'",
                    },
                    "model": {
                        "type": "string",
                        "description": "车型名称，如'SU7'、'海豚'、'Model Y'",
                    },
                },
                "required": ["brand", "model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cars",
            "description": (
                "对比两款车的核心参数，包括价格、续航、加速、智驾。"
                "当用户要求对比两款车、或问'A和B哪个好/哪个更值'时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "car1": {
                        "type": "string",
                        "description": "第一款车的完整名称，如'小米SU7'",
                    },
                    "car2": {
                        "type": "string",
                        "description": "第二款车的完整名称，如'比亚迪海豚'",
                    },
                },
                "required": ["car1", "car2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_budget",
            "description": (
                "根据预算筛选符合条件的车型列表。"
                "当用户说'XX万以内'、'预算XX万'、'XX万左右推荐'时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {
                        "type": "number",
                        "description": "最高预算（单位：万元）",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "最低预算（单位：万元），用户没说下限时填 0",
                    },
                    "category": {
                        "type": "string",
                        "description": "车型类别偏好，如'SUV'、'轿车'，用户没提则留空",
                    },
                },
                "required": ["max_price"],
            },
        },
    },
]


# ============================================================
# 2. Tool 的后端实现（真正执行逻辑的函数）
# ============================================================

# 这些函数是 tool 的"身体"——LLM 发出调用指令，你的代码执行。
# LLM 永远不会执行代码，它只是说"我想调某某函数，参数是某某"。

def _execute_search_car_knowledge(query: str, knowledge_index) -> str:
    """搜索汽车知识库（混合检索 + Reranker）"""
    if knowledge_index is None:
        return json.dumps({"error": "知识库索引未初始化"}, ensure_ascii=False)

    v_idx, bm25_idx, reranker = knowledge_index

    # 混合检索（和第五章完全一致的管线）
    dense = v_idx.search(query, top_k=6)
    sparse = bm25_idx.search(query, top_k=6)
    hybrid = hybrid_rrf(dense, sparse, k=60, top_k=6)

    if reranker and reranker.model:
        hybrid = reranker.rerank(query, hybrid, top_k=3)
    else:
        hybrid = hybrid[:3]

    results = []
    for i, (score, doc) in enumerate(hybrid, 1):
        results.append({
            "rank": i,
            "source": doc.get("source", "未知"),
            "content": doc.get("content", "")[:300],
            "score": round(score, 4),
        })

    return json.dumps({"results": results}, ensure_ascii=False)


def _execute_get_car_price(brand: str, model: str) -> str:
    """查询车型价格（从 CAR_PRICE_DB 查，模拟数据库）"""
    key = f"{brand} {model}"
    if key in CAR_PRICE_DB:
        return json.dumps({
            "car": key,
            "price": CAR_PRICE_DB[key],
            "status": "found",
        }, ensure_ascii=False)
    # 模糊匹配
    for k, v in CAR_PRICE_DB.items():
        if brand in k and model in k:
            return json.dumps({"car": k, "price": v, "status": "found"}, ensure_ascii=False)
    return json.dumps({
        "car": key,
        "price": None,
        "status": "not_found",
        "message": f"未找到 {key} 的价格信息，请尝试其他关键词",
    }, ensure_ascii=False)


def _execute_compare_cars(car1: str, car2: str) -> str:
    """对比两款车（从 CAR_SPEC_DB + CAR_PRICE_DB 取数据）"""
    spec1 = CAR_SPEC_DB.get(car1, "暂无详细参数")
    spec2 = CAR_SPEC_DB.get(car2, "暂无详细参数")
    price1 = CAR_PRICE_DB.get(car1, "暂无价格")
    price2 = CAR_PRICE_DB.get(car2, "暂无价格")

    return json.dumps({
        "car1": {"name": car1, "price": price1, "spec": spec1},
        "car2": {"name": car2, "price": price2, "spec": spec2},
    }, ensure_ascii=False)


def _execute_filter_by_budget(max_price: float, min_price: float = 0,
                               category: str = "") -> str:
    """根据预算筛选车型"""
    matches = []
    for car, price_str in CAR_PRICE_DB.items():
        # 取最低价做筛选（和第五章 apply_filters 策略一致）
        try:
            low_price = float(price_str.split("-")[0].strip())
        except ValueError:
            continue
        if min_price <= low_price <= max_price:
            if category and category not in car:
                continue
            spec = CAR_SPEC_DB.get(car, "")
            matches.append({"car": car, "price": price_str, "spec": spec})

    matches.sort(key=lambda x: float(x["price"].split("-")[0].strip()))
    return json.dumps({"matches": matches, "count": len(matches)}, ensure_ascii=False)


# ============================================================
# 3. 知识点 2：Tool Calling 循环（核心模式）
# ============================================================
# 这是 Day 3 最重要的代码。LLM 调用不是一次性的：
#
#   用户问题
#     → LLM: "我需要调 search_car_knowledge(query='小米SU7续航')"
#     → 你的代码执行 search_car_knowledge → 返回 JSON 结果
#     → 把结果以 role="tool" 追加到 messages
#     → LLM 看到结果后: "好的，小米SU7的续航是..."（生成最终回答）
#
# 循环终止条件：LLM 返回了 content 且没有 tool_calls


def _execute_tool(tool_name: str, arguments: dict, knowledge_index) -> str:
    """
    根据 tool_name 路由到对应的实现函数。

    LLM 输出的是函数名 + JSON 参数，我们负责"翻译"成真正的函数调用。
    如果工具执行出错，返回的 JSON 里带 error 字段——LLM 看到错误会调整策略。
    """
    try:
        if tool_name == "search_car_knowledge":
            return _execute_search_car_knowledge(
                arguments.get("query", ""), knowledge_index
            )
        elif tool_name == "get_car_price":
            return _execute_get_car_price(
                arguments.get("brand", ""), arguments.get("model", "")
            )
        elif tool_name == "compare_cars":
            return _execute_compare_cars(
                arguments.get("car1", ""), arguments.get("car2", "")
            )
        elif tool_name == "filter_by_budget":
            return _execute_filter_by_budget(
                max_price=arguments.get("max_price", 30),
                min_price=arguments.get("min_price", 0),
                category=arguments.get("category", ""),
            )
        else:
            return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)
    except Exception as e:
        # 错误也作为 tool response 返回给 LLM
        # LLM 看到错误后会尝试修正参数或告知用户
        return json.dumps({"error": f"工具执行失败: {str(e)}"}, ensure_ascii=False)


def chat_with_tools(
    user_message: str,
    tools: List[dict],
    tool_choice: str = "auto",
    knowledge_index=None,
    max_turns: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    一次完整的 tool calling 对话。

    参数：
      user_message  — 用户输入
      tools         — 可用工具列表（JSON Schema）
      tool_choice   — "auto" / "required" / "none"
      knowledge_index — (v_idx, bm25_idx, reranker) 元组，工具后端用
      max_turns     — 最大工具调用轮数（防止死循环）
      verbose       — 是否打印中间过程

    返回：
      {"answer": "最终回答", "tool_calls_made": [...], "turns": N}

    tool_choice 三种模式的核心区别：

      auto     — LLM 自主判断：可能调工具，也可能直接回答。
                 while 循环里每次响应可能是 content 或 tool_calls，
                 当 content 出现且无 tool_calls 时循环终止。
                 适用：通用对话。

      required — 强制 LLM 每轮都必须调工具，绝不允许直接输出文本。
                 这意味着 while 循环永远不会自然终止（LLM 永远不能
                 说"我回答完了"）。正确做法：执行完工具后，换一把
                 不带 tools 的调用生成最终回答。
                 适用："先查数据再回答"的严格场景。

      none     — 禁止 LLM 调工具。等于纯 LLM 回答，不会进入循环。
                 适用：不需要外部数据的纯对话。
    """
    # 初始 messages：只有 system + user
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业的汽车导购助手。你可以使用工具查询汽车信息。"
                "当需要查询具体数据时请使用工具，不要编造数据。"
                "当工具返回结果后，请基于结果回答用户问题。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    tool_calls_log = []  # 记录所有工具调用，方便调试
    turn = 0

    while turn < max_turns:
        turn += 1

        if verbose:
            print(f"\n  [第 {turn} 轮 LLM 调用] tool_choice={tool_choice}")

        # ① 调 LLM（带上 tools 参数）
        response = _client.chat.completions.create(
            model=_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if verbose:
            print(f"    finish_reason: {finish_reason}")
            if msg.content:
                print(f"    content: {msg.content[:100]}...")

        # ② 判断：LLM 直接输出了文本 → 对话结束
        #    注意：tool_choice="required" 时永远不会走到这个分支，
        #    因为 required 强制 msg.content 为空、msg.tool_calls 必有值
        if msg.content and not msg.tool_calls:
            if verbose:
                print(f"  [完成] LLM 直接回答了，无需工具")
            return {
                "answer": msg.content,
                "tool_calls_made": tool_calls_log,
                "turns": turn,
            }

        # ③ 判断：LLM 想调工具
        if msg.tool_calls:
            if verbose:
                print(f"  [工具调用] LLM 请求调用 {len(msg.tool_calls)} 个工具")

            # ④ 把 assistant 消息（含 tool_calls）追加到 messages
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

            # ⑤ 逐个执行工具
            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                if verbose:
                    print(f"    → 执行 {func_name}({json.dumps(func_args, ensure_ascii=False)})")

                result = _execute_tool(func_name, func_args, knowledge_index)
                tool_calls_log.append({
                    "name": func_name,
                    "arguments": func_args,
                    "result": result[:200] + "..." if len(result) > 200 else result,
                })

                # ⑥ 把工具结果以 role="tool" 追加回 messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # ═══════════════════════════════════════════════════════
            # ⑦ 关键分支：required vs auto 的不同后续处理
            # ═══════════════════════════════════════════════════════

            if tool_choice == "required":
                # required 模式：LLM 永远不能输出 text，所以不能在 while 里继续循环。
                # 正确做法：工具执行完 → 用 tool_choice="none" 再调一次 LLM，
                # 让它基于工具结果生成最终回答（这次允许输出纯文本）。
                if verbose:
                    print(f"    [required模式] 工具已执行，切换 tool_choice='none' 生成最终回答")

                final_response = _client.chat.completions.create(
                    model=_model,
                    messages=messages,
                    # 不传 tools 参数 → LLM 只能用文本回答
                    temperature=0,
                )
                final_msg = final_response.choices[0].message
                return {
                    "answer": final_msg.content or "(LLM 未返回内容)",
                    "tool_calls_made": tool_calls_log,
                    "turns": turn,
                }
            else:
                # auto 模式：LLM 可能还需要再调工具，也可能直接回答。
                # 回到 while 循环顶部继续。
                continue

        # ⑧ 异常情况：既没有 content 也没有 tool_calls
        if verbose:
            print(f"  [异常] LLM 既没输出文本也没请求工具调用")
        break

    # 超限：达到最大轮数仍没得到最终 answer
    return {
        "answer": "抱歉，工具查询轮次超限，请简化问题重试。",
        "tool_calls_made": tool_calls_log,
        "turns": turn,
    }


# ============================================================
# 4. 知识点 3+4：并行调用 + tool_choice 三种模式对比
# ============================================================

def demo_parallel_tool_calls(knowledge_index):
    """
    场景设计：一个问题需要同时查两个独立的信息源。
    例如："小米SU7多少钱？续航多少？"
    → LLM 可能一次性返回 2 个 tool_calls: get_car_price + search_car_knowledge
    → 两个工具互不依赖，chat_with_tools 里按序执行即可（生产环境可用线程并行）
    """
    print("\n" + "=" * 60)
    print("知识点 3：并行 Tool Calling")
    print("=" * 60)

    query = "小米SU7的价格和续航里程分别是多少"
    print(f"\n[用户] {query}")

    result = chat_with_tools(
        query, TOOLS, tool_choice="auto",
        knowledge_index=knowledge_index, verbose=True,
    )

    print(f"\n[最终回答] {result['answer']}")
    print(f"[工具调用记录] 共 {len(result['tool_calls_made'])} 次:")
    for tc in result["tool_calls_made"]:
        print(f"    {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
        print(f"    → {tc['result'][:100]}...")


def demo_tool_choice_modes(knowledge_index):
    """
    同一问题用三种 tool_choice 模式各跑一遍，对比行为差异。

    auto:     LLM 自己判断要不要调工具 → 灵活，生产默认
    required: 强制 LLM 必须调工具 → 适合"回答前必须查数据"的场景
    none:     禁止 LLM 调工具 → 等于第五章的纯 LLM 回答，不会触发任何 tool
    """
    print("\n" + "=" * 60)
    print("知识点 4：tool_choice 三种模式对比")
    print("=" * 60)

    test_queries = [
        ("需要查询数据的问题", "小米SU7现在卖多少钱"),
        ("纯闲聊问题", "你好，今天心情怎么样"),
    ]

    for desc, query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"[场景] {desc}: \"{query}\"")
        print(f"{'─' * 60}")

        for mode in ["auto", "required", "none"]:
            print(f"\n  --- tool_choice='{mode}' ---")
            result = chat_with_tools(
                query, TOOLS, tool_choice=mode,
                knowledge_index=knowledge_index,
                verbose=False,  # 简略输出
            )
            print(f"  轮次: {result['turns']}")
            print(f"  工具调用: {len(result['tool_calls_made'])} 次")
            if result['tool_calls_made']:
                names = [tc['name'] for tc in result['tool_calls_made']]
                print(f"  调用工具: {names}")
            print(f"  回答: {result['answer'][:120]}...")

    print(f"\n{'─' * 60}")
    print("三种模式总结:")
    print("  auto     → LLM 自主判断，数据问题调工具、寒暄直接答")
    print("            while 循环里 LLM 可能返回 text 或 tool_calls")
    print("  required → 强制先调工具，执行后用 tool_choice='none' 生成最终回答")
    print("            适用场景：'每问必查数据库，不准凭记忆回答'")
    print("  none     → 禁止调工具，等于纯 LLM，不会查任何外部数据")


# ============================================================
# 5. 知识点 5：错误处理 —— 工具执行失败时回传错误
# ============================================================

def demo_error_handling(knowledge_index):
    """
    当工具执行出错时（查不到数据、参数错误），_execute_tool 返回包含 error 的 JSON。
    LLM 看到 error 后应该调整策略，例如告知用户"找不到，换个关键词试试"。

    这里故意构造一个"查不存在的车型"的场景。
    """
    print("\n" + "=" * 60)
    print("知识点 5：错误处理 —— 工具失败时 LLM 如何恢复")
    print("=" * 60)

    # 这辆车不在 CAR_PRICE_DB 里，工具会返回 error
    query = "法拉利SF90的价格是多少"
    print(f"\n[用户] {query}")
    print("  （故意查一个不存在的车型，观察 LLM 如何处理工具错误）")

    result = chat_with_tools(
        query, TOOLS, tool_choice="auto",
        knowledge_index=knowledge_index, verbose=True,
    )

    print(f"\n[最终回答] {result['answer']}")
    print(f"[工具返回] {result['tool_calls_made']}")
    print("\n  要点：LLM 看到 tool response 里的 error 后，应告知用户'未找到'，")
    print("        而非编造一个价格。这正是 Function Calling 防幻觉的核心。")


# ============================================================
# 6. 综合演示：多轮工具调用
# ============================================================

def demo_multi_turn_tool_use(knowledge_index):
    """
    一个更复杂的场景：用户问"预算 20 万以内的 SUV 有哪些，
    选一款最有性价比的推荐"。

    理想过程：
      ① LLM 调 filter_by_budget(max_price=20, category="SUV")
      ② 看到返回列表后，LLM 可能再调 compare_cars 对比前两名
      ③ 综合给出推荐

    这就是 tool calling 的真正威力：LLM 作为"大脑"编排多个工具的调用顺序。
    """
    print("\n" + "=" * 60)
    print("综合演示：多轮工具调用")
    print("=" * 60)

    query = "预算20万以内，推荐一款性价比最高的SUV"
    print(f"\n[用户] {query}")

    result = chat_with_tools(
        query, TOOLS, tool_choice="auto",
        knowledge_index=knowledge_index, verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"[最终回答]")
    print(result['answer'])
    print(f"\n  共 {result['turns']} 轮 LLM 调用，"
          f"执行了 {len(result['tool_calls_made'])} 次工具调用")


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    DATA_DIR = os.path.join(PROJECT_DIR, "data")

    # ---- 初始化知识库索引（给 search_car_knowledge 工具用） ----
    print("=" * 60)
    print("Day 3: 原生 SDK Function Calling")
    print("=" * 60)

    print("\n[加载] 知识库索引...")
    documents = load_data(DATA_DIR)
    documents = embed_documents(documents)
    v_idx = VectorIndex(documents)
    bm25_idx = BM25(documents)
    reranker = Reranker()
    knowledge_index = (v_idx, bm25_idx, reranker)
    print(f"[OK] {len(documents)} 条文档，FAISS={v_idx.index.ntotal} 向量")

    # ---- 演示：打印 tool 定义让用户直观感受 JSON Schema ----
    print("\n[工具列表] LLM 可用的 4 个工具:")
    for t in TOOLS:
        func = t["function"]
        params = func["parameters"]["properties"]
        required = func["parameters"].get("required", [])
        print(f"  • {func['name']}")
        print(f"    描述: {func['description'][:60]}...")
        print(f"    参数: {', '.join(f'{p}{'*' if p in required else ''}' for p in params)}")

    # ---- 知识点 2：基础 Tool Calling 循环 ----
    print("\n" + "=" * 60)
    print("知识点 2：Tool Calling 循环（基础）")
    print("=" * 60)
    result = chat_with_tools(
        "比亚迪海豚现在卖多少钱", TOOLS, tool_choice="auto",
        knowledge_index=knowledge_index, verbose=True,
    )
    print(f"\n[最终回答] {result['answer']}")

    # ---- 知识点 3：并行调用 ----
    demo_parallel_tool_calls(knowledge_index)

    # ---- 知识点 4：tool_choice 三种模式 ----
    demo_tool_choice_modes(knowledge_index)

    # ---- 知识点 5：错误处理 ----
    demo_error_handling(knowledge_index)

    # ---- 综合演示 ----
    demo_multi_turn_tool_use(knowledge_index)
    
