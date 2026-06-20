from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import json
import sys
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# Day 5: ReAct Agent — create_agent 替代手写 StateGraph
from langchain.agents import create_agent

# ---- 路径 & 环境 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "chapters"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 导入检索模块（knowledge_index 初始化 & search_car_knowledge 工具使用）
from naive_rag import load_data
from full_rag_agent import Reranker
from retrieval_test import VectorIndex, BM25, hybrid_rrf
from embedding_test import _embed_model, embed_documents

# 全局索引变量，后续初始化赋值
knowledge_index = None

# 模拟数据库
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

# 状态定义
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def search_car_knowledge(query: str) -> str:
    """搜索汽车知识库，获取车型的详细参数、续航、智驾、空间等信息。
    当用户询问车型的具体配置、性能参数、或是需要了解某款车时调用此工具。
    Args:
        query: 用户查询的车型问题
    """
    global knowledge_index
    if knowledge_index is None:
        return json.dumps({"error": "知识库索引未初始化"}, ensure_ascii=False)

    v_idx, bm25_idx, reranker = knowledge_index
    dense = v_idx.search(query, top_k=6)
    sparse = bm25_idx.search(query, top_k=6)
    hybrid = hybrid_rrf(dense, sparse, k=60, top_k=3)

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

@tool
def get_car_price(brand: str, model: str) -> str:
    """查询指定车型的最新市场指导价。
    当用户询问价格、预算、多少钱、贵不贵时调用此工具。"""
    key = f"{brand} {model}"
    if key in CAR_PRICE_DB:
        return json.dumps({
            "car": key,
            "price": CAR_PRICE_DB[key],
            "status": "found",
        }, ensure_ascii=False)
    for k, v in CAR_PRICE_DB.items():
        if brand in k and model in k:
            return json.dumps({"car": k, "price": v, "status": "found"}, ensure_ascii=False)
    return json.dumps({
        "car": key,
        "price": None,
        "status": "not_found",
        "message": f"未找到 {key} 的价格信息，请尝试其他关键词",
    }, ensure_ascii=False)

@tool
def compare_cars(car1: str, car2: str) -> str:
    """对比两款车的核心参数，包括价格、续航、加速、智驾。
    当用户要求对比两款车、或问'A和B哪个好/哪个更值'时调用此工具。"""
    spec1 = CAR_SPEC_DB.get(car1, "暂无详细参数")
    spec2 = CAR_SPEC_DB.get(car2, "暂无详细参数")
    price1 = CAR_PRICE_DB.get(car1, "暂无价格")
    price2 = CAR_PRICE_DB.get(car2, "暂无价格")

    return json.dumps({
        "car1": {"name": car1, "price": price1, "spec": spec1},
        "car2": {"name": car2, "price": price2, "spec": spec2},
    }, ensure_ascii=False)

@tool
def recommend_cars(budget_min: float, budget_max: float, category: str = "全部",
                   preferred_brand: str = "") -> str:
    """根据预算和偏好推荐车型。当用户询问'XX万买什么车'、'推荐一款'、'有什么选择'时调用。
    Args:
        budget_min: 预算下限（万元），如 15
        budget_max: 预算上限（万元），如 25
        category: 车型类别，如SUV/轿车/MPV，默认全部
        preferred_brand: 偏好品牌关键词，如比亚迪/特斯拉，默认不限
    """
    results = []
    for car_name, price_str in CAR_PRICE_DB.items():
        low_str = price_str.replace(" 万", "").split("-")[0]
        try:
            low_price = float(low_str)
        except ValueError:
            continue
        if budget_min <= low_price <= budget_max:
            if category != "全部" and category not in car_name:
                continue
            if preferred_brand and preferred_brand not in car_name:
                continue
            spec = CAR_SPEC_DB.get(car_name, "暂无参数")
            results.append({"name": car_name, "price": price_str, "spec": spec})
    results.sort(key=lambda x: float(x["price"].split("-")[0]))
    return json.dumps({"count": len(results), "cars": results}, ensure_ascii=False)


@tool
def calculate_ownership_cost(model: str, years: int = 3) -> str:
    """计算购车落地价和年均用车成本。当用户询问'落地多少钱'、'养车费用'、
    '一年花多少钱'、'用车成本'时调用此工具。
    Args:
        model: 车型全名，如'小米 SU7'
        years: 用车年限，默认3年
    """
    price_str = CAR_PRICE_DB.get(model)
    if not price_str:
        # 模糊匹配
        for k, v in CAR_PRICE_DB.items():
            if model in k:
                price_str = v
                model = k
                break
    if not price_str:
        return json.dumps({"error": f"未找到 {model} 的价格信息"}, ensure_ascii=False)
    parts = price_str.replace(" 万", "").split("-")
    mid_price = (float(parts[0]) + float(parts[-1])) / 2  # 取中配价估算
    insurance = mid_price * 0.03          # 年均保险（万）
    maintenance = mid_price * 0.01         # 年均保养（万）
    energy = 0.3 * 20000 / 10000          # 电费 0.3元/km，年2万km，折合0.6万
    annual_cost = insurance + maintenance + energy
    total = mid_price + annual_cost * years
    return json.dumps({
        "model": model,
        "mid_price_wan": round(mid_price, 2),
        "insurance_per_year_wan": round(insurance, 2),
        "energy_maintenance_wan": round(maintenance + energy, 2),
        "annual_cost_wan": round(annual_cost, 2),
        f"total_{years}y_cost_wan": round(total, 2),
    }, ensure_ascii=False)


# Day 4: 基础工具集（3 个）
TOOLS = [search_car_knowledge, get_car_price, compare_cars]
# Day 5: 扩展工具集（5 个） — 覆盖询价/推荐/对比/参数/用车成本
TOOLS_V2 = [search_car_knowledge, get_car_price, compare_cars,
            recommend_cars, calculate_ownership_cost]

# LLM初始化
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
llm_with_tools = llm.bind_tools(TOOLS)

# ========== 1. LLM节点函数 call_model ==========
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ========== 2. 自定义条件判断函数 should_continue ==========
def should_continue(state: AgentState) -> str:
    """判断 LLM 输出是否包含工具调用，决定下一步路由"""
    last_msg = state["messages"][-1]
    # 判断是否存在工具调用 → 去 tools 节点；否则结束
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "__end__"


# ========== 构建流程图 ==========
# 实例化工具执行节点
tool_node = ToolNode(tools=TOOLS)
# 创建图构建器
graph_builder = StateGraph(AgentState)

# 注册2个节点：llm模型节点、工具节点
graph_builder.add_node("llm_node", call_model)
graph_builder.add_node("tool_node", tool_node)

# 起始边：START → llm_node
graph_builder.add_edge(START, "llm_node")

# 条件分支：llm_node 执行完 → 有 tool_calls 走 tools，否则 END
graph_builder.add_conditional_edges(
    source="llm_node",
    path=should_continue,
    path_map={"tools": "tool_node", "__end__": END}
)

# 普通边：工具执行完毕 → 回到模型节点（形成循环）
graph_builder.add_edge("tool_node", "llm_node")

# ---- 编译两个版本 ----
# ① 基础版：无 checkpointer，每次 invoke 都是全新会话
app = graph_builder.compile()

# ② MemorySaver 版：持久化每一步 state，同一 thread_id 可多轮对话
memory = MemorySaver()
app_with_memory = graph_builder.compile(checkpointer=memory)

# ========== 测试覆盖 4 种场景 ==========
if __name__ == "__main__":
    # ================================================================
    # 场景 1：纯对话（无工具调用） → 直接 END，messages 仅 2 条
    # ================================================================
    print("=" * 60)
    print("【场景 1】纯对话 — 不需要调用工具")
    print("=" * 60)
    result = app.invoke({
        "messages": [HumanMessage(content="你好，请介绍一下你自己")]
    })
    print(f"消息数: {len(result['messages'])} (预期=2: HumanMessage + AIMessage)")
    print(f"回复: {result['messages'][-1].content[:120]}...")

    # ================================================================
    # 场景 2：单工具调用 → agent → tools → agent 路径
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 2】单工具调用 — 查价格")
    print("=" * 60)
    result = app.invoke({
        "messages": [HumanMessage(content="小米SU7售价多少？")]
    })
    print(f"消息数: {len(result['messages'])} (预期=4: Human + AIMessage(tool_call) + ToolMessage + AIMessage(final))")
    for idx, msg in enumerate(result["messages"]):
        tag = ""
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            tag = f"  ← 工具调用: {[tc['name'] for tc in msg.tool_calls]}"
        print(f"  [{idx}] {type(msg).__name__}{tag}")
    print(f"最终回复: {result['messages'][-1].content[:200]}")

    # ================================================================
    # 场景 3：多工具调用（可能触发并行 or 对比工具）
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 3】多工具调用 — 对比两款车")
    print("=" * 60)
    result = app.invoke({
        "messages": [HumanMessage(content="对比一下小米SU7和特斯拉Model 3，哪个更值得买？")]
    })
    tool_call_count = sum(
        1 for msg in result["messages"]
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls
    )
    print(f"工具调用轮次: {tool_call_count}")
    for idx, msg in enumerate(result["messages"]):
        tag = ""
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            tag = f"  ← 工具调用: {[tc['name'] for tc in msg.tool_calls]}"
        print(f"  [{idx}] {type(msg).__name__}{tag}")
    print(f"最终回复: {result['messages'][-1].content[:300]}")

    # ================================================================
    # 场景 4：MemorySaver 多轮对话 — 同一 thread_id 记住上文
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 4】MemorySaver 多轮对话 — 上下文记忆")
    print("=" * 60)
    thread_config = {"configurable": {"thread_id": "demo_user_001"}}

    # 第 1 轮
    result1 = app_with_memory.invoke(
        {"messages": [HumanMessage(content="推荐一款20万左右的纯电SUV")]},
        config=thread_config,
    )
    print(f"第1轮消息数: {len(result1['messages'])}")
    print(f"第1轮回复: {result1['messages'][-1].content[:200]}")

    # 第 2 轮 —— 用指代"刚才那款"，验证是否能记住上下文
    result2 = app_with_memory.invoke(
        {"messages": [HumanMessage(content="刚才推荐的那款续航多少？")]},
        config=thread_config,
    )
    print(f"\n第2轮消息数: {len(result2['messages'])} (包含第1轮历史)")
    print(f"第2轮回复: {result2['messages'][-1].content[:300]}")

    # ================================================================
    # 场景 5：Mermaid 流程图可视化
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 5】Mermaid 流程图")
    print("=" * 60)
    print(app.get_graph().draw_mermaid())
    print("\n↑ 复制到 https://mermaid.live 可查看可视化流程图")
    print("结构: START → llm_node ⇄ tool_node → END")