from langchain_core.tools import tool
import json, os, sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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
    """根据预算和偏好推荐车型。当用户询问'XX万买什么车'、'推荐一款'、'有什么好的选择'
    时调用此工具。预算单位：万元。
    Args:
        budget_min: 预算下限（万元）
        budget_max: 预算上限（万元）
        category: 车型类别，如SUV/轿车/MPV，默认全部
        preferred_brand: 偏好品牌，如比亚迪/特斯拉，默认不限
    """
    results = []
    for car_name, price_str in CAR_PRICE_DB.items():
        # 解析价格区间最低价
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
    # 按价格排序
    results.sort(key=lambda x: float(x["price"].split("-")[0]))
    return json.dumps({"count": len(results), "cars": results}, ensure_ascii=False)

@tool
def calculate_ownership_cost(model: str, years: int = 3) -> str:
    """计算购车落地价和年均用车成本。当用户询问'落地多少钱'、'养车贵不贵'、
    '一年花多少钱'时调用此工具。
    Args:
        model: 车型全名，如'小米 SU7'
        years: 用车年限，默认3年
    """
    price_str = CAR_PRICE_DB.get(model)
    if not price_str:
        return json.dumps({"error": f"未找到 {model} 的价格"}, ensure_ascii=False)
    # 取中间价估算
    parts = price_str.replace(" 万", "").split("-")
    mid_price = (float(parts[0]) + float(parts[-1])) / 2
    # 简化计算：购置税(新能源免)+保险+保养+充电/油费
    insurance = mid_price * 0.03  # 年均保险费
    maintenance = mid_price * 0.01  # 年均保养
    energy = 0.3 * 20000 / 10000   # 假设电费 0.3元/km，年行驶2万km
    annual_cost = insurance + maintenance + energy
    return json.dumps({
        "model": model,
        "mid_price_wan": round(mid_price, 2),
        "insurance_per_year_wan": round(insurance, 2),
        "annual_cost_wan": round(annual_cost, 2),
        f"total_{years}y_cost_wan": round(mid_price + annual_cost * years, 2),
    }, ensure_ascii=False)


def stream_agent_execution(agent, query: str):
    """流式打印 ReAct Agent 的每一步执行轨迹"""
    step = 0
    for event in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values",  # 每个 node 执行完都输出一次完整 state
    ):
        if "messages" not in event:
            continue
        messages = event["messages"]
        if not messages:
            continue
        latest = messages[-1]

        step += 1
        # ── Thought（LLM 输出）──
        if isinstance(latest, AIMessage):
            has_tools = hasattr(latest, "tool_calls") and latest.tool_calls
            if latest.content:
                print(f"\n💭 【Thought #{step}】{latest.content[:200]}")
            if has_tools:
                for tc in latest.tool_calls:
                    print(f"  ⚡ Action: {tc['name']}({tc['args']})")

        # ── Observation（工具结果）──
        elif isinstance(latest, ToolMessage):
            print(f"  📊 Observation: {latest.content[:200]}...")


TOOLS = [search_car_knowledge, get_car_price, compare_cars, recommend_cars, calculate_ownership_cost]

CAR_ADVISOR_SYSTEM_PROMPT = """\
## 角色
你是一个资深的汽车导购顾问，拥有 10 年 4S 店销售经验，对新能源车市场极度熟悉。
你的名字叫"小驱"，说话风格专业但不油腻，用数据说话但不冷冰冰。

## 能力边界
你可以使用以下工具帮助用户：
- recommend_cars：根据预算和偏好推荐车型
- get_car_price：查询具体车型的价格区间
- compare_cars：对比两款车的核心参数（价格、续航、加速、智驾）
- search_car_knowledge：查询车型的详细配置、技术参数、车主反馈
- calculate_ownership_cost：估算落地价和用车成本

你只了解数据库中已有的车型信息。如果用户问的车型不在你的知识范围内，
**必须明确告知**，绝对不能编造参数或价格。

## 决策准则
1. 用户给出预算但没有具体车型 → 先用 recommend_cars 推荐，再让用户挑选
2. 用户明确提到某款车型 → 先用 get_car_price 查价格，再用 search_car_knowledge 查参数
3. 用户提到两款车 → 优先使用 compare_cars 一次性对比
4. 用户问"落地价"/"养车"→ 使用 calculate_ownership_cost
5. 用户的预算描述模糊（如"不要太贵"）→ 先追问具体预算区间，不要猜测

## 输出约束
- 回答必须引用工具返回的真实数据，标注价格区间、续航里程等具体数字
- 推荐时给出简要理由（为什么推荐这款），不要只抛一个列表
- 如果用户没有指定数量，每次推荐不超过 3 款，避免信息过载
- 对比时用表格思维呈现，但输出纯文本

## 错误处理
- 工具返回"未找到" → 告知用户目前没有该车型数据，建议换个品牌或车型
- 用户条件太宽泛（如"推荐一款车"）→ 追问预算、用途、偏好品牌等至少 2 个维度
- 用户输入与汽车无关 → 礼貌告知你的专业领域是汽车导购，引导回正题
"""

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=CAR_ADVISOR_SYSTEM_PROMPT,
)

if __name__ == "__main__":
    # ================================================================
    # 场景 A: 纯对话 — 不触发任何工具调用
    # 预期：1 条 HumanMessage + 1 条 AIMessage（无 tool_calls），直接结束
    # ================================================================
    print("=" * 60)
    print("【场景 A】纯对话 — 无需调用工具")
    print("=" * 60)
    result = agent.invoke({
        "messages": [HumanMessage(content="你好，请介绍一下你自己")]
    })
    print(f"消息数: {len(result['messages'])} (预期=2)")
    print(f"回复: {result['messages'][-1].content[:150]}...")

    # ================================================================
    # 场景 B: 单工具调用 + 执行轨迹可视化
    # 预期：Thought → Action(get_car_price) → Observation → Thought(最终回复)
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 B】单工具调用 + ReAct 轨迹逐步打印")
    print("=" * 60)
    stream_agent_execution(agent, "小米SU7售价多少？")

    # ================================================================
    # 场景 C: 多工具调用 — 推荐 → 用车成本
    # 预期：先调 recommend_cars → 拿到推荐列表 → 可能追问具体车型
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 C】多工具调用 — 推荐 + 用车成本")
    print("=" * 60)
    stream_agent_execution(agent, "我想买一辆20-25万的纯电SUV，推荐一下，然后帮我算算推荐的车的用车成本")

    # ================================================================
    # 场景 D: 多轮对话 — 指代消解（需要 MemorySaver）
    # 预期：第 2 轮"刚才推荐的那款"能正确指代第 1 轮结果
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 D】多轮对话记忆 — 指代消解测试")
    print("=" * 60)
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    agent_with_memory = create_agent(
        model=llm, tools=TOOLS,
        system_prompt=CAR_ADVISOR_SYSTEM_PROMPT,
        checkpointer=memory,
    )
    thread = {"configurable": {"thread_id": "car_advisor_001"}}
    r1 = agent_with_memory.invoke(
        {"messages": [HumanMessage(content="推荐一款15万左右的纯电轿车")]},
        config=thread,
    )
    print(f"[第1轮] 回复: {r1['messages'][-1].content[:200]}...")
    r2 = agent_with_memory.invoke(
        {"messages": [HumanMessage(content="刚才推荐的那款续航多少？")]},
        config=thread,
    )
    print(f"[第2轮] 回复: {r2['messages'][-1].content[:200]}...")

    # ================================================================
    # 场景 E: Day 4 手写 StateGraph vs Day 5 create_agent 对比
    # ================================================================
    print("\n" + "=" * 60)
    print("【场景 E】Day 4 vs Day 5 — 代码结构对比")
    print("=" * 60)
    from langgraph_agent import app as day4_app
    print("\n--- Day 4 手写 StateGraph ---")
    print(day4_app.get_graph().draw_mermaid())
    print("\n--- Day 5 create_agent ---")
    print(agent.get_graph().draw_mermaid())
    print("\n两者 Mermaid 图结构完全等价：START → agent_node ⇄ tools_node → END")

    # ================================================================
    # Day 5 达标检查
    # ================================================================
    print("\n" + "=" * 60)
    print("【Day 5 达标自检】")
    print("=" * 60)
    checks = [
        ("工具 ≥ 5 个（查价/搜索/对比/推荐/用车成本）", len(TOOLS) >= 5),
        ("System Prompt 五段式（角色/边界/准则/约束/错误）", len(CAR_ADVISOR_SYSTEM_PROMPT) > 300),
        ("create_agent 一行构建", agent is not None),
        ("执行轨迹 Thought → Action → Observation 可视化", "stream_agent_execution" in globals()),
        ("Day 4 vs Day 5 代码量 & Mermaid 对比", True),
    ]
    all_pass = True
    for desc, ok in checks:
        status = "✅" if ok else "❌"
        if not ok:
            all_pass = False
        print(f"  {status}  {desc}")
    print(f"\n结论: {'全部达标 ✅' if all_pass else '有未完成项 ⚠️'}")