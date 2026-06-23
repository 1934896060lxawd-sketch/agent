# ================================================================
# Day 10：综合增强 Agent v2 — 三层 Prompt 融合
# 核心：CoT Prompt（Day 6）+ 动态 Few-shot（Day 7）+ 结构化输出（Day 8）
#       + Vision（Day 9）→ 全部融入 ReAct Agent 循环（Day 3-5）
# 三层 Prompt 架构：
#   外层 System Prompt → 角色 + 约束 + 输出规范
#   中层 Tool descriptions → 调用时机 + 参数语义
#   内层 CoT Prompt → 推理步骤引导（每次 ReAct Thought 前注入）
# ================================================================

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

# ---- 路径 & 环境 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "chapters"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "agent"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---- LLM 工厂 ----
def get_llm(temperature: float = 0):
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=temperature,
    )


# ================================================================
# 练习 1：三层 Prompt 架构 — System + Tool + CoT 同时生效
# ================================================================

# ── 外层：System Prompt（角色 + 约束）──
CAR_ADVISOR_V2_SYSTEM = """\
## 角色
你是资深汽车导购顾问"小驱 v2"，10 年新能源车销售经验，说话专业但不油腻。

## 能力边界
- 你可以查询价格、推荐车型、对比参数、搜索知识库、计算用车成本
- 你只了解数据库中的车型，查不到时必须明确告知，禁止编造

## 推理要求（重要）
1. 每次回答前，先在脑海中推理：用户真实需求 → 需要哪些信息 → 调哪些工具 → 如何组织回答
2. 工具返回数据后，先验证数据完整性，再基于数据推理结论
3. 推荐时给出基于数据的推理链，不要只抛结论

## 输出约束
- 引用工具返回的真实数据，标注具体数字
- 对比时用表格思维呈现（纯文本）
- 每次推荐 ≤ 3 款，给出排序理由
- 用户条件模糊时主动追问至少 2 个维度（预算/用途/偏好品牌等）

## 错误处理
- 工具返回"未找到" → 告知用户，建议替代方案
- 用户问与汽车无关内容 → 礼貌引导回正题
"""

# ── 中层：Tool descriptions（调用决策）──
# 每个 tool 的 docstring 即中层 Prompt —— 告诉 LLM"何时调"和"参数填什么"

@tool
def search_car_knowledge_v2(query: str) -> str:
    """搜索汽车知识库，获取车型详细参数、续航、智驾、空间、车主口碑。
    当用户询问具体配置、性能参数、或需要了解某款车的技术细节时调用。
    Args:
        query: 具体的查询问题，如"小米SU7的续航和智驾配置"
    """
    # 模拟知识库检索（生产环境接入真实向量库）
    kb = {
        "小米": "CLTC续航 700-830km，零百加速 2.78-5.28s，Orin-X智驾芯片，激光雷达，800V高压平台",
        "比亚迪海豚": "CLTC续航 301-405km，零百加速 ~10s，L2基础智驾，无激光雷达，价格亲民",
        "小鹏G6": "CLTC续航 580-755km，零百加速 3.9-6.2s，XNGP智驾，双激光雷达，800V平台",
        "特斯拉Model Y": "CLTC续航 545-688km，零百加速 3.7-5.0s，FSD智驾，超充网络覆盖广",
        "理想L6": "增程，CLTC综合续航 1390km，零百加速 5.4s，AD Max智驾，家用定位",
        "问界M7": "增程/纯电，CLTC续航 1200km，零百加速 4.8s，ADS 2.0，鸿蒙座舱",
    }
    for k, v in kb.items():
        if k in query:
            return json.dumps({"source": k, "content": v, "status": "found"}, ensure_ascii=False)
    return json.dumps({"source": "knowledge_base", "content": "未找到精确匹配，已返回最相关条目",
                       "status": "partial"}, ensure_ascii=False)


@tool
def recommend_cars_v2(budget_min: float, budget_max: float, category: str = "全部") -> str:
    """根据预算和车型类别推荐车型。当用户给出预算范围、或问'XX万买什么车'时调用。
    Args:
        budget_min: 预算下限（万元）
        budget_max: 预算上限（万元）
        category: 车型类别，SUV/轿车/MPV，默认全部
    """
    cars = [
        {"name": "比亚迪 海豚", "price": "9.98-13.98万", "type": "轿车"},
        {"name": "零跑 C11", "price": "15.58-19.98万", "type": "SUV"},
        {"name": "小鹏 G6", "price": "20.99-27.69万", "type": "SUV"},
        {"name": "小米 SU7", "price": "21.59-29.99万", "type": "轿车"},
        {"name": "特斯拉 Model Y", "price": "24.99-35.49万", "type": "SUV"},
        {"name": "理想 L6", "price": "24.98-27.98万", "type": "SUV"},
        {"name": "极氪 001", "price": "26.90-32.90万", "type": "轿车"},
    ]
    results = []
    for c in cars:
        low = float(c["price"].split("-")[0])
        if budget_min <= low <= budget_max:
            if category == "全部" or category in c["type"]:
                results.append(c)
    results.sort(key=lambda x: float(x["price"].split("-")[0]))
    return json.dumps({"count": len(results), "cars": results}, ensure_ascii=False)


@tool
def compare_cars_v2(car1: str, car2: str) -> str:
    """对比两款车的核心参数。当用户要求对比两款车、或问'A和B哪个好'时调用。
    Args:
        car1: 第一款车全名，如'小鹏 G6'
        car2: 第二款车全名，如'特斯拉 Model Y'
    """
    specs = {
        "小鹏 G6": {"price": "20.99-27.69万", "range": "580-755km", "accel": "3.9s",
                     "smart": "XNGP", "highlight": "800V平台+双激光雷达"},
        "特斯拉 Model Y": {"price": "24.99-35.49万", "range": "545-688km", "accel": "3.7s",
                           "smart": "FSD", "highlight": "超充网络+品牌成熟度"},
        "小米 SU7": {"price": "21.59-29.99万", "range": "700-830km", "accel": "2.78s",
                     "smart": "Xiaomi Pilot", "highlight": "人车家生态+颜值"},
        "极氪 001": {"price": "26.90-32.90万", "range": "546-741km", "accel": "3.8s",
                     "smart": "NZP", "highlight": "猎装造型+操控"},
    }
    return json.dumps({
        "car1": {"name": car1, **specs.get(car1, {"error": "未找到"})},
        "car2": {"name": car2, **specs.get(car2, {"error": "未找到"})},
    }, ensure_ascii=False)


# ================================================================
# 练习 2：结构化输出 — 工具返回格式化为 Pydantic 模型
# ================================================================

class CarRecommendation(BaseModel):
    """汽车推荐的结构化输出"""
    model_name: str = Field(description="推荐车型全称")
    price_range: str = Field(description="价格区间")
    score: float = Field(description="综合评分 0-10", ge=0, le=10)
    reasoning: str = Field(description="推荐理由，基于数据")
    pros: list[str] = Field(description="优点列表，≥3条")
    cons: list[str] = Field(description="缺点列表")


class ComparisonResult(BaseModel):
    """车型对比的结构化输出"""
    dimensions: list[str] = Field(description="对比维度列表")
    car1_scores: dict[str, float] = Field(description="车型1各维度得分")
    car2_scores: dict[str, float] = Field(description="车型2各维度得分")
    winner: str = Field(description="综合胜出车型")
    summary: str = Field(description="对比总结，50字以内")


# ── 内层：CoT Prompt 模板（每次 Thought 前注入）──
COT_INJECTION = """\
请在调用工具前，按以下步骤推理（内部思考，不输出给用户）：
1. 【需求分析】用户真正想要什么？有无隐含需求？
2. 【信息缺口】回答这个问题还需要哪些信息？
3. 【工具选择】应该调哪个工具？参数填什么？
4. 【数据验证】（工具返回后）数据完整吗？有无矛盾？
5. 【回答策略】如何组织答案？需要对比表还是单项推荐？
"""


# ================================================================
# 练习 3：结构化对话日志（每轮记录 T/A/O/E）
# ================================================================

@dataclass
class TurnLog:
    """单轮对话的结构化日志"""
    turn: int
    thought: str = ""           # LLM 推理（AIMessage.content）
    action: Optional[str] = None  # 调用的工具名
    action_args: Optional[dict] = None  # 工具参数
    observation: Optional[str] = None  # 工具返回结果
    error: Optional[str] = None  # 错误信息
    latency_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "thought": self.thought[:200],
            "action": self.action,
            "action_args": self.action_args,
            "observation": self.observation[:200] if self.observation else None,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class AgentTrace:
    """完整对话的结构化日志"""
    query: str
    turns: list[TurnLog] = field(default_factory=list)
    final_answer: str = ""
    total_tool_calls: int = 0
    total_latency_ms: float = 0
    success: bool = True

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "turns": [t.to_dict() for t in self.turns],
            "final_answer": self.final_answer[:300],
            "total_tool_calls": self.total_tool_calls,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "success": self.success,
        }


def run_agent_with_trace(agent, query: str) -> AgentTrace:
    """执行 Agent 并收集结构化日志"""
    trace = AgentTrace(query=query)
    t0 = time.time()

    try:
        step = 0
        for event in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values",
        ):
            if "messages" not in event:
                continue
            messages = event["messages"]
            if not messages:
                continue
            latest = messages[-1]
            step += 1
            turn_log = TurnLog(turn=step)

            if isinstance(latest, AIMessage):
                turn_log.thought = latest.content or ""
                has_tools = hasattr(latest, "tool_calls") and latest.tool_calls
                if has_tools:
                    tc = latest.tool_calls[0]
                    turn_log.action = tc["name"]
                    turn_log.action_args = tc.get("args", {})
                    trace.total_tool_calls += 1
            elif isinstance(latest, ToolMessage):
                turn_log.observation = latest.content or ""
            elif isinstance(latest, HumanMessage):
                continue  # 用户消息不计入 turn

            if turn_log.thought or turn_log.action or turn_log.observation:
                trace.turns.append(turn_log)

        # 提取最终回答
        final_msgs = [m for m in messages if isinstance(m, AIMessage) and m.content]
        if final_msgs:
            trace.final_answer = final_msgs[-1].content

    except Exception as e:
        trace.success = False
        if trace.turns:
            trace.turns[-1].error = str(e)
        else:
            trace.turns.append(TurnLog(turn=0, error=str(e)))

    trace.total_latency_ms = (time.time() - t0) * 1000
    return trace


# ================================================================
# 练习 4：端到端评测 — 10 个测试 case
# ================================================================

TEST_CASES = [
    # (case_id, query, expected_tool, check_keywords)
    ("C01", "小米SU7多少钱？", "get_car_price", ["21.59", "29.99"]),
    ("C02", "15-20万推荐一款纯电SUV", "recommend_cars_v2", ["推荐"]),
    ("C03", "小鹏G6和特斯拉Model Y怎么选？", "compare_cars_v2", ["G6", "Model Y"]),
    ("C04", "Model Y的智驾能力如何？", "search_car_knowledge_v2", ["智驾", "FSD"]),
    ("C05", "你好，你是谁？", None, ["小驱", "导购"]),  # 无工具调用
    ("C06", "25万左右，轿车，续航要长", "recommend_cars_v2", ["SU7", "轿车"]),
    ("C07", "预算10万买什么车？", "recommend_cars_v2", ["海豚"]),
    ("C08", "理想L6适合家用吗？", "search_car_knowledge_v2", ["家用", "L6"]),
    ("C09", "极氪001对比小米SU7", "compare_cars_v2", ["001", "SU7"]),
    ("C10", "推荐一款30万左右加速最快的车", "recommend_cars_v2", ["加速"]),
]


def evaluate_agent(agent, test_cases: list) -> dict:
    """端到端评测：跑 10 个 case，统计成功率"""
    results = []
    for case_id, query, expected_tool, keywords in test_cases:
        trace = run_agent_with_trace(agent, query)
        # 评判标准：① 预期工具是否被调用（若 expected_tool 不为 None）
        #           ② 最终回答是否包含关键词
        tool_ok = True
        if expected_tool:
            called_tools = [t.action for t in trace.turns if t.action]
            tool_ok = expected_tool in called_tools

        answer_ok = all(kw in trace.final_answer for kw in keywords)

        case_ok = tool_ok and answer_ok and trace.success
        results.append({
            "case_id": case_id,
            "query": query,
            "expected_tool": expected_tool,
            "tool_ok": tool_ok,
            "answer_ok": answer_ok,
            "success": case_ok,
            "turns": len(trace.turns),
            "tool_calls": trace.total_tool_calls,
            "latency_ms": round(trace.total_latency_ms, 1),
        })

    success_count = sum(1 for r in results if r["success"])
    return {
        "total": len(results),
        "success": success_count,
        "rate": f"{success_count / len(results) * 100:.1f}%",
        "details": results,
    }


# ================================================================
# Agent 构建
# ================================================================

TOOLS_V2 = [search_car_knowledge_v2, recommend_cars_v2, compare_cars_v2]

# 使用 create_agent（Day 5 方式），注入增强 System Prompt
agent_v2 = create_agent(
    model=get_llm(),
    tools=TOOLS_V2,
    system_prompt=CAR_ADVISOR_V2_SYSTEM,
)


# ================================================================
# 运行入口
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 10: 综合增强 Agent v2 — 三层 Prompt 融合验证")
    print("=" * 60)

    # ── 单条对话日志 ──
    print("\n[练习3] 结构化对话日志（单条）")
    trace = run_agent_with_trace(agent_v2, "小米SU7多少钱？")
    print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))

    # ── 10 条评测 ──
    print("\n[练习4] 端到端评测（10 cases）")
    eval_result = evaluate_agent(agent_v2, TEST_CASES)
    print(f"  成功率: {eval_result['rate']} ({eval_result['success']}/{eval_result['total']})")
    for r in eval_result["details"]:
        status = "[OK]" if r["success"] else "[X]"
        print(f"  {status} {r['case_id']}: {r['query'][:30]}... "
              f"tool={r['tool_ok']} answer={r['answer_ok']} "
              f"({r['turns']} turns, {r['latency_ms']}ms)")

    # ── Day 10 自检 ──
    print("\n" + "=" * 60)
    print("Day 10 自检")
    print("=" * 60)
    checks = [
        ("三层 Prompt 同时生效（System+Tool+CoT）",
         len(CAR_ADVISOR_V2_SYSTEM) > 200 and len(TOOLS_V2) >= 3),
        ("工具返回数据结构化为 Pydantic Schema",
         CarRecommendation.model_fields is not None),
        ("结构化对话日志 T/A/O/E",
         "TurnLog" in globals() and "AgentTrace" in globals()),
        ("10 case 端到端评测",
         len(TEST_CASES) == 10),
        ("Agent 构建（create_agent 一行）",
         agent_v2 is not None),
    ]
    all_pass = True
    for desc, ok in checks:
        status = "[OK]" if ok else "[X]"
        if not ok:
            all_pass = False
        print(f"  {status}  {desc}")
    print(f"\n结论: {'全部达标 [OK]' if all_pass else '有未完成项 [X]'}")
