"""
基于 LangGraph 的 AI Agent 示例
===============================
核心功能：一个能"做数学题"和"联网搜索"的智能助手

整体架构（一张有向图）：
    [用户输入] → [agent 节点] → 有工具调用? → [tools 节点] → 回到 [agent 节点]
                                    ↓ 没有
                                  [结束，输出回答]

关键概念：
  - State（状态）：在整个流程中流转的"对话记录"，每经过一个节点都可能被更新
  - Node（节点）：图里的一个处理步骤，比如"调用大模型"或"执行工具"
  - Edge（边）：节点之间的连线，决定下一步去哪
  - Conditional Edge（条件边）：根据当前状态动态决定下一步
"""

import operator
import os
import ast
from dotenv import load_dotenv
from tavily import TavilyClient
import math
from typing import Annotated, Literal, TypedDict

# ===== LangGraph 核心导入 =====
# StateGraph：用 Python 代码"画"出一张状态流转图
# END：标记流程结束的特殊节点
# ToolNode：LangGraph 内置的工具执行器，自动解析大模型返回的工具调用并执行
# add_messages：消息列表的"合并器"——新消息追加到旧消息末尾，而非覆盖
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# ChatOpenAI：LangChain 对 OpenAI 兼容 API 的统一封装（DeepSeek 等国产模型也走这套接口）
from langchain_openai import ChatOpenAI

# @tool 装饰器：把普通 Python 函数一键注册为"大模型可调用的工具"
from langchain.tools import tool


# ============================================================
# 第一部分：工具定义 —— Agent 的"手脚"
# ============================================================
# 大模型本身只能输出文字，不能真正"做"事。工具就是给它装上手脚：
#   - calculator：执行数学计算（因为大模型直接算数容易出错）
#   - search：联网查资料（大模型知识有截止日期，实时信息必须搜）


# ---------- AST 安全计算器 ----------
# 为什么不用 eval()？eval() 会执行任意 Python 代码，用户输入 rm -rf 也能跑，极度危险
# 这里用 Python 的 ast 模块把字符串"解析成语法树"，只放行白名单里的运算，其余一律拒绝

# 白名单：允许的运算符 → 对应的 Python 运算函数
ALLOWED_OPS = {
    ast.Add: operator.add,      # 加法 +
    ast.Sub: operator.sub,      # 减法 -
    ast.Mult: operator.mul,     # 乘法 *
    ast.Div: operator.truediv,  # 除法 /
    ast.Pow: operator.pow,      # 幂运算 **
    ast.USub: operator.neg,     # 一元负号（如 -5）
}

# 白名单：允许调用的数学函数
ALLOW_FUNC = {
    "sqrt": math.sqrt,   # 开根号
    "pow": pow           # pow(底数, 指数) 通用幂函数
}


def safe_calc(expr: str) -> str:
    """安全地执行数学表达式，只允许白名单内的运算符和函数"""
    expr = expr.strip()
    try:
        # 第一步：用 ast 解析字符串为抽象语法树，mode='eval' 表示这是一个表达式
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return "表达式语法错误"

    # 第二步：递归遍历语法树，逐个节点求值
    # 这就是"解释器模式"的雏形——不用编译执行，而是自己走一遍树
    def _eval(node):
        # 常量数字（如 3、3.14）
        if isinstance(node, ast.Constant):
            return node.value

        # 二元运算（左操作数 op 右操作数，如 1 + 2）
        elif isinstance(node, ast.BinOp):
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持运算符：{type(node.op).__name__}")
            return op_func(_eval(node.left), _eval(node.right))

        # 一元负号（如 -5）
        elif isinstance(node, ast.UnaryOp):
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持运算符：{type(node.op).__name__}")
            return op_func(_eval(node.operand))

        # 函数调用（如 sqrt(9)、pow(2,3)）
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("不支持的函数调用")
            func_name = node.func.id
            if func_name not in ALLOW_FUNC:
                raise ValueError(f"禁止函数:{func_name}")
            # 先对每个参数求值，再调用函数
            args = [_eval(arg) for arg in node.args]
            return ALLOW_FUNC[func_name](*args)

        else:
            raise ValueError(f"不支持语法：{type(node).__name__}")

    try:
        res = _eval(tree.body)
        return str(res)
    except (ValueError, ZeroDivisionError, TypeError) as e:
        return f"计算错误：{str(e)}"


# ---------- 联网搜索 ----------
def external_search(query: str) -> str:
    """调用 Tavily 搜索引擎，返回前 3 条结果的标题+摘要"""
    try:
        # search_depth="basic"：快速搜索，适合实时问题；"advanced"更深入但耗时更长
        resp = tavily.search(query=query, search_depth="basic", max_results=3)
        results = resp.get("results", [])
        if not results:
            return f"未搜索到关于'{query}'的相关内容"

        # 把每条结果的标题和内容（截取前300字）拼成一段返回给大模型
        parts = []
        for i, item in enumerate(results):
            title = item.get("title", "无标题")
            snippet = item.get("content", "")[:300]
            parts.append(f"[{i+1}] {title}\n{snippet}")
        return "\n\n".join(parts)

    except Exception as e:
        return f"搜索失败：{e}"


# ============================================================
# 第二部分：工具注册 —— 告诉 LangGraph 有哪些"手脚"可用
# ============================================================

# @tool 装饰器的作用：
#   1. 自动提取函数名、参数名、docstring 作为"工具描述"发给大模型
#   2. 大模型看到这些描述后，就能决定"什么时候该用哪个工具"
#   3. 参数名需要和函数签名一致，因为大模型会按参数名传参

@tool
def calculator(expression: str) -> str:
    """执行数学表达式计算，支持加减乘除、括号、根号sqrt、平方pow
    expression: 数学表达式，例：(100+123)*234、sqrt(25)
    """
    print(f"【计算器】{expression}")
    return safe_calc(expression)


@tool
def search(expression: str) -> str:
    """联网搜索外部知识、常识、实时数据、科普内容；所有非数学类问题必须优先调用本工具查询
    expression: 需要联网查询的问题、知识点、实时信息
    """
    print(f"【搜索】{expression}")
    return external_search(expression)


# 把所有工具放入列表——这一步相当于"签合同"，告知框架和大模型有哪些能力
tools = [calculator, search]


# ============================================================
# 第三部分：初始化 —— 加载配置、连接各个服务
# ============================================================

load_dotenv()  # 从 .env 文件加载环境变量（API Key 等敏感信息不应写在代码里）

# 从环境变量读取各服务的密钥和地址
api_key = os.getenv("LLM_API_KEY")        # 大模型 API Key（如 DeepSeek 的 key）
base_url = os.getenv("LLM_BASE_URL")      # API 地址（如 https://api.deepseek.com）
tavily_key = os.getenv("TAVILY_API_KEY")  # Tavily 搜索服务的 Key
tavily = TavilyClient(api_key=tavily_key)

# 初始化大模型客户端
# ChatOpenAI 是 LangChain 的通用封装——任何兼容 OpenAI 接口的服务都能用，不限于 OpenAI 本身
llm = ChatOpenAI(
    model="deepseek-chat",   # 模型名称，可按需换成 gpt-4o、qwen 等
    api_key=api_key,
    base_url=base_url,
)

# ⭐ 关键一步：bind_tools 把工具列表"绑定"到大模型上
# 绑定后，每次调用 llm_with_bind 时：
#   1. LangChain 会自动把工具定义（名称+描述+参数）附加到请求里发给大模型
#   2. 大模型如果觉得需要工具，会返回一个"工具调用"而非普通文字
#   3. 大模型如果觉得不需要工具，就正常回复文字
llm_with_bind = llm.bind_tools(tools)


# ============================================================
# 第四部分：定义状态 —— Agent 的"记忆"
# ============================================================

# State 是整张图的"共享内存"，每个节点读写它，修改后的副本传给下一个节点
# Annotated[list, add_messages] 的含义：
#   - 类型是 list（消息列表）
#   - add_messages 是"合并策略"——新的消息不是替换旧的，而是追加到末尾
#   - 这样整个对话历史都能保留，大模型每次都看到完整上下文
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 第五部分：定义节点 —— Agent 的"大脑"和"手脚"
# ============================================================

def call_model(state: State) -> dict:
    """
    核心节点：调用大模型进行"思考"
    ================================
    流程：
      1. 取出当前所有消息（包括用户输入 + 之前的工具结果）
      2. 在开头插入一条 system prompt（系统指令），告诉大模型行为规则
      3. 调用大模型，大模型返回：
         - 可能是一段文字回答（说明不需要工具了）
         - 可能是一个"工具调用请求"（说明需要调用 calculator 或 search）
      4. 把大模型的返回追加到消息列表里，送回图
    """
    msg_list = state["messages"]

    # system prompt：给大模型"定规矩"
    # 这相当于在对话最前面写了一条不可见的指令，规范大模型的行为模式
    sys_prompt = (
        "你是严谨的助手。规则如下：\n"
        "1. 数学计算题 → 调 calculator，开根号用sqrt()、次方用pow(底数,指数)\n"
        "2. 知识/实时信息类问题 → 必须先调 search，拿到结果再作答，禁止编造\n"
        "3. 简单闲聊或已有明确答案的问题可直接回复"
    )

    # 把 system prompt 放在消息列表最前面（角色设为 system）
    full_msg = [{"role": "system", "content": sys_prompt}] + msg_list

    # 调用大模型——因为之前 bind_tools 了，框架会自动处理工具定义的注入
    resp = llm_with_bind.invoke(full_msg)

    # 返回字典，LangGraph 会按 State 的 add_messages 策略把这新消息追加进去
    return {"messages": [resp]}


def should_continue(state: State) -> Literal["tools", "__end__"]:
    """
    条件判断节点：看完大模型的回复，决定下一步走哪条路
    ======================================================
    这是整张图的"岔路口"：
      - 如果大模型说"我要调工具" → 走向 "tools" 节点，执行工具
      - 如果大模型给出了最终回答     → 走向 "__end__"，结束流程
    """
    last_msg = state["messages"][-1]

    # tool_calls 是大模型返回的特殊字段：当它需要工具时，这个字段不为空
    # 里面包含要调用哪个工具、传什么参数
    if last_msg.tool_calls:
        return "tools"
    else:
        return "__end__"


# ============================================================
# 第六部分：组装图 —— 把节点和边"编织"成完整流程
# ============================================================

# 创建一张空白的"状态图"，把 State 作为它的共享数据结构
graph = StateGraph(State)

# 添加两个节点：
#   "agent"  → 大模型思考（大脑）
#   "tools"  → 执行工具调用（手脚）
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))  # ToolNode 自动处理工具调用的解析和执行

# 把 "agent" 设为入口——流程从这里开始
graph.set_entry_point("agent")

# 添加一条"条件边"：从 agent 出发，根据 should_continue 函数的返回值决定去哪
#   - 返回 "tools"  → 去 tools 节点执行工具
#   - 返回 "__end__" → 结束，最终回复返回给用户
graph.add_conditional_edges("agent", should_continue)

# 添加一条"固定边"：tools 节点执行完毕后，无条件回到 agent 节点
# 这样 agent 就能看到工具返回的结果，判断是否还需要再调工具，还是可以最终回答
graph.add_edge("tools", "agent")

# 编译——把上面定义的所有节点和边"固化"成一个可运行的应用
app = graph.compile()


# ============================================================
# 第七部分：运行 —— 启动 Agent
# ============================================================
# 执行流程可视化：
#
#   用户问："2026年国内主流新能源汽车售价？"
#     │
#     ▼
#   [agent] ── 大模型看到问题，发现需要实时数据
#     │         → 返回 tool_calls: [search("2026年国内新能源汽车售价")]
#     │
#     ▼
#   [should_continue] ── 检测到 tool_calls 非空 → 返回 "tools"
#     │
#     ▼
#   [tools] ── ToolNode 解析工具调用，执行 search()，拿到搜索结果
#     │         把搜索结果作为"工具返回消息"追加到消息列表
#     │
#     ▼
#   [agent] ── 大模型看到搜索结果，基于真实数据组织回答
#     │         → 没有 tool_calls 了，返回完整回答
#     │
#     ▼
#   [should_continue] ── 没有 tool_calls → 返回 "__end__"
#     │
#     ▼
#   [END] ── 流程结束，返回答复给用户

if __name__ == "__main__":
    # 初始输入：角色为 "user" 的消息
    # 这和 ChatGPT 的消息格式一致：[("角色", "内容")]
    input_data = {
        "messages": [
            ("user", "2026年国内各种汽油价格？")
        ]
    }

    # app.invoke() 启动整个图的运行
    # 内部会自动循环：agent → tools → agent → tools → ... 直到 should_continue 返回 __end__
    result = app.invoke(input_data)

    # result["messages"] 是完整的对话历史，最后一条就是大模型的最终回答
    print("最终回答：", result["messages"][-1].content)
