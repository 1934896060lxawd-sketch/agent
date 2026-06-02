import operator
import openai
import json
import os
import ast
from dotenv import load_dotenv
from tavily import TavilyClient
import math

load_dotenv()

# 从环境变量读取key，url
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
tavily_key = os.getenv("TAVILY_API_KEY")

# 初始化客户端
client = openai.OpenAI(api_key=api_key, base_url=base_url)
tavily = TavilyClient(api_key=tavily_key)


def build_function_tool(name: str, desc: str, params, required):
    """快速构建 OpenAI 函数调用工具结构"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required
            }
        }
    }


# ========== 定义计算器工具 ==========
calculator_params = {
    "expression": {"type": "string", "description": "数学表达式，例：(100+123)*234、sqrt(25)"}
}
calculator = build_function_tool(
    name="calculator",
    desc="执行数学表达式计算，支持加减乘除、括号、根号sqrt、平方pow",
    params=calculator_params,
    required=["expression"]
)

# ========== 定义搜索工具 ==========
search_params = {
    "expression": {"type": "string", "description": "需要联网查询的问题、知识点、实时信息"}
}
search = build_function_tool(
    name="search",
    desc="联网搜索外部知识、常识、实时数据、科普内容；所有非数学类问题必须优先调用本工具查询",
    params=search_params,
    required=["expression"]
)

# 工具列表
tools = [calculator, search]

# ===== System prompt =====
messages = [
    {
        "role": "system",
        "content": (
            "你是严谨的助手。规则如下：\n"
            "1. 数学计算题 → 调 calculator，开根号用sqrt()、次方用pow(底数,指数)\n"
            "2. 知识/实时信息类问题 → 必须先调 search，拿到结果再作答，禁止编造\n"
            "3. 简单闲聊或已有明确答案的问题可直接回复"
        ),
    },
    {"role": "user", "content": "2026年国内主流新能源汽车售价？"},
]


# ===== AST 安全计算器【新增支持sqrt、pow函数调用】 =====
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
# 允许的内置数学函数白名单
ALLOW_FUNC = {
    "sqrt": math.sqrt,
    "pow": pow
}


def safe_calc(expr: str) -> str:
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return "表达式语法错误"

    def _eval(node):
        # 常量数字
        if isinstance(node, ast.Constant):
            return node.value
        # 二元运算 +-*/**
        elif isinstance(node, ast.BinOp):
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持运算符：{type(node.op).__name__}")
            return op_func(_eval(node.left), _eval(node.right))
        # 一元负号 -5
        elif isinstance(node, ast.UnaryOp):
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持运算符：{type(node.op).__name__}")
            return op_func(_eval(node.operand))
        # 函数调用 sqrt(9) / pow(2,3)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("不支持的函数调用")
            func_name = node.func.id
            if func_name not in ALLOW_FUNC:
                raise ValueError(f"禁止函数:{func_name}")
            args = [_eval(arg) for arg in node.args]
            return ALLOW_FUNC[func_name](*args)
        else:
            raise ValueError(f"不支持语法：{type(node).__name__}")

    try:
        res = _eval(tree.body)
        return str(res)
    except (ValueError, ZeroDivisionError, TypeError) as e:
        return f"计算错误：{str(e)}"


def external_search(query: str) -> str:
    try:
        resp = tavily.search(query=query, search_depth="basic", max_results=3)
        results = resp.get("results", [])
        if not results:
            return f"未搜索到关于‘{query}’的相关内容"
        parts = []
        for i, item in enumerate(results):
            title = item.get("title", "无标题")
            snippet = item.get("content", "")[:300]
            parts.append(f"[{i+1}] {title}\n{snippet}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"搜索失败：{e}"


# ===== Agent 循环 =====
for turn in range(5):
    response = client.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=tools
    )
    msg = response.choices[0].message

    # 无工具调用，输出最终答案
    if not msg.tool_calls:
        print("最终回答：", msg.content)
        break

    # Pydantic对象转标准字典存入上下文
    messages.append(msg.model_dump())

    # 遍历每个工具调用
    for tc in msg.tool_calls:
        tool_name = tc.function.name
        args = json.loads(tc.function.arguments)
        expr_param = args["expression"]  # 统一参数key：全是expression

        if tool_name == "calculator":
            tool_result = safe_calc(expr_param)
            print(f"【计算器】{expr_param} = {tool_result}")

        elif tool_name == "search":
            tool_result = external_search(expr_param)
            print(f"【搜索】{expr_param}，返回字数：{len(tool_result)}")

        else:
            tool_result = "不存在该工具"

        # 工具结果回填上下文
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_result,
        })
else:
    print("达到最大轮次，未获得最终回答")