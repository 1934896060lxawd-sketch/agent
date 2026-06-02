import openai
import json
import os
import re
from dotenv import load_dotenv
from tavily import TavilyClient

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
    "expression": {"type": "string", "description": "数学表达式，例：(100+123)*234"}
}
calculator = build_function_tool(
    name="calculator",
    desc="执行数学表达式计算，支持加减乘除、括号、根号、平方",
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

# =========【重点1：修改system提示词，强制规则】=========
messages = [
    {
        "role": "system",
        "content": """规则：
1. 遇到**数学计算题**：直接调用 calculator 计算；
2. 遇到**常识、科普、实时资讯、知识类问题，不确定答案时，必须优先调用 search 工具查询资料**，拿到搜索结果之后再整理答案；
3. 禁止不调用search直接编造知识性内容；
4. 不能跳过工具直接回复。"""
    },
    # {"role": "user", "content": "计算（100+123）*234的结果"},
    {"role": "user", "content": "现在2026年国内主流新能源汽车售价？"} # 测试搜索用
]


def external_search(q: str) -> str:
    resp = tavily.search(query=q, search_depth="basic")
    res_list = resp["results"]
    content = ""
    for item in res_list:
        content += f"标题：{item['title']}\n摘要{item['content']}\n"
    return content


# 最多5轮工具调用
for turn in range(5):
    response = client.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=tools
    )
    msg = response.choices[0].message

    # 无工具调用，输出最终答案
    if not msg.tool_calls:
        print("最终回答：", msg.content)
        break

    messages.append(msg)

    # =========【重点2：根据工具名分流逻辑】=========
    for tc in msg.tool_calls:
        tool_name = tc.function.name
        args = json.loads(tc.function.arguments)
        expr = args["expression"]
        tool_result = ""

        if tool_name == "calculator":
            # 计算器逻辑
            if not re.match(r"^[\d+\-*/()\s.]+$", expr):
                tool_result = "错误：表达式包含非法字符"
            else:
                tool_result = str(eval(expr))
            print(f"【计算器】{expr} = {tool_result}")

        elif tool_name == "search":
            print(f"【联网搜索】查询：{expr}")
            # 对接真实搜索接口
            tool_result = external_search(expr)

        else:
            tool_result = "不存在该工具"

        # 工具结果塞回上下文
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_result,
        })
else:
    print("达到最大轮次，未获得最终回答")