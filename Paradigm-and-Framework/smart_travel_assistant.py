import datetime
import requests
import os
import re
import pytz
from tavily import TavilyClient
from openai import OpenAI
from dotenv import load_dotenv   # 新增：用于加载 .env 文件

# ---------------- 加载环境变量 ----------------
load_dotenv()   # 自动从项目根目录的 .env 文件中读取配置

# ---------------- 系统提示词（System Prompt）----------------
# 这个提示词定义了智能体的行为、可用的工具以及必须遵循的输出格式
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。
- `get_city_time(city: str)`:获取当前城市的时间。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
- 说话风格幽默

请开始吧！
"""


# ---------------- 工具函数实现 ----------------
def get_city_time(city: str) -> str:
    """
    获取指定城市的当前时间。
    内部维护了一个常用城市到时区的映射表，可以按需扩展。
    """
    city_timezone = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "广州": "Asia/Shanghai",
        "深圳": "Asia/Shanghai",
        "成都": "Asia/Shanghai",
        "杭州": "Asia/Shanghai",
        "纽约": "America/New_York",
        "伦敦": "Europe/London",
        "东京": "Asia/Tokyo",
        "巴黎": "Europe/Paris"
    }

    try:
        tz = city_timezone[city]
        now = datetime.datetime.now(pytz.timezone(tz))
        return f"{city} 当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}"
    except:
        return f"抱歉，暂时不支持 {city} 的时间查询"


def get_weather(city: str) -> str:
    """
    通过 wttr.in 免费 API 获取指定城市的实时天气。
    返回包含天气状况和气温的自然语言字符串。
    """
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    使用 Tavily 搜索 API 获取基于城市和天气的景点推荐。
    需要提前在环境变量中配置 TAVILY_API_KEY。
    """
    # 从环境变量安全读取 API 密钥，不再硬编码
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量，请在 .env 文件中设置。"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 优先返回 Tavily 自动总结的综合回答
        if response.get("answer"):
            return response["answer"]

        # 否则格式化原始结果的前几条
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


# 所有可用工具的字典，键为工具名，值为函数对象
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
    "get_city_time": get_city_time
}


# ---------------- LLM 客户端封装 ----------------
class OpenAICompatibleClient:
    """
    一个通用的 LLM 客户端，兼容所有 OpenAI 接口的服务。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 LLM 生成回复，非流式输出。"""
        print("🧠 正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("✅ 大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


# ---------------- 主程序入口 ----------------
if __name__ == "__main__":
    # 1. 从环境变量读取 LLM 和 Tavily 的配置
    #    请确保在 .env 文件中设置了以下变量：
    #    - LLM_API_KEY
    #    - LLM_BASE_URL
    #    - LLM_MODEL_ID
    #    - TAVILY_API_KEY
    API_KEY = os.getenv("LLM_API_KEY")
    BASE_URL = os.getenv("LLM_BASE_URL")
    MODEL_ID = os.getenv("LLM_MODEL_ID")

    # 必要的环境变量检查，缺失则给出明确提示
    if not all([API_KEY, BASE_URL, MODEL_ID]):
        print("错误：请在 .env 文件中设置 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_ID")
        exit(1)

    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 2. 初始化对话提示与历史
    user_prompt = "你好，请帮我查询一下今天纽约的天气，然后根据天气推荐一个合适的旅游景点,并且给出当地时间。"
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    # 3. 主循环：最多执行 5 轮 Thought-Action-Observation
    for i in range(5):
        print(f"--- 循环 {i + 1} ---\n")

        # 3.1 构建当前提示词（历史记录累积）
        full_prompt = "\n".join(prompt_history)

        # 3.2 调用 LLM，获得 Thought 和 Action
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        # 为防止模型输出多个 Thought-Action 对，截取第一对
        match = re.search(
            r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
            llm_output, re.DOTALL
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("🔔 已截断多余的 Thought-Action 对")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # 3.3 解析 Action 指令
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保回复遵循 'Thought: ... Action: ...' 格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()

        # 判断是否为结束指令
        if action_str.startswith("Finish"):
            final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
            print(f"🎉 任务完成，最终答案: {final_answer}")
            break

        # 3.4 解析工具名称和参数
        tool_match = re.search(r"(\w+)\((.*)\)", action_str)
        if not tool_match:
            observation = "错误: Action 格式不正确，应为 function_name(arg_name=\"arg_value\")"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)

        # 将参数解析为字典，例如 city="纽约" -> {"city": "纽约"}
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        # 3.5 执行工具调用，获取观察结果
        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)