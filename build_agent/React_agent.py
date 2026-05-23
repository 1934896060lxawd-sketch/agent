import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from serpapi import GoogleSearch
import re
from llm_client import HelloAgentsLLM

# ============================================================
# 1. 提示词模板 —— ReAct 范式的核心
# 作用：告诉 LLM 必须按照 “Thought: ... Action: ...” 的格式输出，
#       这样才能被程序稳定解析，实现“思考-行动”循环。
# ============================================================
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


# ============================================================
# 3. 工具函数 —— 智能体能调用的具体能力
# 这里只演示了一个“搜索工具”，实际项目中可以注册多个
# ============================================================
def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误: SERPAPI_API_KEY 未在 .env 文件中配置。"

        # 配置搜索参数：使用 Google 引擎，面向中国地区
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        # 执行搜索并获取结构化结果
        search_instance = GoogleSearch(params)
        results = search_instance.get_dict()

        # ---- 智能解析：按优先级返回最有价值的信息 ----
        # 第一优先：答案摘要框（Google 直接给出的答案）
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]

        # 第二优先：知识图谱（结构化信息，如公司、人物简介）
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]

        # 第三优先：前三条自然搜索结果的标题和摘要
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


# ============================================================
# 4. 工具执行器 —— 工具的“注册中心”和“调度中心”
# 设计思想：统一管理所有工具，新增工具只需注册，不改其他代码
# ============================================================
class ToolExecutor:
    def __init__(self):
        # 用一个字典存储所有工具，键是工具名，值是描述和函数
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """向执行器中注册一个新工具"""
        if name in self.tools:
            print(f"警告: 工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """根据工具名获取对应的函数，如果不存在则返回 None"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """生成所有已注册工具的描述文本，用来填入 Prompt 中的 {tools} 部分"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# ============================================================
# 5. ReAct 智能体 —— 整个框架的大脑和循环引擎
# 核心逻辑：循环执行 “拼接Prompt → LLM思考 → 解析Action →
#           调用工具 → 得到Observation → 写入历史 → 进入下一轮”
# ============================================================
class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        """
        参数:
            llm_client: 封装好的 LLM 调用客户端
            tool_executor: 工具注册与执行器
            max_steps: 最大思考步数，防止无限循环
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []  # 保存每一轮的 Action 和 Observation

    def run(self, question: str):
        """
        启动 ReAct 循环，处理用户问题，直到给出最终答案或步数耗尽。
        """
        self.history = []  # 每次提问都从干净的历史开始
        current_step = 0

        # ---- 主循环：每一步都是一次完整的“思考-行动-观察” ----
        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # ① 组装本次 prompt：工具描述 + 问题 + 历史记录
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # ② 让 LLM 根据当前上下文“思考”，输出 Thought + Action
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # ③ 解析 LLM 的输出，提取 Thought 和 Action
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # ④ 判断 Action 类型并执行
            if action.startswith("Finish"):
                # 如果是 Finish[答案]，直接提取最终答案并结束
                final_answer = re.search(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # 否则就是工具调用，解析工具名和输入参数
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                continue   # 解析失败则跳过本轮，进入下一步（通常会失败）

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            # ⑤ 调用真实的工具函数，得到观察结果
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)
            print(f"👀 观察: {observation}")

            # ⑥ 将本轮的 Action 和 Observation 记入历史
            #    这样下一轮 LLM 就能看到自己做过什么、结果如何
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 如果循环结束仍未返回最终答案
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """
        从 LLM 的原始输出中提取 Thought 和 Action。
        使用正则表达式保证解析的稳定性。
        """
        # Thought 字段：从 "Thought:" 开始，到 "Action:" 或文本末尾结束
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action 字段：从 "Action:" 开始，到文本末尾结束
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        解析 Action 字符串，例如 "Search[华为手机]" -> ("Search", "华为手机")
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


# ============================================================
# 6. 主程序入口 —— 把所有组件拼装起来，运行智能体
# ============================================================
if __name__ == '__main__':
    # ----- 6.1 初始化工具执行器，注册搜索工具 -----
    tool_executor = ToolExecutor()
    search_description = (
        "一个网页搜索引擎。"
        "当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    )
    tool_executor.registerTool("Search", search_description, search)

    print("\n--- 已注册的工具 ---")
    print(tool_executor.getAvailableTools())

    # ----- 6.2 初始化 LLM 客户端（从 .env 读取配置）-----
    try:
        llm = HelloAgentsLLM()
        print(f"✅ LLM 客户端初始化成功，模型：{llm.model}")
    except ValueError as e:
        print(f"❌ LLM 客户端初始化失败：{e}")
        exit(1)

    # ----- 6.3 创建 ReAct 智能体实例 -----
    agent = ReActAgent(
        llm_client=llm,
        tool_executor=tool_executor,
        max_steps=5          # 最多允许 5 轮“思考-行动”
    )

    # ----- 6.4 向智能体提问，启动循环 -----
    question = "小米最新的手机型号及主要卖点是什么？"
    print(f"\n🤖 开始处理问题: {question}\n")

    final_answer = agent.run(question)

    if final_answer:
        print(f"\n🎯 智能体最终答案:\n{final_answer}")
    else:
        print("\n⚠️ 智能体未能在最大步数内得出最终答案。")