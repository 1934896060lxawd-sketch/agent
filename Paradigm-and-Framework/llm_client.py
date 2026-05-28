import os
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv



# ============================================================
#    LLM 客户端 —— 封装与大模型的通信
# 作用：统一管理 API 调用，支持流式输出，让其他模块不需要
#       关心底层细节。
# ============================================================

# 加载 .env 文件中的环境变量（API密钥、模型名等）
load_dotenv()

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        # 如果没有传入参数，就从环境变量中读取
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 必要的配置缺失时直接报错，避免后续调用失败难以排查
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        # 初始化 OpenAI 客户端（兼容大多数国产大模型）
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用 LLM 进行“思考”，返回模型生成的完整文本。
        使用流式输出，边生成边打印，提升用户体验。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,   # 温度设为 0，让输出更稳定、可预测
                stream=True,               # 开启流式输出
            )
            print("✅ 大语言模型响应成功:")
            collected_content = []
            # 逐块接收流式响应，实时打印并拼接完整内容
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 换行
            return "".join(collected_content)
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None