import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
    temperature=0.7,
)

print("✅ 模型初始化成功！\n" + "="*60 + "\n")


# ============================================================
# 练习 1：ChatPromptTemplate — 结构化模板（中译英翻译）
# ============================================================
print("📝 练习 1：ChatPromptTemplate 基本用法")
print("-" * 40)

# 定义翻译模板：包含 system 角色和 user 问题
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位资深的{style}翻译专家，只输出翻译结果，不附加任何解释。"),
    ("user", "请将以下中文翻译成英文：\n\n{text}")
])

# 调用 .invoke() 查看生成的完整消息
# 漏传参langchain会直接报错
filled_prompt = translation_prompt.invoke({
    "style": "文学",
    "text": "床前明月光，疑是地上霜。"
})

print("填充后的模板内容（预览）：")
print(filled_prompt.to_string())


# ============================================================
# 练习 2：LCEL 管道符 `|` — 串联 Prompt | Model | OutputParser
# ============================================================
print("🔗 练习 2：使用 LCEL 串联链")
print("-" * 40)

# 定义一个简单的翻译链
chain = translation_prompt | model | StrOutputParser()

result = chain.invoke({
    "style": "口语化",
    "text": "今天天气真好啊，我们出去散步吧。"
})

print("翻译结果（口语化风格）：")
print(result)


# ============================================================
# 练习 3：RunnableParallel — 并行跑三种翻译风格
# ============================================================
print("⚡ 练习 3：RunnableParallel 并行执行三种风格")
print("-" * 40)

# 定义三种不同风格的提示词模板
prompt_direct = ChatPromptTemplate.from_messages([
    ("system", "你是翻译专家，请进行【直译】：严格按字面意思翻译，保持原文语法结构。"),
    ("user", "翻译：{text}")
])

prompt_free = ChatPromptTemplate.from_messages([
    ("system", "你是翻译专家，请进行【意译】：用地道、自然的目标语言表达，不拘泥于原文语序。"),
    ("user", "翻译：{text}")
])

prompt_academic = ChatPromptTemplate.from_messages([
    ("system", "你是翻译专家，请进行【学术翻译】：使用正式、严谨、书面化的学术语言。"),
    ("user", "翻译：{text}")
])

# 构建三条独立的链
chain_direct = prompt_direct | model | StrOutputParser()
chain_free = prompt_free | model | StrOutputParser()
chain_academic = prompt_academic | model | StrOutputParser()

# 使用 RunnableParallel 并行执行
parallel_chain = RunnableParallel(
    direct=chain_direct,
    free=chain_free,
    academic=chain_academic
)

# 一次 invoke，同时拿到三个结果（底层并发请求，总耗时约等于最慢的那一个）
input_text = "这个项目虽然困难重重，但团队依然迎难而上，最终取得了突破性进展。"
results_3 = parallel_chain.invoke({"text": input_text})

print(f"原文：{input_text}\n")
print(f"【直译】：{results_3['direct']}")
print(f"【意译】：{results_3['free']}")
print(f"【学术】：{results_3['academic']}")


# ============================================================
# 练习 4：.bind() — 固定模型参数（temperature=0）
# ============================================================
print("🎛️ 练习 4：使用 .bind() 固定模型参数")
print("-" * 40)

# 绑定 temperature=0，让模型输出尽可能确定（无随机性）
model_stable = model.bind(temperature=0, max_tokens=100)

# 构建一个独立的“稳定链”
stable_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是数学老师，请用一句话回答。"),
    ("user", "1 + 1 等于几？")
])

stable_chain = stable_prompt | model_stable | StrOutputParser()

# 连续调用两次，看看结果是否完全一致
result_4a = stable_chain.invoke({})
result_4b = stable_chain.invoke({})

print(f"第一次调用：{result_4a}")
print(f"第二次调用：{result_4b}")
print("是否完全一致？", result_4a == result_4b)
print("\n✅ temperature=0 时模型几乎没有随机性，输出极其稳定。")