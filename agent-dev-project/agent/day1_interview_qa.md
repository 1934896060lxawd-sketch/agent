# Day 1 面试题：Chain 与 Prompt Template

> 对应文件：`agent/langchain_basics.py`
> 核心能力：ChatPromptTemplate、LCEL 管道、RunnableParallel、.bind()、原生 SDK vs LCEL 对比

---

## Q1：`ChatPromptTemplate` + `MessagesPlaceholder` 解决了什么问题？和手写字符串拼接有什么本质区别？

**一句话**：手写字符串拼接把 prompt、role、变量混在一起，`ChatPromptTemplate` 把三者解耦——结构（role）由模板定义、内容（text）由变量注入、格式由 LangChain 自动处理。

**代码对比**：

```python
# ❌ 手写字符串拼接
prompt = f"你是{style}翻译专家。\n请翻译: {text}"
# 问题 1: 没有 role 概念（system/user/assistant 混在一起）
# 问题 2: {text} 里如果包含特殊字符可能破坏 prompt 结构
# 问题 3: 无法和 LangChain 生态的其他组件串联

# ✅ ChatPromptTemplate（Day 1 代码）
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位资深的{style}翻译专家，只输出翻译结果，不附加任何解释。"),
    ("user", "请将以下中文翻译成英文：\n\n{text}")
])
# 优势 1: role 明确分离（system vs user）
# 优势 2: 变量注入安全（LangChain 内部处理转义）
# 优势 3: 返回值是 ChatPromptValue，可以直接喂给 ChatModel
```

**`MessagesPlaceholder` 的作用**：当变量值本身是消息列表时使用（而不是字符串）。Day 2 的 Memory 注入就依赖它——`MessagesPlaceholder(variable_name="history")` 把整个对话历史列表插入到指定位置。

**面试话术**："Prompt Template 的核心价值不是少写几行字符串代码，而是把 prompt 变成可组合、可复用、可测试的组件。一个设计良好的 ChatPromptTemplate 可以被任何 ChatModel 消费，可以被塞进任何 LCEL 管道，可以独立单元测试每个变量的注入是否触发期望的报错。"

---

## Q2：LCEL 的 `|` 管道符是什么？它的设计哲学是什么？

**一句话**：`|` 是 LangChain 的 Runnable 协议的核心语法——把 prompt、model、parser 像 Unix 管道一样串联，数据从左流到右。

**代码中的体现**：

```python
# Day 1 代码 — LCEL 链
chain = translation_prompt | model | StrOutputParser()
#        ① PromptTemplate    ② ChatModel     ③ OutputParser
#        (dict → messages)   (messages → AIMessage)  (AIMessage → str)

result = chain.invoke({"style": "口语化", "text": "今天天气真好啊"})
# result 是纯字符串，不是 AIMessage 对象
```

**数据流转过程**：

```
invoke({"style": "口语化", "text": "..."})
  → ChatPromptTemplate: dict → ChatPromptValue（含 system + user messages）
  → ChatOpenAI: ChatPromptValue → AIMessage(content="It's such a nice day...")
  → StrOutputParser: AIMessage → "It's such a nice day..."
```

**设计哲学 — Runnable 协议**：每个组件都实现三个标准方法：

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `.invoke()` | 单条数据 | 单条结果 | 同步单次调用 |
| `.batch()` | 多条数据 | 多条结果 | 批量处理 |
| `.stream()` | 单条数据 | 迭代器 | 流式输出 |

**面试话术**："LCEL 的 `|` 本质是函数组合——`f | g` 等价于 `g(f(x))`。它的核心价值不是语法糖，而是让数据流的每一步都显式可见：你一眼就能看出 prompt 产出什么、model 消费什么、parser 产出什么。这是声明式编程在 AI 管道中的最佳实践。"

---

## Q3：`RunnableParallel` 的内部是怎么工作的？和 Python 的 `asyncio.gather` 有什么关系？

**一句话**：`RunnableParallel` 在 `invoke()` 时并发执行所有分支，底层用 `asyncio.gather` 或线程池实现。总耗时 ≈ 最慢分支的耗时，而非各分支之和。

**代码中的体现**：

```python
# Day 1 代码 — 三种翻译风格并行
parallel_chain = RunnableParallel(
    direct=chain_direct,      # 直译链
    free=chain_free,          # 意译链
    academic=chain_academic,  # 学术链
)
results = parallel_chain.invoke({"text": input_text})
# results = {"direct": "...", "free": "...", "academic": "..."}
# 三个 LLM 调用同时发起，总耗时 ≈ max(T1, T2, T3)
```

**底层机制**：LangChain 检查当前运行环境——如果在 async 上下文（如 FastAPI），用 `asyncio.gather` 并发；如果在同步上下文，用线程池并发。三种翻译风格各自独立（不共享中间结果、不互相等待）。

**和 Day 3 并行 Tool Calling 的关系**：Day 3 的并行 tool calling 也是并发执行——LLM 一次返回多个 `tool_calls`，互不依赖的工具可以同时执行。`RunnableParallel` 是 LangChain 框架层的并发原语，Day 3 是手写应用层的并发。两者底层思想一致。

**面试话术**："`RunnableParallel` 的价值不只是并发提速——它让你在一张图中表达'这三个分析维度互相独立'，这对复杂 Agent 的架构设计尤其重要。面试官如果问性能，回答：N 个独立 LLM 调用的并发延迟 ≈ max(N 个延迟)，而非 sum(N 个延迟)。"

---

## Q4：`.bind()` 在 LangChain 中的作用是什么？和直接在 `ChatOpenAI()` 构造时传参有什么区别？

**一句话**：`.bind()` 创建一个**新 Runnable** 并固定部分参数，不修改原对象。这让你可以从一个基础 model 派生出多个不同配置的变体。

**代码中的体现**：

```python
# Day 1 代码 — .bind() 固定 temperature
model = ChatOpenAI(model="deepseek-chat", temperature=0.7)  # 基础 model

model_stable = model.bind(temperature=0, max_tokens=100)  # 派生：确定性输出
# model 本身未被修改！model.temperature 还是 0.7

stable_chain = stable_prompt | model_stable | StrOutputParser()
result_4a = stable_chain.invoke({})  # temperature=0
result_4b = stable_chain.invoke({})  # temperature=0
print(result_4a == result_4b)  # True — 确定性输出
```

**和构造时传参的对比**：

| 方式 | 是否修改原对象 | 是否可复用 | 适用场景 |
|------|:---:|:---:|------|
| `ChatOpenAI(temperature=0)` | N/A（新建） | 需每次构造 | 只需要一种配置 |
| `model.bind(temperature=0)` | 否 | 从一个 model 派生多个 | 需要多种配置（如稳定/创意） |

**面试话术**："`.bind()` 体现了函数式编程的不可变性哲学——原始 model 保持不变，每次 `.bind()` 返回一个新对象。这在复杂 Agent 中很重要：同一个 model 实例可以被 bind 出多套参数配置（查询用 temperature=0、闲聊用 temperature=0.8、摘要用 max_tokens=200），各配置互不干扰。"

---

## Q5：`StrOutputParser` 做了什么？如果没有它会怎样？

**一句话**：`StrOutputParser` 把 `AIMessage` 对象转成纯字符串——去掉 `content` 外的所有包装（message 元数据、token usage、finish_reason 等）。

**代码体现**：

```python
# 不加 StrOutputParser
chain_no_parser = translation_prompt | model
result = chain_no_parser.invoke({"style": "...", "text": "..."})
print(type(result))   # <class 'langchain_core.messages.AIMessage'>
print(result)         # AIMessage(content="It's...", response_metadata={...}, id='...')

# 加 StrOutputParser
chain_with_parser = translation_prompt | model | StrOutputParser()
result = chain_with_parser.invoke({"style": "...", "text": "..."})
print(type(result))   # <class 'str'>
print(result)         # "It's a nice day today..."
```

**LangChain 的 OutputParser 体系**：

| Parser | 输入 | 输出 | 场景 |
|--------|------|------|------|
| `StrOutputParser` | AIMessage | str | 纯文本回答 |
| `JsonOutputParser` | AIMessage | dict | 结构化 JSON |
| `PydanticOutputParser` | AIMessage | BaseModel | Pydantic 对象（Day 8 前身） |

**面试话术**："`StrOutputParser` 看起来简单，但它代表了一个重要设计模式——'解析器和模型解耦'。同一个 model 的输出可以接不同的 parser，拿到的结果可以是字符串、JSON、Pydantic 对象。这就是 LangChain 的'管道可组合性'的具体体现。"

---

## Q6：原生 SDK vs LCEL 实现同一功能，各有什么优劣？什么时候该用哪个？

**一句话**：原生 SDK 给你完全控制权（适合理解原理和极致性能优化），LCEL 给你可组合的抽象层（适合快速开发和团队协作）。

**代码对比**（Day 1 练习 5 的核心）：

```python
# 原生 SDK（OpenAI 官方 client）
from openai import OpenAI
client = OpenAI(api_key=..., base_url=...)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": f"你是{style}翻译专家"},
        {"role": "user", "content": f"翻译：{text}"},
    ],
)
answer = response.choices[0].message.content

# LCEL（LangChain）
chain = translation_prompt | model | StrOutputParser()
answer = chain.invoke({"style": style, "text": text})
```

**对比表**：

| 维度 | 原生 SDK | LCEL |
|------|---------|------|
| 代码量 | ~10 行 | ~3 行 |
| 可组合性 | 手动管理 | 声明式管道 |
| 可测试性 | 需 mock API | 每步可独立测试 |
| 模型切换 | 可能改代码 | 换 ChatModel 子类即可 |
| 性能 | 无框架开销 | 微量代理开销 |
| 理解成本 | 看 OpenAI 文档 | 需理解 Runnable 协议 |
| 调试难度 | 简单（直接看 HTTP） | 复杂（需追踪管道） |

**选型原则**："原型验证用 LCEL（快速迭代），生产核心路径用原生 SDK（减少依赖），教学阶段先手写原生再学 LCEL（理解框架帮你省了什么）。Day 1-3 就是一个从原生 → LCEL → LangGraph 的渐进式学习路径。"

**面试话术**："框架的价值不是帮你省代码——是让你团队的人用同一种范式思考和协作。原生 SDK 你只能一个人快速写，但 LCEL 的管道声明式语法让 Code Review 时一眼看出数据流。这是从个人英雄主义到团队工程的转变。"

---

## Q7：`ChatPromptTemplate.from_messages()` 支持哪些 role 组合？`MessagesPlaceholder` 和普通变量有什么区别？

**一句话**：支持 `system`、`user`、`assistant`、`ai`、`human` 五种 role，以及 `MessagesPlaceholder` 作为消息列表的动态插入点。

**支持的 role 类型**：

```python
ChatPromptTemplate.from_messages([
    ("system", "你是..."),                    # 系统角色设定
    ("user", "用户问题：{question}"),          # 用户消息（别名："human"）
    ("assistant", "上次回答：{last_answer}"),  # AI 消息（别名："ai"）
    MessagesPlaceholder(variable_name="history"),  # 消息列表动态插入
])
```

**`MessagesPlaceholder` vs 普通 `{variable}`**：

| 特性 | `{variable}` | `MessagesPlaceholder` |
|------|-------------|----------------------|
| 变量类型 | 字符串（str） | 消息列表（List[BaseMessage]） |
| 插入方式 | 字符串替换 | 列表展开 |
| 典型用途 | 注入用户输入、参数 | 注入对话历史（Memory） |
| 生成的消息数 | 始终 1 条 | N 条（取决于列表长度） |

**面试话术**："`MessagesPlaceholder` 是 Memory 注入的底层机制。没有它，你只能在 prompt 里把整个对话历史拼成一个超长字符串——LLM 分不清哪些是历史、哪些是新问题。有了它，历史消息以原生 message 格式插入，每条保留独立 role，LLM 能正确区分上下文。"

---

## Q8：如果 `invoke()` 时漏传了模板中的变量会怎样？这个设计好在哪里？

**一句话**：LangChain 会直接抛出 `KeyError`，而不是默默用空字符串填充——这保证了 prompt 的完整性和可调试性。

**代码验证**：

```python
# Day 1 练习 1 的注释："漏传参 langchain 会直接报错"
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{style}翻译专家"),
    ("user", "翻译：{text}")
])

# 漏传 style
translation_prompt.invoke({"text": "床前明月光"})
# → KeyError: 'style'  ← 直接报错！不是默默用空字符串

# 多传无关变量
translation_prompt.invoke({"style": "文学", "text": "...", "extra": "..."})
# → 正常执行（多余变量被忽略）
```

**设计好处**：

1. **Fail Fast**：prompt 错误在开发阶段就暴露，不会被 delay 到 LLM 返回奇怪答案后才排查
2. **类型安全**：类似 TypeScript 的编译期检查——prompt 模板是"类型定义"，invoke 参数是"调用点"
3. **可调试**：KeyError 清楚告诉你缺了哪个变量，而不是"LLM 为什么表现奇怪"

**面试话术**："这个设计体现了 LangChain 的工程哲学——宁愿在开发阶段报错，也不要在生产环境默默产生低质量的回答。相比之下，手写 f-string 时忘了传变量，Python 也不会报错，LLM 看到的 prompt 就是残缺的。这就是框架的'安全网'价值。"

---

### Day 1 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | ChatPromptTemplate vs 手写字符串拼接的本质区别？MessagesPlaceholder 的作用？ | □ |
| 2 | LCEL `\|` 的数据流转过程？Runnable 协议三个标准方法？ | □ |
| 3 | RunnableParallel 的并发机制？和 asyncio.gather 的关系？ | □ |
| 4 | .bind() 和构造时传参的区别？为什么需要不可变性？ | □ |
| 5 | StrOutputParser 做了什么？LangChain 的 OutputParser 体系？ | □ |
| 6 | 原生 SDK vs LCEL 各有什么优劣？什么时候用哪个？ | □ |
| 7 | from_messages 支持哪些 role？MessagesPlaceholder vs 普通变量的区别？ | □ |
| 8 | invoke 漏传变量会怎样？这个设计好在哪里？ | □ |
