# Day 6 — Agent 集成：ReAct 循环 + Function Calling + 流式输出

> **今日目标**：用 DeepSeek API 的原生 Function Calling 实现 ReAct Agent，接入 Day 5 的混合检索引擎，替换 Phase 2 的占位回答。这是 Phase 3 的收尾——从今天起，系统能基于真实知识库给出真实回答。

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [ReAct Agent 架构](#2-react-agent-架构)
3. [Function Calling 机制](#3-function-calling-机制)
4. [5 个工具详解](#4-5-个工具详解)
5. [流式输出管道](#5-流式输出管道)
6. [chat.py：前后端打通](#6-chatpy前后端打通)
7. [核心技术原理](#7-核心技术原理)
8. [初学者常见疑问](#8-初学者常见疑问)
9. [面试模拟问答](#9-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `agent/tools.py` | ~200 | 5 个工具定义(OpenAI Schema) + ToolExecutor 执行器 |
| `agent/prompts.py` | ~40 | 五段式 System Prompt (角色/边界/准则/约束/错误) |
| `agent/advisor.py` | ~120 | CarAdvisorAgent: ReAct 循环 + 流式 token 输出 |
| `api/deps.py` (改) | +35 | 新增 get_agent() 依赖：加载索引 → 创建 Agent 单例 |
| `api/routes/chat.py` (改) | ~110 | 替换占位生成器为真实 Agent 流式调用 |

---

## 2. ReAct Agent 架构

### 2.1 什么是 ReAct？

ReAct = **Re**asoning + **Act**ing。LLM 不是一次性生成答案，而是：

```
用户: "25万家用SUV推荐"
  → Thought (推理): 用户有预算没有具体车型，先推荐
  → Action (行动): recommend_cars(budget_min=20, budget_max=28, category="SUV")
  → Observation (观察): [3款推荐车型返回]
  → Thought (推理): 有3款适合，但需要进一步查看参数
  → Action (行动): search_car_knowledge("理想L6 比亚迪宋L 问界M7 配置对比")
  → Observation (观察): [详细参数文档返回]
  → Final Answer (回答): "根据25万预算，推荐以下3款SUV..."（流式输出）
```

**与普通 LLM 对话的区别**：普通对话是"问题→答案"一次完成。ReAct 是"问题→推理→行动→观察→推理→行动→...→答案"的多次循环。模型在行动中获取外部信息，在观察中修正推理方向。

### 2.2 为什么不用 LangGraph？

参考代码使用 `langgraph.agents.create_agent()` 一行构建 ReAct Agent。但本项目的选择是**手动实现 ReAct 循环**，理由：

| 维度 | LangGraph create_agent() | 手动 Function Calling |
|------|--------------------------|----------------------|
| 依赖 | langchain + langgraph + langchain-openai (~50MB) | openai (~5MB) |
| 可控性 | 黑盒——stream 事件格式固定 | 完全控制事件格式和循环逻辑 |
| 调试 | 需要理解 LangGraph 内部状态机 | 透明——就是一个 while 循环 |
| 定制 | 受限于 LangGraph API | 可以插任意中间步骤（缓存、审计、限流） |
| 学习价值 | 会用工具 | 理解原理 |

**关键原则**：面试官会问"为什么不直接用 LangChain？"——答案不是"LangChain 不好"，而是"对于 function calling + while 循环这种简单模式，手动实现更轻量、可控性更强、更符合企业级系统的透明性要求"。

### 2.3 ReAct 循环实现

```python
async def stream_chat(self, query, history):
    messages = [system_prompt, *history, user_query]

    for iteration in range(MAX_ITERATIONS):  # 最多 5 轮
        # ① 调 LLM（非流式——需要判断有没有工具调用）
        response = await self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOL_SCHEMAS,        # ← 告诉 LLM 有哪些工具可用
            tool_choice="auto",        # ← LLM 自己决定要不要用工具
        )
        msg = response.choices[0].message

        # ② 有工具调用 → 执行 → 注入结果 → 继续循环
        if msg.tool_calls:
            messages.append(msg)  # 把 assistant 的 tool_calls 加入对话
            for tc in msg.tool_calls:
                result = await executor.execute(tc.function.name, args)
                messages.append({"role": "tool", "content": result})
            continue  # ← 回到循环顶部，LLM 看到工具结果后继续推理

        # ③ 无工具调用 → 最终回答，退出循环
        break

    # ④ 流式输出最终回答
    stream = await self.llm.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True,  # ← 逐 token 返回
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield {"type": "token", "content": chunk.choices[0].delta.content}
```

**三个关键设计决策**：

1. **第 1 次调用用 `stream=False`**：因为需要完整读取 `msg.tool_calls` 才能决定是否执行工具。如果 stream=True，tool_calls 是分 chunk 过来的，需要手动拼接——复杂度远大于收益。

2. **第 2 次调用用 `stream=True`**：最终回答不需要继续调用工具了（已经确认 `msg.tool_calls` 为 None），直接流式输出 token 给前端。

3. **`MAX_ITERATIONS=5`**：防止 LLM 在工具调用之间无限循环（如：推荐 → 不满意 → 重新推荐 → 不满意 → ...）。

---

## 3. Function Calling 机制

### 3.1 什么是 Function Calling？

Function Calling 不是 LLM 真的"调用"了你的函数。它只是让 LLM **输出一个结构化的 JSON 对象**，表示"我想调用这个函数，参数是这些"：

```
正常 LLM 输出: "推荐比亚迪宋L，价格15.98-20.98万..."

启用 Function Calling 后 LLM 可能输出:
{
  "tool_calls": [{
    "function": {
      "name": "recommend_cars",
      "arguments": '{"budget_min": 20, "budget_max": 28, "category": "SUV"}'
    }
  }]
}
```

**你的代码拿到这个 JSON 后，自己执行 `recommend_cars(20, 28, "SUV")`，把结果塞回对话。** LLM 全程不知道函数内部做了什么——它只是"提议"调用某个函数。

### 3.2 工具 Schema 定义

每个工具需要提供 OpenAI 格式的 JSON Schema：

```json
{
    "type": "function",
    "function": {
        "name": "recommend_cars",
        "description": "根据预算和偏好推荐车型。当用户询问'XX万买什么车'时调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "budget_min": {"type": "number", "description": "预算下限（万元）"},
                "budget_max": {"type": "number", "description": "预算上限（万元）"},
                "category": {"type": "string", "description": "车型类别，如SUV/轿车"}
            },
            "required": ["budget_min", "budget_max"]
        }
    }
}
```

**`description` 字段至关重要**——它决定 LLM **什么时候**决定调用这个工具。如果 description 写"推荐车"，LLM 可能在用户问"你觉得特斯拉怎么样"时也调用 recommend_cars（不合适）。应该写清楚触发条件："当用户询问'XX万买什么车'时调用此工具"。

### 3.3 对话消息的角色演变

Function Calling 在标准对话的基础上引入了一个新角色 `tool`：

```
[{"role": "system", "content": "你是汽车导购..."},          ← 系统提示
 {"role": "user", "content": "25万SUV推荐"},               ← 用户问题
 {"role": "assistant", "content": null,                     ← LLM 决定调工具
     "tool_calls": [{"function": {"name": "recommend_cars", ...}}]},
 {"role": "tool", "tool_call_id": "call_001",               ← 工具执行结果
     "content": '{"count": 3, "cars": [...]}'},
 {"role": "assistant", "content": "根据25万预算..."}]        ← LLM 最终回答
```

传统对话：`user ↔ assistant` 交替。Function Calling：`user → assistant(tool_calls) → tool(results) → assistant(answer)`。`tool` 角色让 LLM 知道"这是你刚才要调的那个函数的返回结果"。

---

## 4. 5 个工具详解

### 4.1 工具全景

| 工具 | 触发场景 | 输入 | 输出 | 数据源 |
|------|---------|------|------|--------|
| `search_car_knowledge` | 查看配置/参数/评价 | query 字符串 | Top-3 文档片段 | FAISS+BM25+RRF+Rerank |
| `get_car_price` | 问价格 | brand + model | 指导价区间 | CAR_PRICE_DB 字典 |
| `compare_cars` | 对比两款车 | car1 + car2 | 双车参数并排 | CAR_PRICE_DB + CAR_SPEC_BRIEF |
| `recommend_cars` | XX万推荐 | budget + 类别/品牌 | 匹配车型列表 | CAR_PRICE_DB 遍历过滤 |
| `calculate_ownership_cost` | 落地价/养车 | model + 年限 | 年均成本+累计 | 中间价 × 费率公式 |

### 4.2 search_car_knowledge：RAG 管线的 Agent 入口

这是 5 个工具中最复杂的一个——它把整个 Day 4 + Day 5 的检索管线打包成一个工具：

```python
def _tool_search_car_knowledge(self, query):
    dense = self.retriever.search(query, top_k=6)    # FAISS 向量
    sparse = self.bm25.search(query, top_k=6)         # BM25 关键词
    hybrid = hybrid_rrf(dense, sparse, k=60, top_k=5) # RRF 融合
    reranked = rerank(query, hybrid, top_k=3)          # CrossEncoder 精排
    return json.dumps({"results": [...]})
```

**一个工具调用 = 整个 RAG 管线执行一遍**。Agent 负责决策"什么时候需要查知识库"，工具负责"怎么查"。职责分离清晰。

### 4.3 ToolExecutor：工具执行器

```python
class ToolExecutor:
    def __init__(self):
        self.retriever = None   # 由 deps.py 注入
        self.bm25 = None

    async def execute(self, name, arguments):
        method = getattr(self, f"_tool_{name}")  # 按名称分发
        return method(**arguments)                 # 解包参数调用
```

**为什么用 getattr 反射而不是 if-elif？**

```python
# if-elif 方式 — 加一个工具要改 3 处
if name == "search_car_knowledge":
    return self._tool_search_car_knowledge(**args)
elif name == "get_car_price":
    ...

# getattr 方式 — 加一个工具只需添加 _tool_xxx 方法
method = getattr(self, f"_tool_{name}", None)
```

工具是可扩展的——Phase 4 如果要加 `calculate_loan`（贷款计算）工具，只需加一个 `_tool_calculate_loan` 方法和对应的 Schema 定义，`execute()` 不用改。

---

## 5. 流式输出管道

### 5.1 全链路数据流

```
用户: "25万SUV推荐"  (HTTP POST /chat)
  │
  ▼
chat.py → agent.stream_chat(query, history)
  │
  ▼  异步生成器逐事件产出
  ├─ {"type": "source", "documents": [...]}   ← 工具调用时推送
  ├─ {"type": "token", "content": "推"}        ← 最终回答逐字推送
  ├─ {"type": "token", "content": "荐"}
  ├─ ...
  └─ {"type": "done", "total_tokens": 342}
  │
  ▼
sse_generator(event_generator) → "data: {...}\n\n" 字符串流
  │
  ▼
StreamingResponse(sse_strings, media_type="text/event-stream")
  │
  ▼
客户端 EventSource: event.data → 打字机效果
```

### 5.2 为什么最终回答用 stream=True 而工具决策用 stream=False？

```
第 1 次调用 (stream=False):
  目的: 判断"LLM 要调用工具还是直接回答？"
  需要: 完整读取 message.tool_calls 字段
  如果 stream=True: tool_calls 分散在多个 chunk 中，需要手动拼接 + 状态机

第 2 次调用 (stream=True):
  目的: 向用户展示逐字输出
  前提: 已经确认 msg.tool_calls is None（不会再调用工具了）
  如果 stream=False: 用户要等完整回答生成完才能看到——几秒的白屏
```

这是企业级 Agent 的经典模式——**决策用非流式，输出用流式**。

### 5.3 _stream_and_save 闭包模式

```python
async def _stream_and_save():
    full_parts = []
    async for sse_str in sse_generator(agent_generator(query, history, agent)):
        if '"type":"token"' in sse_str:
            # 并行任务1: 收集 token 用于保存
            full_parts.append(extract_content(sse_str))
        # 并行任务2: 立即推送给前端
        yield sse_str

    # 流结束后保存完整回答到 Redis
    await session_mgr.add_message(session_id, "assistant", "".join(full_parts))
```

**为什么用闭包而不是两个独立函数？**

SSE 的 StreamingResponse 一旦 return，控制权就全交给 FastAPI 了。你不能再在路由函数里 `await session_mgr.add_message()`——那个代码永远不会执行。闭包把"边推送边收集"的逻辑嵌入到生成器内部，是唯一正确的方式。

---

## 6. chat.py：前后端打通

### 6.1 改造前后对比

**Phase 2 (占位):**
```python
async def _placeholder_generator(query):
    yield {"type": SSE_SOURCE, "documents": []}
    for char in f"收到您的问题「{query}」...":
        yield {"type": SSE_TOKEN, "content": char}
        await asyncio.sleep(0.03)
```

**Phase 3 (真实 Agent):**
```python
async def _agent_generator(query, history, agent):
    async for event in agent.stream_chat(query, history):
        yield event  # 直接透传 Agent 产出的事件
```

替换点极其干净——因为 Phase 2 设计时就预埋了相同的 SSE 事件协议（SSE_SOURCE/SSE_TOKEN/SSE_DONE）。**好的架构让替换变成插拔，而不是重写。**

### 6.2 对话历史的加载

```python
# 从 Redis 加载最近 20 条消息
raw_history = await session_mgr.get_history(body.session_id, limit=20)
history = []
for msg in raw_history[:-1]:  # 排除刚加入的当前用户消息
    data = json.loads(msg)
    history.append({"role": data["role"], "content": data["content"]})
```

为什么传给 Agent 历史消息？因为 DeepSeek API 是无状态的——每次调用都是一个全新的对话。如果不传历史，LLM 不知道上一轮说了什么。这就是"对话记忆"的本质：把 Redis 里的历史消息拼到当前请求的 messages 列表里。

---

## 7. 核心技术原理

### 7.1 为什么用 AsyncOpenAI 而不是 requests？

```python
# requests（同步阻塞）:
response = requests.post(url, json=data)  # 阻塞整个线程
# 100 个并发请求 = 100 个线程

# AsyncOpenAI（异步非阻塞）:
response = await client.chat.completions.create(...)  # 让出控制权
# 100 个并发请求 = 1 个线程上的 100 个协程
```

FastAPI 是 asyncio 框架，路由函数是 async 的。如果用同步 HTTP 客户端（requests/httpx sync），调用 LLM API 的 3-5 秒内整个线程被阻塞——其他请求全部排队等待。`AsyncOpenAI` 在等待网络响应时让出控制权，asyncio 事件循环可以处理其他请求。

### 7.2 System Prompt 工程五段式

```
## 角色         ← 你是谁？说话风格？
## 能力边界      ← 你能做什么？不能做什么？
## 决策准则      ← 遇到什么情况该调哪个工具？
## 输出约束      ← 回答应该长什么样？
## 错误处理      ← 遇到意外情况怎么办？
```

这不是模板——是经过验证的 Prompt Engineering 最佳实践。每段解决一个具体问题：
- 没有"角色"定义 → LLM 可能拒绝回答（"我是 AI 助手，不能推荐汽车"）
- 没有"决策准则" → LLM 可能不调用工具（直接编造回答）或过度调用工具（每个问题都查知识库）
- 没有"错误处理" → LLM 遇到工具返回空结果时可能继续编造数据（幻觉）

### 7.3 为什么 Agent 是单例？

```python
_agent: CarAdvisorAgent | None = None  # 模块级单例

async def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    # 首次调用: 加载 FAISS 索引 (0.1s) + BM25 建索引 (0.5s) + 创建 Agent
    _agent = CarAdvisorAgent(executor)
    return _agent
```

Agent 内部持有 FAISS 索引和 BM25 索引——这些对象不能每个请求创建一次（102 条文档的 BM25 建索引约 0.5s × 100 并发 = 50s）。单例保证所有请求共享同一个 Agent 实例。这和 `get_embedding_model()` 的单例模式是一致的设计理念。

---

## 8. 初学者常见疑问

**Q: Function Calling 和 Prompt Engineering 有什么区别？**

用 Prompt Engineering 也能让 LLM "调用函数"——在 prompt 里写"如果你想查询价格，请输出 `PRICE: 车型名`"，然后在代码里正则匹配 `PRICE:`。问题：① 正则匹配不如 JSON 解析可靠；② LLM 可能忘记格式；③ 参数复杂时正则无法处理嵌套结构。Function Calling 是模型层面支持的——DeepSeek/OpenAI 在训练时专门优化了工具调用的准确率。

**Q: `tool_choice="auto"` 是什么意思？**

让 LLM 自己决定"需要调工具"还是"直接回答"。如果设 `tool_choice="required"`，LLM 必须调工具——即使用户说"你好"，LLM 也会强行调用某个工具。`auto` 更灵活：简单聊天不调工具，复杂问题自动调。

**Q: 为什么先 `stream=False` 再 `stream=True`？不能全程 stream 吗？**

可以但没必要。全程 stream 需要：① 边收 chunk 边判断有没有 tool_calls（tool_calls 可能跨多个 chunk）；② 如果检测到 tool_calls，中断流、执行工具、重新发起流式请求。实现复杂度翻倍，而 UX 收益极小（工具执行阶段用户看不到任何 token——因为 LLM 在"想"用什么工具时本来就不输出内容）。分两次调用是工程上的务实选择。

**Q: 对话历史为什么要限制 20 条？**

DeepSeek 的上下文窗口是有限的（deepseek-chat 支持 64K tokens）。20 条消息约 2000-4000 tokens，再加上 system prompt（约 500 tokens）、知识库检索结果（约 1000 tokens）、当前回答的空间——总计约 5000-8000 tokens，远在 64K 限制内。如果无限制累积历史，几十轮对话后必然超窗口截断。

**Q: 工具返回的结果 LLM 是怎么"理解"的？**

工具返回的是 JSON 字符串，以 `{"role": "tool", "content": json_str}` 的形式注入对话。LLM 把这段 JSON 当作对话的"观察"部分——和它训练时见过的工具调用模式是相同的格式。JSON 结构让 LLM 能准确提取字段值（如价格 15.98 万），而不是从自然语言中猜测数字。

---

## 9. 面试模拟问答

> **Q: 解释一下你们的 Agent 是怎么工作的？**

我们基于 DeepSeek 的 Function Calling 实现了 ReAct Agent。核心是一个 while 循环：LLM 收到用户问题后判断是否需要调用工具（如查价格、搜知识库），如果需要就输出结构化 JSON 指明工具名和参数，我们的 ToolExecutor 执行工具并把结果注入对话，LLM 看到结果后继续推理，直到能给出最终回答。循环上限 5 轮防止无限循环。决策阶段用非流式调用（需要完整读取 tool_calls），最终回答用流式调用（逐 token 推送给前端）。

> **Q: 为什么不用 LangChain/LangGraph？**

Function Calling + while 循环这种模式足够简单，手动实现比引入 LangChain 更轻量：① 依赖更少（openai 一个包 vs langchain+langgraph 多个包）；② 调试更透明（状态就是 messages 列表，不需要理解 LangGraph 内部状态机）；③ 定制更灵活（可以插入日志、缓存、审计等中间步骤）。引入框架的收益（一行 create_agent）抵不上理解框架内部机制和排查框架 bug 的成本。

> **Q: 如果 Agent 的 ReAct 循环超过了 MAX_ITERATIONS 会怎样？**

循环退出，messages 列表中最后一条是 LLM 的思考结果（可能包含或不包含 tool_calls）。如果还有 tool_calls 未执行，它们会被忽略——LLM 的最终流式输出会基于当前 messages 给出回答。这相当于"超时降级"：即使推理不充分，也比无限等待好。生产环境可以加日志告警（"Agent 在第 5 轮仍未收敛"），排查是否某个工具返回结果质量有问题导致 LLM 反复调用。

> **Q: 你们的 System Prompt 是怎么设计的？**

五段式结构：角色定义（说话风格+专业背景）、能力边界（明确可用工具和数据范围，防止编造）、决策准则（什么时候该调哪个工具的具体规则）、输出约束（推荐数量、引用格式）、错误处理（工具返回空、用户输入模糊、偏离领域时的应对策略）。每段解决一个具体问题——角色防止拒绝回答，边界防止幻觉，准则防止不调工具或过度调用，约束保证输出格式一致，错误处理提升健壮性。

> **Q: Agent 调用 search_car_knowledge 和用户直接调用检索有什么区别？**

Agent 调用多了一层"智能决策"。用户说"有没有比特斯拉便宜的电动车"，Agent 会：① 先调 get_car_price("特斯拉", "Model 3") 获得特斯拉价格；② 用特斯拉价格作为预算上限调 recommend_cars；③ 如果需要具体参数再调 search_car_knowledge。用户不用手动分解查询步骤——Agent 自动做任务规划。这就是 ReAct 的核心价值：不是一次检索回答所有问题，而是多步推理逐步缩小答案空间。

> **Q: 如果 DeepSeek API 挂了，系统怎么处理？**

当前异常处理在 `stream_chat()` 的 try/except 中：捕获所有异常后 yield `{"type": "error", "message": str(e)}`，sse_generator 会将其格式化为 SSE error 事件推送给前端。前端可以根据 error 类型决定是重试还是提示用户。如果要做更高可用性，可以在 Agent 层加 circuit breaker（类似 Day 2 的 Redis 熔断器），或配置 DeepSeek 的 fallback endpoint。

---

## 附：今日文件依赖关系

```
prompts.py ─────────────────────┐
tools.py (TOOL_SCHEMAS) ────────┤
                                ├──→ advisor.py (CarAdvisorAgent)
retriever.py (VectorIndex+BM25)─┤         │
reranker.py ────────────────────┘         │
                                          │
deps.py (get_agent 单例) ─────────────────┤
                                          │
chat.py (_agent_generator) ───────────────┘
    │
    ▼
POST /chat → SSE StreamingResponse (真实 Agent 流式回答)
```

Day 6 是 Phase 3 的收尾——之前的 6 天（Day 1-3 基础设施 + Day 4-5 RAG 检索 + Day 6 Agent 集成）全部汇聚到 `/chat` 这一个端点上。从 curl 发起一个请求到看到打字机效果的回答，中间经过了：Redis 会话 → 限流检查 → FAISS 向量检索 → BM25 关键词 → RRF 融合 → CrossEncoder 精排 → DeepSeek ReAct → 流式 token 推送。这是全链路 AI Agent 系统的完整闭环。
