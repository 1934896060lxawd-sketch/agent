# Day 2 面试题：Memory 与 RAG 封装

> 对应文件：`agent/langchain_rag.py`
> 核心能力：RunnableWithMessageHistory、三种 Memory 策略、RunnableLambda、BaseRetriever、RunnablePassthrough、原生 vs LCEL 对比

---

## Q1：`RunnableWithMessageHistory` 是怎么工作的？它帮我们省了什么？

**一句话**：它在每次 `invoke()` 前后自动做两件事——调用前从 store 取出历史消息注入 prompt，调用后把本轮对话写回 store。把 Day 1 手动维护 `self.chat_history` 的逻辑自动化了。

**数据流**：

```
invoke({"question": "小米SU7多少钱"})
  → ① 从 store 取历史: get_session_history(session_id="user_123")
       → [HumanMessage("你好"), AIMessage("你好！..."), HumanMessage("推荐SUV"), ...]
  → ② 历史注入到 prompt 的 MessagesPlaceholder 位置
  → ③ prompt + 历史 → model → AIMessage
  → ④ 把本轮 HumanMessage + AIMessage 写回 store
  → ⑤ 返回 AIMessage.content
```

**代码简化对比**：

```python
# ❌ Day 1 方式：手动维护历史
chat_history = []
chat_history.append(HumanMessage(content=query))
result = chain.invoke(...)
chat_history.append(AIMessage(content=result))
# 每次对话要手动 push，要手动传给 prompt，要手动截断

# ✅ Day 2 方式：RunnableWithMessageHistory 自动管理
chain_with_memory = RunnableWithMessageHistory(
    base_chain, get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
result = chain_with_memory.invoke(
    {"question": query},
    config={"configurable": {"session_id": "user_123"}}
)
# 历史存取全自动，只需传 session_id
```

**面试话术**："`RunnableWithMessageHistory` 的核心价值是把'历史管理'从业务逻辑中解耦出来。你的 chain 不需要知道历史存在哪里、怎么截断、怎么格式化——它只需要声明 `MessagesPlaceholder(variable_name="history")`，框架负责填。这是 AOP（面向切面编程）在 Agent 管道中的典型应用。"

---

## Q2：Buffer vs Summary vs Window 三种 Memory 策略各有什么优劣？生产环境怎么选？

**一句话**：Buffer 全量保留（短对话最佳）、Summary 压缩保留（长对话首选但多一次 LLM 调用）、Window 只保留最近 K 轮（折中方案但可能丢失远距离上下文）。

**三种策略对比**：

| 维度 | BufferMemory | SummaryMemory | WindowMemory(K=3) |
|------|:---:|:---:|:---:|
| 传给 LLM 的内容 | 全部历史消息 | 压缩摘要文本 | 最近 6 条消息 |
| token 增长 | 线性增长（O(n)） | 恒定 | 恒定 |
| 额外 LLM 调用 | 0 | 每 N 轮 1 次（摘要） | 0 |
| 远距离上下文 | 保留 | 保留（仅摘要级） | 丢弃 |
| 适用对话长度 | < 10 轮 | 任意 | 5-20 轮 |
| 丢失风险 | 无丢失 | 摘要可能丢细节 | 窗口外的历史丢失 |
| 延迟影响 | 逐轮增长 | 摘要时 spike | 无影响 |

**代码中的实际行为**：

```python
# Day 2 代码 — demo_memory_comparison()
# 5 轮对话后的效果：

# Buffer:    历史消息数 = 10 条（5轮×2），token 持续增长
# Window(K=2): 传给 LLM 的历史 = 最近 4 条（2轮），老对话直接丢弃
# Summary:    传给 LLM 的 = 一段摘要文本（固定长度），不增长
```

**生产环境选型决策树**：

```
用户场景是什么？
├── 单次问答（不需要 Memory）
│   └── 不用任何 Memory 策略
├── 短对话（< 10 轮，如智能搜索）
│   └── BufferMemory — 简单直接，token 可控
├── 中长对话（10-30 轮，如客服场景）
│   └── WindowMemory(K=5) — 平衡上下文和成本
└── 长对话（> 30 轮，如长期陪伴）
    └── SummaryMemory — 唯一的可持续方案
        注意：摘要质量取决于 LLM 能力，弱模型可能丢关键信息
```

**面试话术**："选 Memory 策略的本质是在 token 成本、上下文完整度、响应延迟三者之间做平衡。生产环境最常见的方案是 Summary + Window 混合：用 Window 保留最近 3 轮完整对话，用 Summary 保留更早历史的摘要。这样近距离对话精确、远距离上下文不丢、token 可控。"

---

## Q3：`RunnableLambda` 是什么？它怎么把普通 Python 函数变成 LangChain 可串联的组件？

**一句话**：`RunnableLambda` 是一个适配器——把任何签名为 `dict → dict` 的 Python 函数包装成 Runnable 对象，获得 `.invoke()` / `.batch()` / `.stream()` 能力，可以和其他 Runnable 用 `|` 串联。

**代码中的体现**：

```python
# Day 2 代码 — build_rag_chain()
def step_rewrite(state: dict) -> dict:
    """Step ①: Query 改写"""
    rewritten = rewrite_query(state["question"], state.get("history", []))
    return {"rewritten": rewritten}  # 返回的部分会合并到 state

def step_retrieve(state: dict) -> dict:
    """Step ②: 两路并行检索"""
    dense = vector_index.search(state["rewritten"], top_k=top_k * 3)
    sparse = bm25_idx.search(state["rewritten"], top_k=top_k * 3)
    return {"dense_docs": dense, "sparse_docs": sparse}

# 把普通函数包装成 Runnable，然后用 | 串联
chain = (
    RunnableLambda(step_rewrite)     # dict → dict（加了 rewritten）
    | RunnableLambda(step_retrieve)  # dict → dict（加了 dense/sparse）
    | RunnableLambda(step_fusion)    # dict → dict（加了 hybrid_docs）
    | RunnableLambda(step_filter)    # ...
    | RunnableLambda(step_rerank)
    | RunnableLambda(step_assemble)
    | rag_prompt | llm | StrOutputParser()
)
```

**关键设计：state dict 传递模式**

每个 `step_*` 函数返回的 dict 会**合并**到上游的 state dict 中（类似 spread operator）。这让你可以逐步丰富 state：

```python
# 初始 state: {"question": "小米SU7续航", "history": []}
# step_rewrite 返回: {"rewritten": "小米SU7 续航里程"}
#   → state 变为: {"question": "小米SU7续航", "history": [], "rewritten": "..."}
# step_retrieve 返回: {"dense_docs": [...], "sparse_docs": [...]}
#   → state 变为: {..., "dense_docs": [...], "sparse_docs": [...]}
```

**面试话术**："`RunnableLambda` 是 LangChain 生态的'万能适配器'——任何已有的 Python 函数都能无缝融入 LCEL 管道。这让渐进式迁移成为可能：你可以先保留核心算法的纯函数实现（方便单独测试），用 `RunnableLambda` 包装后快速编排成管道。等管道稳定后，再考虑把关键步骤替换为更专用的 LangChain 组件。"

---

## Q4：自定义 `HybridRetriever`（继承 `BaseRetriever`）有什么价值？和直接用函数有什么区别？

**一句话**：继承 `BaseRetriever` 让你的检索器成为 LangChain 生态的一等公民——可以和其他组件用 `|` 串联，支持 `.invoke()` / `.batch()`，可以被 `EnsembleRetriever` 等工具消费。

**代码中的实现**：

```python
# Day 2 代码 — HybridRetriever
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_index, bm25_idx, reranker=None, top_k=3):
        super().__init__()
        self._vector_index = vector_index
        self._bm25_idx = bm25_idx
        self._reranker = reranker
        self._top_k = top_k

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # ① 两路检索 → ② RRF 融合 → ③ 过滤 → ④ Reranker → ⑤ 转 Document 格式
        dense = self._vector_index.search(query, top_k=self._top_k * 3)
        sparse = self._bm25_idx.search(query, top_k=self._top_k * 3)
        hybrid = hybrid_rrf(dense, sparse, k=60, top_k=self._top_k * 2)
        # ... 过滤 + 精排 ...
        return [Document(page_content=doc["content"], metadata={...}) for _, doc in hybrid]
```

**对比：函数 vs Retriever**：

| 维度 | 函数 `search(query) → list` | `BaseRetriever.invoke(query) → List[Document]` |
|------|---------|------|
| 与 LCEL 串联 | 需 `RunnableLambda` 包装 | 直接 `retriever \| prompt \| model` |
| 批量处理 | 需手动 for 循环 | `.batch()` 一行搞定 |
| 被其他组件消费 | 不能 | `EnsembleRetriever`、`ContextualCompressionRetriever` 等 |
| 返回类型 | 自定义元组 | `List[Document]`（标准格式） |

**面试话术**："继承 `BaseRetriever` 是'做 LangChain 生态的良民'——你的组件能被生态中的其他工具发现和使用。而返回 `Document` 而非自定义元组，意味着下游组件（如 Reranker、ContextCompressor）能直接消费。这就是框架生态的价值：你遵循约定，整个生态为你服务。"

---

## Q5：`RunnablePassthrough` 解决的问题是什么？什么场景下会用它？

**一句话**：当 prompt 需要多个输入源（如 context 来自检索、question 来自用户），而管道上下游只产出其中一部分时，`RunnablePassthrough` 把需要透传的字段原样穿过。

**代码中的体现**：

```python
# Day 2 代码 — demo_passthrough()
chain = (
    {
        "context": retriever,              # 检索器产出 context
        "question": RunnablePassthrough(),  # 用户问题原样穿过
    }
    | rag_prompt    # 需要 context + question 两个字段
    | llm
    | StrOutputParser()
)

result = chain.invoke("小米SU7的续航里程是多少")
# invoke 的输入 "小米SU7..." 被 RunnablePassthrough() 透传给 question 字段
```

**不用 RunnablePassthrough 的困境**：

```python
# ❌ 不用 RunnablePassthrough — 必须额外写函数透传
def add_question_to_state(state):
    # 但问题来了：question 从哪取？它不在 retriever 的输出里
    # 你得在链的最开始就通过其他方式保存原始 question
    ...

# ✅ 用 RunnablePassthrough — 一行解决问题
{
    "context": retriever,
    "question": RunnablePassthrough(),  # 直接从 invoke() 的输入参数拿
}
```

**典型场景**：

1. **RAG 链**：context（检索产出） + question（用户输入） → prompt
2. **多源融合**：某些字段来自 LLM 生成，某些来自外部 API，某些来自用户
3. **条件注入**：根据开关决定是否注入某个字段

**面试话术**："`RunnablePassthrough` 体现了 LangChain 设计中的一个核心洞察——数据流中有些字段是'生成'的（经过管道处理），有些是'透传'的（原封不动穿过）。把两者在同一个字典构造中显式区分，代码意图非常清晰。"

---

## Q6：Day 2 的原生 RAGAgent vs LCEL Chain 对比实验揭示了什么？

**一句话**：LCEL 链的代码更紧凑、每步可独立测试，但多了一层 Runnable 包装的微量开销。核心算法保持纯函数 + LCEL 做编排是最佳实践。

**对比实验的核心发现**：

```python
# Day 2 代码 — demo_comparison()
# 同一批问题，两种实现方式的结果：
# 1. 响应质量：几乎一致（底层都是同一套检索算法）
# 2. 延迟：LCEL 略高（RunnableLambda 包装的微量开销），实际可忽略
# 3. 代码量：LCEL 更紧凑，声明式管道一眼看出数据流
```

**架构建议**：

| 层 | 用什么 | 原因 |
|----|--------|------|
| 核心算法 | 纯 Python 函数 | 方便单元测试、无框架依赖 |
| 编排逻辑 | LCEL / LangGraph | 可视化数据流、方便扩展 |
| 框架组件 | BaseRetriever 等 | 融入生态、被其他工具消费 |

**面试话术**："Day 2 的对比实验不是证明'LCEL 更好'，而是证明'编排和算法应该解耦'。核心检索算法永远用纯函数实现（可以在任何框架中复用），编排逻辑用 LCEL 声明式表达（方便团队理解和修改）。这就是工程上的 separation of concerns。"

---

## Q7：Memory 的 session_id 是怎么管理的？多用户并发时如何避免历史串台？

**一句话**：`session_id` 是隔离键——每个 session_id 对应一个独立的对话历史存储。LangChain 用 `config["configurable"]["session_id"]` 传递，你负责实现 `get_session_history(session_id)` 的存储后端。

**代码中的机制**：

```python
# Day 2 代码 — BufferMemory 的 dict 存储
store_a: Dict[str, List] = {}  # {session_id: [messages]}

def get_history_a(session_id: str) -> List:
    return store_a.get(session_id, [])

chain_a.invoke(
    {"question": q},
    config={"configurable": {"session_id": "demo"}}  # ← 隔离键
)
```

**从 dict → Redis 的升级路径**：

```python
# 开发环境：dict
store = {}

# 生产环境：Redis（Day 13 会讲）
import redis
r = redis.Redis()
def get_history_redis(session_id: str):
    data = r.get(f"chat:{session_id}")
    return json.loads(data) if data else []
```

**并发安全考虑**：
- dict 存储：单进程安全，多进程不共享
- Redis 存储：天然支持多进程，但需要注意同 session 的并发写入（用分布式锁）
- 生产环境：session_id 应该来自认证系统（JWT / API Key），不能信任客户端传入

**面试话术**："`session_id` 本质上是一个 namespace——所有对话历史的读写都在这个命名空间内。它的设计哲学和 HTTP Session 完全一致。生产环境的三层考虑：存储后端选型（dict → Redis → DB）、过期清理（TTL）、并发写入防护（分布式锁）。"

---

## Q8：为什么 Day 2 要单独创建一个 `temperature=0` 的 `llm_stable` 用于 Summary？这个细节说明了什么？

**一句话**：Summary 需要确定性（每次总结同一段对话应该输出相似结果），而翻译/对话需要一定的创造性。用不同的 `temperature` 满足不同子任务的需求。

**代码体现**：

```python
# Day 2 代码
llm = ChatOpenAI(model="deepseek-chat", temperature=0.3)  # 翻译用：有一点灵活性

llm_stable = ChatOpenAI(model="deepseek-chat", temperature=0)  # 摘要用：严格确定
```

**temperature 在不同任务中的意义**：

| 任务 | temperature | 原因 |
|------|:---:|------|
| 对话/翻译 | 0.3-0.7 | 需要一定的表达多样性 |
| 摘要 | 0 | 必须稳定——同样对话不能每次总结出不同结论 |
| 结构化输出 | 0 | 格式必须严格一致 |
| 创意写作 | 0.8-1.2 | 需要多样性 |

**面试话术**："同一个模型、不同的 temperature 用于不同子任务——这是在 token 经济上的 micro-optimization。生产环境中，你可能同时并行 3 个不同 temperature 的调用：t=0 用于数据抽取（如价格查询）、t=0.3 用于常规回答、t=0.7 用于创意推荐。temperature 不是全局设置，是 per-call 的配置。"

---

### Day 2 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | RunnableWithMessageHistory 的数据流？它帮我们省了什么？ | □ |
| 2 | 三种 Memory 策略各有什么优劣？生产环境怎么选？ | □ |
| 3 | RunnableLambda 怎么把普通函数变成 Runnable？state dict 传递模式？ | □ |
| 4 | 继承 BaseRetriever 比直接用函数有什么价值？ | □ |
| 5 | RunnablePassthrough 解决什么问题？典型使用场景？ | □ |
| 6 | 原生 RAGAgent vs LCEL Chain 对比实验揭示了什么？架构建议？ | □ |
| 7 | session_id 的隔离机制？从 dict 到 Redis 的升级路径？ | □ |
| 8 | 为什么单独创建 temperature=0 的 llm_stable？temperature 的 per-task 配置？ | □ |
