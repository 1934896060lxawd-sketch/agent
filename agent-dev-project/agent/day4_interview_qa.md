# Day 4 面试题：LangGraph StateGraph 核心

> 对应文件：`agent/langgraph_agent.py`
> 核心能力：StateGraph 三要素、ToolNode 自动执行、条件边路由、MemorySaver 持久化

---

## Q1：LangGraph 的 StateGraph 是怎么工作的？State、Node、Edge 分别是什么？

**一句话**：StateGraph 把 Agent 建模成有向图，State 是流动的数据，Node 是操作，Edge 决定数据流向。

**展开**：

| 要素 | 含义 | 代码体现 |
|------|------|---------|
| **State** | 在节点间流动的数据，用 `TypedDict` 定义 | `class AgentState(TypedDict): messages: Annotated[list, add_messages]` |
| **Node** | 一个纯函数，接收 State 返回部分 State | `def call_model(state) → {"messages": [response]}` |
| **Edge** | 分两种：普通边（固定路由 A→B）和条件边（根据函数返回值决定 A→B 还是 A→C） | `add_edge(START, "agent")` / `add_conditional_edges(...)` |

**关键细节**：`add_messages` 是一个 reducer，控制新旧状态如何合并——新消息追加到列表末尾，而不是覆盖。没有它，每个节点返回的新 messages 会覆盖旧的，历史对话全部丢失。

**图循环的执行轨迹**（一次 `invoke` 内部发生了什么）：
```
invoke({"messages": [HumanMessage("小米SU7续航多少")]})

→ START → "agent" 节点
    llm_with_tools.invoke(messages)
    → AIMessage(tool_calls=[get_car_price("小米","SU7")])

→ 条件边判断: should_continue → 有 tool_calls → 返回 "tools"

→ "tools" 节点
    ToolNode 执行 get_car_price
    → ToolMessage(content="小米 SU7 | 售价 21.59-29.99万 | 续航 700-830km")

→ 普通边: tools → agent（回到 agent）

→ 再次 "agent" 节点
    llm 看到 ToolMessage → 生成最终回答
    → AIMessage(content="小米SU7的续航...")

→ 条件边判断: should_continue → 无 tool_calls → 返回 "__end__"
→ END
```

**执行了 2 次 agent、1 次 tools、2 次条件边判断。没有 while True，图自动完成循环。**

---

## Q2：LangGraph 的 ToolNode 内部做了什么？如果让你自己实现一个 ToolNode，你会怎么设计？

**一句话**：ToolNode 自动解析 AIMessage 中的 tool_calls、执行对应函数、返回 ToolMessage。

**ToolNode 内部四步**：

1. 取最后一条消息，检查 `hasattr(msg, "tool_calls")`
2. **并行执行**所有 tool_calls（无依赖关系的工具不应该串行等）
3. 每个工具执行结果包装成 `ToolMessage(content=result, tool_call_id=tc["id"])`
4. 返回 `{"messages": [tool_msg_1, tool_msg_2, ...]}`，由 `add_messages` reducer 自动追加

**自己实现的要点**：

```python
class MyToolNode:
    def __init__(self, tools: list):
        self._tool_map = {t.name: t for t in tools}

    def __call__(self, state):
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            tool = self._tool_map[tc["name"]]
            try:
                args = json.loads(tc["arguments"])    # ① 参数校验：JSON 字符串 → dict
                result = tool.invoke(args)             # ② 执行
            except Exception as e:
                result = f"工具执行失败: {e}"          # ③ 异常隔离：单工具失败不影响其他
            results.append(ToolMessage(
                content=str(result)[:4000],            # ④ 结果截断：防止撑爆上下文
                tool_call_id=tc["id"],
            ))
        return {"messages": results}
```

**面试加分点**：提到并行执行（`ThreadPoolExecutor`）、超时控制（`signal.alarm` 或 `concurrent.futures.TimeoutError`）、结果截断（防止 token 爆炸）。

---

## Q3：LangGraph 和图灵完备的关系——为什么用图而不是链？

**一句话**：链（Chain）是 DAG（有向无环图），只能单向走；Agent 需要循环（有环图），LangGraph 的 StateGraph 天然支持环和条件分叉。

**展开**：

- LCEL 的 `|` 管道是**单向的**：数据从左流到右，中间不会回到起点
- Agent 的核心模式是"LLM 思考 → 调工具 → LLM 再思考 → 可能再调工具"——这是一个**循环**
- LangGraph 通过 `tools → agent` 的回边实现了这个循环
- `should_continue` 条件边控制了**何时跳出循环**（无 tool_calls → END）

**图比链的表达能力强在哪里**：

```python
# 可以自由加节点，不改变原有逻辑：
graph_builder.add_node("security_check", check_tool_params)   # 安全检查节点
graph_builder.add_node("audit_log", log_tool_call)            # 审计日志节点
# 插入到 tools → agent 之间，形成 tools → security → audit → agent
```

链做不到这种"中间插入"，因为链只有一条道走到底。

---

## Q4：`add_messages` reducer 的作用是什么？不用它会怎样？

**一句话**：`add_messages` 是一个 reducer，控制新状态和旧状态如何合并——新消息追加到列表末尾，而不是覆盖。

**对比代码**：

```python
# ✅ 有 add_messages：
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
# 行为：state["messages"] = [msg1, msg2]
#       node 返回 {"messages": [msg3]}
#       → 新 state = [msg1, msg2, msg3]  ← 追加

# ❌ 没有 add_messages（默认 reducer = 覆盖）：
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
# 行为：state["messages"] = [msg1, msg2]
#       node 返回 {"messages": [msg3]}
#       → 新 state = [msg3]  ← 覆盖！msg1/msg2 丢了！
```

**后果**：不加 `add_messages`，历史对话全部丢失，LLM 永远只看到最后一条消息，完全丧失多轮对话能力。

---

## Q5：Checkpointer（MemorySaver）和 LangChain 的 Memory 有什么区别？用哪个？

**一句话**：Memory 管的是"传什么给 LLM"，Checkpointer 管的是"图执行到哪一步的状态持久化"。

| 维度 | LangChain Memory | LangGraph Checkpointer |
|------|-----------------|----------------------|
| 工作层面 | Prompt 层——控制哪些历史注入到 prompt | 图层——保存每一步的完整 state |
| 粒度 | 轮次级别 | 节点级别（每个 node 执行完都可能存） |
| 恢复能力 | 重跑时加载历史 | 图执行中断后，从最近的成功节点恢复 |
| 适用场景 | 简单多轮对话 | Agent 循环 + 工具调用 + 中断/人机协同 |

**代码体现**：

```python
# LangChain Memory（Day 2）
chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="input"
)

# LangGraph Checkpointer（Day 4）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user_001"}}
app.invoke({"messages": [HumanMessage("推荐一款20万SUV")]}, config=config)
app.invoke({"messages": [HumanMessage("刚才那款续航多少")]}, config=config)
# ↑ 同一 thread_id，LangGraph 自动从 MemorySaver 取出第1轮的 messages
```

**选型建议**：LangGraph Agent 用 Checkpointer。普通 Chain（无循环/无复杂路由）用 Memory。

---

## Q6：你在项目中怎么调试 LangGraph Agent 的执行过程？Agent 调了错误的工具怎么排查？

**一句话**：三层调试——打印 messages 列表追踪每步状态 + 条件边返回日志 + LangSmith/LangFuse 可视化回放。

**层 1：invoke 后打印完整 messages 链条**：

```python
result = app.invoke({"messages": [HumanMessage(query)]})
for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"[{i}] {msg_type} → tool_calls: {[tc['name'] for tc in msg.tool_calls]}")
    else:
        print(f"[{i}] {msg_type} → {str(msg.content)[:100]}...")
```

**层 2：在 should_continue 里加日志**：

```python
def should_continue(state):
    last = state["messages"][-1]
    has_tools = hasattr(last, "tool_calls") and last.tool_calls
    print(f"[DEBUG] should_continue: has_tool_calls={has_tools}")
    return "tools" if has_tools else "__end__"
```

**层 3：生产环境接 LangSmith** — 每次 invoke 自动记录完整 trace，出问题可以回放每一步的 state。

**排查工具调用错误**：看 messages 列表中的 ToolMessage——如果工具返回了 `"未找到 XX"` 或报错信息，而 LLM 下一步还在追问同一个问题，说明**工具描述和实际行为不一致**（LLM 以为工具能做某件事，实际不能）。

**修复策略**：
1. 改工具的 docstring，明确写清楚"什么情况会返回什么"
2. 工具返回错误时携带建议，如 `{"error": "未找到", "suggestion": "尝试用 recommend_cars 模糊搜索"}`
3. 在 System Prompt 中写清楚调用优先级
