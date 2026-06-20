# Day 5 面试题：ReAct Agent（汽车导购实战）

> 对应文件：`agent/car_advisor_agent.py`
> 核心能力：ReAct 模式、create_agent 框架、System Prompt 五段式、执行轨迹可视化、手写 vs 框架对比

---

## Q1：ReAct 和 Chain-of-Thought（CoT）的区别是什么？为什么 Agent 场景用 ReAct 而不是单纯的 CoT？

**一句话**：CoT 是"脑子里想"，ReAct 是"边想边做"。

| 维度 | CoT | ReAct |
|------|-----|-------|
| 推理方式 | 纯文本推理，一气呵成 | 推理 + 工具调用交替进行 |
| 信息来源 | 仅 LLM 内部知识 | LLM + 外部工具返回的真实数据 |
| 幻觉风险 | 高（闭门造车，数据可能过时） | 低（每次行动看真实反馈，及时纠偏） |
| 可纠正性 | 无法纠正（一次输出） | 可纠正（Observation 发现错误，下一步 Thought 调整策略） |
| 适用场景 | 数学/逻辑推理题 | 需要查数据库、调用 API 的实际应用 |

**具体例子**：用户问"小米 SU7 和 Model 3 哪个值得买"

- **CoT**：LLM 凭训练数据回忆参数来推理。可能过时（比如 SU7 刚调价）、可能错误（记错续航数字）、可能编造（训练时没有的型号）
- **ReAct**：LLM 先调用 `compare_cars("小米 SU7", "特斯拉 Model 3")` 拿到实时价格和参数，再基于真实数据做推理和推荐

**核心洞察**：CoT 是 LLM 的"内心独白"，ReAct 是 LLM 的"边做边想"。Agent 场景的本质不是推理本身，而是**推理 + 获取外部信息的能力**。

---

## Q2：`create_agent`（旧版 `create_react_agent`）内部帮你做了什么？如果不用它，你需要实现哪些组件？

**一句话**：它自动生成了一个 StateGraph，和你 Day 4 手写的图结构完全等价。

**你需要手写但 create_agent 帮你省掉的**：

1. **`AgentState` TypedDict 定义** — messages + add_messages reducer
2. **`call_model(state)` 节点函数** — 调用 LLM（带 tools binding）
3. **`should_continue(state)` 路由函数** — 判断最后一条消息是否有 tool_calls
4. **`ToolNode(tools)` 工具执行节点** — 解析 tool_calls、执行、返回 ToolMessage
5. **图构建**：
   ```python
   graph_builder = StateGraph(AgentState)
   graph_builder.add_node("agent", call_model)
   graph_builder.add_node("tools", ToolNode(tools))
   graph_builder.add_edge(START, "agent")
   graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
   graph_builder.add_edge("tools", "agent")
   app = graph_builder.compile()
   ```

**对比**：

| 组成部分 | Day 4 手写 | Day 5 create_agent |
|---------|-----------|-------------------|
| 状态定义 | `AgentState` TypedDict（3行） | 自动生成（等价） |
| 节点函数 | `call_model`（~3行） | 内置（等价） |
| 路由函数 | `should_continue`（~3行） | 内置（等价） |
| 图构建 | `add_node`×2 + `add_edge`×2 + 条件边（~10行） | 内置（等价） |
| 编译 | `compile()`（1行） | 内置（等价） |
| **总计** | **~30 行图代码** | **1 行** |

**验证方法**：分别输出 Mermaid 图，你会发现两张图**完全一样**。

```python
# Day 4 手写
print(app.get_graph().draw_mermaid())
# Day 5 create_agent
print(agent.get_graph().draw_mermaid())
# 两张图等价：START → agent_node ⇄ tools_node → END
```

**面试话术**："`create_agent` 不是黑魔法——它内部就是调用了 StateGraph API，生成的图和手写完全一样。框架的价值不是帮你少写代码，而是标准化了 Agent 的图结构，让团队所有人都用同一套模式。"

---

## Q3：你在 System Prompt 里是怎么防止 Agent 乱调工具的？有哪些策略？

**一句话**：五段式 Prompt 设计，每一段都有自己的防御职责。

**五段防御体系**：

```
┌─────────────────────────────────────────────┐
│ 段 1: 角色定义   → 设定专业边界，超出领域不回答    │
├─────────────────────────────────────────────┤
│ 段 2: 能力边界   → 明确列出可用工具，写清楚"能做什么"│
│                   和"绝对不能做什么"               │
├─────────────────────────────────────────────┤
│ 段 3: 决策准则   → 优先级规则：什么情况调哪个工具     │
│                   模糊条件 → 追问，不要猜测          │
├─────────────────────────────────────────────┤
│ 段 4: 输出约束   → 必须引用真实数据、推荐带理由       │
│                   限制推荐数量，防止信息过载          │
├─────────────────────────────────────────────┤
│ 段 5: 错误处理   → 工具返回失败时告知用户而非编造     │
│                   条件不足时主动追问至少 2 个维度      │
└─────────────────────────────────────────────┘
```

**具体策略**：

1. **能力边界明确**："你只了解数据库中已有的车型信息。如果用户问的车型不在你的知识范围内，**必须明确告知**，绝对不能编造参数或价格。"

2. **决策准则优先级**：
   - 用户给预算但没给车型 → 先用 `recommend_cars`，不要直接猜
   - 用户给车型 → 先查价再查参数
   - 用户比较两款 → 优先用 `compare_cars` 一次性对比
   - 用户问落地价 → 用 `calculate_ownership_cost`

3. **追问机制**："用户的预算描述模糊（如'不要太贵'）→ 先追问具体预算区间，不要猜测"

4. **负面示例**（进阶做法，在 Prompt 中给 1-2 个反面教程）：
   ```
   ❌ 错误：用户说"不要太贵"，你猜预算 20 万直接调 recommend_cars
   ✅ 正确：先问"您的预算大概什么范围？10-15万还是15-25万？"
   ```

---

## Q4：Agent 调错了工具怎么办？你有哪些排查和修复手段？

**一句话**：三层排查——打印执行轨迹定位问题 → 改工具描述修复根因 → 加防御层防止再犯。

**调试层（定位问题）**：

通过 `stream_agent_execution` 打印每一步的 Thought → Action → Observation：
```
💭 Thought #1: 用户想了解小米SU7的价格
  ⚡ Action: get_car_price(brand='小米', model='SU7')
  📊 Observation: {"car": "小米 SU7", "price": "21.59-29.99 万", "status": "found"}
💭 Thought #2: 已获取价格信息，整理回复...
```

通过这个轨迹你一眼就能看到：LLM 调了什么工具、传了什么参数、拿到了什么结果。如果走偏了（比如明明应该调 `compare_cars` 却调了 `get_car_price` 两次），马上能定位。

**修复层（根治问题）**：

| 问题类型 | 根因 | 修复 |
|---------|------|------|
| LLM 调了错误的工具 | 工具 docstring 描述不精确 | 在 docstring 写清"什么情况调用"，补具体示例 |
| 工具名有歧义 | 两个工具功能重叠 | 改名区分（如 `search_car` → `get_car_price`），或合并 |
| 参数传错了 | 参数名或类型描述不清 | 在 docstring 的 Args 段写清类型和示例 |
| LLM 看不到工具返回结果就继续调 | 结果格式不好理解 | 返回结构化 JSON + suggestion 字段 |

**防御层（防止再犯）**：

- System Prompt 决策准则段写清调用优先级
- 工具返回错误时携带建议：
  ```json
  {"error": "未找到", "suggestion": "尝试用 recommend_cars 模糊搜索"}
  ```
  而不仅仅是 `{"error": "未找到"}` ——让 LLM 知道"下一步该怎么办"

---

## Q5：手写 StateGraph 和用 `create_agent`，你在实际项目中选哪个？为什么？

**一句话**：标准场景用 `create_agent`，需要自定义节点或复杂路由时手写 StateGraph。我在项目里两种都写了，知道各自的适用边界。

**选 `create_agent` 的场景**：

- 标准 ReAct 循环，不需要额外节点
- 团队快速原型验证
- 工具数量和类型稳定，不需要复杂路由
- 代码量敏感，追求可维护性和团队一致性

**选手写 StateGraph 的场景**：

- 需要在 `agent → tools` 之间插入节点：安全审查、预算检查、审计日志
- 需要不同工具调用后走不同路线：调工具 A 进节点 X，调工具 B 进节点 Y
- 需要 Human-in-the-Loop 中断：调工具前暂停等待人工审批
- 需要非标准图结构：多 Agent 协作、子图嵌套
- 团队需要最大灵活性和可见性

**决策框架**：

```
需要加自定义节点/复杂路由？
  ├── 是 → 手写 StateGraph
  └── 否 → 用 create_agent
             └── 后面需要扩展？
                   ├── 是 → 从 create_agent 生成的图出发，加节点
                   └── 否 → 保持 create_agent
```

**面试能说的关键点**："我在 Day 4 手写了 StateGraph，理解了图结构；在 Day 5 用了 create_agent，理解了框架的封装价值。这两种经验让我在实际项目中能做正确的工程取舍，而不是盲目选一个。"

---

## Q6：ReAct Agent 的 Tool Calling 循环怎么防止无限循环？

**一句话**：三层防护——LLM 层面（Prompt 约束）+ 图层面（recursion_limit）+ 应用层面（重复调用检测）。

**层 1：LLM 层面 — Prompt 约束**：

在 System Prompt 中明确写：
> "如果你已经获得了足够信息来回答用户，请直接给出答案，不要继续调用工具。"

让 LLM 在 Thought 阶段就自我终止，而不是无脑继续调工具。

**层 2：图层面 — recursion_limit**：

LangGraph 默认 `recursion_limit=25`，超过抛出 `GraphRecursionError`。可以按需调整：

```python
# 编译时设置
app = graph_builder.compile()  # 默认 recursion_limit=25

# 或 invoke 时覆盖
app.invoke(messages, config={"recursion_limit": 10})
```

**层 3：应用层面 — 重复调用检测**：

```python
def safe_invoke(agent, messages, max_same_tool_calls=3):
    """带重复调用保护的 invoke"""
    try:
        result = agent.invoke(
            messages,
            config={"recursion_limit": 10}
        )
        return result
    except GraphRecursionError:
        return {"messages": messages["messages"] + [
            AIMessage(content="抱歉，我暂时无法完成这个查询，请换一种问法。")
        ]}
```

更细粒度的做法是在 `should_continue` 中统计连续调用同一工具且返回相同结果的次数，超过阈值就强制终止。

**为什么这个问题在实际面试中重要**：面试官想看你是否理解 Agent 是"不稳定的控制系统"——LLM 可能陷入循环、可能反复调同一个工具、可能永远不停。知道三层防护说明你做过生产级考虑。
