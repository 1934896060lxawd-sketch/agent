# Day 3 面试题：原生 SDK Function Calling

> 对应文件：`agent/function_calling_raw.py`
> 核心能力：Tool 定义（JSON Schema）、Tool Calling 循环、并行调用、tool_choice 三种模式、错误处理、与 LangChain/LangGraph 的关系

---

## Q1：Tool 的 JSON Schema 定义中，哪些字段最关键？写不好会有什么后果？

**一句话**：`name`（函数标识）、`description`（调用时机）、`parameters.properties` 及其子字段的 `description`（参数语义）——三者缺一不可，任何一个写得不好都会导致 LLM 错误调用或不调用。

**关键字段详解**：

```python
# Day 3 代码 — TOOLS 定义
{
    "type": "function",
    "function": {
        "name": "get_car_price",           # ① LLM 用它指定要调哪个工具
        "description": (                    # ② 告诉 LLM "什么时候该调这个工具"
            "查询指定车型的最新市场指导价。"
            "当用户询问价格、预算、多少钱、贵不贵时调用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brand": {
                    "type": "string",
                    "description": "品牌名称，如'小米'、'比亚迪'、'特斯拉'",  # ③ 告诉 LLM 这个参数应该填什么
                },
                "model": {
                    "type": "string",
                    "description": "车型名称，如'SU7'、'海豚'、'Model Y'",
                },
            },
            "required": ["brand", "model"],  # ④ 标记必填字段
        },
    },
}
```

**写不好的后果**：

| 问题 | 后果 | 例子 |
|------|------|------|
| `description` 没写"何时调" | LLM 该调时不调 | 用户问"多少钱"，LLM 直接编价格 |
| `parameters` 的 `description` 缺失 | LLM 填错参数值 | brand 填成"小鹏G6"（车名当品牌） |
| `required` 漏标 | LLM 漏传关键参数 | 不传 model 参数导致查不到 |
| `type` 写错 | LLM 可能填错类型 | number 写成 string，"20万"当 20 |

**面试话术**："Tool 的 JSON Schema 本质上是给 LLM 看的 API 文档——`description` 就是文档里'何时使用此 API'和'参数含义'的说明。很多团队工具定义写得很随意，导致 LLM 该调工具时不调、调了又填错参数——根因就在 Schema 设计的细节上。和 Day 8 的 Pydantic Field description 是同一个道理。"

---

## Q2：手写 Tool Calling 循环的 `while True` 逻辑是怎样的？终止条件是什么？

**一句话**：核心是 `while True` 循环——每轮调 LLM，检查返回的是 `content`（直接回答）还是 `tool_calls`（想调工具）。如果是 tool_calls，执行工具并把结果以 `role="tool"` 追加回 messages，继续循环。

**完整循环流程**（Day 3 最核心的代码）：

```python
# Day 3 代码 — chat_with_tools() 的核心循环
messages = [
    {"role": "system", "content": "你是汽车导购助手..."},
    {"role": "user", "content": user_message},
]

while turn < max_turns:
    turn += 1
    
    # ① 调 LLM（带 tools 参数）
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,  # auto / required / none
        temperature=0,
    )
    
    msg = response.choices[0].message
    
    # ② 终止条件：LLM 直接输出了文本，且没有 tool_calls
    if msg.content and not msg.tool_calls:
        return {"answer": msg.content, "turns": turn}
    
    # ③ LLM 想调工具
    if msg.tool_calls:
        # ④ 把 assistant 消息（含 tool_calls）追加到 messages
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })
        
        # ⑤ 逐个执行工具
        for tc in msg.tool_calls:
            func_args = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, func_args, knowledge_index)
            
            # ⑥ 把工具结果以 role="tool" 追加回 messages
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        
        continue  # ← 回到 while 顶部，LLM 看到工具结果后决定下一步
```

**三种终止路径**：

| 路径 | 条件 | 说明 |
|------|------|------|
| 直接回答 | `msg.content and not msg.tool_calls` | LLM 觉得不需要工具，直接回答 |
| 工具后回答 | 执行工具 → 下一轮 LLM 返回 content | 查了数据后给出答案 |
| 超限 | `turn >= max_turns` | 防止死循环，返回兜底话术 |

**面试话术**："手写的 while True 循环让你完全掌控每一步——什么时候调 LLM、什么时候执行工具、什么时候终止。Day 4 的 LangGraph 其实就是把这个循环建模成了有向图：状态节点（调 LLM / 执行工具）+ 条件边（判断终止还是继续）。理解了 Day 3 的手写循环，再看 LangGraph 的 StateGraph 就是'声明式的循环'。"

---

## Q3：并行 Tool Calling 是什么？什么条件下可以并行？代码里怎么实现的？

**一句话**：当 LLM 一次返回多个 `tool_calls` 且它们互不依赖时，可以并行执行。这比串行执行省掉了 N-1 次 LLM 调用的延迟。

**代码中的体现**：

```python
# Day 3 代码 — demo_parallel_tool_calls()
# 用户问："小米SU7的价格和续航里程分别是多少"
# LLM 返回 2 个 tool_calls:
#   1. get_car_price(brand="小米", model="SU7")
#   2. search_car_knowledge(query="小米SU7续航里程")
# → 两个工具互不依赖，可以并发执行！

for tc in msg.tool_calls:  # 逐个执行（Day 3 的实现是串行的）
    func_args = json.loads(tc.function.arguments)
    result = _execute_tool(tc.function.name, func_args, knowledge_index)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

**并行判断条件**：

```
可以并行 ⇔ 工具 A 的输入不依赖工具 B 的输出
         ⇔ 工具调用的 arguments 中不包含其他工具的结果
         ⇔ LLM 在一次响应中同时发出多个 tool_calls（说明它认为互不依赖）
```

**生产级并行实现**（Day 3 代码的升级版）：

```python
# Day 3 是串行执行，生产环境可升级为线程池并行
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=len(msg.tool_calls)) as executor:
    futures = {
        executor.submit(_execute_tool, tc.function.name, 
                       json.loads(tc.function.arguments), knowledge_index): tc
        for tc in msg.tool_calls
    }
    for future in as_completed(futures):
        tc = futures[future]
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": future.result(),
        })
```

**面试话术**："并行 Tool Calling 是 Agent 性能优化的关键——一次 LLM 调用中返回的所有 tool_calls 都可以并发执行。这个优化在生产环境中把 5 工具调用的延迟从 sum(T1..T5) 降到 max(T1..T5)。LLM 本身天然知道哪些工具互不依赖（否则它不会在一次响应中返回），代码只需要响应式地并行执行即可。"

---

## Q4：`tool_choice` 的三种模式（auto / required / none）有什么区别？各自的适用场景？

**一句话**：auto 让 LLM 自主判断（灵活但不可控），required 强制 LLM 必须调工具（严格但需要额外处理），none 禁止 LLM 调工具（纯文本回答）。

**三种模式的行为差异**：

| 维度 | `auto` | `required` | `none` |
|------|--------|-----------|--------|
| LLM 能直接回答吗 | ✅ 可以 | ❌ 不能 | ✅ 可以 |
| LLM 能调工具吗 | ✅ 可以 | ✅ 必须调 | ❌ 不能 |
| while 循环能终止吗 | ✅ msg.content 出现即终止 | ❌ 永远不会自然终止 | ✅ 不用进循环 |
| 适用场景 | 通用对话 | "先查后答" | 纯闲聊/无工具场景 |

**`tool_choice="required"` 的特殊处理**（Day 3 代码的关键设计）：

```python
# Day 3 代码 — required 模式需要额外步骤
if tool_choice == "required":
    # 工具执行完后，不能继续 while 循环（LLM 永远不能输出 text）
    # 正确做法：用 tool_choice="none" 再调一次 LLM，生成最终回答
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        # 不传 tools 参数 → LLM 只能用文本回答
        temperature=0,
    )
    return {"answer": final_response.choices[0].message.content}
```

**选型决策**：

```
用户对话场景？
├── 通用助手（可能查数据、可能闲聊）
│   └── tool_choice="auto" — 默认，LLM 自主判断
├── 严格 RAG（每问必查，不准凭记忆）
│   └── tool_choice="required" — 强制先查再答
├── 纯闲聊（打招呼、情感支持）
│   └── tool_choice="none" — 省掉 tools 参数的 token
└── 混合场景
    └── 第一轮 auto（让 LLM 判断），如果返回 answer 为空
        → 第二轮 required（强制查数据）
```

**面试话术**："`tool_choice` 本质上是控制 LLM 的'行为自由度'。auto 是最大自由度（适合大多数场景），required 是零自由度（适合需要硬保证的场景如'回答必须基于数据库'），none 是完全不涉及工具的路径。这个参数和 Day 8 的 `response_format` 在设计哲学上一致——都是通过 API 参数约束 LLM 的输出空间。"

---

## Q5：Tool Calling 中的错误处理是怎么做的？为什么要把错误信息回传给 LLM？

**一句话**：工具执行失败时，返回包含 `error` 字段的 JSON（而不是抛异常），让 LLM 看到错误后调整策略——告知用户"未找到"或尝试修正参数。

**代码中的错误处理链路**：

```python
# Day 3 代码 — _execute_tool() 的错误处理
def _execute_tool(tool_name: str, arguments: dict, knowledge_index) -> str:
    try:
        if tool_name == "get_car_price":
            return _execute_get_car_price(arguments["brand"], arguments["model"])
        # ...
    except Exception as e:
        # 错误也作为 tool response 返回给 LLM！
        return json.dumps({"error": f"工具执行失败: {str(e)}"}, ensure_ascii=False)

# _execute_get_car_price 查不到时的返回
def _execute_get_car_price(brand: str, model: str) -> str:
    key = f"{brand} {model}"
    if key in CAR_PRICE_DB:
        return json.dumps({"car": key, "price": CAR_PRICE_DB[key], "status": "found"})
    # 查不到 ≠ 抛异常，返回 status="not_found"
    return json.dumps({
        "car": key, "price": None, "status": "not_found",
        "message": f"未找到 {key} 的价格信息，请尝试其他关键词",
    })
```

**为什么要把错误回传给 LLM**：

```
工具执行失败（如查不到法拉利SF90的价格）
  → 回传: {"status": "not_found", "message": "未找到..."}
  → LLM 看到后能告知用户："抱歉，目前数据库中暂无法拉利SF90的价格信息"
  → 而非编造一个价格

如果直接抛异常（不传给 LLM）：
  → LLM 不知道工具执行失败了
  → LLM 要么等待超时，要么基于不完整信息编造答案
```

**面试话术**："Function Calling 的防幻觉核心就是——让 LLM 看到错误而非隐藏错误。传统编程思维是'遇到错误抛异常'，但在 Agent 场景中，错误信息是给 LLM 做决策的重要信号。Day 3 的错误处理设计体现了'把 LLM 当成协作方而非调用方'的思维方式。"

---

## Q6：Tool Calling 循环和 Day 4 LangGraph StateGraph 是什么关系？两者如何对应？

**一句话**：Day 3 的 while True 循环 = Day 4 的 StateGraph。while 循环中的"判断 → 调 LLM → 执行工具 → 回传"被建模为图中的节点（agent / tools）和条件边（should_continue）。

**映射关系**：

```
Day 3 (命令式 while 循环)          Day 4 (声明式 StateGraph)
═══════════════════════════        ═══════════════════════
while turn < max_turns:     ←→    图编译后的自动循环
  response = client.create() ←→    agent 节点（call_model）
  if msg.content: break      ←→    should_continue → END
  if msg.tool_calls:         ←→    should_continue → tools 节点
    execute tools            ←→    ToolNode 自动执行
    messages.append(result)  ←→    边自动传递 state
    continue                 ←→    边 tools → agent（循环）
```

**代码量对比**：

```python
# Day 3: ~80 行手写循环
while turn < max_turns:
    response = client.chat.completions.create(...)
    if msg.content and not msg.tool_calls: ...
    if msg.tool_calls:
        for tc in msg.tool_calls: ...
        if tool_choice == "required": ...
    # ... 各种判断、messages 拼接、错误处理

# Day 4: ~10 行声明式图
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")
app = workflow.compile(checkpointer=MemorySaver())
```

**面试话术**："Day 3 → Day 4 的递进是'理解框架帮你省了什么'的典型案例。手写循环让你理解每一步的细节（messages 如何拼接、tool_calls 如何解析、tool_choice 如何影响循环逻辑），LangGraph 让你在生产环境中不重复造轮子。能说清两者的对应关系，说明你既懂原理又懂工程。"

---

## Q7：多轮 Tool Calling 编排（先筛选→再对比→再推荐）是怎么实现的？LLM 如何"自主决定"调用顺序？

**一句话**：LLM 每轮看到当前 messages（含之前所有工具调用的结果）后，自主判断下一步该调哪个工具。代码不预设调用顺序——LLM 自己编排。

**Day 3 综合演示的完整过程**：

```
用户: "预算20万以内，推荐一款性价比最高的SUV"

第 1 轮:
  LLM → tool_calls: [filter_by_budget(max_price=20, category="SUV")]
  执行 → 返回: {matches: [{零跑C11: 15.58-19.98万}, {埃安Y: 11.98-18.98万}]}

第 2 轮:
  LLM 看到返回列表 → tool_calls: [compare_cars(car1="零跑C11", car2="埃安Y")]
  执行 → 返回: {car1: {...}, car2: {...}}

第 3 轮:
  LLM 看到对比数据 → content: "推荐零跑C11，空间更大、配置更高..."
  → content 且无 tool_calls → 循环终止
```

**关键洞察**：代码只定义了 tools 和执行逻辑，没有定义"先调 filter → 再调 compare → 最后推荐"的顺序。这个顺序是 LLM 根据当前上下文自主决定的。这就是 Agent 和传统 if-else 规则引擎的本质区别。

**面试话术**："多轮 Tool Calling 编排是 Agent 智能的核心体现——LLM 像一个项目经理，根据当前掌握的信息决定下一步该做什么。代码的角色从'编排者'变成了'能力提供者 + 执行者'。这正是 ReAct 框架的理论基础：Reasoning（LLM 思考）→ Action（代码执行）→ Observation（结果反馈）→ 循环。"

---

## Q8：Day 3 用原生 SDK、Day 4 用 LangGraph、Day 5 用 `create_react_agent`——这三层抽象分别帮我们省了什么？

**一句话**：Day 3 让你理解每一个字节的流动（完全控制），Day 4 把循环建模为图（可观测可恢复），Day 5 把图封装成一行 API（最快开发）。

**三层对比**：

| 维度 | Day 3: 原生 SDK | Day 4: LangGraph | Day 5: create_react_agent |
|------|:---:|:---:|:---:|
| 代码量 | ~80 行（手写循环） | ~30 行（图定义） | ~3 行（一行 API） |
| 控制力 | 100%（每步可控） | 80%（图节点可控） | 30%（只能调参数） |
| 可视化 | 无（需脑补） | Mermaid 流程图 | Mermaid 流程图 |
| 可恢复性 | 无（中断=丢失） | Checkpointer 自动 | Checkpointer 自动 |
| 学习价值 | ★★★★★ 理解本质 | ★★★★ 理解架构 | ★★ 会用 API |
| 生产推荐 | 原型验证 | ★ 推荐 | 简单场景可用 |

**渐进式学习路径**：

```
Day 3 (原生 SDK)
  └→ 理解 tool_calls 的 JSON 结构
  └→ 理解 messages 拼接的每一步
  └→ 理解 tool_choice 如何影响循环
      │
      ▼
Day 4 (LangGraph)
  └→ 把 while 循环建模为 StateGraph
  └→ 用条件边替代 if 判断
  └→ 用 ToolNode 替代手动执行
  └→ 加上 Checkpointer 持久化
      │
      ▼
Day 5 (create_react_agent)
  └→ 一行代码搞定 Day 4 的整张图
  └→ 但不妨碍你在需要时降级到手写图
```

**面试话术**："Day 3 → 4 → 5 的递进反映了一个核心工程原则：'先理解底层，再用好上层抽象'。很多面试者对 `create_react_agent` 用得很熟，但一问 tool_calls 的 JSON 结构、messages 的 role 类型、tool_choice 如何影响循环逻辑，就答不上来。Day 3 的手写循环就是填这个坑——面试官要的不是'你会用 LangChain'，而是'你理解 LangChain 背后发生了什么'。"

---

### Day 3 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | Tool JSON Schema 三个最关键字段？写不好会有什么后果？ | □ |
| 2 | Tool Calling 循环的 while True 逻辑？终止条件有哪三种？ | □ |
| 3 | 并行 Tool Calling 的触发条件？生产环境怎么实现？ | □ |
| 4 | tool_choice 三种模式的行为差异和适用场景？required 为什么要额外处理？ | □ |
| 5 | 工具执行失败后为什么把错误回传 LLM 而不是抛异常？ | □ |
| 6 | Day 3 的 while 循环和 Day 4 的 StateGraph 如何对应？ | □ |
| 7 | 多轮 Tool Calling 编排中，LLM 如何"自主决定"调用顺序？ | □ |
| 8 | Day 3/4/5 三层抽象分别帮我们省了什么？渐进式学习路径？ | □ |
