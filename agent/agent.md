# 从零手写 Agent：踩坑实录

> 一份给初学者的诚实记录——每个 bug 都是自己撞出来的。

---

## 1. 工具定义的 JSON Schema 为什么那么深

**踩坑**：第一次看到 `tools` 的四层嵌套觉得莫名其妙。

**理解**：每一层都有语义，不是套娃。

```python
tools =  [                          # ① 数组——可以传多个工具
  {
    "type": "function",             # ② 类型声明——告诉 API 这是函数工具
    "function": {                   # ③ 函数的"说明书"
      "name": "calculator",         # ④ 函数名——模型返回这个名表示要调它
      "description": "……",          # ⑤ 描述——模型靠这段文字决定调不调
      "parameters": {               # ⑥ JSON Schema——告诉模型参数长什么样
        "type": "object",
        "properties": { ... },      # ⑦ 每个参数的类型和说明
        "required": [...]
      }
    }
  }
]
```

**关键认知**：`parameters` 里面的 `type`、`properties`、`required` 是 JSON Schema 标准，不是 OpenAI 发明的。LLM 训练数据里有大量 JSON Schema，所以天然能理解。

**教训**：遇到"看起来复杂的嵌套"，先问自己"每一层回答了什么不同的问题"。如果两层回答的是同一件事，那才叫设计问题。

---

## 2. `eval()` 是定时炸弹——哪怕加了正则

**踩坑**：写了个正则 `^[\d+\-*/()\s.]+$` 觉得安全了，模型说支持"根号"，但 `sqrt(4)` 里的 `sqrt` 全是字母，正则直接拦。

更致命的是——`__import__('os').system('rm -rf /')` 全是字母、括号、引号、点。正则全放行。

**修法**：用 `ast` 模块做白名单解析，不在白名单的语法直接拒绝。

```python
import ast
import operator

ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def safe_calc(expr: str) -> str:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return "表达式语法错误"

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的操作符: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的操作符: {type(node.op).__name__}")
            return op(_eval(node.operand))
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    try:
        return str(_eval(tree))
    except (ValueError, ZeroDivisionError) as e:
        return f"计算错误: {e}"
```

**教训**：正则表达式不是安全边界。对外部输入做校验，用白名单不要用黑名单。

---

## 3. `messages.append(msg)` 是隐式兼容——不可靠

**踩坑**：直接把 Pydantic 对象 `msg` 塞进 list。能跑是因为 OpenAI SDK 内部做了 `model_dump()`，但这是隐式行为。

**修法**：

```python
# 错误
messages.append(msg)

# 正确
messages.append(msg.model_dump())
```

**教训**：框架的"隐式兼容"不写在文档里就可能下个版本消失。显式转换。

---

## 4. 变量名混淆——`result` 和 `tool_result` 傻傻分不清

**踩坑**：在工具路由里定义了 `tool_result = ""`，计算结果却写给了 `result`，搜索也一样。最后塞回 messages 的是空字符串。

```python
# 错误
tool_result = ""         # ← 初始化
if tool_name == "calculator":
    result = safe_calc()   # ← 写给了另一个变量
    # tool_result 还是 ""

messages.append({"role": "tool", "content": tool_result})  # ← 空字符串
```

**修法**：一个分支里只用一个变量名。

```python
if name == "calculator":
    tool_result = safe_calc(args["expression"])
elif name == "search":
    tool_result = external_search(args["query"])
else:
    tool_result = f"未知工具: {name}"

messages.append({"role": "tool", "content": tool_result})
```

**教训**：变量名定了就用到底，别在分支里"顺便"换个名字。

---

## 5. 工具参数名不一致——`expression` vs `query`

**踩坑**：工具定义里搜索参数叫 `expression`，执行时读的是 `args["query"]`。`KeyError`。

```python
# 工具定义
search_params = {"expression": {"type": "string", ...}}
search = build_function_tool(..., required=["expression"])

# 执行时
args["query"]   # ← 模型返回的字段名是 expression，KeyError
```

**修法**：参数名反映语义。搜索词的语义是"查询词"，不该复用计算器的"表达式"。

```python
# 统一成 query
search_params = {"query": {"type": "string", "description": "搜索关键词或问题"}}

# 解析时
if name == "calculator":
    expr = args["expression"]
elif name == "search":
    query = args["query"]
```

**教训**：参数名应该有业务含义。不同工具的参数不要复用同一个名字，否则模型把"搜索词"传给"数学表达式"就是灾难。

---

## 6. 外部 API 调用没做异常处理

**踩坑**：`tavily.search()` 网络挂了直接抛异常，整个 Agent 循环崩溃。`resp["results"]` 如果 API 返回错误格式，`KeyError`。

**修法**：

```python
def external_search(query: str) -> str:
    try:
        resp = tavily.search(query=query, search_depth="basic", max_results=3)
        results = resp.get("results", [])
        if not results:
            return f"未搜索到关于「{query}」的相关内容"
        parts = []
        for i, item in enumerate(results):
            title = item.get("title", "无标题")
            snippet = item.get("content", "")[:300]
            parts.append(f"[{i+1}] {title}\n{snippet}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"搜索失败: {e}"
```

**四个改进点**：
- `try/except` 兜住所有异常，把错误变成字符串塞回 messages，Agent 不会崩
- `resp.get("results", [])` 用 `.get()` 而不是 `["results"]`，防止 KeyError
- `item.get("title", "无标题")` 同上
- `[:300]` 截断，防止搜索结果太长撑爆 context window

**教训**：Agent 调用外部工具时，**工具返回的永远是字符串**。所有异常在工具内部消化，别让它炸到 Agent 循环。

---

## 7. System prompt 写"禁止跳过工具"太粗暴

**踩坑**：写了"不能跳过工具直接回复"，导致 `1+1` 这种模型可以直接答的问题也要调一次 calculator。浪费一轮 API 调用。

**修法**：按类别做约束，而不是一刀切。

```python
"你是严谨的助手。规则如下：\n"
"1. 数学计算题 → 调 calculator\n"
"2. 知识/实时信息类问题 → 必须先调 search，拿到结果再作答，禁止编造\n"
"3. 简单闲聊或已有明确答案的问题可直接回复"
```

**教训**：System prompt 是约束模型行为的第一道防线，但约束应该精确。不加区分的"禁止"会让 Agent 变蠢。

---

## 8. Vibe Coding 的边界——什么时候该手写，什么时候该用 AI

**最大的踩坑**：这个问题本身。

**结论**：

| 阶段 | 做法 | 为什么 |
|---|---|---|
| 学习期（第 1-2 个月） | 手写为主，AI 当 mentor | 你需要建立"肌肉记忆" |
| 工作期（2 个月后） | Vibe 为主，手动做架构决策 | 你已经能识别 AI 的错误 |

**手写至少一遍能让你获得**：
- 理解 `messages` 数组的增长过程
- 理解 Agent 循环的终止条件
- 理解 LLM 只输出 JSON、执行是你做的
- 能 debug tool call 失败（而不是把 error 贴回给 AI）

**Vibe coding 用对的方式**：不是"帮我写一个 Agent"，而是"帮我把这个 AST 计算器接进 calculator 分支，只改这一块，其他地方不要动"。

**教训**：手写给你的是"知道我不知道什么"的能力。Vibe coding 给你的是速度。顺序不能反。

---

## Agent 的核心认知：它只是一个 while 循环

```python
for turn in range(MAX_TURNS):
    response = llm.chat(messages, tools)

    if not response.tool_calls:
        break    # 模型觉得可以回答了

    for each tool_call:
        result = execute(tool_call)
        messages.append(result)

# 最终回答
print(response.content)
```

搞懂这 10 行代码，就是 Agent 的"第一性原理"。框架只是在这上面加了状态管理、错误恢复、流式输出——核心还是这个循环。

---

## 修改清单（代码到可交付状态的 7 项）

| # | 检查项 | 为什么重要 |
|---|---|---|
| 1 | `eval()` → AST 解析 | 安全 |
| 2 | `msg.model_dump()` | 显式转换，不依赖框架隐式行为 |
| 3 | Tavily 加异常处理 + `.get()` + 截断 | 外部调用不炸 Agent |
| 4 | 工具参数名有语义（`query` vs `expression`） | 模型和代码的契约清晰 |
| 5 | 变量名统一（`tool_result` 一个名字用完） | 不出 bug |
| 6 | System prompt 按类别约束，不一刀切 | Agent 不浪费调用 |
| 7 | 搜索结果加 `[1] [2] [3]` 编号 + 截断 300 字 | LLM 能更准确地引用来源 |

---

## 学习路径总结

```
阶段 1：裸写 API 调用 + 一个工具          ← 手写
阶段 2：加第二个工具，撞工具路由的坑      ← 手写  
阶段 3：接真实 API，撞异常处理的坑        ← 手写
阶段 4：拿框架（LangGraph / Claude Agent SDK）重写  ← 现在可以用 AI 辅助了
阶段 5：做一个完整项目，写 README 讲清楚踩过的坑 ← 放简历
```

---

*最后更新：2026-06-02*
