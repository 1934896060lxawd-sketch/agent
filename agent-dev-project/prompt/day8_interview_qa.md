# Day 8 面试题：结构化输出

> 对应文件：`prompt/structured_output.py`
> 核心能力：Pydantic Schema 定义、response_format 约束、with_structured_output、嵌套结构、校验重试

---

## Q1：Structured Output 和 JSON Mode 的区别是什么？为什么 Structured Output 更可靠？

**一句话**：JSON Mode 只保证输出是合法 JSON（`{...}`），Structured Output 保证输出符合你指定的 JSON Schema（字段名、类型、约束全对）。

**三层递进（从弱到强）**：

| 层级 | 方式 | 可靠性 | 说明 |
|------|------|:---:|------|
| Prompt Only | System Prompt 写"请返回 JSON" | ~80% | LLM 可能加前缀"好的，结果是：" |
| JSON Mode | `response_format: {"type": "json_object"}` | ~95% | 保证 `{ }` 合法，但字段可能拼错或缺漏 |
| Structured Output | `response_format: {"type": "json_schema", "json_schema": {...}}` | ~99.9% | 保证字段名、类型、约束全部匹配 |

**JSON Mode 为什么不够**：

```json
// LLM 在 JSON Mode 下的输出（合法 JSON，但 Schema 不对）
{
  "car_name": "小鹏G6",      // ← 要求的 key 是 model_name，LLM 自己起了 car_name
  "price": "20.99",          // ← 要求 float，LLM 给了 string
  // 缺了 score、pros、cons 字段
}
```

**Structured Output 的底层机制 — 受限解码（Constrained Decoding）**：

在 token 采样阶段，API 持有完整的 JSON Schema。每生成一个 token，API 会计算哪些 token 能导向合法的 Schema 匹配，然后只从这些 token 中采样。从概率空间上杜绝了 Schema 不匹配的可能。

```
普通生成：下一个 token 可以是词表(32000)中任意一个 → 按概率采样
受限解码：下一个 token 只能是 {能导向合法 Schema 的子集(50-200个)} → 强制合法
```

**面试加分点**：能说出"受限解码（constrained decoding）"这个底层机制——不是在 prompt 里"建议"LLM 输出 JSON，而是在 token 采样时就约束可选 token 集合。

**代码中的体现**：

```python
# structured_output.py — structured_output_native()
# 由于 DeepSeek 不支持 response_format: json_schema，代码中用 Function Calling 兼容实现
# 原理一致：把 Schema 包装成 tool → tool_choice 强制 → LLM 以 tool call 形式输出
fake_tool = {
    "type": "function",
    "function": {
        "name": "CarRecommendation",
        "parameters": schema_class.model_json_schema(),
    },
}
response = llm_client.invoke(prompt, tools=[fake_tool],
    tool_choice={"type": "function", "function": {"name": "CarRecommendation"}})
# LLM 被强制"调用"CarRecommendation 工具 ← 效果等同于 Structured Output
```

---

## Q2：Pydantic Field 的 `description` 参数为什么重要？不写会怎样？

**一句话**：`description` 是 LLM 理解"这个字段应该填什么"的唯一依据——它会被写入 JSON Schema 传给 LLM。不写等于让 LLM"猜"每个字段的含义。

**具体机制**：

```python
# 有 description
score: float = Field(description="综合评分，0-10分，10分代表同价位最佳")
# → JSON Schema 中: {"score": {"type": "number", "description": "综合评分，0-10分..."}}
# LLM 读到 description 后理解：我需要给一个 0-10 的分数

# 无 description
score: float  # 没有任何上下文提示
# → JSON Schema 中: {"score": {"type": "number"}}
# LLM 看到后：不知道评分范围，可能给 85（百分制）、4.5（5分制）或 9.2（10分制）
```

**description 的三个作用**：

1. **定义语义契约**：告诉 LLM 这个字段代表什么意思（是"价格"还是"价格区间"？）
2. **隐含格式约束**：如 `"价格区间，如'20.99-27.69万'"`——既说明了语义，又暗示了格式
3. **提供示例**：description 中的示例是给 LLM 最直接的"应该怎么填"的示范

**代码中完整的 Field 定义**：

```python
# structured_output.py — CarRecommendation
class CarRecommendation(BaseModel):
    model_name: str = Field(description="推荐车型全称，如'小鹏G6 755超长续航Max'")
    price_range: str = Field(description="价格区间，如'20.99-27.69万'")
    score: float = Field(description="综合评分，0-10分", ge=0, le=10)
    pros: list[str] = Field(description="优点列表，至少3条", min_length=3)
    cons: list[str] = Field(description="缺点列表")
    best_for: str = Field(description="最适合的人群描述")
```

**面试话术**："`description` 是 Schema 和 LLM 之间的接口文档。它的核心价值在于降低 LLM 的'语义猜测'——每个字段都有明确的语义契约。生产环境中，description 的措辞本身就是一个需要迭代优化的对象：同一个字段，不同的 description 表述会导致 LLM 填充质量的显著差异。"

---

## Q3：`with_structured_output()` 背后做了什么？如果让你手写实现，你会怎么做？

**一句话**：四个步骤——提取 Schema → 注入 API 约束 → 调用 LLM → 解析校验。

**手写等价实现**（面试可能要你口述）：

```python
def my_with_structured_output(llm, pydantic_model):
    """手写 with_structured_output 的等价逻辑"""
    class StructuredRunnable:
        def invoke(self, prompt: str):
            # Step 1: Pydantic Model → JSON Schema
            schema = pydantic_model.model_json_schema()
            
            # Step 2: 包装成 tool + 强制 tool_choice
            fake_tool = {
                "type": "function",
                "function": {
                    "name": pydantic_model.__name__,
                    "description": f"输出一个 {pydantic_model.__name__} 结构",
                    "parameters": schema,
                },
            }
            
            # Step 3: 调用 LLM
            response = llm.invoke(
                prompt,
                tools=[fake_tool],
                tool_choice={"type": "function", "function": {"name": pydantic_model.__name__}},
            )
            
            # Step 4: 从 tool_calls 提取 JSON → Pydantic 校验
            tool_args = response.tool_calls[0]["args"]  # ← 注意：不是 additional_kwargs！
            return pydantic_model.model_validate_json(tool_args)
    
    return StructuredRunnable()
```

**代码中的两种实现**（对比理解）：

```python
# 方式 1: 手写（structured_output_native）— 理解底层每一步
# 约 15 行代码，显式处理 Schema 转换 → tool 构造 → 调用 → 解析

# 方式 2: LangChain 封装（structured_output_langchain）— 一行搞定
structured_llm = llm_client.with_structured_output(CarRecommendation, method="function_calling")
result = structured_llm.invoke(prompt)
# 1 行代码，LangChain 自动处理所有步骤
```

**`method` 参数的技术细节**：

| method | 实现方式 | 模型要求 | 延迟 |
|--------|---------|---------|:---:|
| `"json_schema"` | `response_format: {"type": "json_schema"}` | 需模型支持（GPT-4 等） | 更低 |
| `"function_calling"` | 包装成 tool + `tool_choice` 强制 | 支持 Function Calling 即可（DeepSeek 兼容） | 稍高 |

**DeepSeek 兼容性踩坑**：DeepSeek API 返回 `400: "This response_format type is unavailable now"` 不支持 `json_schema` 模式，所以 Day 8 代码全用 `method="function_calling"`。这是面试中能展示"理解兼容性差异"的加分点。

**面试话术**："`with_structured_output()` 的 value 不是省代码——是标准化了 Schema→调用→解析→校验这条管道。更深层的价值是：你可以在 LCEL 管道中像使用普通 Runnable 一样使用结构化输出——`prompt | structured_llm | validation | db_save`。这是 LangChain Runnable 协议的威力。"

---

## Q4：嵌套 Pydantic Model 在传给 LLM 时是什么样的？LLM 能正确填充嵌套结构吗？

**一句话**：Pydantic 的 `model_json_schema()` 递归展开所有嵌套子对象，生成 `$defs` + `$ref` 引用的树形 Schema。LLM 看到的是一个完整的结构化契约。

**代码中的三层嵌套**：

```python
# structured_output.py — 练习 4
class SmartDrive(BaseModel):
    """智驾子对象"""
    chip: str = Field(description="智驾芯片型号，如'Orin-X'")
    computing_power_tops: int = Field(description="算力，单位TOPS")
    lidar_count: int = Field(description="激光雷达数量")
    features: list[str] = Field(description="智驾功能列表")

class CarParams(BaseModel):
    """车辆参数子对象"""
    length_mm: int = Field(description="车长(mm)")
    wheelbase_mm: int = Field(description="轴距(mm)")
    range_km: int = Field(description="CLTC续航(km)")
    # ...

class DetailedCarInfo(BaseModel):
    """完整车型信息 — 第二层嵌套"""
    model_name: str
    params: CarParams        # ← 嵌套子对象
    smart_drive: SmartDrive  # ← 嵌套子对象
```

**生成的 JSON Schema（简化）**：

```json
{
  "type": "object",
  "properties": {
    "model_name": {"type": "string"},
    "params": {"$ref": "#/$defs/CarParams"},       // 引用嵌套定义
    "smart_drive": {"$ref": "#/$defs/SmartDrive"}
  },
  "$defs": {
    "CarParams": {
      "type": "object",
      "properties": {
        "length_mm": {"type": "integer", "description": "车长(mm)"},
        "range_km": {"type": "integer", "description": "CLTC续航(km)"}
      }
    },
    "SmartDrive": {
      "type": "object",
      "properties": {
        "chip": {"type": "string", "description": "智驾芯片型号"},
        "lidar_count": {"type": "integer"}
      }
    }
  }
}
```

**LLM 填充嵌套结构的可靠性**：

| 嵌套深度 | 可靠性 | 说明 |
|:---:|:---:|------|
| 1 层（扁平） | 99% | 几乎不出错 |
| 2 层（主→子） | 95% | 很少出错 |
| 3 层（主→子→孙） | 85% | 深层字段 description 信息可能被"稀释" |
| 4+ 层 | < 70% | 不建议——认知负荷过大 |

**设计原则**：嵌套不超过 3 层，子对象表达内聚概念（物理参数是一组、智驾是另一组），而非按数据来源或存储结构来拆。

**面试话术**："嵌套 Schema 的设计和数据库范式化思路一样——每个子对象封装一个内聚的语义概念。但 LLM 不是数据库，过深的嵌套会让 LLM 丢失 context。超过 3 层时应该拆成多次调用：第一次拿摘要，后续每次深入一个子树。"

---

## Q5：结构化输出校验失败后怎么重试？说说你的重试策略设计。

**一句话**：把 Pydantic 的 `ValidationError`（包含具体哪个字段、哪种错误）作为反馈注入下一轮 Prompt，让 LLM 自我修正。重试上限 3 次，超过则降级返回 None 并记录日志。

**代码中的重试策略**（`structured_output_with_retry`）：

```python
for attempt in range(1, max_retries + 1):
    try:
        return structured_llm.invoke(current_prompt)
    except OutputParserException as e:
        # JSON 格式错误（极少见）→ 提醒 LLM 严格遵守 JSON
        current_prompt = f"{prompt}\n\n[第{attempt}次修正] 上次输出不是合法的 JSON..."
    except ValidationError as e:
        # 字段校验错误 → 把具体错误反馈给 LLM
        error_details = json.dumps(e.errors(), ensure_ascii=False)
        current_prompt = f"{prompt}\n\n[第{attempt}次修正] 上次校验失败:\n{error_details}\n请修正..."

# 重试耗尽 → 降级
print(f"[WARN] 重试 {max_retries} 次后仍失败")
return None
```

**三种错误类型及处理策略**：

| 错误类型 | 原因 | 频率 | 处理 |
|----------|------|:---:|------|
| `OutputParserException` | LLM 返回的不是合法 JSON | 极少（API 层已约束） | 重试 + 加强 Prompt |
| `ValidationError` | JSON 合法但字段不满足约束（如 score=15 超出 0-10） | 偶尔 | **把 `e.errors()` 反馈给 LLM**，让它自我修正 |
| 语义错误 | JSON 合法、字段合法，但内容不对（虚构车型） | 常见 | `@field_validator` 加业务规则校验 |

**`@field_validator` 的业务规则**（代码中的例子）：

```python
@field_validator("model_name")
@classmethod
def must_be_real_car(cls, v: str) -> str:
    """车型名必须包含已知品牌（防止 LLM 虚构车型）"""
    known_brands = ["比亚迪", "特斯拉", "小鹏", "理想", "蔚来", "问界", ...]
    if not any(brand in v for brand in known_brands):
        raise ValueError(f"'{v}' 不包含已知汽车品牌，可能是 LLM 虚构的车型")
    return v
```

**生产级重试的四个关键考虑**：

1. **错误分类**：区分格式错误 vs 校验错误 vs 语义错误——处理策略不同
2. **退避策略**：不是无脑重试同一 prompt，每次都要把**具体的失败原因**注入
3. **成本控制**：每次重试 = 一次 API 调用。上限 3 次是经验值——超过通常是 Schema 本身有问题
4. **降级兜底**：重试耗尽后不能崩溃，返回 None 或带默认值的部分结果 + 记录日志

**面试话术**："重试不是机械重放——关键是每次都给 LLM 新的信息（具体的校验错误）。这和 Self-Consistency 不同：Self-Consistency 是并行多次采样取投票结果（解决随机性），结构化重试是串行逐步修正（解决确定性错误）。"

---

## Q6：什么时候用 Structured Output，什么时候用 Function Calling？两者能互相替代吗？

**一句话**：Structured Output 用于"让 LLM 产出结构化数据"（终点），Function Calling 用于"让 LLM 调用外部工具"（中间点）。两者底层机制相通，但设计意图不同。

**对比表**：

| 维度 | Structured Output | Function Calling |
|------|:---:|:---:|
| 目的 | 产出符合 Schema 的 JSON 数据 | 产出工具调用指令 + 参数 |
| 输出的消费方 | 代码直接使用（前端、数据库、下游逻辑） | 工具执行器 → 结果再喂回 LLM |
| Schema 角色 | 定义输出数据格式 | 定义工具签名（参数约束） |
| 后续动作 | 解析 → 存入数据库/返回前端 | 执行工具 → 把 Observation 喂回 LLM |
| API 实现 | `response_format: {"type": "json_schema"}` | `tools: [...]` + `tool_choice` |
| 典型场景 | 信息抽取、表单填充、报告生成 | Agent 调 API、查数据库、执行计算 |

**两者能互相替代吗？**

技术上完全可以——这也是 Day 8 代码用 `method="function_calling"` 的原因：

```python
# 用 Function Calling 模拟 Structured Output（代码实际做法）
# 把 Schema 包装成 tool，tool_choice 强制调用 → LLM 输出 tool call = 结构化数据
fake_tool = {"function": {"name": "CarRecommendation", "parameters": schema}}
response = llm.invoke(prompt, tools=[fake_tool],
    tool_choice={"type": "function", "function": {"name": "CarRecommendation"}})
# tool_calls[0]["args"] 就是结构化数据！
```

反之，也可以用 Structured Output 模拟 Function Calling——让 LLM 输出 `{"tool": "get_car_price", "args": {"model": "G6"}}`。Day 3 手写 tool calling 循环本质上就是这个思路。

**选型原则**："看输出是终点还是中间点。如果 LLM 产出的 JSON 就是最终结果（展示/存储），用 Structured Output。如果 JSON 是用来触发下一步动作（调 API/查数据库），用 Function Calling。Agent 场景中两者经常共存——Function Calling 选工具，Structured Output 格式化 Observation 返回结果。"

**面试话术**："理解了 Structured Output 和 Function Calling 在底层的一致性，你就能在任何模型上实现结构化输出——不支持 `response_format: json_schema` 就用 function_calling 方式，不支持 function calling 就用 prompt + 正则降级。这就是 Day 3 和 Day 8 串联后的工程判断力。"

---

## Q7：如果 LLM 不支持 `response_format: json_schema`（如旧模型或本地模型），怎么保证结构化输出？

**一句话**：降级到四层防御体系——Prompt 约束 → JSON 提取 → 多次采样投票 → 校验兜底。

**代码中的降级方案**（`extract_json_from_text`）：

```python
def extract_json_from_text(text: str) -> dict:
    # 层 1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 层 2: 提取 ```json ... ``` 代码块
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # 层 3: 提取最外层 { ... }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    # 层 4: 修复常见 JSON 错误（尾部逗号、单引号、注释）后重试
    fixed = re.sub(r',\s*}', '}', text)       # 尾部逗号
    fixed = re.sub(r',\s*]', ']', fixed)       # 数组尾部逗号
    fixed = re.sub(r'//.*?\n', '\n', fixed)     # 单行注释
    ...
```

**完整的降级路线图**：

```
最佳方案：response_format: json_schema
    ↓ 模型不支持
兼容方案：method="function_calling"（把 Schema 包装成 tool）
    ↓ 模型不支持 Function Calling
降级方案：Prompt 要求 JSON + extract_json_from_text + 多次采样
    ↓ 全部失败
兜底方案：返回带默认值的对象 + _parse_error: True 标记
```

**面试话术**："生产环境中不假设所有模型都支持最新特性。我的做法是 API 层先检测模型能力（是否支持 json_schema / function_calling），自动选择最高效的路径。关键是降级路径要透明——记录每种降级策略的触发频率，作为模型选型和升级的决策依据。"

---

## Q8：结构化输出在 Agent 架构中处于什么位置？它和其他组件怎么配合？

**一句话**：结构化输出是 Agent 的"输出格式化层"——把 LLM 的文本输出转成代码可以直接消费的数据结构，位于 LLM 和下游消费者之间。

**Agent 架构中的位置**：

```
用户输入 → [Prompt 组装] → LLM 推理 → [结构化输出层] → 下游消费
                                          ↑
                              Pydantic Schema（由下游需求定义）
                              
下游消费者：
  ├── 工具调用：tool_calls 本身就是结构化输出的一种（内置 Schema）
  ├── 状态更新：AgentState 字段更新需要解析 LLM 输出为结构化数据
  ├── 前端展示：JSON → UI 组件渲染（推荐卡片、对比表格）
  └── 数据库写入：结构化数据 → ORM → SQL
```

**和 Day 5 Agent 的串联**：

```python
# Day 5 Agent 输出（自由文本）
"Thought: 综合对比后推荐小鹏G6\nAction: recommend_car(name='小鹏G6')\n..."

# Day 8 增强后（结构化）
result: CarRecommendation = structured_llm.invoke(agent_context)
# → result.model_name = "小鹏G6 755超长续航Max"
# → result.score = 8.5
# → result.pros = ["智驾同价位最强", "800V超快充", ...]
# 前端直接把 result.model_dump() 渲染成推荐卡片
```

**和 Day 3 Function Calling 的串联**：

```
Function Calling（Day 3）
  LLM 输出 tool_calls ← 系统预定义的 JSON Schema
  ↓
  代码执行工具 → 返回 Observation
  ↓
Structured Output（Day 8）
  把 Observation 格式化成结构化上下文 → 喂回 LLM
  ↓
  LLM 最终推荐 → 用 Structured Output 格式化为 CarRecommendation 对象
  ↓
  前端渲染
```

**面试话术**："结构化输出在 Agent 中扮演的是'翻译官'角色——把 LLM 的思考翻译成代码能直接消费的数据结构。位置上看，它是 LLM 输出的最后一步、下游消费的第一步。设计上看，Schema 的定义应该由下游消费者决定（前端需要什么字段、数据库存什么字段），而不是由 LLM 的能力决定。这个原则叫 Consumer-Driven Schema Design。"

---

## Q9（附加）：DeepSeek 为什么不支持 `response_format: json_schema`？你如何发现并解决的？

**一句话**：DeepSeek API 当前版本不支持 `json_schema` 类型的 response_format（返回 400 错误）。解决方案是用 `method="function_calling"` 替代——把 Schema 包装成 tool 强制 LLM 输出。

**发现过程**（踩坑实录）：

```python
# 尝试 1: 直接传 response_format（理想方案）
response = llm.invoke(prompt, response_format={
    "type": "json_schema",
    "json_schema": {"name": "CarRec", "schema": {...}},
})
# → DeepSeek 400: "This response_format type is unavailable now"

# 尝试 2: 降级到 function_calling（兼容方案）
structured_llm = llm.with_structured_output(CarRecommendation, method="function_calling")
result = structured_llm.invoke(prompt)
# → 成功！因为 Function Calling 是所有模型的基础能力
```

**兼容性思维**：不同模型支持的 API 特性不同，这是生产中的常态。工程上的正确做法是：

1. 先写能力检测（Capability Detection）
2. 根据检测结果选择最优实现路径
3. 记录降级事件用于监控

**面试话术**："这个踩坑恰恰展示了'理解机制 > 会用 API'的价值。如果只知道 `with_structured_output()` 一行搞定，遇到 DeepSeek 报错就束手无策。理解了它底层是通过 function_calling 实现的，你就能主动切换 method、手写等价实现，甚至给团队输出一份模型兼容性矩阵。"

---

### Day 8 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | Structured Output vs JSON Mode 的核心区别？受限解码是什么？ | □ |
| 2 | Pydantic Field description 的三个作用？不写会怎样？ | □ |
| 3 | with_structured_output 背后四个步骤？method 参数两种模式的区别？ | □ |
| 4 | 嵌套 Pydantic Model 在 JSON Schema 中如何表示？设计原则是什么？ | □ |
| 5 | 结构化输出的重试策略？三种错误类型及不同处理方式？ | □ |
| 6 | Structured Output vs Function Calling 的关系？能互相替代吗？ | □ |
| 7 | 不支持 json_schema 时的四层降级方案？ | □ |
| 8 | 结构化输出在 Agent 架构中的位置？Consumer-Driven Schema 是什么意思？ | □ |
| 9 | DeepSeek 兼容性问题如何发现和解决？对你的启示？ | □ |
