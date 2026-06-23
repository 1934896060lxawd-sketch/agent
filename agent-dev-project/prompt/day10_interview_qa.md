# Day 10 面试题：综合增强 Agent v2 —— 三层 Prompt 融合

> 对应文件：`prompt/car_advisor_v2.py`
> 核心能力：三层 Prompt 架构、CoT+StructuredOutput+ToolCalling 融合、结构化对话日志、端到端评测体系、Week 1+2 全链路整合

---

## 为什么需要 Agent v2？

Day 5 的 car_advisor_agent 已经能跑通 ReAct 循环，但存在三个生产级痛点：

1. **Prompt 改不动**：System Prompt 混在一起，改角色可能影响工具调用逻辑
2. **错了查不到**：`print()` 日志人眼可读但机器不可搜索，定位"第 3 轮为什么调错工具"靠翻控制台
3. **改了不知道好不好**：没有评测基线，优化全靠感觉

Day 10 的 v2 把 Week 2 的 Prompt 工程能力（CoT / Few-shot / 结构化 / Vision）全部注入 Agent 循环——不是加功能，而是加深层次。

---

## 三层 Prompt 架构图

```
┌─────────────────────────────────────────────────────────┐
│ 外层：System Prompt                                      │
│ 角色定义 + 能力边界 + 输出约束 + 错误处理                  │
│ "你是谁，能做什么，不能做什么，怎么说话"                     │
├─────────────────────────────────────────────────────────┤
│ 中层：Tool Descriptions                                  │
│ 每个 tool 的 name + description + parameters              │
│ "什么时候该调哪个工具，参数填什么"                          │
├─────────────────────────────────────────────────────────┤
│ 内层：CoT Prompt                                         │
│ 推理步骤引导（需求分析→信息缺口→工具选择→验证→策略）       │
│ "怎么想，怎么推理，怎么组织答案"                            │
└─────────────────────────────────────────────────────────┘
```

**三层为什么必须分离，不能全写 System Prompt 里**：

| 问题 | 后果 |
|------|------|
| System Prompt 太长 | LLM 注意力分散，"迷失在中间"效应 |
| Tool description 有独立 JSON Schema 通道 | 放 System Prompt 浪费 token |
| CoT Prompt 只在 Thought 阶段注入 | 放 System Prompt 会导致闲聊也套推理模板 |

---

## CoT + Structured Output + Tool Calling 融合时序

```
User Query
  │
  ▼
┌─ Thought ──────────────────────────────┐
│  ← CoT Prompt（Day 6）注入              │
│  "先分析需求→找信息缺口→选工具"          │
│  LLM 推理输出 AIMessage.content         │
└──────────────┬──────────────────────────┘
               ▼
┌─ Action ───────────────────────────────┐
│  ← Tool descriptions（Day 3）引导        │
│  LLM 输出 tool_calls                    │
│  代码执行工具                            │
└──────────────┬──────────────────────────┘
               ▼
┌─ Observation ──────────────────────────┐
│  ← Structured Output（Day 8）校验        │
│  工具返回 JSON → Pydantic 验证           │
│  不合法 → retry / 降级                   │
└──────────────┬──────────────────────────┘
               │
          (回到 Thought，循环)
               │
               ▼ (无更多 tool_calls)
┌─ Final Answer ─────────────────────────┐
│  ← CarRecommendation Schema 约束        │
│  结构化输出（品牌/价格/理由/评分）        │
└────────────────────────────────────────┘
```

---

## Day 5 v1 → Day 10 v2 核心 Diff

```
Day 5 v1                              Day 10 v2
════════                             ════════
System Prompt 一段式                  System Prompt 三层架构
  角色+规则全混在一起                   外层(角色+约束) + 中层(tool desc) + 内层(CoT)

5 个工具（平铺）                      3 个工具 + Pydantic Schema 校验
  get_car_price/compare_cars/         search_car_knowledge_v2/compare_cars_v2/
  recommend_cars/calculate_cost/      recommend_cars_v2
  search_car_knowledge

print() 轨迹                          @dataclass TurnLog + AgentTrace
  人眼可读，机器不可处理                  JSON 可序列化，可统计，可持久化

无评测                                 10 case 评测体系
  跑通了 = "感觉对了"                   成功率 > 80% = 有量化基准
```

---

## 结构化日志设计

```
TurnLog（单轮快照）
├── turn: int              — 轮次编号
├── thought: str           — Thought（LLM 推理）
├── action: str            — Action（调了哪个工具）
├── action_args: dict      — Action（工具参数）
├── observation: str       — Observation（工具返回）
├── error: str             — Error（异常信息）
└── latency_ms: float      — 该轮耗时

AgentTrace（完整交互审计记录）
├── query: str             — 用户输入
├── turns: list[TurnLog]   — 每轮详情
├── final_answer: str      — 最终回答
├── total_tool_calls: int  — 工具调用总次数
├── total_latency_ms: float — 端到端延迟
└── success: bool          — 是否成功
```

---

## 10 条分层评测 Case

| Case | Query | 预期工具 | 测试目标 |
|------|-------|---------|---------|
| C01 | 小米SU7多少钱？ | 单工具查询 | 工具调用正确性 |
| C02 | 15-20万推荐一款纯电SUV | recommend_cars_v2 | 条件过滤 |
| C03 | 小鹏G6和特斯拉Model Y怎么选？ | compare_cars_v2 | 对比场景 |
| C04 | Model Y的智驾能力如何？ | search_car_knowledge_v2 | 知识检索 |
| C05 | 你好，你是谁？ | 无（直接回答） | 不调工具场景 |
| C06 | 25万左右，轿车，续航要长 | recommend_cars_v2 | 多条件过滤 |
| C07 | 预算10万买什么车？ | recommend_cars_v2 | 低价区间 |
| C08 | 理想L6适合家用吗？ | search_car_knowledge_v2 | 定性判断 |
| C09 | 极氪001对比小米SU7 | compare_cars_v2 | 双车对比 |
| C10 | 推荐一款30万左右加速最快的车 | recommend_cars_v2 | 排序+推荐 |

**两层评判标准**：过程层（工具是否按预期调用）+ 结果层（回答是否包含关键数据），两层都过才算通过。

---

## 四层错误边界

```
第 1 层：工具层
  工具失败 → 返回 {"status": "not_found"} JSON
  不抛异常，LLM 收到结构化错误后自行调整策略

第 2 层：Agent 循环层
  循环超限 → 强制终止，输出已有信息
  LLM 无输出 → 兜底话术

第 3 层：结构化校验层
  工具返回缺字段 → 补充默认值，标记 partial
  最终输出不符合 Schema → retry（最多 2 次）

第 4 层：评测层
  单 case 异常 → try/catch，标记 success=False
  整体成功率 < 80% → 告警
```

---

## Week 1 + Week 2 全链路串联

```
Week 1: Agent 框架基础                    Week 2: Prompt 工程
═════════════════════                    ═══════════════════

Day 1: Chain + PromptTemplate           Day 6: 四种 CoT 策略
Day 2: Memory + RAG                     Day 7: 动态 Few-shot
Day 3: 原生 SDK Tool Calling            Day 8: 结构化输出
Day 4: LangGraph StateGraph             Day 9: Vision RAG
Day 5: create_agent（汽车导购）           Day 10: Agent v2（三层融合）
              │                                   │
              └──────────── 汇合 ─────────────────┘
                              │
                    生产级 Agent 完整能力
```

每个前序 Day 在 v2 中的位置：

| Day | 在 Day 10 中的角色 |
|-----|-------------------|
| Day 3 Tool Calling | 中层 Tool descriptions + ReAct 循环骨架 |
| Day 4 StateGraph | `create_agent` 底层的图表建模 |
| Day 5 create_agent | 一行代码组装 Agent |
| Day 6 CoT | 内层 CoT Prompt → 每次 Thought 前注入 |
| Day 7 Few-shot | System Prompt 中固定 2-3 条轨迹示例 |
| Day 8 Structured Output | Pydantic Schema 校验工具返回 + 最终输出 |
| Day 9 Vision | 可扩展 `analyze_car_image` 作为新 tool |

---

## v2 到生产环境的五个缺口

| 缺口 | 现状 | 需要的 | 对应 |
|------|------|--------|------|
| 部署 | 命令行 `python` | HTTP API 服务 | Day 11 FastAPI |
| 流式输出 | `agent.invoke()` 同步阻塞 | SSE 逐 token 推送 | Day 12 SSE |
| 会话管理 | 单次调用，无会话 | Redis 持久化 + TTL | Day 13 Redis |
| 知识库 | 硬编码 dict | 真实向量库 + RAG | Day 16 Milvus |
| 安全 | 无防护 | Prompt Injection 四层防御 | Day 17 |

---

## Q1：生产级 Agent 的"三层 Prompt 架构"是什么？每层各解决什么问题？

**一句话**：外层 System Prompt 定角色和约束边界，中层 Tool descriptions 引导调用决策，内层 CoT Prompt 保证推理质量——三层各司其职，互不重叠。

**每层的不可替代性**：

| 层 | 解决的问题 | 如果缺失会怎样 |
|----|-----------|---------------|
| System Prompt | Agent 的行为边界 | LLM 可能回复非汽车内容、编造数据 |
| Tool descriptions | 何时调工具 | 该调时不调、调了填错参数、乱调 |
| CoT Prompt | 推理深度 | 跳过分析直接给结论、工具结果不验证 |

**面试话术**："三层 Prompt 架构的本质是关注点分离——每层有自己的职责边界和注入时机。面试官问'你的 Prompt 怎么设计的'，如果你说'全写 system 里'，就暴露了没有生产经验。Day 10 的三层设计让每个 Prompt 组件可独立评测和迭代。"

---

## Q2：Day 10 如何把 CoT（Day 6）+ Structured Output（Day 8）+ Tool Calling（Day 3）三者融合？融合的难点在哪？

**一句话**：三者在 ReAct 循环的**不同阶段**分别注入——Thought 前注入 CoT 推理模板、Action 后对工具返回做结构化校验、Final Answer 前套用 Pydantic Schema 约束输出格式。

**融合的三大难点**：

| 难点 | 说明 | Day 10 的解法 |
|------|------|-------------|
| 上下文冲突 | CoT 让 LLM"多思考"，但 System Prompt 要它"简洁" | 把 CoT 标记为"内部思考，不输出给用户" |
| Token 膨胀 | 三层 Prompt + 对话历史 + 工具结果 → 轻松超 context | CoT 只在当前 Thought 注入，不进历史 |
| 结构化校验时机 | 工具返回后校验 vs 最终输出前校验 | 工具返回时做轻量校验（字段齐全），最终输出前做完整 Schema 校验 |

**面试话术**："Day 3/6/8 各自独立练习时，你学会的是单项能力。Day 10 的难点在于让三者**在同一个 ReAct 循环里不打架**——CoT 多了 token 超了、校验严了重试多了延迟涨了、System Prompt 太长了 LLM 注意力散了。真正的 Prompt 工程就是在这些约束中找到平衡点。"

---

## Q3：结构化对话日志（每轮 T/A/O/E）的设计思路是什么？为什么不能用 print？

**一句话**：`print()` 是人看的，`TurnLog` 数据结构是机器读的。结构化日志是评测、调试、审计的基础设施——没有它，你没法回答"第 3 轮为什么调了错误的工具"。

**为什么不能只用 print**：

| 需求 | print | 结构化日志 |
|------|-------|-----------|
| 统计"5 次对话中 recommend 被调了几次" | 人肉数 | `grep \| wc -l` 一行 |
| 定位"第 3 轮 tool_calls 参数 brand 为什么是空" | 翻控制台 | 按 turn 索引 |
| 计算 P95 延迟 | 不可能 | `numpy.percentile` |
| 接入 ELK/Prometheus | 解析 print → 噩梦 | JSON → 原生支持 |
| 回归测试：改了 Prompt 后成功率降了没 | 靠感觉 | CI 自动对比 |

**面试话术**："`print()` 是原型阶段的调试手段，结构化日志是生产系统的可观测性基础设施。Day 10 的 `TurnLog` + `AgentTrace` 虽然简单，但体现了'观测先行'的工程思维——先有日志，再谈评测，最后才是优化。没有观测的优化就是瞎调。"

---

## Q4：10 个测试 case 的端到端评测是怎么设计的？为什么不能只看"最终回答像不像"？

**一句话**：评测分两层——过程层（工具是否按预期调用）和结果层（回答是否包含关键数据），两层都通过才算 case 通过。

**只检查 final_answer 的陷阱**：

```
用户问"小米SU7多少钱？"
LLM 直接编了一个"21.59万"（没调 get_car_price）
→ answer_ok = True（答案碰巧对了）
→ 但 tool 没调 = 幻觉隐患

两层都检查：
→ tool_ok = False（没调工具）
→ case 不通过 [X]
```

**面试话术**："评测 Agent 和评测分类模型不同——Agent 的正确性不止看最终结果，还要看过程是否合理。一个没调工具就给出答案的 Agent，这次蒙对了，下次就编造。过程层的评测保证了 Agent 行为的**可解释性**和**可复现性**。"

---

## Q5：Day 5 的 car_advisor_agent 升级到 Day 10 的 v2，具体改了什么？改动的价值在哪？

**一句话**：v1 → v2 不是加功能，而是加深层次——System Prompt 从"指令列表"升级为"三层架构"、日志从 print 升级为结构化、从无评测升级为有评测。

**改动的价值**：

| 维度 | v1 | v2 | 价值 |
|------|----|----|------|
| Prompt 可维护性 | 改了角色可能影响工具调用逻辑 | 三层独立修改 | 改 CoT 不影响 System Prompt |
| 可调试性 | 翻控制台找"第三轮为什么错了" | `trace.turns[2]` | 调试效率 ×10 |
| 可评测性 | 靠人工感觉"好像更好了" | 跑一次 10 case 看成功率 | 优化有方向 |
| Token 效率 | System Prompt 太长浪费 token | CoT 只在 Thought 注入 | 平均省 15-20% token |

**面试话术**："v1→v2 的升级不是为了炫技，而是解决 v1 在真实使用中的三个痛点——Prompt 改不动（牵一发动全身）、错了查不到（日志不可搜索）、改了不知道好不好（没 baseline）。这三个痛点恰恰是面试官判断你'有没有真的在生产环境用过 Agent'的关键信号。"

---

## Q6：工具返回数据如何格式化为"结构化上下文"？为什么 LLM 能更好理解？

**一句话**：工具返回从自然语言句子改为 JSON 结构化数据，每个字段有明确的 key，LLM 不需要从句子中"提取"信息，直接引用 key 即可。

**对比**：

```python
# v1 风格：自然语言返回
"比亚迪海豚价格区间为9.98万到13.98万元，CLTC续航最高405公里，
零百加速约10秒，智驾为L2级别基础功能，没有激光雷达"

# v2 风格：结构化 JSON 返回
{
  "name": "比亚迪 海豚",
  "price": {"min": 9.98, "max": 13.98, "unit": "万"},
  "spec": {
    "range_km": 405,
    "accel_s": 10,
    "smart_drive": "L2",
    "lidar": false
  },
  "status": "found"
}
```

**为什么 LLM 更好理解结构化返回**：

| 维度 | 自然语言返回 | 结构化 JSON 返回 |
|------|------------|-----------------|
| 信息提取 | LLM 需从句子中"再解析"数字 | LLM 直接引用 `price.min` |
| 多工具组合 | 两个工具返回混在一起，LLM 分不清 | 每个字段有来源标记 |
| 幻觉风险 | LLM 可能误解"约10秒"为"10.0秒" | 精确值无歧义 |
| 错误处理 | "未找到价格"掺杂在自然语言中 | `{"status": "not_found"}` 明确信号 |

**面试话术**："工具返回的结构化不是'为了好看'，而是为了减少 LLM 的二次解析负担。自然语言返回要求 LLM 先理解句子、再提取数字、再验证不编造——每一步都可能出错。结构化 JSON 每个字段精确无歧义，LLM 直接引用即可。这是 Day 8 Pydantic 思想从'输出'延伸到'中间数据流'的体现。"

---

## Q7：动态 Few-shot（Day 7）在 Agent 场景中怎么用？为什么不像 Day 7 那样做 MMR 选择？

**一句话**：Agent 场景的 Few-shot 不如纯推理场景密集——Agent 的行为由工具返回数据驱动，Few-shot 示例只能引导推理风格，不能替代数据。

**Agent 场景的 Few-shot 策略差异**：

| 维度 | Day 7 纯推理 | Day 10 Agent |
|------|-------------|-------------|
| Few-shot 示例内容 | 完整的问答对 | Thought→Action→Observation 轨迹 |
| 示例密度 | 每条 query 选 3-5 条 | System Prompt 固定 2-3 条 |
| MMR 选择 | 有价值（避免示例同质化） | 价值有限（工具返回决定行为） |
| 更新频率 | 每次 query 动态选 | 按场景类型静态配置 |
| 示例作用 | 格式化推理过程 | 示范工具调用顺序 |

**为什么 Agent 场景不做复杂 MMR**：

```
Day 7 MMR 场景：
  用户问 → 选 3 条最相关+最多样的示例 → 注入 prompt → LLM 模仿推理
  适用条件：答案模式因问题类型而异，示例多样性直接影响答案质量

Agent 场景不用 MMR 的原因：
  1. Agent 的行为由工具返回数据驱动，不是由示例驱动
  2. 每轮上下文已经有很多 tool 返回数据，Few-shot 示例占比小
  3. Agent 的多样性来自工具返回数据本身，不需要示例来保证
```

**面试话术**："Day 7 的 MMR 选择器在纯推理场景（如分类、摘要）很有效，但 Agent 场景的 Few-shot 只起'风格引导'作用——示范怎么推理、怎么用工具返回数据。Agent 的核心约束是三层 Prompt 架构 + 工具返回的数据，Few-shot 是锦上添花不是核心。知道什么时候不该用复杂技术，比知道怎么用更重要。"

---

## Q8：Agent v2 的错误边界设计是什么？从 v1 到 v2 在鲁棒性上做了什么改进？

**一句话**：v2 把错误分成四层处理——工具层、Agent 循环层、结构化校验层、评测层。每层有独立的降级策略，错误不向上传播。

**v1 → v2 鲁棒性改进**：

| v1 问题 | v2 改进 | 效果 |
|---------|--------|------|
| 工具抛异常 → Agent 崩溃 | 工具返回结构化 error JSON | Agent 不会因单工具失败而终止 |
| 日志 print → 无错误追踪 | `TurnLog.error` 字段 | 知道哪一轮、哪个工具出错 |
| 无重试机制 | 校验失败 → retry（最多 2 次） | 临时输出格式问题自愈 |
| 无兜底 | 循环超限 → 输出已有信息 | 用户不会看到空白 |

**面试话术**："Agent 的错误处理不是 try/catch 就完了——关键是'错误信息的传递链'。工具失败后，错误信息要以 LLM 能理解的格式回传，循环要能优雅终止，输出要经过校验，整个过程要可追溯。这个四层框架和 Day 17 的 Prompt Injection 四层防御是一脉相承的设计思想。"

---

## Q9：Day 10 作为 Week 2 的收尾，如何把 Week 1（Agent 框架基础）和 Week 2（Prompt 工程）串起来？

**一句话**：Week 1 教会你"Agent 是怎么跑起来的"（骨架），Week 2 教会你"怎么让 Agent 跑得更好"（血肉），Day 10 是骨头和肉合在一起，形成一个完整、可评测、可迭代的 Agent 系统。

**Day 10 使用了 Week 1+2 的哪些**：

| Day | 在 Day 10 中的位置 |
|-----|-------------------|
| Day 3 Tool Calling | 中层 Tool descriptions + ReAct 循环骨架 |
| Day 4 StateGraph | `create_agent` 底层的图表建模 |
| Day 5 create_agent | 一行代码组装 Agent |
| Day 6 CoT | 内层 CoT Prompt → 每次 Thought 前注入 |
| Day 7 Few-shot | System Prompt 中固定 2-3 条轨迹示例 |
| Day 8 Structured Output | Pydantic Schema 校验工具返回 + 最终输出 |
| Day 9 Vision | 可扩展 `analyze_car_image` 作为新 tool |

**面试话术**："Day 10 不是一个新功能，而是一次整合——你前面 9 天学的每一个技术都不是孤立的，它们在 Day 10 的 Agent v2 中各就各位。面试官不会让你'展示 Day 6 的 CoT 代码'，但会让你解释'你的 Agent 怎么保证推理质量'——这时候你脑子里应该浮现三层 Prompt 架构图，而不是 CoT 的 5 行代码。"

---

## Q10：Agent v2 距离真正的生产环境还缺什么？下一步要补哪些能力？

**一句话**：v2 解决了"Agent 怎么想"的问题，但还没有解决"Agent 怎么可靠地跑在服务器上"——Week 3 的 FastAPI、SSE、Redis、Docker、熔断限流就是补齐这些。

**v2 在成熟度阶梯上的位置**：

```
原型 (v0) ──→ 可运行 (v1) ──→ 可评测 (v2) ──→ 可部署 (Week 3) ──→ 可放量 (Week 4-6)
  │              │               │               │                    │
Day 1-4       Day 5          Day 10          Day 11-15            Day 16-30
"能跑通"      "能跑通"       "知道好不好"    "能上线服务"          "能扛住流量"
```

**面试话术**："Day 10 的 Agent v2 在'逻辑正确性'上已经成熟——有架构设计、有 Prompt 层次、有结构化日志、有评测基线。但从'逻辑正确'到'生产可用'还差工程化六件套：部署、流式、会话、向量库、评测自动化、安全。Week 3-4 的目标就是把这六件补齐，让 Agent 从'我机器上能跑'变成'别人也能用'。"

---

### Day 10 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | 三层 Prompt 架构是哪三层？每层解决什么问题？为什么不能全写 System Prompt 里？ | □ |
| 2 | CoT + Structured Output + Tool Calling 在 ReAct 循环中如何时序配合？融合的三大难点？ | □ |
| 3 | 结构化日志 TurnLog/AgentTrace 的设计？为什么不能用 print？ | □ |
| 4 | 10 case 评测为何分过程层和结果层？只看最终回答有什么陷阱？ | □ |
| 5 | Day 5 v1 升级到 Day 10 v2 的五个改动？每个改动的价值？ | □ |
| 6 | 工具返回自然语言 vs JSON 结构化，为什么 LLM 更好理解后者？ | □ |
| 7 | Day 7 的 MMR Few-shot 为什么在 Agent 场景不必做？Agent 场景的 Few-shot 策略是什么？ | □ |
| 8 | Agent v2 的四层错误边界？v1 → v2 在鲁棒性上的改进？ | □ |
| 9 | Day 10 如何把 Week 1（框架）+ Week 2（Prompt）串联？每个前序 Day 在 v2 中处于什么位置？ | □ |
| 10 | Agent v2 到生产还缺什么？下一步要补的五个缺口是什么？ | □ |
