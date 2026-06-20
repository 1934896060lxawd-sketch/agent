# Agent 开发完整学习路线 —— 大厂面试对标版

> **学习目标**：建立 Agent 工程判断力，看懂 Agent 循环、状态机、上下文管理、MCP 协议、Skill 工具调用、记忆与权限控制；最终独立完成一套可演示、可评测、可写入简历的全栈垂直 Agent 落地项目。
>
> **核心思路**：先建立架构判断能力，再写代码；优先掌握方案设计思维，而非只钻研对话功能。
>
> **前置基础**：已完成 RAG 五章（naive rag → embedding → retrieval → hybrid RRF → full rag agent）。

---

## 五套核心学习项目（贯穿全程）

| 项目 | 作用 | 对应阶段 |
|------|------|---------|
| **hello-agents** | 入门扫盲，熟悉 Agent 术语、通用流程 | 第 1 周 |
| **learn-claude-code** | 掌握 Agent Loop、状态机、工具调用底层实现 | 第 1-2 周 |
| **learn-harness-engineering** | 学习工程边界、输入输出校验、观测日志、失败恢复、评测体系 | 第 3 周 |
| **craft-agents-oss** | 完整全栈 Agent 架构样板（Electron + MCP + 多后端抽象） | 第 4 周 |
| **Zread** | 批量解析开源仓库，快速建立全局认知，减少无效读源码 | 贯穿全程 |

---

## 七周概览

| 周 | 阶段主题 | 产出目录 | 大厂面试维度覆盖 |
|----|---------|----------|-----------------|
| 1 | Agent 框架基础（LangChain → LangGraph → ReAct） | `agent/` | Agent 核心架构、Tool Calling、ReAct 框架 |
| 2 | Prompt 工程（CoT / Few-shot / 结构化 / 多模态） | `prompt/` | Prompt 设计、结构化输出、多模态 RAG |
| 3 | 生产化部署（FastAPI / SSE / Redis / Docker / 熔断） | `api/` | 工程化落地、流式输出、会话管理、容错 |
| 4 | 进阶能力（Milvus / Agent 安全 / LoRA / Streamlit） | `chapters/` + `finetune/` | RAG 进阶、Agent 安全、微调基础 |
| 5 | 算法原理补课（Transformer / RLHF / GRPO / 量化） | `algo/` | Transformer 原理、对齐训练、推理优化 |
| 6 | 工程深度（MCP 协议 / 多 Agent / 消融实验 / 后端基础） | `api/` + `agent/` | MCP/A2A 协议、多 Agent 协作、项目量化 |
| 7 | 垂直落地 + 面试冲刺（全链路串联 / 模拟面试） | 项目复盘 | 系统设计、项目故事线、简历话术 |

---

## 第 1 周：Agent 框架基础

> **目标**：手写 Agent 核心循环，对比原生 SDK vs LangChain vs LangGraph 三种实现方式。
>
> **参读项目**：hello-agents（概念扫盲）、learn-claude-code（Agent Loop 机制）

### Day 1 — Chain 与 Prompt Template

**文件**：`agent/langchain_basics.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | `ChatPromptTemplate` + `MessagesPlaceholder` | 变量注入正确，缺变量时报错 |
| 2 | LCEL `prompt \| model \| StrOutputParser()` | 输出纯字符串，不包含 choices[0].message.content |
| 3 | `RunnableParallel` 同时跑三种翻译风格 | 一次 invoke 拿到三个结果 |
| 4 | `.bind(temperature=0)` 固定模型参数 | temperature 生效 |
| 5 | 原生 SDK vs LCEL 实现同一功能，写注释分析差异 | 理解抽象层价值 |

**面试话术**："LangChain 的价值不是封装 API 调用，而是通过 Runnable 协议把 prompt、model、parser 组合成可复用的管道。LCEL 的 `|` 语法让数据流显式可见。"

---

### Day 2 — Memory 与 RAG 封装

**文件**：`agent/langchain_rag.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | `RunnableWithMessageHistory` + 简易 dict 存储 | 第二轮回答能引用第一轮内容 |
| 2 | Buffer vs Summary vs WindowMemory 三种策略 token 消耗对比 | 打印每种策略的 messages 长度 |
| 3 | `RunnableLambda` 封装第五章检索函数，串联全链路 | 端到端跑通 |
| 4 | `HybridRetriever`（继承 `BaseRetriever`） | 注册后可与任何 LangChain 组件串联 |
| 5 | `RunnablePassthrough` 优化数据流 | 消除多余透传函数 |

**面试话术**："Memory 不是简单存历史消息，生产项目用 SummaryMemory 做 token 消耗和上下文质量的最佳平衡。LangChain 的 Runnable 协议让 RAG 管线从命令式变为声明式。"

---

### Day 3 — 原生 SDK Function Calling

**文件**：`agent/function_calling_raw.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 定义 3 个汽车导购工具，手写 JSON Schema | description 和 parameters 完整 |
| 2 | 手写 tool calling 循环：`while True` + 判断 `content` / `tool_calls` | 单轮工具调用跑通 |
| 3 | 连续两次工具调用场景（先查车型→再查价格） | 多轮 tool calling 跑通 |
| 4 | 触发并行 tool calling（一次请求返回多个 tool_calls） | 并行执行，结果都喂回 |
| 5 | 对比 `tool_choice: "auto"` vs `"required"` vs `"none"` | 三种模式行为差异清晰 |
| 6 | 错误处理：工具执行失败时返回错误信息给 LLM | LLM 能基于错误调整策略 |

**面试话术**："Function Calling 的本质是 LLM 输出结构化指令而非文本，代码负责执行并反馈结果。手写过完整的 tool calling 循环才能理解 LangChain/LangGraph 在背后帮你省了什么。"

---

### Day 4 — LangGraph StateGraph 核心

**文件**：`agent/langgraph_agent.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | `AgentState` TypedDict + `add_messages` reducer | 理解消息追加机制 |
| 2 | `@tool` 定义工具 + `ToolNode` 自动执行 | 替代 Day 3 手写解析 |
| 3 | 组图：`add_node` × 2 + `add_conditional_edges` + `add_edge` | 图编译成功 |
| 4 | 无工具输入 → 直接 END | messages 只有 2 条 |
| 5 | 单工具调用 → agent→tools→agent 路径 | 有 ToolMessage |
| 6 | 多轮工具调用 → 图自动循环 | 循环次数正确 |
| 7 | `MemorySaver` Checkpointer，同 thread_id 多轮对话 | 第二轮引用第一轮 |
| 8 | 生成 Mermaid 流程图 | 与 Day 3 while 循环对照理解 |

**面试话术**："LangGraph 的核心是把 Agent 建模成有向图：State 是流动的数据，Node 是操作，条件边是路由决策。相比手写 while 循环，图的可组合性和可观测性是生产落地的关键优势。"

---

### Day 5 — ReAct Agent（汽车导购实战）

**文件**：`agent/car_advisor_agent.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 定义汽车导购专有工具集（≥5 个工具） | 覆盖询价、推荐、对比、评测、参数查询 |
| 2 | System Prompt 五段式设计（角色+能力边界+决策准则+输出约束+错误处理） | Prompt 设计有依据 |
| 3 | `create_react_agent` 一行构建 ReAct Agent | 与 Day 4 手写 StateGraph 对比 |
| 4 | Agent 执行轨迹打印（Thought → Action → Observation 每步可视化） | 可调试 |
| 5 | 对比 `create_react_agent` vs 手写 StateGraph 的代码量差异 | 理解框架价值 |

**面试话术**："ReAct 框架的核心是 Think → Act → Observe 循环。LangGraph 把这三步建模为图中的节点，条件边处理路由。框架的价值不是节省代码，而是让系统行为可预测、可观测、可恢复。"

---

## 第 2 周：Prompt 工程与模型调用

> **目标**：掌握生产级 Prompt 设计方法论，从"拼字符串"升级为"可评测可迭代的 Prompt 工程"。
>
> **参读项目**：learn-claude-code（Prompt 组装上下文裁剪机制）

### Day 6 — 四种 CoT 策略对比

**文件**：`prompt/cot_comparison.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Zero-shot CoT："请一步步思考" | 输出包含推理步骤 |
| 2 | Few-shot CoT：3 个示例引导推理格式 | 格式与示例一致 |
| 3 | Self-Consistency：同问题跑 5 次取多数答案 | 投票机制生效 |
| 4 | Auto-CoT：自动聚类 + 选代表性问题生成示例 | 多样性覆盖 |
| 5 | 四种策略在同一测试集上对比准确率 | 有量化对比表格 |

**面试话术**："CoT 不是简单加一句'请思考'——Zero-shot 省 token 但推理质量不稳定，Few-shot 效果好但示例挑选是瓶颈，Self-Consistency 用采样+投票解决单次推理的随机性。"

---

### Day 7 — 动态 Few-shot 选择器

**文件**：`prompt/few_shot_selector.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 基于关键词的简单选择器 | 关键词匹配正确 |
| 2 | 基于语义相似度的选择器（Embedding 选 Top-K） | 语义相近的示例被选中 |
| 3 | MMR（最大边际相关度）选择器 — 兼顾相似度 + 多样性 | 示例不重复 |
| 4 | 对比三种选择器的选择结果差异 | 有对比分析 |

**面试话术**："Few-shot 示例不是越多越好，关键是多样性和代表性。MMR 算法在语义相关度和示例间差异度之间做了平衡，避免选出的 3 个示例回答思路完全一样。"

---

### Day 8 — 结构化输出

**文件**：`prompt/structured_output.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Pydantic BaseModel 定义输出结构 | Schema 定义完整 |
| 2 | `response_format: {"type": "json_schema"}` 强制 JSON 输出 | 100% 合法 JSON |
| 3 | LangChain `with_structured_output()` 一行绑定 | 返回 Pydantic 对象 |
| 4 | 多层嵌套结构（车型→参数→价格→智驾子对象） | 嵌套解析正确 |
| 5 | 错误处理：字段校验失败时重试 | 重试后得到合法结果 |

**面试话术**："结构化输出让 LLM 从'生成文本'变成'填充 Schema'。json_schema 模式在 API 层面就保证输出合法 JSON，配合 Pydantic 做第二层校验，双层保险。"

---

### Day 9 — 图片识别 + RAG

**文件**：`prompt/vision_rag.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Base64 编码图片 → `image_url` 传给多模态模型 | 模型正确返回图片描述 |
| 2 | 图片描述文本 → Embedding → 向量检索 | 图片内容参与检索 |
| 3 | 端到端：上传汽车图片 → 识别车型 → RAG 检索参数 → 回答 | 全链路跑通 |
| 4 | 对比纯文本 RAG vs Vision RAG 在图片场景的差异 | 有对比分析 |

**面试话术**："多模态 RAG 的核心是把图片先转成文本描述再做 Embedding，图片本身不直接进向量库。关键设计点：图片描述的质量直接决定后续检索的召回率。"

---

### Day 10 — 综合增强 Agent v2

**文件**：`prompt/car_advisor_v2.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | CoT Prompt + Structured Output + Tool Calling 三者融合 | 三种能力同时生效 |
| 2 | 工具返回数据格式化为结构化上下文 | LLM 能理解工具返回 |
| 3 | Agent 对话日志：每轮记录 Thought / Action / Observation / Error | 日志结构化 |
| 4 | 10 个测试 case 的端到端评测 | 成功率 > 80% |

**面试话术**："生产级 Agent 需要三层 Prompt 协同——外层 System Prompt 定角色和约束，中层工具描述引导调用决策，内层 CoT Prompt 保证推理质量。"

---

## 第 3 周：生产化部署

> **目标**：把 RAG Agent 从命令行脚本变成可部署的 API 服务。
>
> **参读项目**：learn-harness-engineering（工程边界、校验、观测、失败恢复）

### Day 11 — FastAPI 包装 RAG 服务

**文件**：`api/main.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | `/chat` 同步接口（POST → RAG → 返回） | curl 能调通 |
| 2 | `/health` 健康检查 | 返回 200 |
| 3 | Pydantic Request/Response Schema 定义 | 自动生成 OpenAPI 文档 |
| 4 | middleware：请求日志 + 耗时统计 | 日志有 timestamp + latency |
| 5 | `BackgroundTasks` 异步写入日志/统计 | 不影响主请求延迟 |

**面试话术**："FastAPI 包装 Agent 的关键不是能调通，而是 Request/Response Schema 约束——Agent 入参和出参都要结构化定义，方便前端对接和自动化测试。"

---

### Day 12 — SSE 流式输出

**文件**：`api/stream.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | `StreamingResponse` + `text/event-stream` | 浏览器 EventSource 能接收 |
| 2 | LLM `stream=True` → 逐 token yield | 首 token 延迟 < 2s |
| 3 | 流式 + 非流式双模式（`?stream=true`） | 两种模式都正常 |
| 4 | 中断处理：客户端断开 → 停止生成 | 不浪费 token |
| 5 | 流式过程中传输检索来源（`"type": "source"` 事件） | 前端能展示来源 |

**面试话术**："SSE 相比 WebSocket 的优势是单向就够了——LLM 生成是单向数据流。关键指标是 TTFT（首 token 延迟），生产环境要求 < 1.5 秒。"

---

### Day 13 — Redis 会话管理

**文件**：`api/session_manager.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Redis 存储对话历史（session_id → messages JSON） | 对话持久化 |
| 2 | TTL 自动过期（默认 30 分钟） | 过期后新对话开始 |
| 3 | 用户身份校验（API Key / JWT） | 未认证请求返回 401 |
| 4 | 并发限制：同用户最多 N 个并发请求 | 超过限制返回 429 |
| 5 | 会话列表/删除/重命名 API | CRUD 完整 |

**面试话术**："Redis 做会话存储的三个关键：TTL 自动清理过期会话（省内存）、Pipeline 批量读写（降低 RTT）、分布式锁防止同 session 并发写入导致的 history 混乱。"

---

### Day 14 — Docker 容器化

**文件**：`Dockerfile` + `docker-compose.yml`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Dockerfile：multi-stage build（build → runtime） | 镜像体积优化 |
| 2 | docker-compose：app + Redis 双容器 | `docker compose up` 一键启动 |
| 3 | 模型文件挂载 volume（不打包进镜像） | 镜像 < 2GB |
| 4 | 环境变量注入（.env → docker-compose env_file） | API key 不入镜像 |
| 5 | 健康检查配置（`HEALTHCHECK` + `depends_on`） | Redis 就绪后 app 才启动 |

**面试话术**："Multi-stage build 的核心思路：build 阶段装编译依赖，runtime 阶段只保留运行需要的文件。模型文件不打包进镜像，通过 volume 挂载，方便热更新不重建镜像。"

---

### Day 15 — 熔断 + 限流

**文件**：`api/resilience.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Circuit Breaker 三态：CLOSED → OPEN → HALF_OPEN | 连续失败 N 次后熔断 |
| 2 | HALF_OPEN 探活：放行少量请求测试恢复 | 成功后关闭熔断 |
| 3 | Rate Limiter：滑动窗口 + Token Bucket 两种实现 | 超过限制返回 429 |
| 4 | LLM API 重试策略：指数退避（1s → 2s → 4s → 8s） | 临时故障自动恢复 |
| 5 | 降级兜底：LLM 不可用时返回预设话术 | 不返回 500 |

**面试话术**："熔断器的核心不是'保护 LLM API'，而是防止级联故障——LLM 挂了导致你的服务线程全堵在等待超时，进而拖垮整个服务。HALF_OPEN 探活机制保证恢复时不会瞬间打满。"

---

## 第 4 周：进阶能力

> **目标**：补齐生产级向量库、Agent 安全、模型微调基础、全功能界面。
>
> **参读项目**：craft-agents-oss（全栈 Agent 架构样板）

### Day 16 — 生产级向量库 Milvus

**文件**：`chapters/milvus_index.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Milvus Lite 本地部署（pip install pymilvus） | 无需 Docker |
| 2 | 建 Collection + 灌数据 + 建索引（IVF_FLAT） | 写入 + 查询正常 |
| 3 | 标量过滤：价格区间 + 类别 + 品牌混合过滤 | filter 表达式正确 |
| 4 | FAISS vs Milvus 性能对比（10 万条文档级） | QPS + 延迟对比 |
| 5 | 数据迁移脚本：第五章 JSON → Milvus Collection | 迁移无丢失 |

**面试话术**："Milvus 和 FAISS 不是替代关系——FAISS 是单机向量检索库，Milvus 是分布式向量数据库。核心区别：Milvus 支持标量过滤下推、动态数据增删、多副本高可用。"

---

### Day 17 — Agent 安全：Prompt Injection 防御

**文件**：`api/semantic_firewall.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 攻击样本集：指令覆盖、角色篡改、间接注入、编码绕过 | ≥ 10 种攻击模式 |
| 2 | 输入层：正则黑名单 + 语义防火墙（小模型二分类） | 拦截率 > 90% |
| 3 | 工具层：危险操作分级（只读/修改/删除）+ 确认机制 | 高危操作需二次确认 |
| 4 | 输出层：敏感信息脱敏（手机号、金额、Key） | 正则 + NER 双重脱敏 |
| 5 | 全链路安全测试：攻击样本 → 防火墙 → 工具审查 → 输出清洗 | 每一层日志可审计 |

**面试话术**："Prompt Injection 防御不能靠一层正则，需要四层体系：前置语义防火墙（拦截已知攻击模式）→ 工具调用参数审查（防止通过工具执行注入指令）→ 输出脱敏（防止泄露检索到的敏感信息）→ 全量审计日志（保证可追溯）。"

---

### Day 18 — LoRA 微调

**文件**：`finetune/lora_car_terms.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 构造汽车领域 SFT 数据集（instruction-input-output triplet） | ≥ 50 条高质量样本 |
| 2 | LoRA 配置：rank=8, target_modules=["q_proj", "v_proj"] | 参数量 < 1% |
| 3 | 训练 + 保存 adapter（不修改基座模型） | adapter 文件 < 10MB |
| 4 | 对比微调前后在汽车术语理解上的准确率 | 有量化提升 |
| 5 | 理解 LoRA 原理：`W' = W + (alpha/r) × A × B` 每项含义 | 能讲清公式 |

**面试话术**："LoRA 的核心假设：微调时的权重更新矩阵 ΔW 是低秩的。不用更新整个 4096×4096 矩阵，只训练两个小矩阵 A（4096×8）和 B（8×4096），参数量降 >99%，推理时合并回原权重零额外延迟。"

---

### Day 19 — Streamlit 全功能界面

**文件**：`app.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | `st.chat_input` + `st.chat_message` 对话界面 | 多轮对话正常 |
| 2 | 侧边栏：会话管理（新建/切换/删除）+ 检索来源展示 | 功能完整 |
| 3 | 工具调用可视化：展开看每步 Thought/Action/Observation | 可调试 |
| 4 | 设置面板：top_k / temperature / 检索策略切换 | 参数可调 |
| 5 | `st.session_state` 管理对话状态 | 刷新不丢失 |

**面试话术**："Streamlit 适合 Agent Demo 的快速搭建——`st.chat_message` 原生支持对话 UI，`st.session_state` 管理会话状态。生产前端换 React/Vue，但原型验证阶段 Streamlit 效率最高。"

---

### Day 20 — 联调 + Week 1-4 复盘

**文件**：`README.md`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 端到端跑通：Streamlit → FastAPI → RAG → LLM | 全链路无报错 |
| 2 | 20 天所有模块的 import 链路测试 | 模块间无循环依赖 |
| 3 | README：架构图 + 快速启动 + API 文档索引 | 新人能看懂 |
| 4 | Week 1-4 面试题自测（30 题） | 每题能答 2 分钟 |
| 5 | 记录当前技术短板 → Week 5-6 针对性补强 | 有明确清单 |

---

## 第 5 周：算法原理补课（大厂分水岭）

> **目标**：补齐 Transformer 原理、RLHF/GRPO 对齐训练、模型量化三大算法缺口。这是"会用工具"和"懂底层原理"之间的分界线，也是大厂二面/三面的核心区分度。
>
> **新增目录**：`algo/`

### Day 21 — Transformer 手写（Self-Attention 完整实现）

**文件**：`algo/transformer_scratch.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 纯 numpy 实现 `ScaledDotProductAttention(Q, K, V)` | 与 PyTorch 结果一致 |
| 2 | `MultiHeadAttention`：d_model=512, h=8, d_k=64 | 输出 shape 正确 |
| 3 | `PositionalEncoding`（Sinusoidal）+ `LayerNorm` | 位置编码公式手写 |
| 4 | Q/K/V 三个投影矩阵的维度推导 | 能在白板上画出计算图 |
| 5 | 时间复杂度分析：O(n²d) 为什么是瓶颈 | 能解释瓶颈来源 |

**面试必考题**："Self-Attention 的 Q/K/V 是怎么来的？Multi-Head 本质是什么？时间复杂度为什么是 O(n²d)？"

---

### Day 22 — KV Cache + RoPE + Flash Attention

**文件**：`algo/kv_cache_demo.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | KV Cache 原理：推理时缓存已计算的 K/V 矩阵 | 理解为什么省计算 |
| 2 | KV Cache 内存占用计算：2 × n_layers × n_heads × d_head × seq_len | 能估算显存 |
| 3 | RoPE（旋转位置编码）：为什么比 Sinusoidal 更好 | 理解相对位置编码 |
| 4 | Flash Attention 核心思想：分块计算 + 不存储完整 Attention 矩阵 | 会用，不手写 |
| 5 | 笔记输出：一页纸画清 Attention 演变路线 | 面试时能快速回顾 |

**面试话术**："KV Cache 的核心洞察：自回归生成时，之前的 K/V 不需要重新算。KV Cache 的显存占用随序列长度线性增长，这是长上下文推理的主要瓶颈。RoPE 的核心优势是只依赖相对位置，外推能力更强。"

---

### Day 23 — SFT → RLHF → PPO

**文件**：`algo/rlhf_intro.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | SFT 数据格式：instruction-input-output triplet | 能构造训练样本 |
| 2 | RLHF 三阶段流程图：SFT → RM → PPO | 画清每阶段输入/输出 |
| 3 | PPO Clip 机制手写 demo（numpy 实现公式） | `min(ratio × advantage, clip(ratio))` |
| 4 | 理解 Clip 的作用：防止策略更新步子太大导致 reward 崩溃 | 能举例说明 |
| 5 | Reference Model 的作用：KL 惩罚项防止模型偏离太远 | 理解 KL 散度约束 |

**面试话术**："PPO 的 Clip 机制本质是信任域——当新旧策略差异超过 ε 时停止更新，防止一次 bad update 毁掉整个模型。KL 惩罚是另一个约束：生成的 token 分布不能和 reference model 差太远。"

---

### Day 24 — GRPO + DPO（2026 面试热点）

**文件**：`algo/grpo_dpo_notes.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | GRPO 核心公式：组内相对排名替代绝对 reward | 与 PPO 对比图 |
| 2 | GRPO 为什么不需要 Critic：同一 prompt 采样多条，组内归一化 | 省一半显存 |
| 3 | DPO 原理：直接从偏好对学习，绕过 Reward Model | 与 RLHF 流程对比 |
| 4 | SFT 时为什么 mask observation tokens | 理解训练目标 |
| 5 | PPO → GRPO → DPO 演进路线图 + 各自适用场景 | 能画图讲清演进逻辑 |

**面试话术**："GRPO 是 DeepSeek-R1 提出的，核心创新是去掉了 Critic 模型——同一 prompt 下采样 G 条回答，组内相对排名本身就是 reward 信号，不需要额外训一个 Reward Model。这比 PPO 训练更稳定、显存少一半。"

---

### Day 25 — 模型量化 + 推理加速

**文件**：`algo/quantization_notes.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 量化本质：FP32 → INT8/INT4，内存/显存降 4 倍 | 能画精度对比图 |
| 2 | GPTQ vs AWQ vs SmoothQuant 三种方案对比 | 各方案适用场景清晰 |
| 3 | vLLM PagedAttention 原理：KV Cache 分页管理 | 理解为什么提高吞吐 |
| 4 | GGUF/llama.cpp 本地 CPU 推理 | 实际跑通一次量化推理 |
| 5 | 推理性能三指标：TTFT / TPOT / Throughput | 能解释各自含义 |

**面试话术**："vLLM 的 PagedAttention 借鉴了操作系统的虚拟内存思想：KV Cache 不再是一整块连续显存，而是切成固定大小的 page，按需分配、动态回收。这解决了碎片化问题，让并发吞吐量提升 20 倍以上。"

---

## 第 6 周：工程深度 + 项目量化

> **目标**：补齐 MCP 协议、多 Agent 协作、后端基础、消融实验四个工程缺口。这是从"能跑 Demo"到"能上生产"的关键。
>
> **参读项目**：craft-agents-oss（MCP 协议层、多 Agent 通信）

### Day 26 — MCP 协议实战

**文件**：`api/mcp_server.py` + `api/mcp_client.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | MCP Server：注册 Tools/Resources/Prompts | `list_tools` 返回工具列表 |
| 2 | MCP Client：发现 → 调用工具 → 获取结果 | 端到端跑通 |
| 3 | stdio vs SSE 两种传输方式对比 | 各自适用场景清晰 |
| 4 | MCP vs 传统 Function Calling 架构对比图 | O(N×M) → O(N+M) |
| 5 | MCP 安全：工具返回内容清理 + 来源校验 | 防 Context Poisoning |

**面试话术**："MCP 解决的不是'怎么调工具'，而是'怎么管理工具'。传统 Function Calling 每个 Agent 要对每个工具单独对接（O(N×M) 复杂度），MCP 通过标准协议把工具治理降到 O(N+M)。MCP 是垂直连接（Agent→工具），A2A 是水平连接（Agent↔Agent），两者分层协作。"

---

### Day 27 — 多 Agent 协作

**文件**：`agent/multi_agent.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 主从模式（Supervisor + 2 Workers）：LangGraph SuperGraph | 主 Agent 能路由到正确的 Worker |
| 2 | 共享消息池（Mailbox）通信 | Worker 间互不干扰 |
| 3 | 上下文隔离：每个子 Agent 只看到自己的 state | 不泄露其他 Agent 的信息 |
| 4 | 对比 Subagents vs Agent Teams 模式 | 能讲清选型依据 |
| 5 | 故障隔离：子 Agent 崩溃不影响主 Agent | 返回结构化错误信息 |

**面试话术**："多 Agent 架构的核心挑战不是通信，而是上下文隔离和故障隔离。每个子 Agent 只持有自己的 state，通过 mailbox 机制与主 Agent 通信。这样单个子 Agent 崩溃或死循环不会污染其他 Agent 的状态。"

---

### Day 28 — 消融实验 + RAGAs 幻觉评估

**文件**：`chapters/ablation_study.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 消融实验表格：完整 RAG / 去掉 Reranker / 去掉 Query Rewrite / 去掉 RRF / 只用单路 | 每个变体的 Hit Rate + MRR |
| 2 | RAGAs 框架：Faithfulness + Answer Relevance + Context Precision | 三项指标都有数据 |
| 3 | 延迟分析：P50 / P95 / P99 端到端延迟 | 有性能分布图 |
| 4 | 幻觉率统计：多少比例的回答包含不在 context 中的数据 | < 5% |
| 5 | 对比基线：与 naive RAG（第一章纯关键词）对比 | 提升有量化数字 |

**面试话术**："消融实验的结论不是'Reranker 有用'，而是'Reranker 在短查询场景提升 15%，在长文档场景提升 8%'——精确到场景的量化结论才让面试官信服你是在做工程，而不是抄 Demo。"

---

### Day 29 — 后端基础速通

**文件**：`api/backend_basics.py`（笔记 + 代码片段）

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | MySQL 索引：B+ 树原理 + 联合索引最左前缀原则 | 能 EXPLAIN 分析查询计划 |
| 2 | Redis 数据结构选型：String/Hash/List/Set/Sorted Set | 每种有 2 个应用场景 |
| 3 | 缓存穿透/击穿/雪崩 三种问题 + 解决方案 | 能画出架构图 |
| 4 | Redis 分布式锁：SET NX + Lua 脚本释放 | 锁不会误删 |
| 5 | 消息队列基础：Kafka Topic/Partition/Consumer Group | 理解消费语义 |

**面试话术**："MCP 解决的不是'怎么调工具'，而是'怎么管理工具'。传统 Function Calling 每个 Agent 要对每个工具单独对接（O(N×M) 复杂度），MCP 通过标准协议把工具治理降到 O(N+M)。"

---

### Day 30 — 成本控制 + Week 5-6 复盘

**文件**：`api/cost_tracker.py`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | Token 计数 + 费用实时追踪（输入/输出分别计价） | 每次请求有 cost 字段 |
| 2 | 缓存策略：相同 query 命中缓存跳 LLM 调用 | 缓存命中率统计 |
| 3 | 模型降级策略：满负荷时自动切轻量模型 | 降级对用户透明 |
| 4 | 成本优化建议文档（分级：小团队→中型→规模化） | 可执行 |
| 5 | Week 5-6 面试题自测（20 题算法 + 10 题工程） | 每题能答 2 分钟 |

---

## 第 7 周：垂直落地 + 面试冲刺

> **目标**：把所有模块串联成一个完整的业务闭环，写出能打动面试官的项目简历，完成面试逐字稿。
>
> **参读项目**：craft-agents-oss（全栈整合最终参照）

### Day 31 — 业务闭环：汽车导购 Agent 全功能集成

**文件**：项目全链路联调

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 所有工具收拢为专属业务动作（询价/推荐/对比/评测/参数/用车成本） | ≥ 8 个业务工具 |
| 2 | 完整的业务对话流：需求探索 → 车型推荐 → 深度对比 → 下单引导 | 能走完完整流程 |
| 3 | 工具调用的异常边界测试 | 每种工具有失败兜底 |
| 4 | 用户模糊意图的主动澄清（"大空间"→ 追问"几人家庭"） | Agent 能主动追问 |
| 5 | 拒绝不合理请求（"推荐最好的车"→ 追问预算和使用场景） | 不盲目调用工具 |

---

### Day 32 — 评测集构建

**文件**：`data/eval_set.json`

| 练习 | 内容 | 验证标准 |
|------|------|---------|
| 1 | 正常场景 × 30：不同预算/车型/场景的导购问答 | 对比车、推荐车、查参数 |
| 2 | 边界场景 × 10：无匹配车型、超预算、冷门需求 | Agent 不编造 |
| 3 | 攻击场景 × 10：Prompt Injection / 越权 / 诱导错误 | 防火墙拦截 |
| 4 | 评测指标体系：任务成功率 + 工具调用准确率 + 用户满意度 + 幻觉率 + 平均延迟 | 每项有 baseline |
| 5 | 评测脚本自动化：一键跑完全部 case → 出报告 | CI 可集成 |

---

### Day 33 — 系统设计面试模拟

**练习**：模拟 3 道大厂常见系统设计题

| 题目 | 要覆盖的要点 |
|------|------------|
| "设计一个智能客服 Agent 系统" | 五层架构、MCP 工具管理、会话记忆、人工兜底、熔断限流、多 Agent 路由 |
| "设计一个 RAG 系统每天处理 100 万次查询" | 向量库选型、缓存策略、多路召回、异步索引更新、灰度发布、成本控制 |
| "多 Agent 系统 Agent 间如何通信" | Supervisor/Peer-to-Peer/Hierarchical、Mailbox 模式、上下文隔离、共识机制 |

**每道题输出**：架构图 + 技术选型表格 + 关键数据流描述 + 边界/异常处理

---

### Day 34 — 简历项目书写 + GitHub 展示页

**文件**：项目 README 精修

**简历三块结构**（避免"基于大模型实现问答"这种泛泛描述）：

**① 架构表达**（证明你懂系统）：
> "基于 LangGraph 实现 ReAct Agent，包含 Agent Loop 循环、条件边路由、Tool Call 并行执行、Checkpointer 状态持久化。工具层以 MCP 协议暴露，支持动态发现和热插拔。"

**② 业务表达**（证明你能落地）：
> "聚焦汽车导购垂直场景，封装 8 个专属业务工具（K 线行情/持仓查询/市场扫描/预警通知/基本面分析等），完成从需求探索→车型推荐→深度对比→下单引导的完整业务闭环。"

**③ 结果表达**（证明你有数据意识）：
> "构建 50 条评测集（正常 30 + 边界 10 + 攻击 10），任务成功率 86%，RAGAs 忠实度 0.92，消融实验验证 Reranker 贡献 +15% Hit Rate（P95 延迟仅增加 120ms）。单卡成本控制在 0.003 元/次查询。"

---

### Day 35 — 终极面试逐字稿

**文件**：面试准备笔记

| 准备项 | 内容 |
|--------|------|
| 1 分钟自我介绍 | 我是谁 → 做过什么（Agent 完整项目）→ 量化成果（一句话）→ 为什么投这个岗 |
| 3 分钟项目介绍 | 背景 → 架构 → 我的贡献 → 踩过的坑 → 量化结果 → 反思改进 |
| 高频 30 题逐字稿 | 每道题准备"一句话结论 + 展开 + 代码例子"三段式回答 |
| "你还有什么问题" | 准备 3 个展示深度的问题（团队技术栈/Agent 架构演进/评估体系） |

---

## 大厂面试维度完整度矩阵

| 面试维度 | 权重 | 覆盖周 | 覆盖内容 | 完整度 |
|---------|------|-------|---------|-------|
| Agent 架构 + 推理框架 | 25% | Week 1-2, 7 | ReAct/LangGraph/StateGraph/Tool Calling/Agent Loop | ●●●●● 95% |
| RAG 全链路 | 20% | 前5章 + Week 1-2, 4 | Hybrid RRF/Milvus/Reranker/消融/RAGAs | ●●●●● 95% |
| Prompt 工程 | 15% | Week 2 | CoT/Few-shot/Structured/Vision/Pydantic | ●●●●● 90% |
| 工程化部署 | 20% | Week 3, 6 | FastAPI/SSE/Redis/Docker/熔断/MCP/多Agent | ●●●●○ 85% |
| 算法原理 | 20% | Week 5 | Transformer/RLHF/GRPO/DPO/量化/vLLM | ●●●●○ 80% |
| 后端基础 | 附加分 | Week 6 Day 29 | MySQL索引/Redis/消息队列 | ●●○○○ 50%（够用，需持续补） |
| 项目量化 | 核心区分 | Week 6-7 | 消融/评测集/RAGAs/成本/Ablation | ●●●●● 90% |

**加权总分：约 88 分。大厂面试过关线约 75-80 分。**

---

## 推荐阅读顺序

```
概念扫盲（hello-agents）
  → Agent 运行框架（learn-claude-code）
    → 工程边界（learn-harness-engineering）
      → 全栈架构（craft-agents-oss）
        → Spec/Plan 方案设计
          → 垂直场景落地 + 评测优化
```

---

## 简历项目书写模板

> **项目名称**：汽车导购智能 Agent 系统
>
> **架构表达**：基于 LangGraph 实现 ReAct Agent 循环，StateGraph 管理状态流转，ToolNode 处理工具并行调用，MemorySaver 持久化多轮对话。工具层以 MCP 协议标准化暴露，支持动态发现与热插拔。部署层采用 FastAPI + Redis + Docker Compose，实现 SSE 流式输出、会话 TTL 管理、熔断限流、Prompt Injection 四层防御。
>
> **业务表达**：聚焦汽车导购垂直场景，封装 8 个专属业务工具（车型查询/价格对比/参数检索/用户评价/智能推荐/用车成本/保值率/试驾预约），完成从需求探索→车型推荐→深度对比→下单引导的完整业务闭环。
>
> **结果表达**：构建 50 条分层评测集（正常 30 + 边界 10 + 攻击 10），任务成功率 86%，RAGAs 忠实度 0.92，Hit@3=90%，MRR=0.73。消融实验验证 Reranker 贡献 +15% Hit Rate（P95 额外延迟仅 120ms），Query Rewrite 在多轮场景贡献 +22% Recall。单次查询成本控制在 0.003 元。

---

## 学习补充原则

1. **先建立工程判断力，再写代码**：先看懂成熟 Agent 运行框架，理解模型/工具/记忆/权限协同逻辑
2. **落地产品必须配套**：工具约束 + 评测集 + 故障分析 + 成本管控，不能只做对话 Demo
3. **量化一切成果**：准确率、召回率、幻觉率、延迟 P50/P95/P99，拒绝"提升了一些"
4. **善用 AI 编程辅助**：Claude Code 生成方案、文档、代码，完成迭代
5. **最终目标**：能独立解释 Agent 整套运行机制，产出可演示、可复盘、可写进简历的完整工程作品
6. **面试反模式**：只会调 API、没有量化指标、不懂底层原理、Java 后端基础一片空白
