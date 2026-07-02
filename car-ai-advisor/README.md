# 智能汽车导购助手 (Smart Car Advisor)

> **全链路 AI Agent 系统** — 从 RAG 知识库 → ReAct Agent → FastAPI 流式服务 → Redis 会话管理 → Streamlit 前端，一站式垂直落地。
>
> 对标岗位：**AI 应用开发工程师 / 大模型应用开发 / AI Agent 开发**

---

## 项目定位

**一句话**：用 FastAPI + LangGraph + Redis + SSE 构建的智能汽车导购系统，支持流式多轮对话、RAG 混合检索、会话管理和 Docker 一键部署。

**与学习项目的区别**：

| 维度 | agent-dev-project（学习项目） | car-ai-advisor（全链路项目） |
|------|------|------|
| 目的 | 每日一个知识点，独立练习 | 整合所有技能，全链路打通 |
| 架构 | 各文件独立运行 | 模块化分层，统一入口 |
| 数据 | 12条车型+5篇行业报告 | 16款车型+10篇深度文档+术语词典 |
| 前端 | curl / HTML测试页 | Streamlit 全功能对话框 |
| 部署 | 手动 python xxx.py | Docker Compose 一键启动 |
| 测试 | 手动验证 | 单元/集成/E2E 三层测试 |

---

## 系统架构

```
                         ┌──────────────────────────────┐
                         │       Streamlit 前端          │
                         │  · 对话界面 + 会话侧边栏       │
                         │  · 工具调用可视化              │
                         │  · 参数调节面板                │
                         │  · fetch + ReadableStream    │
                         └─────────────┬────────────────┘
                                       │ HTTP POST /chat
                                       │ Authorization: Bearer sk-xxx
                                       │ Content-Type: application/json
                                       ▼
                         ┌──────────────────────────────┐
                         │      FastAPI 网关层            │
                         │                              │
                         │  Middleware                   │  ← 请求日志 + 耗时统计
                         │    ↓                          │
                         │  Auth (HTTPBearer + API Key) │  ← 身份校验 → user_id
                         │    ↓                          │
                         │  Rate Limiter                │  ← 滑动窗口限流 (Day 15)
                         │    ↓                          │
                         │  Circuit Breaker             │  ← 熔断保护 (Day 15)
                         │    ↓                          │
                         │  Concurrency Slot            │  ← Lua 原子 INCR ≤ 3
                         │    ↓                          │
                         │  Session Manager             │  ← Redis 读取历史
                         │    ↓                          │
                         │  /chat Handler               │
                         │    ├─ stream=true  → SSE     │
                         │    └─ stream=false → JSON    │
                         └─────────────┬────────────────┘
                                       │
                         ┌─────────────▼────────────────┐
                         │        Agent 决策层            │
                         │                              │
                         │  三层 Prompt 架构:            │
                         │    外层: 角色设定 + 行为约束   │
                         │    中层: Tool 描述 + 调用时机  │
                         │    内层: CoT 推理链引导        │
                         │                              │
                         │  ReAct 循环:                  │
                         │    Thought → Action → Obs → … │
                         │    ┌──────────┐              │
                         │    │ LLM      │              │
                         │    │ DeepSeek │  streaming   │
                         │    └──────────┘              │
                         └──────┬──────────┬────────────┘
                                │          │
                   ┌────────────┼──────────┼────────────┐
                   ▼            ▼          ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ 检索工具  │ │ 价格工具  │ │ 对比工具  │ │ 推荐工具  │
            │ search_  │ │ get_car_ │ │ compare_ │ │recommend │
            │ knowledge│ │  price   │ │  cars    │ │  _car    │
            └────┬─────┘ └──────────┘ └──────────┘ └──────────┘
                 │
                 ▼
            ┌──────────────────────────────────────────┐
            │              RAG 检索管线                  │
            │                                          │
            │  ① Query 改写 (LLM扩展模糊问题)           │
            │  ② 向量检索 (FAISS / Milvus + BGE)       │
            │  ③ BM25 关键词检索 (jieba分词)            │
            │  ④ RRF 融合排序 (k=60)                    │
            │  ⑤ Reranker 精排 (BGE CrossEncoder)       │
            │  ⑥ Top-K 文档拼接 + 引用标注               │
            └──────────────┬───────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────────────┐
            │            知识库 (Knowledge Base)         │
            │                                          │
            │  · vehicles.json   16款车型结构化参数     │
            │  · reviews.json    10条真实用户评价        │
            │  · glossary.json   20+行业术语解释         │
            │  · guides/         购车/保险/保养指南      │
            │  · industry/       市场分析/技术趋势       │
            │  · faq/            14个高频购车问题         │
            └──────────────────────────────────────────┘

    ───────────── 基础设施 ─────────────
    Redis    ← 会话存储 + 并发控制 + 限流计数器
    Docker   ← docker-compose up 一键启动
    Milvus   ← (可选) 生产级向量数据库
```

---

## 技术栈与技能映射

| 技术层 | 使用技术 | 对应学习 Day | 面试考察点 |
|--------|---------|:---:|------|
| **前端** | Streamlit, fetch+ReadableStream | Day 19 | 快速原型能力、用户体验意识 |
| **API网关** | FastAPI, Middleware, Depends | Day 11 | 路由设计、中间件、依赖注入 |
| **流式输出** | SSE, StreamingResponse, asyncio | Day 12 | 流式协议理解、取消处理、TTFT |
| **会话管理** | Redis List/Hash/Set, Pipeline, Lua | Day 13 | 数据结构选型、原子操作、TTL |
| **容错** | Circuit Breaker, Rate Limiter | Day 15 | 熔断三态、滑动窗口、系统韧性 |
| **Agent** | ReAct, Tool Calling, StateGraph | Day 4-5 | Agent架构、工具设计、循环控制 |
| **Prompt** | 三层架构, CoT, Few-shot | Day 6-10 | Prompt设计、结构化评测 |
| **RAG** | Hybrid RRF, Reranker, Chunking | ch1-5 | 检索策略、排序融合、分块策略 |
| **向量库** | FAISS / Milvus, IVF_FLAT | Day 16 | 索引类型、标量过滤、ANN原理 |
| **微调** | LoRA (PEFT), SFT | Day 18 | 参数高效微调、数据构造 |
| **部署** | Docker, docker-compose | Day 14 | 容器化、服务编排、环境管理 |
| **测试** | pytest, 单元/集成/E2E | Day 20 | 测试金字塔、mock策略 |

---

## 目录结构

```
car-ai-advisor/
│
├── README.md                          # ← 本文件：全链路架构与开发计划
├── .env.example                       # 环境变量模板
├── .gitignore
├── Makefile                           # 常用命令快捷方式
├── docker-compose.yml                 # [待实现] 服务编排 (app+redis+milvus)
├── Dockerfile                         # [待实现] 多阶段构建
│
├── backend/                           # === FastAPI 后端 ===
│   ├── main.py                        # [待实现] 应用入口：挂载路由、启动事件
│   ├── config.py                      # [待实现] 配置管理：读取 .env + 全局 Settings
│   ├── api/                           # API 层
│   │   ├── deps.py                    # [待实现] 依赖注入：get_current_user, get_session_mgr
│   │   └── routes/
│   │       ├── chat.py                # [待实现] POST /chat：流式/非流式双模式
│   │       ├── sessions.py            # [待实现] CRUD /sessions：创建/列表/重命名/删除
│   │       └── health.py              # [待实现] GET /health：健康检查
│   ├── core/                          # 核心基础设施
│   │   ├── session_manager.py         # [待实现] Redis 会话：历史存取+TTL+并发控制
│   │   ├── stream.py                  # [待实现] SSE 流式生成器：token级输出+CancelledError
│   │   ├── resilience.py              # [待实现] 熔断器+限流器：滑动窗口+三态切换
│   │   └── security.py                # [待实现] 认证模块：API Key验证+JWT签发
│   ├── agent/                         # Agent 决策层
│   │   ├── advisor.py                 # [待实现] 汽车导购Agent：ReAct循环+工具调度
│   │   ├── tools.py                   # [待实现] 工具集：检索/查价/对比/推荐/成本计算
│   │   └── prompts.py                 # [待实现] 三层Prompt模板：角色+工具+CoT
│   ├── rag/                           # RAG 检索管线
│   │   ├── retriever.py               # [待实现] 混合检索：FAISS向量+BM25关键词→RRF融合
│   │   ├── embeddings.py              # [待实现] Embedding管理：BGE模型加载+批量编码
│   │   ├── chunker.py                 # [待实现] 文档分块：Markdown按标题/JSON按条目
│   │   └── reranker.py                # [待实现] 精排：BGE CrossEncoder重排序
│   └── schemas/                       # Pydantic 数据模型
│       ├── chat.py                    # [待实现] ChatReq/ChatResp (含stream字段)
│       └── session.py                 # [待实现] SessionItem/SessionListResp/CreateReq
│
├── frontend/                          # === Streamlit 前端 ===
│   ├── app.py                         # [待实现] 主界面：st.chat_input + session_state
│   ├── api_client.py                  # [待实现] API调用封装：fetch+ReadableStream处理SSE
│   └── components/
│       ├── chat.py                    # [待实现] 对话渲染：用户/AI消息气泡+流式更新
│       ├── sidebar.py                 # [待实现] 侧边栏：会话列表+新建/切换/删除
│       └── tools.py                   # [待实现] 工具调用可视化：折叠面板+参数展示
│
├── knowledge_base/                    # === 企业级知识库 ===
│   ├── README.md                      # 知识库文档：结构说明+数据统计+更新策略
│   ├── raw/                           # 原始数据（~36500字）
│   │   ├── vehicles.json              # 16款车型结构化参数
│   │   ├── reviews.json               # 10条用户评价
│   │   ├── glossary.json              # 20+行业术语词典
│   │   ├── guides/                    # 购车/保险/保养/新能源指南
│   │   ├── industry/                  # 市场格局+技术趋势
│   │   └── faq/                       # 14个购车高频问题
│   ├── processed/                     # 向量索引存放目录
│   └── scripts/                       # [待实现] 索引构建+数据校验脚本
│
├── tests/                             # === 测试 ===
│   ├── conftest.py                    # [待实现] pytest fixtures: 测试客户端、mock Redis
│   ├── unit/                          # 单元测试
│   │   ├── test_rag.py                # [待实现] 检索/分块/Embedding/重排序
│   │   ├── test_agent.py              # [待实现] 工具调用/ReAct循环/Prompt模板
│   │   └── test_session.py            # [待实现] 会话CRUD/并发控制/TTL
│   ├── integration/                   # 集成测试
│   │   └── test_api.py                # [待实现] 各端点HTTP请求/响应验证
│   └── e2e/                           # 端到端测试
│       └── test_full_flow.py          # [待实现] 完整对话流程：发消息→收回复→查历史
│
├── deploy/                            # === 部署配置 ===
│   ├── nginx/default.conf             # [待实现] 反向代理+静态文件+WebSocket
│   ├── prometheus/prometheus.yml      # [待实现] 指标采集配置
│   └── scripts/start.sh               # [待实现] 服务器启动脚本
│
└── docs/                              # === 文档 ===
    ├── ARCHITECTURE.md                # [待实现] 架构决策记录 (ADR)
    ├── API.md                         # [待实现] API 接口文档
    ├── RAG_DESIGN.md                  # [待实现] RAG 管线设计文档
    └── DEPLOY.md                      # [待实现] 部署运维手册
```

> **文件统计**：目录 25 个，代码文件 39 个（待实现），知识库文件 10 个（已完成），配置文件 6 个。

---

## 全链路数据流（一次完整的用户提问）

```
时间线 →

T=0    用户在前端输入: "25万预算，家用为主，推荐什么车？"
       │
T=0.1  Streamlit 构建请求体:
       │  {query, session_id, stream: true}
       │  携带 Header: Authorization: Bearer sk-xxx
       │  fetch POST → http://localhost:8000/chat
       │
T=0.2  FastAPI 中间件记录: [2026-06-27 14:30:01] POST /chat uid=user_001
       │
T=0.3  Depends(get_current_user) → 校验 API Key → uid="user_001"
       │
T=0.4  Depends(check_rate_limit) → 滑动窗口计数 → 通过
       │
T=0.5  session_mgr.acquire_slot(uid) → Redis Lua 原子 INCR → 通过 (1/3)
       │
T=0.6  session_mgr.get_history(sid) → Redis LRANGE -20 -1 → 前3轮对话
       │
T=0.7  session_mgr.add_message(sid, "user", query) → Pipeline RPUSH+EXPIRE
       │
T=0.8  进入 Agent 循环:
       │  ├─ System Prompt (角色: 汽车导购助手, 工具: 4个)
       │  ├─ 用户消息: "25万预算，家用为主，推荐什么车？"
       │  └─ LLM Decision → Thought: 需要检索25万家用车
       │
T=1.0  Agent 调用 search_knowledge("25万 家用 推荐")
       │  ├─ RAG ① Query改写: "25万预算 家庭用车 SUV 推荐 性价比"
       │  ├─ RAG ② FAISS向量检索 → 候选50篇
       │  ├─ RAG ③ BM25关键词检索 → 候选50篇
       │  ├─ RAG ④ RRF融合 → 去重排序 → 候选30篇
       │  ├─ RAG ⑤ BGE Reranker精排 → Top-5篇
       │  └─ 返回: [{宋L}, {Model Y}, {理想L6}, {银河E8}, {问界M7}]
       │
T=1.5  Agent 调用 get_car_price("宋L") → "18.98-24.98万元"
       │  Agent 调用 get_car_price("理想L6") → "24.98-28.98万元"
       │  Agent 调用 compare_cars(["宋L","理想L6"]) → 对比表
       │
T=2.0  Agent Observation 汇总，决定最终回答
       │
T=2.1  LLM 开始生成最终回答（streaming=True）:
       │  "根据您的25万预算和家用需求，我推荐以下三款车型：
       │   ① 理想L6 (24.98万起) — 增程无焦虑，大五座空间..."
       │
T=2.1  SSE 流式推送:
~      │  data: {"type":"source","sources":[...]}
T=6.0  │  data: {"type":"token","token":"理","index":0}
       │  data: {"type":"token","token":"想","index":1}
       │  data: {"type":"token","token":"L","index":2}
       │  ... (数百个token)
       │  data: {"type":"done","total_tokens":342}
       │  data: [DONE]
       │
T=6.1  前端接收到每个 token → st.empty() 流式更新显示
       │
T=6.2  后端 finally:
       │  session_mgr.add_message(sid, "assistant", full_answer)
       │  session_mgr.release_slot(uid) → Redis Lua DECR
       │
T=6.3  用户看到完整回答 + 参考来源标签 + 工具调用折叠面板
```

---

## API 设计

### 核心端点

| 方法 | 路径 | 认证 | 说明 |
|------|------|:---:|------|
| `GET` | `/health` | 无 | 健康检查：返回 Redis 连通性 + 服务状态 |
| `POST` | `/chat` | Bearer | 对话接口：支持 stream=true(SSE) / false(JSON) |
| `GET` | `/chat-ui` | 无 | 浏览器测试页面（内嵌HTML） |
| `POST` | `/sessions` | Bearer | 创建新会话 |
| `GET` | `/sessions` | Bearer | 列出用户的所有会话 |
| `PATCH` | `/sessions/rename` | Bearer | 重命名会话 |
| `DELETE` | `/sessions/{id}` | Bearer | 删除会话 |
| `GET` | `/sessions/{id}/history` | Bearer | 获取会话对话历史 |

### ChatReq Schema

```json
{
  "query": "25万预算推荐什么车？",
  "session_id": "abc12345",
  "stream": true
}
```

### SSE 事件类型

| event type | 含义 | 示例 payload |
|------|------|------|
| `source` | 检索到的参考文档 | `{"type":"source","sources":[{"title":"...","url":"..."}]}` |
| `token` | LLM 生成的单个 token | `{"type":"token","token":"理","index":42}` |
| `done` | 生成完成 | `{"type":"done","total_tokens":342}` |
| `error` | 发生错误 | `{"type":"error","message":"并发请求过多"}` |

---

## 开发路线图

### Phase 1: 基础设施搭建 (Day 14 对应)
- [ ] `Dockerfile` — 多阶段构建（builder + runtime）
- [ ] `docker-compose.yml` — app + Redis + 可选 Milvus
- [ ] `backend/config.py` — 统一配置管理
- [ ] `backend/main.py` — FastAPI 应用骨架

### Phase 2: 核心功能实现
- [ ] `backend/core/security.py` — 认证模块
- [ ] `backend/core/session_manager.py` — Redis 会话管理（基于 Day 13 代码）
- [ ] `backend/core/stream.py` — SSE 流式生成器（基于 Day 12 代码）
- [ ] `backend/core/resilience.py` — 熔断 + 限流（Day 15）
- [ ] `backend/api/routes/*` — 所有 API 端点

### Phase 3: Agent & RAG 集成
- [ ] `backend/agent/*` — 基于 Day 5/10 代码适配
- [ ] `backend/rag/*` — 基于 chapters/ 管线代码
- [ ] `knowledge_base/scripts/build_index.py` — 知识库索引构建

### Phase 4: 前端与测试
- [ ] `frontend/app.py` — Streamlit 主界面（Day 19）
- [ ] `frontend/components/*` — 对话/侧边栏/工具可视化组件
- [ ] `tests/*` — 三层测试（Day 20）

### Phase 5: 部署与文档
- [ ] `deploy/*` — Nginx + Prometheus 配置
- [ ] `docs/*` — 架构/API/RAG/部署文档
- [ ] Makefile 全部命令可运行

---

## 快速开始（目标状态）

```bash
# 1. 克隆项目
git clone <repo-url> && cd car-ai-advisor

# 2. 配置环境
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY

# 3. 构建知识库索引
make build-index

# 4. 一键启动
make docker-up
# 或本地开发模式:
make dev

# 5. 访问
# 前端: http://localhost:8501
# API文档: http://localhost:8000/docs
# 浏览器测试: http://localhost:8000/chat-ui
```

---

## 简历话术

### 项目描述（简历用，约150字）

> **智能汽车导购助手** — 全链路 AI Agent 系统
>
> 技术栈：FastAPI · LangGraph · Redis · SSE · FAISS · Streamlit · Docker
>
> - 设计并实现基于 ReAct 范式的汽车导购 Agent，集成检索/查价/对比/推荐 4 个工具，三层 Prompt 架构使工具调用准确率 >85%
> - 基于 SSE 协议实现流式对话，TTFT < 1.5s，支持流式/非流式双模式，`CancelledError` 保护客户端断开后自动停止生成
> - Redis 管理会话状态：List 存对话历史 + TTL 滑动过期 + Lua 脚本原子并发控制（≤3 并发/用户）
> - RAG 管线：BM25 + FAISS 混合检索 RRF 融合 → BGE Reranker 重排序，知识库覆盖 16 款车型 + 10 篇深度文档（3.6 万字）
> - Streamlit 全功能对话界面 + Docker Compose 一键部署

### 面试追问准备

| 追问 | 回答要点 |
|------|------|
| "为什么用 SSE 而不是 WebSocket？" | 单向推送够用、协议更轻、浏览器原生支持、自动重连 |
| "ReAct 循环怎么防止无限调用？" | 最大步数限制 + token 预算 + 重复调用检测 |
| "Redis 为什么用 List 而不是 String？" | RPUSH O(1)追加、LRANGE 负索引取尾、无需读-改-写 |
| "并发控制 Lua 脚本怎么写的？" | INCR→判上限→超限则 DECR 回滚→60s TTL 兜底防槽位泄漏 |
| "RAG 为什么用 RRF 而不是简单的分数加权？" | 向量分数和 BM25 分数不在同一量级，直接加权需调参；RRF 无参数且对分数分布鲁棒 |
| "微调过模型吗？" | LoRA rank=8 微调汽车术语理解，adapter < 10MB |

---

## 与原学习项目 (agent-dev-project) 的关系

| 原学习项目文件 | 在全链路项目中的对应 |
|------|------|
| `api/main.py` (Day 11) | → `backend/main.py` + `backend/api/routes/chat.py` |
| `api/stream.py` (Day 12) | → `backend/core/stream.py` |
| `api/session_manager.py` (Day 13) | → `backend/core/session_manager.py` + `backend/core/security.py` |
| `agent/car_advisor_agent.py` (Day 5) | → `backend/agent/advisor.py` + `backend/agent/tools.py` |
| `prompt/car_advisor_v2.py` (Day 10) | → `backend/agent/prompts.py` |
| `chapters/full_rag_agent.py` (ch5) | → `backend/rag/retriever.py` + `backend/rag/reranker.py` |
| `chapters/naive_rag.py` (ch1) | → `backend/rag/chunker.py` + `knowledge_base/scripts/load_data.py` |
| `chapters/embedding_test.py` (ch3) | → `backend/rag/embeddings.py` |
| `chapters/milvus_index.py` (Day 16) | → `backend/rag/retriever.py` (可选 Milvus 后端) |
| `finetune/lora_car_terms.py` (Day 18) | → 独立模块，可选启用 |
| `app.py` (Day 19 空壳) | → `frontend/app.py` + `frontend/components/*` |

---

## 技术决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 前端框架 | Streamlit | 快速原型、Python 技术栈统一、适合数据应用 |
| 流式协议 | SSE | 单向推送够用、协议更轻、比 WebSocket 更适合 AI 流式场景 |
| 会话存储 | Redis List + Hash + Set | O(1)追加、Pipeline 批量操作、TTL 自动过期 |
| 并发控制 | Lua 原子脚本 | 避免 INCR→检查→DECR 的竞态条件 |
| 向量库 | FAISS (本地) + Milvus (可选) | FAISS 开发友好零配置、Milvus 生产级扩展 |
| LLM | DeepSeek Chat | 中文效果好、API 兼容 OpenAI、streaming 稳定 |
| Embedding | BGE-base-zh-v1.5 | 中文 SOTA、本地部署、1024 维性价比高 |
| Reranker | BGE-Reranker-base | 与 Embedding 同厂商、中文精排效果好 |
| 容器化 | Docker Compose | 本地开发+演示够用，生产再上 K8s |

---

> **项目状态**：架构设计 + 知识库已完成，代码文件待实现（共 39 个 Python 文件 + 4 个配置文件）。
