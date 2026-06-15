# Agent 开发完整闭环 — 学习计划

> 适用场景：已完成 RAG 基础（5章），对标 JD 补齐 Agent 框架 / Function Calling / 生产部署 / 进阶能力
>
> 详细内容见 [learning_plan.md](learning_plan.md) — 本文件为目录索引

## 四周概览

| 周 | 主题 | 产出目录 | 覆盖 JD |
|----|------|----------|---------|
| 1 | Agent 框架（LangChain→LangGraph→ReAct） | `agent/` | ② 大模型与Agent核心 |
| 2 | Prompt工程（CoT/Few-shot/结构化/多模态） | `prompt/` | ④ Prompt与模型调用 |
| 3 | 生产化（FastAPI/SSE/Redis/Docker/熔断限流） | `api/` | ⑤ 后端 / ⑥ 中间件 |
| 4 | 进阶（Milvus/Agent安全/LoRA/Streamlit） | `chapters/` + `finetune/` + `app.py` | 加分项 |

## 每日任务速查

### 第一周
- Day 1: `agent/langchain_basics.py` — Chain 与 Prompt Template
- Day 2: `agent/langchain_rag.py` — Memory 与 RAG 封装
- Day 3: `agent/function_calling_raw.py` — 原生 SDK 实现 tool calling
- Day 4: `agent/langgraph_agent.py` — StateGraph 核心
- Day 5: `agent/car_advisor_agent.py` — ReAct Agent

### 第二周
- Day 6: `prompt/cot_comparison.py` — 四种 CoT 策略对比
- Day 7: `prompt/few_shot_selector.py` — 动态 few-shot
- Day 8: `prompt/structured_output.py` — Pydantic + json_schema
- Day 9: `prompt/vision_rag.py` — 图片识别 + RAG
- Day 10: `prompt/car_advisor_v2.py` — 综合增强 Agent

### 第三周
- Day 11: `api/main.py` — FastAPI 包装 RAG 服务
- Day 12: `api/stream.py` — SSE 流式输出
- Day 13: `api/session_manager.py` — Redis 会话管理
- Day 14: `Dockerfile` + `docker-compose.yml` — 容器化
- Day 15: `api/resilience.py` — 熔断 + 限流

### 第四周
- Day 16: `chapters/milvus_index.py` — 生产级向量库
- Day 17: `api/semantic_firewall.py` — Agent 安全
- Day 18: `finetune/lora_car_terms.py` — LoRA 微调
- Day 19: `app.py` — Streamlit 全功能界面
- Day 20: `README.md` — 联调 + 面试复盘
