# Naive RAG 完整流程

## 什么是 RAG？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种让大语言模型能够回答"它没学过"的问题的技术。

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. 加载数据   │ ──→ │  2. 检索相关   │ ──→ │  3. 组 Prompt │ ──→ │  4. 调 LLM   │
│  load_data()  │     │  search()     │     │ build_prompt()│     │  ask_llm()   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       ↑                    ↑                    ↑                    ↑
   cars_specs.json     jieba 分词           system prompt         openai API
   industry_reports/   关键词匹配             + context            流式 / 非流式
   user_reviews.json   Top-K 排序            + query              temperature=0
```

核心理念：**不让 LLM 凭记忆回答 → 先把相关资料找出来 → 把资料和问题一起喂给 LLM → LLM 照资料回答。**

---

## 数据流总览

```
用户输入 query
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  search(query, documents)                            │
│  ① jieba 分词 → 关键词列表                           │
│  ② 遍历文档，统计每个关键词在 content 中的命中次数     │
│  ③ 按分数降序 → 取 top_k                             │
│  ④ 拼成上下文字符串                                   │
└─────────────────────────────────────────────────────┘
      │ context (字符串)
      ▼
┌─────────────────────────────────────────────────────┐
│  build_prompt(query, context)                        │
│  → [{"role":"system","content":"你是导购..."},       │
│     {"role":"user","content":"【参考资料】...\n       │
│                              【用户问题】..."}]       │
└─────────────────────────────────────────────────────┘
      │ messages (列表)
      ▼
┌─────────────────────────────────────────────────────┐
│  ask_llm(messages)                                   │
│  → openai.chat.completions.create(...)               │
│  → response.choices[0].message.content               │
└─────────────────────────────────────────────────────┘
      │ answer (字符串)
      ▼
  输出给用户
```

---

## 逐函数详解

### 1. `load_data(data_dir)` — 数据加载与拍平

**输入**：数据目录路径  
**输出**：`list[dict]`，每个 dict 格式为 `{"content": "文本", "source": "来源名", "type": "类型"}`

**为什么需要统一格式？**
项目有三种异构数据源，但后续检索只认"文本"。统一格式让检索逻辑对三种数据一视同仁。

| 数据源 | 原始格式 | 拍平函数 | 输出 type |
|--------|---------|---------|-----------|
| `cars_specs.json` | 嵌套 dict（20款车） | `car_to_text()` | `car_spec` |
| `industry_reports/*.txt` | 纯文本（5份报告） | 直接 `f.read()` | `industry_report` |
| `user_reviews.json` | 嵌套 dict（N条评价） | `review_to_text()` | `user_review` |

#### `car_to_text(car)` — 结构化 → 自然语言

```python
# 输入
{"full_name": "比亚迪海豹08", "price_range": "25-32万元",
 "powertrain": {"battery_capacity_kwh": 100, "cltc_range_km": 720, ...},
 "key_features": ["兆瓦闪充", "激光雷达", ...]}

# 输出
"""比亚迪海豹08 中大型智能运动旗舰轿车，中大型纯电轿车，售价25-32万元。
动力方面：纯电动，100kWh电池，CLTC续航720km，充电：兆瓦级闪充，5分钟补能400km。
性能：380kW电机，零百加速3.8秒。
智驾：L2+，DiPilot 300，有激光雷达。
亮点：兆瓦闪充、激光雷达、Ocean-S概念车设计、云辇智能车身控制。"""
```

**关键技巧**：
- 全部用 `.get(key, default)` 而不是 `[key]`，字段缺失时不崩
- 嵌套字段先取父级（`p = car.get("powertrain", {})`），再拿子字段
- `"、".join(list)` 把列表转成中文顿号分隔的字符串

#### `review_to_text(review)` — 评价 → 自然语言

```python
# 输入
{"model": "小米SU7", "rating": 4.5,
 "pros": ["续航扎实", "加速惊艳"],
 "cons": ["后排头部空间一般"],
 "advice": "城市通勤为主，标准版足够"}

# 输出
"""车型：小米SU7 Max 2026款，车主：深圳车主，购车1个月，评分：4.5/5。
优点：续航扎实、加速惊艳。
缺点：后排头部空间一般。
建议：城市通勤为主，标准版足够"""
```

---

### 2. `search(query, documents, top_k=3)` — 关键词检索

这是整个 RAG 管线中最核心的环节。检索质量直接决定 LLM 回答质量。

**流程**：

```
query = "续航超过700公里的纯电轿车"
         │
         ▼ jieba.cut()
["续航", "超过", "700", "公里", "的", "纯电", "轿车"]
         │
         ▼ 过滤停用词
["续航", "700", "公里", "纯电", "轿车"]   ← keywords
         │
         ▼ 对每篇文档打分
documents[i]:
  if "续航" in content: score += 1
  if "700" in content:  score += 1
  ...
         │
         ▼ 按 score 降序，取 top_k
[(3, doc_a), (2, doc_b), (2, doc_c)]  ← 前 3 且 score > 0
         │
         ▼ 拼成上下文字符串
"""【来源：比亚迪 海豹08】
比亚迪海豹08，...续航720km...
---
【来源：小米 SU7】
小米SU7，...续航902km..."""
```

**关键技术决策**：

| 决策点 | 选了 | 为什么 | 代价 |
|--------|------|--------|------|
| 分词工具 | `jieba` | 中文无空格，必须用分词器 | 引入额外依赖 |
| 匹配方式 | 子串包含 `kw in content` | 简单直观 | "700" 匹配不到 "720" |
| 打分方式 | 命中次数（0/1） | 快 | 高频词淹没信号 |
| 排序 | 按 score 降序 | 最直接的信号 | 无"预算约束"语义理解 |

**停用词**：过滤掉出现频率太高、无区分度的词（如"的""了""纯电""轿车"），让真正有区分度的关键词（品牌名、数字）发挥筛选作用。

---

### 3. `build_prompt(query, context)` — 组装对话

**输入**：用户问题 + 检索到的上下文  
**输出**：标准 OpenAI Chat Messages 格式

```python
[
    {
        "role": "system",
        "content": "你是一个专业的汽车导购助手。请严格根据下面提供的上下文信息来回答..."
    },
    {
        "role": "user",
        "content": "【参考资料】\n...\n【用户问题】\n...\n请根据以上参考资料回答。"
    }
]
```

**System Prompt 的设计要点**：
- **角色定义**："专业的汽车导购助手"——约束回答风格
- **信息来源约束**："严格根据上下文，不要编造"——防幻觉
- **引用要求**："引用具体的数据来源和数字"——可追溯

---

### 4. `ask_llm(messages)` — 调用大模型

**配置来源**：`.env` 文件，与项目其他模块共享

| 环境变量 | 作用 | 示例 |
|----------|------|------|
| `LLM_API_KEY` | API 密钥 | `sk-xxx` |
| `LLM_BASE_URL` | API 地址（兼容 OpenAI 接口） | `https://api.openai.com/v1` |
| `LLM_MODEL_ID` | 模型名 | `gpt-4o` / `deepseek-chat` |

**关键参数**：
- `temperature=0`：输出稳定、确定性强，适合事实型问答
- 非流式调用（`stream=False`）：简单直接

---

## 踩坑记录

| 坑 | 现象 | 根因 | 解法 |
|----|------|------|------|
| 路径错误 | `No such file or directory` | 脚本在 `chapters/` 下运行，`data/` 在上一级 | `os.path.join(os.path.dirname(__file__), "..", "data")` |
| 括号错位 | `join() got unexpected keyword argument 'encoding'` | `os.path.join(data, "file.json", "r", encoding="utf-8")` 把 5 个参数都传给了 join | `join()` 的括号在文件名后闭合：`join(data, "file.json")` |
| 中文分不出词 | 检索结果为空 | `"续航超过700公里".split()` 返回一整句 | 引入 `jieba.cut()` |
| 列表直接打印 | `['兆瓦闪充', '激光雷达']` 带方括号 | `str(list)` 保留了 Python 语法 | `"、".join(list)` |
| 停用词淹没信号 | 所有结果得分相近 | "纯电""轿车"几乎所有文档都有 | 加停用词表过滤 |
| LLM 编造信息 | "终端优惠可能进入 25 万以内" | 检索上下文缺乏真正符合条件的车，LLM 硬凑答案 | system prompt 加强约束 + 改善检索 |

---

## 从 Naive 到 Better：各章节递进关系

```
第 1 章：Naive RAG（本章）
├── 关键词匹配检索
├── 整篇文档作为检索单元
├── 简单 System Prompt
└── 问题：召回不精准、长文档命中太多噪音

第 2 章：文本分块（Chunking）
├── 把长文本切成小块
├── 对比固定大小 / 段落 / 章节三种策略
└── 效果：检索粒度更细，减少无关信息

第 3 章：向量嵌入（Embedding）
├── 用 embedding 模型把文本转成向量
├── 用余弦相似度代替关键词匹配
└── 效果："25万以内"能语义匹配到"13-18万" 的车

第 4 章：向量检索（Retrieval）
├── 向量数据库 / FAISS 索引
├── 混合检索（关键词 + 向量）
└── 效果：大规模数据下的高效召回

第 5 章：完整 RAG Agent
├── 多轮对话 + 记忆
├── 改写查询（Query Rewriting）
├── 重排序（Re-ranking）
└── 效果：生产级 RAG 系统
```

---

## 快速启动

```bash
cd rag-project/chapters

# 确保 .env 在 rag-project/ 下
# LLM_API_KEY=你的key
# LLM_BASE_URL=https://api.xxx.com/v1
# LLM_MODEL_ID=模型名

python 01_naive_rag.py
```

```
✅ 加载 26 条文档

🔍 请输入你的问题（输入 q 退出）：推荐一款25万以内续航超过600公里的轿车
🤖 根据您提供的上下文，推荐以下车型...
```

---

## 依赖

```
openai
python-dotenv
jieba
```
