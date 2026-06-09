# 第二章：文本分块（Chunking）

## 从第一章遗留下来的问题

第一章的 Naive RAG 能跑了，但有一个明显的短板：

```
用户问："2026年买新能源车要交多少购置税？"

分块前：整份 report_01（1300字，覆盖市场、出口、技术、智驾、政策、购置税）
         → 全部塞给 LLM → 1200字噪音 + 100字答案
         → token 浪费 + LLM 容易在无关段落里"迷路"

分块后：report_01 被切成 7 块
         → 只召回 sec_6（161字，恰好就是"政策环境变化"章节）
         → LLM 收到精准聚焦的 161 字
         → 回答又快又准
```

**核心问题：检索的最小单元太大。** 一份报告不是一个话题，是多个话题的集合。把整份报告当一个文档，检索就失去了精度。

---

## 三种分块策略

### 策略 1：固定大小分块（Fixed-size Chunking）

**原理**：设定 `chunk_size` 和 `overlap`，像切豆腐一样切成等长片段。

```python
def chunk_size(text, size=400, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append({"content": text[start:end].strip(), "chunk_id": f"c_{start}"})
        start += size - overlap   # 下一块起点，overlap 让相邻块有重叠
    return chunks
```

```
原文: |████████████████████████████████████████|
块1:  |████████████████|           ← 0 - 400
块2:       |████████████████|      ← 320 - 720  (重叠 80 字)
块3:            |████████████████| ← 640 - 1040
...
```

**优点**：
- 实现最简单，无需理解文本结构
- 所有块大小均匀，适合向量化（embedding 模型对长度有限制）
- `overlap` 减少信息在边界断裂

**缺点**：
- 可能在句子中间切断语义
- 完全不考虑文本的自然边界（段落、章节）
- 同一个段落的内容可能被切到两个块里

---

### 策略 2：按段落分块（Paragraph-based Chunking）

**原理**：以空行为分隔符，尊重作者的自然分段。

```python
def chunk_paragraph(text):
    paragraphs = text.split("\n\n")
    chunks = []
    for i, p in enumerate(paragraphs):
        p = p.strip()
        if p:
            chunks.append({"content": p, "chunk_id": f"para_{i}"})
    return chunks
```

**优点**：
- 不破坏作者有意设定的段落边界
- 每块相对语义完整

**缺点**：
- 块的大小极不均匀——一句话（8字）和一段长论述（800字）混在一起
- 写作风格决定块质量：行间距大的文本会变成一堆碎片
- 有些作者一个段落包含多个话题，仍然不够细

---

### 策略 3：按章节标题分块（Section-based Chunking）⭐ 最终选用

**原理**：匹配中文报告的"一、二、三、..."标题，以章节为边界切分。

```python
import re

def chunk_sections(text):
    pattern = r"([一二三四五六七八九十]+、)"
    parts = re.split(pattern, text)
    # parts → ["前言文字", "一、", "市场规模...", "二、", "出口数据..."]
    
    chunks = []
    # 处理开头（在第一个标题之前的内容）
    if parts and not re.match(pattern, parts[0]):
        chunks.append({"content": parts[0].strip(), "chunk_id": "sec_preamble"})
    
    # 标题 + 正文配对
    i = 1
    while i < len(parts):
        if re.match(pattern, parts[i]):      # 是标题
            title = parts[i]
            body = parts[i+1] if i+1 < len(parts) else ""
            chunks.append({
                "content": title + "\n" + body,
                "chunk_id": f"sec_{len(chunks)}"
            })
            i += 2
        else:
            i += 1
    return chunks
```

**优点**：
- 每块 = 一个话题，检索粒度最合理
- 大小相对均匀（这批数据里 161-243 字）
- 充分利用了作者已经做好的结构

**缺点**：
- 依赖标题格式，如果报告没写"一、二、三"就无效
- 正则可能漏掉"1.""①""第一节"等非标准标题
- 不通用——换个数据源就要调整正则

---

## 实测对比

对 `report_01_market_overview.txt`（1308 字符）三种策略的输出：

```
策略                 块数     平均长度     最短     最长
固定大小 400+80        5      314      27     400
按段落                15     85       8      217
按章节                 7      185      59     243

按章节切出的块标题：
  sec_preamble: 2026年中国新能源汽车市场全景概览  (59字)
  sec_1: 一、市场规模与销量  (243字)
  sec_2: 二、出口成为核心增长极  (192字)
  sec_3: 三、技术路线变革  (238字)
  sec_4: 四、智能化进入关键爆发期  (218字)
  sec_5: 五、行业格局加速分化  (190字)
  sec_6: 六、政策环境变化  (161字)
```

| 策略 | 判定 | 理由 |
|------|:--:|------|
| 固定大小 | ❌ | 末尾块只剩 27 字碎片，且切断了"五、"和"六、"的完整结构 |
| 按段落 | ❌ | 15 块太碎，8 字的那块只是"来源：xxx"一行孤立文本 |
| **按章节** | ✅ | 7 块每块都是完整话题，作者已标好边界，直接用 |

---

## 集成回 RAG 管线

第二章的核心价值不在独立测试，而在把分块逻辑嵌入第一章的 `load_data()`：

### 改动前（整篇塞）

```python
for report_path in glob.glob(os.path.join(data, "industry_reports", "*.txt")):
    with open(report_path, "r", encoding="utf-8") as f:
        documents.append({
            "content": f.read(),                         # 整篇 1300 字
            "source": os.path.basename(report_path),
            "type": "industry_report"
        })
# → 5 份报告 = 5 条文档
```

### 改动后（分块塞）

```python
for report_path in glob.glob(os.path.join(data, "industry_reports", "*.txt")):
    with open(report_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    for chunk in chunk_sections(raw_text):
        documents.append({
            "content": chunk["content"],                 # 每块 160-240 字
            "source": f"{os.path.basename(report_path)}#{chunk['chunk_id']}",
            "type": "industry_report_chunk"
        })
# → 5 份报告 × ~7 块 = ~35 条文档
```

### 效果对比

| 维度 | 分块前 | 分块后 |
|------|--------|--------|
| 文档总数 | ~26 条 | ~55 条 |
| 检索命中单位 | 整份报告（1300字） | 单个章节（~200字） |
| 答案来源可追溯 | `report_01.txt` | `report_01.txt#sec_6` |
| 无关内容 | 90% | ~10% |
| LLM 幻觉风险 | 高（噪音多，容易编） | 低（上下文精准聚焦） |

---

## 关键认知

### 1. Chunking 没有银弹

适合你的数据的切法，不一定适合别人的数据：

| 数据类型 | 推荐策略 | 原因 |
|----------|---------|------|
| 中文研报、政策文件 | 按章节标题 | 作者已用"一二三"标好结构 |
| 对话记录、客服日志 | 按发言轮次 | 每句话独立，语义边界清晰 |
| 代码文件 | 按函数/类 | AST 解析，每个函数一个 chunk |
| 纯英文技术文档 | 按 `##` / `###` 标题 | Markdown heading 分级 |
| 没有结构的长文本 | 固定大小 + overlap | 没办法时最保底的选择 |

### 2. 分块直接影响检索质量

检索不是在"文档"上做的，是在"块"上做的。块的粒度决定了检索的精度：

- **太粗**：召回太多无关内容，LLM 被噪音淹没
- **太细**：丢失上下文，LLM 看不懂孤立的碎片

**一个好的 chunk = 一个能独立回答问题的最小语义单元。**

### 3. 重叠（overlap）是一种保底手段

固定大小分块一定会切碎句子。overlap 让相邻块共享一小段文字，修补边界。但它不是替代好策略的理由——与其靠 overlap 补救，不如先用按章节/按段落来避免问题。

---

## 与各章节的关系

```
第 1 章：Naive RAG
└── 加载 → 关键词检索 → 组 Prompt → 调 LLM
    痛点：检索粒度太粗

第 2 章：文本分块（本章）
└── 把长文本切成语义完整的小块
    成果：检索精度从"整篇报告"提升到"单个章节"

第 3 章：向量嵌入（Embedding）
└── 用向量相似度替代关键词匹配
    要解决的问题："25 万以内" 语义上应该匹配到 "13-18 万"

第 4 章：向量检索（Retrieval）
└── FAISS / ChromaDB 高效检索 + 混合检索

第 5 章：完整 RAG Agent
└── 多轮对话、查询改写、重排序
```

---

## 依赖

本章无额外依赖，只用 Python 标准库 `re`。
