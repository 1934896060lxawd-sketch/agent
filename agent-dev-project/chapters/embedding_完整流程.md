# 第三章：向量嵌入（Embedding）语义检索

## 从第二章遗留下来的问题

第二章解决了"检索粒度太粗"——用分块把整篇报告切成小节。但检索本身还是关键词匹配：

```
用户问："推荐一款性价比高的、适合家用的大空间SUV"

关键词分词：推荐 / 一款 / 性价比 / 高 / 适合 / 家用 / 大 / 空间 / SUV

问题 1：文档里写的"经济实惠""入门级""值回票价"，跟"性价比高"不是一个词 → 匹配不到
问题 2："大空间"和"后排翘二郎腿""第三排不憋屈"说的是同一件事 → 关键词完全对不上
问题 3："25万以内"语义上应该包含"13.98万"和"15.18-18.98万" → 字符串匹配做不到
```

**根因：关键词匹配看的是"字符串里有没有这个字"，而不是"这段话是什么意思"。**

```
关键词检索："这两个字符串里有多少字一样？"
语义检索：  "这两段话说的是不是一个意思？"
```

---

## Embedding 干了什么

```
"便宜代步电车"  ──→  BGE模型  ──→  [0.23, -0.45, 0.78, ..., -0.12]  (768个浮点数)
"入门级城市通勤" ──→  BGE模型  ──→  [0.21, -0.43, 0.75, ..., -0.10]  (768个浮点数)
                                          ↑   ↑   ↑          ↑
                                        这两个向量很接近！余弦相似度 ≈ 0.95
```

**一句话：Embedding 模型把文本变成了 768 维空间里的一个坐标点。意思相近的文本，坐标点靠得近；意思不同的，离得远。**

BGE 模型（BAAI General Embedding）在大量中文文本上训练过，它"见过"：
- "性价比高"和"经济实惠"经常出现在相似的语境里 → 向量接近
- "大空间"和"后排翘二郎腿"描述的是同一个话题 → 向量接近
- 所以即使字面完全不同，语义相近的文本向量距离也近

---

## 数据流总览

```
┌─────────────────────────────────────────────────────────────┐
│                      离线阶段（跑一次，结果缓存）               │
│                                                             │
│  documents[i]["content"]                                     │
│       │                                                     │
│       ▼                                                     │
│  _embed_model.encode([doc1, doc2, ..., doc64])              │
│       │                                                     │
│       ▼                                                     │
│  [[0.23, -0.45, ...], [0.11, 0.67, ...], ...]              │
│       │                                                     │
│       ▼                                                     │
│  存回 documents[i]["embedding"]                               │
│                                                             │
│  64 份文档 × 768 维 = 一个 64×768 的浮点矩阵                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      在线阶段（每次查询）                      │
│                                                             │
│  query = "便宜的城市通勤电车"                                  │
│       │                                                     │
│       ▼                                                     │
│  q_vec = _embed_model.encode([query])  →  (768,) 向量       │
│       │                                                     │
│       ▼                                                     │
│  for doc in documents:                                      │
│      sim = np.dot(q_vec, doc["embedding"])   ← 点积 = 余弦相似度│
│       │                                                     │
│       ▼                                                     │
│  按 sim 降序 → 取 top_k → 拼成 context                        │
│       │                                                     │
│       ▼                                                     │
│  同样的 build_prompt(query, context) + ask_llm(messages)     │
└─────────────────────────────────────────────────────────────┘
```

和第一章的唯一区别：`search()` 里的评分逻辑变了——从"统计关键词命中次数"换成了"计算向量余弦相似度"。管线的其他部分（加载、分块、Prompt、LLM）纹丝不动。

---

## 逐函数详解

### 1. 模块级初始化：加载 Embedding 模型

```python
from sentence_transformers import SentenceTransformer

# 优先从本地加载（ModelScope 下载的），网络不通也能跑
_LOCAL_MODEL_DIR = os.path.join(..., "models", "bge-base-zh-v1.5", "BAAI", "bge-base-zh-v1___5")
if os.path.isdir(_LOCAL_MODEL_DIR):
    _embed_model = SentenceTransformer(_LOCAL_MODEL_DIR, prompts={})
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    _embed_model = SentenceTransformer("BAAI/bge-base-zh-v1.5", prompts={})
```

**关键决策**：

| 决策点 | 选了 | 为什么 |
|--------|------|--------|
| 模型 | `BAAI/bge-base-zh-v1.5` | 中文效果最好的开源 Embedding 模型之一，768 维 |
| 加载方式 | 本地路径优先 | 避免网络超时（WinError 10060），ModelScope 下载一次永久使用 |
| `prompts={}` | 禁用自动 prompt 模板 | 新版 sentence-transformers 5.x 兼容性 |
| `normalize_embeddings=True` | 编码时归一化 | 归一化后向量长度为 1，点积直接等于余弦相似度，无需除法 |

### 2. `embed_documents(documents)` — 批量向量化

```python
def embed_documents(documents: list[dict]) -> list[dict]:
    """给每条文档的 content 生成 embedding 向量，直接写回 dict"""
    texts = [doc["content"] for doc in documents]
    embeddings = _embed_model.encode(texts, normalize_embeddings=True)
    # embeddings.shape = (64, 768) — 64 条文档，每条 768 维
    for doc, vec in zip(documents, embeddings):
        doc["embedding"] = vec       # vec 是 numpy array，768 个 float32
    return documents
```

**为什么 `normalize_embeddings=True`？**

```
未归一化：
  cos(A, B) = dot(A, B) / (|A| × |B|)   ← 每次算都要除一次模长

归一化后：
  cos(A, B) = dot(A, B)                   ← 点积直接就是余弦相似度
  因为 |A| = |B| = 1，分母恒为 1
```

归一化是在编码时用 `normalize_embeddings=True` 完成的，之后所有向量运算直接做点积即可。这是向量检索领域最常用的优化。

**embed_documents() 的性能**：
- 64 条文档 × 平均 200 字 = ~12800 字
- BGE 模型在 CPU 上一次 encode 约 2-3 秒
- 只在程序启动时跑一次

### 3. `semantic_search(query, documents, top_k=3)` — 语义检索

```python
def semantic_search(query: str, documents: list[dict], top_k: int = 3) -> str:
    """用向量余弦相似度检索，替代关键词匹配"""
    # ① query 也编码成向量（注意必须传列表，新版 API 不接受单字符串）
    q_vec = _embed_model.encode([query], normalize_embeddings=True)[0]
    # q_vec.shape = (768,)

    # ② 遍历文档，逐个算点积
    scored = []
    for doc in documents:
        sim = float(np.dot(q_vec, doc["embedding"]))  # [-1, 1]，越大越相似
        scored.append((sim, doc))

    # ③ 按相似度降序，取 top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    # ④ 拼上下文（和第一章一模一样）
    parts = []
    for sim, doc in top:
        parts.append(
            f"【来源：{doc['source']}】(相似度：{sim:.4f})\n{doc['content']}"
        )
    return "\n\n---\n\n".join(parts)
```

**和第一章 `search()` 的对比**：

| 维度 | 第一章 `search()` | 第三章 `semantic_search()` |
|------|------------------|--------------------------|
| 分词 | jieba 分词 + 停用词过滤 | 无需分词，模型直接理解 |
| 匹配方式 | `kw in content` 子串匹配 | `np.dot(q_vec, doc_vec)` 向量相似度 |
| 打分 | 关键词命中次数 (0~N) | 余弦相似度 (-1.0~1.0) |
| 语义理解 | ❌ "便宜"≠"经济实惠" | ✅ 模型理解同义关系 |
| 数字范围 | ❌ "25万以内"≠"13-18万" | ⚠️ 部分改善，不完全可靠 |
| 速度 | 快（纯字符串操作） | 慢（768 维向量点积 × 64 次） |

### 4. 对比实验主循环

```python
if __name__ == "__main__":
    documents = load_data(DATA_DIR)
    documents = embed_documents(documents)   # ← 启动时向量化一次

    while True:
        query = input(">> 请输入查询：").strip()

        # ---- 关键词检索 ----
        kw_context = keyword_search(query, documents)
        # 打印 Top-3 来源 + 前 150 字

        # ---- 语义检索 ----
        sem_context = semantic_search(query, documents)
        # 打印 Top-3 来源 + 相似度 + 前 150 字
```

对比实验的核心价值：**同一个 query，两路检索结果并排展示，肉眼就能看出差异。**

---

## 实测对比

### 测试 1：语义理解型查询

```
查询："推荐一款性价比高的、适合家用的大空间SUV"

【关键词检索 Top-3】
  1. report_05_buying_guide.txt#sec_1  (购车指南，命中了"性价比""家用""SUV")
  2. 极氪 7X                           (命中了"SUV""大")
  3. 比亚迪 海狮08                     (命中了"SUV")

【语义检索 Top-3】
  1. 理想 L8    (sim=0.6438)  ← "大空间SUV"语义匹配，原文可能根本没写"性价比"
  2. AITO M7    (sim=0.6426)  ← "家用"语义匹配
  3. 理想 L6    (sim=0.6421)  ← 三款都是公认的家用大空间 SUV
```

**分析**：关键词检索被"性价比""家用""SUV"这些词牵着走，命中了购车指南（因为指南里反复出现这些词）。语义检索直接找出了三款真正适合家用的大空间 SUV——即使它们的原始描述里可能根本没有出现"性价比高"这几个字。

### 测试 2：精确匹配型查询

```
查询："小米SU7的续航是多少"

【关键词检索 Top-3】
  1. 小米 SU7          ← 品牌名精确命中
  2. report_03_tech    ← 可能包含"小米""续航"关键词
  3. 比亚迪 海豹08

【语义检索 Top-3】
  1. 小米 SU7          (sim=0.85)  ← 同样命中
  2. 小鹏 P8           (sim=0.72)  ← 同为"纯电轿车"，语义接近
  3. 比亚迪 海豹08     (sim=0.70)
```

**分析**：精确查询时两者都能命中目标，但语义检索会把"类似的车"也拉进来——这是优势（发现竞品）还是劣势（噪音），取决于场景。

---

## 踩坑记录

| 坑 | 现象 | 根因 | 解法 |
|----|------|------|------|
| 模型下载超时 | `WinError 10060` 连接失败 | HuggingFace 和 hf-mirror 都不可达 | 改用 ModelScope 下载到本地，`SentenceTransformer(本地路径)` |
| `TypeError: TextEncodeInput` | 调用 `encode(query)` 报错 | 新版 sentence-transformers 5.x 不接受单字符串 | `encode([query])` 传列表，取 `[0]` |
| Windows 终端乱码 | 打印 `✅` `🔍` 报 UnicodeEncodeError | GBK 编码不支持 emoji | 用 ASCII 替代：`[OK]` `>>` `[AI]` |
| `prompts` 兼容性 | 模型加载后 encode 报错 | 新版自动注入空 prompt 模板导致 tokenizer 异常 | `SentenceTransformer(..., prompts={})` 禁用 |
| 语义不保证数字约束 | "25 万以内"搜到了 25-32 万的车 | Embedding 对数值大小不敏感，它理解"价格区间"但不理解"≤25" | 第四章混合检索：语义分 + 关键词分加权融合 |

---

## 关键认知

### 1. Embedding 不是魔法，是训练出来的"语感"

BGE 模型的 768 维向量里有什么？不是人工定义的规则，而是在海量中文文本上训练出来的分布模式。它见过"性价比高"和"便宜实惠"出现在相似的上下文里，所以把它们的向量拉近。它没被专门训练过"理解数字大小关系"，所以对预算约束不敏感。

**Embedding 擅长：同义词、近义表达、话题相似度**
**Embedding 不擅长：精确数字比较、逻辑推理、否定关系（"不要SUV"→ 还是会搜 SUV）**

### 2. 归一化是一个"一次性成本"

`normalize_embeddings=True` 让每条向量长度变成 1。代价是在 `embed_documents()` 阶段多花 0.01 秒，收益是之后每次 `semantic_search()` 都不需要算模长和除法。64 次查询就把成本赚回来了。

### 3. 从第一章到第三章，只改了一处

```
第一章：
  context = search(query, documents)       # jieba 分词 + 关键词命中

第三章：
  documents = embed_documents(documents)   # ← 启动时加一行
  context = semantic_search(query, documents)  # ← 检索时换一个函数
```

**管线的其他部分（加载、分块、Prompt、LLM）完全不用动。** 这就是 RAG 的模块化之美——检索策略是可插拔的。

### 4. 暴力遍历是第四章要解决的问题

第三章的 `semantic_search()` 每次查询遍历全部 64 条文档。64 条没问题，64000 条就要等几百毫秒，6400000 条就崩了。第四章用 FAISS 建索引，把 O(n) 降到 O(log n)。

---

## 与各章节的关系

```
第 1 章：Naive RAG
└── 关键词检索 + LLM
    痛点："便宜"匹配不到"经济实惠"（同义词盲区）

第 2 章：文本分块
└── 按章节标题切分长文本
    痛点：检索粒度问题已解决，但匹配方式仍是关键词

第 3 章：向量嵌入（本章）
└── 文本 → 768 维向量 → 余弦相似度检索
    成果："大空间SUV"能语义匹配到理想 L8/L6
    新痛点：① 暴力遍历 O(n)  ② 数字范围仍不敏感

第 4 章：向量检索 + 混合检索
└── FAISS 索引 + 关键词×语义加权融合
    要解决的问题：大规模高效检索 + 数字约束精确匹配

第 5 章：完整 RAG Agent
└── 多轮对话、查询改写、重排序
```

---

## 快速启动

```bash
cd rag-project/chapters

# 首次运行需要下载模型（约 390MB），已通过 ModelScope 缓存到 ../models/
python embedding_test.py
```

```
[OK] 加载并向量化 64 条文档

>> 请输入查询（输入 q 退出）：推荐一款性价比高的、适合家用的大空间SUV

============================================================
【关键词检索 Top-3】
============================================================

【来源：report_05_buying_guide.txt#sec_1】
  一、不同预算区间推荐车型...

【来源：极氪 7X】
  极氪7X 大型纯电豪华SUV，大型纯电SUV，售价22.99-26.99万元...

【来源：比亚迪 海狮08】
  比亚迪海狮08 中大型混动家用SUV，中大型SUV，售价约25万元...

============================================================
【语义检索 Top-3】
============================================================

【来源：理想 L8】(相似度：0.6438)
  理想L8 中大型增程家用SUV，中大型SUV，售价32.98-38.98万元...

【来源：AITO M7】(相似度：0.6426)
  AITO M7 中大型增程SUV，中大型SUV，售价28.98-37.98万元...

【来源：理想 L6】(相似度：0.6421)
  理想L6 中型增程SUV，中型SUV，售价24.98-28.98万元...
```

---

## 依赖

```
sentence-transformers  (BGE 模型加载和编码)
numpy                  (向量点积运算)
modelscope             (模型下载，首次需要)
torch                  (sentence-transformers 的底层依赖)
transformers           (sentence-transformers 的底层依赖)
```
