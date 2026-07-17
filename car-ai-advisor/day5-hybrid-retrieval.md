# Day 5 — 混合检索引擎：FAISS + BM25 + RRF

> **今日目标**：实现企业级混合检索管线——向量语义检索（FAISS）+ 关键词检索（BM25）+  Reciprocal Rank Fusion 融合。Day 4 的嵌入向量终于有了用武之地。

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [检索管线全景图](#2-检索管线全景图)
3. [FAISS 向量索引](#3-faiss-向量索引)
4. [BM25 关键词检索](#4-bm25-关键词检索)
5. [RRF 混合融合](#5-rrf-混合融合)
6. [Reranker 精排](#6-reranker-精排)
7. [核心技术原理](#7-核心技术原理)
8. [初学者常见疑问](#8-初学者常见疑问)
9. [面试模拟问答](#9-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `rag/retriever.py` | ~150 | VectorIndex(FAISS) + BM25 + hybrid_rrf() 混合融合 |
| `rag/reranker.py` | ~70 | BGE CrossEncoder 单例 + rerank() 精排 |
| `scripts/build_index.py` | ~90 | 编排 load_data → embed → 建 FAISS → 保存到磁盘 |

---

## 2. 检索管线全景图

```
用户 query: "25万家用SUV推荐"
        │
        ├──→ [向量路] embed_query() → FAISS.search() → Top-10
        │        语义: "家用" ≈ "家庭用户"，"SUV" ≈ "运动型多功能"
        │
        ├──→ [关键词路] jieba.cut() → BM25.search() → Top-10
        │        精确: "25万" 精确命中价格数字，"SUV" 命中类别
        │
        ▼
    hybrid_rrf(vec_results, bm25_results, k=60) → Top-10 候选
        │
        ▼
    rerank(query, candidates, top_k=5) → Top-5 精排结果
        │
        ▼
    喂给 LLM 作为参考上下文
```

**为什么需要两路检索？**

- **向量检索**擅长语义匹配：用户说"家用"，能匹配到文档里的"适合家庭用户""大空间""后排舒适"
- **关键词检索**擅长精确匹配：用户说"25万"，能精确匹配到文档里的价格数字。"SU7"这种专有名词在向量空间里不常见

单独使用任何一路都有盲区。两路互补才是企业级方案。

---

## 3. FAISS 向量索引

### 3.1 为什么需要 FAISS？

Day 4 做 embedding 时用的是 `sentence-transformers` 的 `encode()`。如果检索时也用暴力 for 循环：

```python
# Day 3 级别的检索：O(N) 暴力遍历
q_vec = model.encode([query])
for doc in documents:               # 遍历所有文档
    sim = np.dot(q_vec, doc.embedding)  # 逐个算相似度
    scored.append((sim, doc))
scored.sort(reverse=True)[:5]       # 排序取 top-5
```

102 个文档约 0.1ms，10 万个文档约 100ms，1000 万个文档约 10s。FAISS 用 C++ 优化 + 近似算法将百万级检索压缩到毫秒级。

### 3.2 IndexFlatIP 原理

```python
class VectorIndex:
    def __init__(self, documents):
        # Stack: list of (768,) → matrix (102, 768)
        embeddings = np.stack([doc.embedding for doc in documents])
        self.index = faiss.IndexFlatIP(dim=768)  # IP = Inner Product
        self.index.add(embeddings)                # 灌入索引
```

**为什么是 IndexFlatIP（内积）而不是 IndexFlatL2（欧氏距离）？**

Day 4 做 embedding 时 `normalize_embeddings=True`——所有向量都被归一化到单位长度（模长=1）。在单位向量上：

```
余弦相似度 = (A·B) / (|A| × |B|) = A·B / (1 × 1) = A·B = 内积
```

`IndexFlatIP` 做内积搜索等价于余弦相似度搜索，但比 `IndexFlatL2` 少一次开方运算。

### 3.3 Flat 到 IVF 到 HNSW 的演进

| 索引类型 | 算法 | 时间复杂度 | 适用规模 | 精度 |
|----------|------|:----------:|:--------:|:----:|
| IndexFlatIP | 暴力计算所有向量 | O(N×D) | < 10万 | 100% |
| IndexIVFFlat | 先聚类，只在最近簇里搜 | O(√N×D) | 10万-1000万 | ~95% |
| IndexHNSWFlat | 图索引，沿边游走 | O(log N×D) | 100万-10亿 | ~98% |

当前知识库 102 条文档用 Flat 完全足够。等知识库超过 10 万条时，只需把 `faiss.IndexFlatIP` 替换为 `faiss.IndexIVFFlat`，`search()` 调用方式不变——这是 FAISS 最大的价值：**索引透明**。

### 3.4 索引持久化

```python
# 保存（build_index.py 离线执行）
faiss.write_index(index, "faiss_index.bin")
pickle.dump(documents, open("documents.pkl", "wb"))

# 加载（服务启动时）
index = faiss.read_index("faiss_index.bin")
documents = pickle.load(open("documents.pkl", "rb"))
```

为什么不每次都重建？embedding 是确定性的——同一文档+同一模型=同一向量。离线建索引一次，服务启动时加载只需 0.1 秒。如果每次启动都编码 102 个文档（~20s），容器重启时间不可接受。

---

## 4. BM25 关键词检索

### 4.1 从"命中计数"到 BM25

最简单的关键词检索：

```python
# 第一章级别：命中一次加一分
score = 0
for keyword in query_keywords:
    if keyword in doc:
        score += 1  # "小米"出现1次和出现100次得分一样
```

三个致命缺陷：

| 缺陷 | 例子 | BM25 如何解决 |
|------|------|-------------|
| 无 IDF | "SUV"（出现在 60% 文档）和 "小米"（出现在 5% 文档）权重一样 | `IDF(t) = log(1 + (N-df)/(df+0.5))` → 稀有词高分 |
| 无词频饱和 | "续航"出现 100 次 = 100 分，但出现 3 次已经够说明相关性了 | TF 饱和曲线: f=1→~1, f=3→~1.5, f=100→~2.5 |
| 无长度归一化 | 长文档天然包含更多词（即使是噪音词），比短文档"占便宜" | 除以文档长度: `dl/avgdl` → 长文档被压扁 |

### 4.2 BM25 公式拆解

```
BM25(doc, query) = Σ IDF(t) × TF_saturated(t, doc) × length_norm(doc)
                     ↑             ↑                        ↑
                  词的重要性    出现次数（饱和）         文档长度惩罚
```

**IDF 项**：
```python
def idf(self, term):
    n = self.df.get(term, 0)   # 这个词出现在多少篇文档
    return log(1 + (N - n + 0.5) / (n + 0.5))

# N=102:
# "小米" 出现在 3 篇 → idf = log(1 + 99.5/3.5) ≈ 3.35 (高分)
# "汽车" 出现在 80 篇 → idf = log(1 + 22.5/80.5) ≈ 0.25 (低分)
```

**TF 饱和项**：
```python
tf_sat = f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
#                                          ↑ 长度归一化藏在这里
# f=1 → ~1.0, f=3 → ~1.5, f=10 → ~2.3, f=100 → ~2.5
# k1=1.5 控制饱和速度; b=0.75 控制长度归一化强度
```

`k1=1.5, b=0.75` 是经过数十年信息检索研究验证的经验最优值，几乎不需要改动。

### 4.3 jieba 分词的角色

```python
# BM25 建索引时
self.doc_tokens = [list(jieba.cut(doc.content)) for doc in documents]

# 检索时
query_tokens = list(jieba.cut("25万家用SUV"))
# → ["25", "万", "家用", "SUV"]
```

BM25 需要**词**作为基本匹配单元。中文没有空格分隔词，jieba 完成这个切分。算法核心是基于前缀词典的最大概率路径——如 "家用SUV" 切为 ["家用", "SUV"] 而不是 ["家", "用", "SUV"]，因为前者的路径概率更高。

---

## 5. RRF 混合融合

### 5.1 为什么需要融合？

两路检索结果不能直接合并——分数不在同一尺度：

```
向量路分数: [0.85, 0.72, 0.70, ...]  范围 ~[-1, 1]（实际 0.5~0.9）
BM25 路分数: [23.5, 18.2, 15.7, ...]  范围 ~[0, 几百]
```

如果直接 `0.5×向量分 + 0.5×BM25分`，BM25 分数会完全碾压向量分数。

### 5.2 两种融合策略

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| 加权求和 | Min-Max 归一化后 `α×vec + (1-α)×bm25` | α 可调，有标注数据时最优 | 需要归一化，需要调参 |
| RRF | `Σ 1/(k + rank_i)` | 无需归一化，无需调参 | 丢弃分数值，只用排名 |

### 5.3 RRF 公式与直觉

```
RRF(doc) = Σ 1/(k + rank_i)

k=60: 排名1→1/61=0.0164, 排名2→1/62=0.0161, 排名10→1/70=0.0143
```

**为什么 k=60？**

k 控制"相邻排名之间的分差"：
- k=0: 排名1→1/1=1.0, 排名2→1/2=0.5 — 差太大，排名1 的结果统治一切
- k=60: 排名1→0.0164, 排名2→0.0161 — 差很小，排名 1-10 几乎平权
- k=∞: 所有排名同等权重 — RRF 退化为平权投票

k=60 是在 TREC、MS MARCO 等标准数据集上验证过的通用最优值。它平衡了"高排名文档更重要"和"单一检索源不会统治融合结果"。

**RRF 的最大优势：容错**

假设某文档在向量路排第 1（.0164 分），但在 BM25 路排第 15（1/75=0.0133）。如果向量路的第 1 名是噪音（比如语义相似但实际无关），BM25 路的第 3 名（1/63=0.0159）可以通过在另一路的好排名追上。**单一检索源的极端高分不会破坏融合结果**——这是 RRF 在企业场景被广泛采用的核心原因。

---

## 6. Reranker 精排

### 6.1 Bi-Encoder vs Cross-Encoder

```
Bi-Encoder (embeddings.py — 粗排):
  Query: "小米SU7续航" ──→ [Encoder] ──→ q_vec (768,)
  Doc:   "小米SU7..." ──→ [Encoder] ──→ d_vec (768,)
  Score = cos(q_vec, d_vec)                   ← 独立编码，然后比较

Cross-Encoder (reranker.py — 精排):
  Pair:  ["小米SU7续航", "小米SU7..."] ──→ [Encoder] ──→ score
                                               ← 拼接后联合编码
```

Cross-Encoder 把 query 和 doc 拼在一起编码——模型能学习到"query 中的'续航'与 doc 中的'700km'之间的细粒度语义关系"。Bi-Encoder 各编各的，无法捕捉这种交互。

代价是速度：Top-20 候选 × 每个 pair ~20ms = 400ms。所以只在候选池上做精排，不在全库上做。

### 6.2 降级设计

```python
def get_reranker():
    if local_path.exists():
        return CrossEncoder(local_path)   # 本地模型
    try:
        return CrossEncoder("BAAI/bge-reranker-base")  # HuggingFace
    except:
        return None   # ← 精排跳过，核心功能不受影响
```

Reranker 是"锦上添花"的组件。模型未下载时自动降级——粗排结果直接返回，回答质量略有下降但不影响可用性。

---

## 7. 核心技术原理

### 7.1 向量检索的本质：高维空间最近邻

768 维空间中，语义相似的文本对应的点距离近。FAISS 做的就是在这 768 维空间中快速找到离查询点最近的 K 个文档点。Flat 索引做的是精确最近邻（exact nearest neighbor）——计算查询向量与所有文档向量的内积，取 top-K。

### 7.2 BM25 的本质：概率检索模型

BM25 属于概率检索模型家族，源自 1970 年代的"概率排序原则"（Probability Ranking Principle）。它不关心文档"关于"什么，只关心文档包含哪些词以及这些词在多大程度上预示相关性。

### 7.3 混合检索的本质：信号融合

向量检索和关键词检索提供了两种正交的信号：
- 向量信号：**全局语义**——"家用" ≈ "家庭用户" ≈ "大空间"（即使没有公共词汇）
- 关键词信号：**精确事实**——"25万" 就是"25万"，不可能混淆为"30万"

RRF 在不改变信号的前提下完成融合——它只是说"如果你在向量路排第 1，在 BM25 路排第 3，那你的总排名应该比只在向量路排第 5 的文档高"。

---

## 8. 初学者常见疑问

**Q: FAISS 和 ChromaDB/Milvus 有什么区别？**

FAISS 是向量索引**算法库**（底层引擎），ChromaDB/Milvus 是向量**数据库**（加了持久化、查询 API、元数据过滤）。FAISS 更轻量（1 个 pip 包），适合嵌入式场景。大数据量（100 万+）或需要分布式时用 Milvus。当前 102 条文档用 FAISS 足够了。

**Q: BM25 的 k1 和 b 参数需要调吗？**

不需要。k1=1.5, b=0.75 是数十年 IR（信息检索）研究验证的经验最优值。它们在 TREC 标准评测集上对各类查询（短查询、长查询、不同语言）都表现稳定。除非你有标注数据做网格搜索，否则不要动它们。

**Q: RRF 的 k=60 是怎么来的？**

来自学术界 2009 年发表后被反复验证的经验值。实验表明 k 在 50-70 范围内融合效果几乎无差别，60 是中间值。k 太小 → 排名靠前的文档主导一切（退化回单路检索），k 太大 → 所有排名平权（融合失去意义）。

**Q: 为什么 reranker 不直接替代 embedding？全部用 CrossEncoder 检索不行吗？**

CrossEncoder 无法做索引——它必须把 query 和 doc 拼在一起编码，意味着每次检索都要重新编码所有文档。1000 个文档 × 20ms = 20 秒。Bi-Encoder 可以预计算所有文档向量存入 FAISS，查询时只需编码 query（~5ms）。所以管线必须是"粗排（Bi-Encoder）→ 精排（CrossEncoder）"而不是反过来。

**Q: `np.stack()` 和 `np.array()` 有什么区别？**

```python
embeddings_list = [np.array([1,2,3]), np.array([4,5,6])]  # list of (3,)

np.array(embeddings_list)   # → shape (2, 3)  OK
np.stack(embeddings_list)   # → shape (2, 3)  结果一样

# 但如果 list 里元素已经是等长1D数组，两者等价
# stack 更语义明确："我把这些向量堆成一个矩阵"
```

---

## 9. 面试模拟问答

> **Q: 你们的 RAG 系统用什么检索方案？为什么这样设计？**

我们用混合检索：FAISS 向量检索 + BM25 关键词检索，然后 RRF 融合，最后 BGE CrossEncoder 精排。设计原则是"两路互补"——向量检索覆盖语义匹配（用户说"家用"，匹配到"家庭用户""大空间"），BM25 覆盖精确匹配（价格数字、专有名词"小米 SU7"）。RRF 融合不需要归一化和调参（k=60 通用），CrossEncoder 在粗排候选池上做精排提高精度。整套管线从 102 条文档到 10 万条文档只需换 FAISS 索引类型，其他代码不变。

> **Q: 解释一下 BM25 的三个核心机制？**

① IDF：稀有词权重高。在全库 102 篇文档中，"小米"只出现在 3 篇（IDF≈3.35），"汽车"出现在 80 篇（IDF≈0.25）。② TF 饱和：词频贡献不是线性的——出现 1 次得 1 分，出现 3 次得 1.5 分，出现 100 次得 2.5 分。防止某个词无限重复主导得分。③ 长度归一化：长文档天然包含更多词，除以文档长度做公平比较。参数 k1=1.5 控制饱和速度，b=0.75 控制长度归一化强度——都是数十年经验最优值，不需要调整。

> **Q: RRF 和加权求和怎么选？**

没有标注数据 → RRF。有标注数据（query + 正确答案对）→ 可以先网格搜索最优 α，再用加权求和。RRF 的优势在于零配置上线——k=60 在所有标准数据集上通用。实践中大多数团队用 RRF 是因为它"够好且不用维护"。只有当检索质量成为瓶颈时，才收集标注数据调加权求和的 α。

> **Q: 你们的检索是怎么做持久化的？为什么这样做？**

`build_index.py` 离线执行一次：加载文档 → 向量化 → 建 FAISS 索引 → `faiss.write_index()` 存磁盘。服务启动时 `faiss.read_index()` 加载（~0.1s）。为什么不是每次启动重建？embedding 是确定性的——同一文档+同一模型=同一向量。预计算和缓存是 RAG 系统的基础优化，避免每次容器重启等 20 秒重新编码所有文档。

> **Q: FAISS IndexFlatIP 和 IndexFlatL2 有什么区别？应该用哪个？**

IndexFlatIP 做内积搜索，IndexFlatL2 做欧氏距离搜索。在归一化向量（模长=1）上，余弦相似度 = 内积。我们的 embedding 管线的 `normalize_embeddings=True` 保证所有向量归一化，所以用 IndexFlatIP。IP 比 L2 计算快（少一次开方），且语义上与"相似度"直觉一致——分数越高越相关，而不是距离越小越相关。

> **Q: 你们的 Reranker 加载失败了会怎样？**

Reranker 在 `get_reranker()` 中有三层降级：本地路径 → HuggingFace → None。如果都失败，`rerank()` 函数检测到 `model is None`，直接返回原始候选列表——相当于跳过精排步骤。回答质量略有下降，但系统核心功能不受影响。这是企业级系统的"优雅降级"模式：辅助组件失败不影响主流程。

---

## 附：今日文件依赖关系

```
chunker.py (Document) ──┐
                         ├──→ retriever.py (VectorIndex + BM25 + hybrid_rrf)
embeddings.py ──────────┘         │
                                   ├──→ reranker.py (CrossEncoder精排)
load_data.py ──────────────────────┤
                                   │
                                   └──→ build_index.py (编排+持久化)
                                            │
                                            ▼
                                   knowledge_base/processed/
                                     faiss_index.bin + documents.pkl
                                            │
                                            ▼  Day 6 消费
                                   deps.py → get_agent() 加载索引
```
