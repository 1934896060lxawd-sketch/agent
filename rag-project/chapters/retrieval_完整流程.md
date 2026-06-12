# 第四章：FAISS 向量索引 + BM25 混合检索

## 从第三章遗留下来的问题

第三章用 BGE Embedding 实现了语义检索，"大空间SUV"能匹配到理想 L8/L6。但留下两个新痛点：

```
痛点①：暴力遍历 O(n)

  第三章每次查询：
    for doc in documents:                           # Python for 循环
        sim = np.dot(q_vec, doc["embedding"])       # 逐个算点积
        scored.append((sim, doc))

  64 条文档没事（~0.5ms），640 万条就崩了（~5 秒）。
  不是算法慢——是 Python for 循环本身慢。


痛点②：数字约束仍然不敏感

  用户问："25万以内的SUV"
  向量检索返回：理想 L8（32.98万） ← 语义最相关，但超预算了
  Embedding 理解"大空间≈理想L8"，但不理解"≤25万"。
  它没被训练过数值大小比较。
```

第四章干两件事：**FAISS 消除 Python 循环开销，BM25 补齐精确匹配盲区，最后 RRF 融合取两路长项。**

---

## FAISS 干了什么（解决痛点①）

### 核心思路

```
第三章（Python）：
  for doc in documents:
      sim = np.dot(q_vec, doc["embedding"])    ← Python 解释器：取对象、调函数、append

第四章（FAISS C++）：
  index.search(q_vec, top_k)                   ← C++ 矩阵运算 + 堆排序，一句替代 for 循环
```

**IndexFlatIP 没有降低算法复杂度（仍然是 O(n) 暴力计算），但把 Python for 循环换成了 C++ 矩阵乘法，常数因子差了两个数量级：**

| | 第三章 | 第四章 IndexFlatIP |
|---|---|---|
| 语言 | Python for 循环 | C++ + BLAS |
| 向量运算 | 逐个 np.dot() | 一次矩阵乘法 q × M^T |
| 排序取 top-k | Python sorted() | C++ 堆排序 |
| 64 条文档 | ~0.5ms | ~0.01ms |
| 64 万条 | ~5 秒 | ~50ms |
| 算法复杂度 | O(n) | O(n)（一样！） |
| 精度 | 100% | 100%（一样！） |

**什么时候换近似索引？**

```
IndexFlatIP      O(n)        精度 100%  ← 64 条文档的最优选择
    ↓
IndexIVFFlat     O(√n)       精度 ~95%  ← 先聚类再在最近簇里搜
    ↓
IndexHNSW        O(log n)    精度 ~98%  ← 图索引，工业界最常用
```

---

## BM25 干了什么（解决痛点②）

### 第一章关键词检索的三个致命缺陷

第一章 `search()` 打分逻辑：

```python
if kw in content:
    score += 1    # 命中一次加一分
```

| 缺陷 | 问题 | 例子 |
|------|------|------|
| 没有 IDF | "轿车"和"小米SU7"命中一次都是 +1 分，但"小米SU7"明显更精确 | 搜"小米SU7"，"的"命中 20 次得 20 分，比"小米"命中 1 次得 1 分高 |
| 没有词频饱和 | 一个词出现 100 次得 100 分，但出现 3 次后就应该饱和了 | 购车指南里"SUV"出现几十次，任何跟 SUV 相关的查询都把它排第一 |
| 没有长度归一化 | 长文档天然包含更多词，在关键词计数规则下不公平 | 600 字的报告切块比 80 字的车主评价天然多得几十分 |

### BM25 核心公式

```
score(doc, query) = Σ IDF(t) × TF_saturated(t) × length_norm

IDF(t)：       log(1 + (N - df + 0.5) / (df + 0.5))
               稀有词（df 小）→ IDF 大 → 命中它几乎锁定答案
               常见词（df 大）→ IDF 接近 0 → 基本没贡献

TF_saturated： f × (k1 + 1) / (f + k1 × (1 - b + b × dl / avgdl))
               f=1 → ~1分，f=2 → ~1.3分，f=100 → ~2.5分（饱和！）
               长文档分母更大 → 词频被压扁 → 和短文档公平比较
```

**三个参数，一个都不用改，但面试可能会问：**

| 参数 | 默认值 | 含义 | 调大 | 调小 |
|------|--------|------|------|------|
| k1 | 1.5 | 词频饱和速度 | 词频可以继续涨分 | 更快饱和 |
| b | 0.75 | 长度归一化强度 | 长文档被压得更狠 | 长文档占优 |
| avgdl | 自动算 | 所有文档的平均词数 | — | — |

---

## 混合融合干了什么

两路检索各返回自己的 Top-K，需要合并成一个最终的排序。

核心问题：**两路分数不在一个量级，不能直接加权。**

```
向量相似度范围： 0.5 ~ 0.9（归一化后在 [0,1] 内）
BM25 分数范围：  0 ~ 几十（跟文档集大小和 query 长度有关）
直接加权：      BM25 分碾压向量分 → 加权毫无意义
```

### 方法一：加权求和（需调参）

```
① 两路各自 Min-Max 归一化到 [0, 1]
② merged[doc] = alpha × 向量分 + (1-alpha) × BM25分
③ 按融合分降序取 top_k

alpha=0   → 纯 BM25（精确查询："小米SU7续航多少"）
alpha=0.5 → 均衡（默认）
alpha=1   → 纯向量（模糊查询："推荐一款家用车"）
```

**缺点**：alpha 需要调，不同数据集的最优值不同。有标注数据时可以网格搜索。

### 方法二：RRF（生产首选）

```
RRF(doc) = Σ 1 / (k + rank_i)

k=60 是经验值，rank 从 1 开始。

例子：
  某文档在向量路排第 1 → 贡献 1/(60+1) = 0.0164
  某文档在 BM25 路排第 3 → 贡献 1/(60+3) = 0.0159
  RRF 总分 = 0.0323
```

**为什么 RRF 是生产首选？**

| | 加权求和 | RRF |
|---|---|---|
| 需要归一化 | ✅ Min-Max | ❌ 只用排名 |
| 需要调参 | ✅ alpha 要试 | ❌ k=60 基本通用 |
| 用到分数值 | ✅ | ❌ 只用排名 |
| 受极端值影响 | ⚠️ 向量 0.99 分可能主导 | ✅ 排名 1 和 2 差距不大 |
| 生产推荐 | 有标注数据时更好 | **快速上线首选** |

---

## 数据流总览

```
┌─────────────────────────────────────────────────────────────┐
│                    离线阶段（启动时跑一次）                      │
│                                                             │
│  documents = load_data()                                     │
│       │                                                     │
│       ├──→ embed_documents(documents)  ← 第三章：文本→768维向量│
│       │         │                                           │
│       │         ▼                                           │
│       │    FAISS IndexFlatIP ← np.stack → astype(np.float32) │
│       │         │                  index.add(embeddings)      │
│       │         ▼                                           │
│       │    向量索引就绪                                       │
│       │                                                     │
│       └──→ BM25 索引 ← jieba 分词 → DF 统计 → avgdl         │
│                │                                            │
│                ▼                                            │
│            关键词索引就绪                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    在线阶段（每次查询）                         │
│                                                             │
│  query = "25万以内的大空间SUV"                                 │
│       │                                                     │
│       ├──→ v_idx.search(query, top_k=6)   向量路 Top-6       │
│       │                                                     │
│       ├──→ bm25_idx.search(query, top_k=6)  BM25路 Top-6     │
│       │                                                     │
│       └──→ hybrid_rrf(vec_results, bm25_results, k=60)       │
│                │               │                            │
│                ▼               ▼                            │
│            RRF 融合：1/(60+rank) 累加 → 降序 → Top-3          │
│                │                                            │
│                ▼                                            │
│          build_prompt(query, context) + ask_llm(messages)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 逐组件详解

### 1. `VectorIndex` — FAISS 向量索引

```python
class VectorIndex:
    def __init__(self, documents):
        # ① 堆叠：list of (768,) 变成 (64, 768) numpy 矩阵
        self.embeddings = np.stack([doc["embedding"] for doc in documents])
        self.embeddings = self.embeddings.astype(np.float32)
        #    ↑ astype(np.float32) 很关键——FAISS 只认 float32

        # ② 建索引：IP = Inner Product（内积=余弦相似度，因为向量已归一化）
        dim = self.embeddings.shape[1]  # 768
        self.index = faiss.IndexFlatIP(dim)
        #   等价写法：faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)

        # ③ 灌入
        self.index.add(self.embeddings)
        # index.ntotal → 64（已索引的向量数）

    def search(self, query, top_k=3):
        # ① query → 768维向量
        q_vec = _embed_model.encode([query], normalize_embeddings=True)
        q_vec = q_vec.reshape(1, -1)  # (768,) → (1, 768)，FAISS 要二维输入

        # ② 一句替代 for 循环
        scores, ids = self.index.search(q_vec, top_k)
        # scores[0] = [0.85, 0.72, 0.70]  ← 相似度
        # ids[0]    = [3,    17,   42  ]  ← 在 documents 里的下标

        # ③ 组装
        return [(float(scores[0][i]), self.documents[ids[0][i]])
                for i in range(len(ids[0])) if ids[0][i] >= 0]
```

**为什么用 `IndexFlatIP` 而不是 `IndexFlatL2`？**

第三章编码时已经 `normalize_embeddings=True`，向量模长 = 1。此时：
- `IndexFlatIP`：内积 = 余弦相似度，越高越相关
- `IndexFlatL2`：欧氏距离，越低越相关 — 也能用但语义不对齐（"越相似"期望"分数越高"）

### 2. `BM25` — 关键词检索

```python
class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        # ① 分词：和第一章用同样的 jieba
        self.doc_tokens = [list(jieba.cut(doc["content"])) for doc in documents]

        # ② 统计
        self.N = len(self.doc_tokens)                              # 文档总数
        self.avgdl = sum(len(d) for d in self.doc_tokens) / self.N # 平均长度

        # ③ DF：用 set 去重，同一文档出现多次只算 1
        self.df = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score_one(self, query_tokens, doc_tokens):
        dl = len(doc_tokens)
        score = 0.0
        for t in query_tokens:
            if t not in doc_tokens:
                continue
            f = doc_tokens.count(t)
            tf_sat = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += self.idf(t) * tf_sat
        return score

    def search(self, query, top_k=3):
        tokens = list(jieba.cut(query))
        scores = [self._score_one(tokens, doc) for doc in self.doc_tokens]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(score, self.raw_docs[idx]) for idx, score in ranked[:top_k] if score > 0]
```

**IDF 的实际效果（假设 64 篇文档）：**

| 词 | 出现在几篇 | IDF | 含义 |
|----|----------|-----|------|
| "小米" | 2 篇 | 3.22 | 稀有词，命中它几乎确定答案 |
| "SUV" | 40 篇 | 0.49 | 常见词，有一定区分度 |
| "的" | 64 篇 | 0.008 | 基本不贡献分，相当于自动被过滤 |
| 不存在 | 0 篇 | ~4.9 | 如果 query 有这词但库中没有，IDF 给最高分（但你找不到） |

### 3. 混合融合 — 加权 vs RRF

```python
# ---- 加权求和 ----
def hybrid_weighted(v_scored, b_scored, alpha=0.5, top_k=3):
    # ① Min-Max 归一化到 [0,1]
    v_norm = min_max_norm(v_scored)
    b_norm = min_max_norm(b_scored)

    # ② 加权合并，用 source 字段去重
    merged = defaultdict(float)
    for score, doc in v_norm:
        merged[doc["source"]] = alpha * score
    for score, doc in b_norm:
        merged[doc["source"]] += (1 - alpha) * score

    # ③ 排序
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---- RRF（生产首选）----
def hybrid_rrf(v_scored, b_scored, k=60, top_k=3):
    scores = defaultdict(float)
    for rank, (_, doc) in enumerate(v_scored):
        scores[doc["source"]] += 1.0 / (k + rank + 1)
    for rank, (_, doc) in enumerate(b_scored):
        scores[doc["source"]] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**候选池为什么要取 top_k × 2？**

```
最终要 Top-3，但粗排阶段两路各取 Top-6。

原因：
  - 向量路排第 4 的文档，可能和 BM25 路排第 1 的是同一篇 → RRF 分累加后冲到第 1
  - 如果粗排只取 Top-3，这篇就漏了

这不是浪费——多拉的 3 个候选只参与融合计算，不暴露给用户。
```

---

## 实测对比

### 测试 1：语义理解型查询

```
查询："大空间SUV"

【关键词检索 Top-3】
  来源：极氪 7X                   ← "SUV""大"关键词命中
  来源：比亚迪 海狮08             ← "SUV"命中
  来源：report_05_buying_guide    ← "SUV"反复出现几十次，堆积成高分

【向量检索 Top-3】
  来源：理想 L8    (0.6438)      ← 大空间SUV语义匹配，但 32.98 万超预算
  来源：AITO M7    (0.6426)      ← 家用语义匹配
  来源：理想 L6    (0.6421)      ← 大空间语义匹配，24.98 万

【混合 RRF Top-3】
  来源：理想 L6    (0.0251)      ← 语义高分 + 价格合理，两路综合最优
  来源：比亚迪 海狮08 (0.0183)   ← 关键词精确命中价格区间
  来源：极氪 7X    (0.0156)      ← 两路都有得分
```

**分析**：关键词检索被"SUV"这个词牵着走（购车指南里 SUV 出现几十次，关键词计数下分数虚高）。向量检索找到了真正的大空间 SUV，但把超预算的理想 L8 排了第一。混合 RRF 把 L6 提到了第一——它在两路排名都靠前（语义第二 + 关键词也有得分）。

### 测试 2：精确匹配型查询

```
查询："小米SU7的续航是多少"

【关键词检索 Top-3】
  来源：小米 SU7                  ← "小米""SU7"精确命中，IDF 高
  来源：report_03_tech            ← "续航"命中
  来源：比亚迪 海豹08

【向量检索 Top-3】
  来源：小米 SU7    (0.8500)     ← 同样精准命中
  来源：小鹏 P8     (0.7200)     ← 同为纯电轿车，语义接近但可能是噪音
  来源：比亚迪 海豹08 (0.7000)

【混合 RRF Top-3】
  来源：小米 SU7    (0.0333)     ← 两路都是第一，融合分最高
  来源：report_03_tech (0.0167)  ← 行业报告里有续航相关内容
  来源：比亚迪 海豹08 (0.0159)
```

**分析**：精确查询时三路都能命中目标。混合检索的优势在于——当向量路拉入了"类似的车"（小鹏 P8）作为噪音时，BM25 路不会给它高分（因为没有"小鹏"这个关键词），RRF 融合后噪音被压低。

### 测试 3：数字约束型查询（混合检索最大的优势场景）

```
查询："25万以内的SUV推荐"

【关键词检索 Top-3】
  来源：report_05_buying_guide    ← "25万""推荐"命中购车指南
  来源：比亚迪 海狮08             ← "25万"命中
  来源：极氪 7X                   ← "22.99-26.99万"，部分命中

【向量检索 Top-3】
  来源：理想 L8    (0.6438)      ← "SUV推荐"语义匹配，但 32.98 万超预算！
  来源：AITO M7    (0.6426)      ← 28.98 万，仍然超预算
  来源：理想 L6    (0.6421)      ← 24.98 万在预算内

【混合 RRF Top-3】
  来源：理想 L6    (0.0251)      ← 语义相关 + 价格合适，两路综合最优
  来源：比亚迪 海狮08 (0.0183)   ← 关键词精确命中价格
  来源：极氪 7X    (0.0156)      ← 22.99 万在预算内，两路都有得分
```

**分析**：纯向量检索把 32.98 万的理想 L8 排第一——它的"大空间SUV"语义最强，但价格约束完全不满足。混合检索通过 BM25 的精确数字匹配把理想 L6（24.98 万）和海狮 08（约 25 万）提了上来。**这是混合检索相比纯向量的最大增量价值。**

---

## 踩坑记录

| 坑 | 现象 | 根因 | 解法 |
|----|------|------|------|
| FAISS dtype 不匹配 | `add()` 后 search 返回乱码分数 | BGE 输出 float32，但如果中途转了 float64/float16，FAISS 静默失败 | `astype(np.float32)` 显式转换 |
| `IndexFlatIP` vs `IndexFlatL2` | 分数排序反了 | L2 是距离（越小越好），IP 是内积（越大越好），用错度量排序就反了 | 归一化后用 IP |
| BM25 忘记 set 去重 | 常见词的 IDF 被压低过头 | DF 统计时没去重，一篇文档里出现 50 次 "SUV" 算了 50 次 df | `for term in set(tokens)` |
| jieba 分词和向量模型的 tokenizer 不一致 | 两路对"同一概念"的切分不同 | jieba 按词切，BGE 按子词切（WordPiece），粒度不同 | 不影响，两路各自独立工作，融合时只看排名 |
| FAISS `search()` 返回 -1 | `ids[0]` 里出现负数 | 候选不足 top_k（索引里文档总数 < top_k） | `if doc_id < 0: continue` |
| RRF rank 忘 +1 | 所有分数偏大但排序正确 | rank 从 0 开始，没 +1 影响绝对值但不影响排序 | 最坏只是数值不对，排序仍然正确 |

---

## 关键认知

### 1. FAISS Flat 没降算法复杂度，降的是常数因子

**IndexFlatIP 仍然是 O(n) 暴力计算。** 加速来源是把 Python for 循环换成了 C++ 矩阵运算。64 条文档差异不大，64 万条差距 100 倍。真正降复杂度（O(n) → O(log n)）要换 IndexHNSW 等近似索引——那是工业界的事，当前数据量不需要。

### 2. BM25 是第一章关键词检索的"正确版本"

第一章的 `score += 1` 只是让你理解"关键词能检索"。BM25 的三个修正（IDF、饱和词频、长度归一化）让关键词检索从"玩具"变成"可用"。

### 3. RRF 背后是"让好文档被任何一路发现"

```
加权求和："我觉得你的分数可信，我们加权平均一下"
RRF：     "我不信你的分数值，但信你的排名——排名高的就给它加分"
```

RRF 不依赖分数绝对值，所以不受分数尺度差异影响。这就是为什么不用归一化、不用调参。

### 4. 混合检索的本质上"语义理解 + 精确匹配互补"

```
向量检索擅长：话题相似（"大空间"≈"后排翘二郎腿"）
              同义表达（"便宜"≈"经济实惠"）

BM25 擅长：    专有名词（"小米SU7"——IDF 极高）
              数字范围（"25万"——字面匹配）
              缩写/代号（"L2""ACC"——模型不一定理解，关键词匹配一定能）

RRF 融合：     两边排得都高 → 真正相关
              只有一边高   → 至少被一边找到了
              两边都低     → 确实不相关
```

### 5. 从第三章到第四章，管线的变化

```
第三章：
  documents = embed_documents(documents)     # ← 启动时向量化
  context = semantic_search(query, documents) # ← 暴力遍历余弦相似度

第四章：
  documents = embed_documents(documents)     # ← 不变
  v_idx = VectorIndex(documents)             # ← 建索引（替代 for 循环）
  bm25_idx = BM25(documents)                 # ← 建关键词索引（升级第一章）
  context = hybrid_rrf(                       # ← 混合检索（新增）
      v_idx.search(query, top_k=6),
      bm25_idx.search(query, top_k=6)
  )
```

**Prompt 和 LLM 部分纹丝不动。** 和第三章一样——检索策略是可插拔的。

---

## 与各章节的关系

```
第 1 章：Naive RAG
└── 关键词检索 + LLM
    痛点："便宜"匹配不到"经济实惠"

第 2 章：文本分块
└── 按章节标题切分长文本
    痛点：检索粒度已解决，但匹配方式仍是关键词

第 3 章：向量嵌入
└── 文本 → 768 维向量 → 余弦相似度检索
    成果：语义理解
    新痛点：① Python for 循环太慢  ② 数字约束不敏感

第 4 章：FAISS + BM25 混合检索（本章）
└── FAISS 消除 Python 循环开销 + BM25 精确匹配 + RRF 融合
    成果：生产级检索管线
    新痛点：① 没有评估指标  ② 没有 Reranker  ③ 没有多轮对话

第 5 章：完整 RAG Agent
└── 评估 + Reranker + Query改写 + 多轮对话
```

---

## 快速启动

```bash
cd rag-project/chapters

# 首次运行需安装
pip install faiss-cpu

python retrieval_test.py
```

```
[1/4] 加载文档...
      加载 64 条文档
[2/4] 向量化（BGE Embedding）...
      完成，向量维度 768
[3/4] 建 FAISS 向量索引...
      完成，索引内 64 个向量
[4/4] 建 BM25 关键词索引...
      完成，64 篇文档，平均长度 85 词
      词表大小 1523

===========================================================================
[OK] FAISS + BM25 + 混合检索就绪
===========================================================================

>> 请输入查询（输入 q 退出）：25万以内的大空间SUV

 来源                   分数       内容预览
---------------------------------------------------------------------------
[关键词检索 Top-3]
  report_05_buying_gu    -        一、不同预算区间推荐车型...
  极氪 7X                 -        极氪7X 大型纯电豪华SUV，大型纯电SUV...
  比亚迪 海狮08           -        比亚迪海狮08 中大型混动家用SUV...

[向量检索 Top-3]
  理想 L8                0.6438    理想L8 中大型增程家用SUV，中大型SUV...
  AITO M7                0.6426    AITO M7 中大型增程SUV，中大型SUV...
  理想 L6                0.6421    理想L6 中型增程SUV，中型SUV...

[混合 RRF Top-3]
  理想 L6                0.0251    理想L6 中型增程SUV，中型SUV...
  比亚迪 海狮08          0.0183    比亚迪海狮08 中大型混动家用SUV...
  极氪 7X                0.0156    极氪7X 大型纯电豪华SUV，大型纯电SUV...
```

---

## 依赖

```
faiss-cpu              (向量索引，替代第三章的 for 循环)
jieba                  (BM25 分词，和第一章一致)
numpy                  (向量堆叠和运算)
（复用第三章的：sentence-transformers / torch / transformers）
```

---

## 三种索引类型速查（面试用）

| 索引 | 复杂度 | 精度 | 原理 | 适用量级 |
|------|--------|------|------|---------|
| IndexFlatIP | O(n) | 100% | 暴力算全部内积 | < 10 万 |
| IndexIVFFlat | O(√n) | ~95% | K-Means 聚类，只在最近簇搜 | 10万 ~ 1000万 |
| IndexHNSW | O(log n) | ~98% | 图索引，沿邻居边贪心跳转 | 100万 ~ 亿级 |
