"""
第四章：FAISS 向量索引 + BM25 混合检索

解决第三章遗留的两个问题：
  痛点①：暴力遍历 O(n) → FAISS IndexFlatIP 索引，O(log n)
  痛点②：数字/专名匹配弱 → BM25 关键词检索补语义盲区
  最后：两路融合（加权求和 + RRF），取各自长项

管线对比：
  第一章：keyword_search()  → jieba 分词 + 命中计数
  第三章：semantic_search() → for 循环 np.dot() 暴力遍历
  第四章：hybrid_rrf()      → FAISS + BM25 + RRF 融合

依赖安装：pip install faiss-cpu
"""

import os
import sys
import math
import jieba
import numpy as np
import faiss
from collections import defaultdict

# 复用前两章的模块级单例
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_rag import load_data, search as keyword_search, build_prompt, ask_llm
from embedding_test import _embed_model, embed_documents


# ============================================================
# 1. FAISS 向量索引 —— 替代第三章的 for 循环遍历
# ============================================================

class VectorIndex:
    """
    把 embed_documents() 生成的向量灌进 FAISS，查询时一次 search() 调用
    替代第三章的手动 for 循环 + np.dot()。

    什么是 IndexFlatIP？
      - Flat：暴力计算，不压缩不近似，精度 100%
      - IP：Inner Product（内积），因为我们向量已归一化（模长=1），
        内积直接等于余弦相似度，不需要再除以模长

    什么时候换别的索引？
      - 64 条文档：Flat 完全够用，0.1ms
      - 64 万条：换 IndexIVFFlat（先聚类再在最近的簇里搜，精度换速度）
      - 6400 万条：换 IndexHNSWFlat（图索引，工业界最常用）
    """

    def __init__(self, documents: list[dict]):
        """
        建索引三步：
          ① 把所有 embedding 堆成矩阵 (N, 768)
          ② new 一个 IndexFlatIP，告诉它向量维度是 768
          ③ add() 把矩阵灌进去
        """
        # ① 堆叠：把 list of (768,) 变成 (64, 768) 的 numpy 矩阵
        self.embeddings = np.stack([doc["embedding"] for doc in documents]).astype(np.float32)
        #     ↑ astype(np.float32) 很重要——FAISS 只认 float32，
        #     sentence-transformers 默认也是 float32，所以通常是一致的，但显式转换更安全

        self.documents = documents

        # ② 建索引：d=768 是向量维度，IP 是内积度量
        dim = self.embeddings.shape[1]  # 768
        self.index = faiss.IndexFlatIP(dim)
        #     ↑ 等价写法 faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)

        # ③ 灌入：这一步只跑一次，O(N*dim)
        self.index.add(self.embeddings)
        #     ↑ add() 之后，索引里就有 64 个向量了，index.ntotal 可以看总数

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """
        一次 index.search() 替代第三章的：
            for doc in documents:
                sim = np.dot(q_vec, doc["embedding"])
                scored.append((sim, doc))

        FAISS 内部用高度优化的 C++ 实现，百万级向量也是毫秒级。

        返回：[(相似度, doc_dict), ...]，按相似度降序
        """
        # ① query → 向量（和第三章完全一样，注意传 list + 取 [0]）
        q_vec = _embed_model.encode([query], normalize_embeddings=True)
        q_vec = q_vec.reshape(1, -1)  # (768,) → (1, 768)，FAISS 要求二维输入

        # ② 一句代码替代 for 循环
        scores, ids = self.index.search(q_vec, top_k)
        #   scores.shape = (1, top_k)，scores[0] = [0.85, 0.72, 0.70]
        #   ids.shape    = (1, top_k)，ids[0]    = [3,    17,   42  ]
        #   ids[0][i] 就是 documents 里的下标

        # ③ 组装结果
        results = []
        for i in range(len(ids[0])):
            doc_id = ids[0][i]
            if doc_id < 0:       # FAISS 用 -1 表示"没搜到足够结果"
                continue
            results.append((float(scores[0][i]), self.documents[doc_id]))
        return results

    def format_results(self, scored: list[tuple[float, dict]]) -> str:
        """格式化输出，和第三章 semantic_search() 保持一致的格式"""
        parts = []
        for sim, doc in scored:
            parts.append(
                f"【来源：{doc['source']}】(相似度：{sim:.4f})\n{doc['content']}"
            )
        return "\n\n---\n\n".join(parts)


# ============================================================
# 2. BM25 关键词检索 —— 升级第一章的简单命中计数
# ============================================================

class BM25:
    """
    第一章的关键词检索只统计"命中了几次"，有三个严重问题：
      ① 没有 IDF：搜索"小米SU7"和搜索"轿车"，后者应该被降权（太常见），但第一章一视同仁
      ② 没有词频饱和：一个词出现 100 次不应该得 100 分，出现 2-3 次就应该接近饱和
      ③ 没有长度归一化：长文档天然包含更多词，应该除以文档长度做公平比较

    BM25 核心公式：
      score(doc, query) = Σ IDF(t) × TF_saturated(t, doc) × length_norm(doc)
      
      其中：
        IDF(t)        = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
                        稀有词（df 小）→ IDF 大，常见词（df 大）→ IDF 接近 0
        TF_saturated  = f × (k1 + 1) / (f + k1 × (1 - b + b × dl / avgdl))
                        f=1→约1分，f=2→约1.3分，f=100→接近k1+1=2.5分（饱和了）
        length_norm   = 文档越长分母越大 → 长文档的词频被压扁

    k1 和 b 的含义：
      k1=1.5：词频饱和速度，越大饱和得越慢（常见词出现多次还能继续涨分）
      b=0.75：长度归一化强度，0=不归一，1=完全按比例压缩，0.75 是经验最优值
    """

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        """
        建 BM25 索引：
          ① 每篇文档用 jieba 分词（和第一章保持一致）
          ② 统计 DF（Document Frequency）：每个词出现在多少篇文档里
          ③ 算 avgdl（平均文档长度）：给后面的长度归一化用
        """
        self.k1 = k1
        self.b = b
        self.raw_docs = documents

        # ① 分词
        self.doc_tokens = [list(jieba.cut(doc["content"])) for doc in documents]

        self.N = len(self.doc_tokens)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / self.N

        # ② 统计 DF：注意用 set 去重——同一个词在一篇文档里出现多次只算 1
        self.df = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1

    def idf(self, term: str) -> float:
        """
        标准 BM25 IDF 公式。
        
        例子（假设 N=64）：
          "小米" 出现在 2 篇 → idf = log(1 + 62.5/2.5) ≈ 3.22 ← 稀有词，分值高
          "SUV" 出现在 40 篇 → idf = log(1 + 24.5/40.5) ≈ 0.49 ← 常见词，分值低
          "的" 出现在 64 篇 → idf = log(1 + 0.5/64.5) ≈ 0.008 ← 基本没贡献
        """
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score_one(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """
        对一篇文档算 BM25 分。
        
        核心逻辑：遍历 query 里的每个词，累加 IDF × 饱和词频。
        如果 query 里的词在这篇文档里没出现 → 不贡献分。
        """
        dl = len(doc_tokens)  # 这篇文档的长度（词数）
        score = 0.0

        for t in query_tokens:
            if t not in doc_tokens:
                continue

            f = doc_tokens.count(t)  # 词频：这个词在这篇文档里出现了几次

            # 饱和变换：f=1→~1, f=2→~1.3, f=10→~2.3, f→∞→k1+1=2.5
            tf_sat = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            #                              ↑ 长度归一化藏在分母里：
            #                              文档比平均长 → 分母变大 → tf_sat 压低
            #                              文档比平均短 → 分母变小 → tf_sat 抬高

            score += self.idf(t) * tf_sat

        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """
        对 query 分词 → 对每篇文档算分 → 按分降序取 top_k。

        和第一章 search() 的对比：
          第一章：score += 1 (if kw in content)  ← 命中一次加一分，简单粗暴
          BM25：  score += IDF(kw) × 饱和词频       ← 考虑词的重要性和出现次数饱和
        """
        tokens = list(jieba.cut(query))

        # 对所有文档打分
        scores = [self._score_one(tokens, doc_tok) for doc_tok in self.doc_tokens]

        # 按分数降序排序，取 top_k（跳过 0 分的）
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(score, self.raw_docs[idx])
                for idx, score in ranked[:top_k] if score > 0]


# ============================================================
# 3. 混合融合 —— 把两路检索结果合并成一个排序
# ============================================================

def min_max_norm(scored: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """
    Min-Max 归一化，把所有分数线性映射到 [0, 1] 区间。

    为什么需要归一化？
      向量余弦相似度范围：[-1, 1]（实际大多是 0.5~0.9）
      BM25 分数范围：       [0, 几十到几百]
      直接加权：BM25 分数碾压向量分数，加权毫无意义
      归一化后：两路都在 [0, 1]，alpha 才有真正的调节作用

    公式：x_norm = (x - min) / (max - min)
    边界情况：如果所有分一样（max=min），统一给 0.5
    """
    if not scored:
        return []
    vals = [s for s, _ in scored]
    v_min, v_max = min(vals), max(vals)
    if v_max == v_min:
        return [(0.5, doc) for _, doc in scored]
    return [((s - v_min) / (v_max - v_min), doc) for s, doc in scored]


def hybrid_weighted(v_scored: list[tuple[float, dict]],
                    b_scored: list[tuple[float, dict]],
                    alpha: float = 0.5,
                    top_k: int = 3) -> list[tuple[float, dict]]:
    """
    加权求和融合。

    流程：
      ① 两路各自 Min-Max 归一化到 [0, 1]
      ② merged[doc_id] = alpha × 向量分 + (1-alpha) × BM25分
      ③ 按融合分降序，取 top_k

    alpha 调节：
      alpha=0   → 纯 BM25（适合精确查询："小米SU7续航多少"）
      alpha=0.5 → 均衡（默认，大多数情况好用）
      alpha=1   → 纯向量（适合模糊查询："推荐一款家用车"）

    生产建议：如果有标注数据（query + 正确答案），网格搜索 0.1~0.9 找最优 alpha
    """

    # ① 归一化
    v_norm = min_max_norm(v_scored)
    b_norm = min_max_norm(b_scored)

    # ② 加权合并。用 doc 的 source 做去重 key
    #    （同一篇文档可能在两路都出现，分数叠加）
    merged = defaultdict(float)
    doc_map = {}

    for score, doc in v_norm:
        key = doc["source"]
        merged[key] = alpha * score
        doc_map[key] = doc

    for score, doc in b_norm:
        key = doc["source"]
        # max 保护：如果向量检索已经给过这文档分，BM25 的分数叠加
        # 如果向量检索没给过这文档分（只出现在 BM25 路），从 0 开始加
        merged[key] = merged.get(key, 0) + (1 - alpha) * score
        doc_map[key] = doc

    # ③ 排序取 top_k
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(score, doc_map[key]) for key, score in ranked]


def hybrid_rrf(v_scored: list[tuple[float, dict]],
               b_scored: list[tuple[float, dict]],
               k: int = 60,
               top_k: int = 3) -> list[tuple[float, dict]]:
    """
    RRF (Reciprocal Rank Fusion) 融合。

    公式：RRF(doc) = Σ 1/(k + rank_i)
      其中 rank_i 是文档在第 i 路检索结果里的排名（从 1 开始，不是从 0）

    举例：
      某文档在向量路排第 1 → 贡献 1/(60+1) = 0.0164
      某文档在 BM25 路排第 3 → 贡献 1/(60+3) = 0.0159
      RRF 总分 = 0.0164 + 0.0159 = 0.0323

    为什么 RRF 是生产首选？
      ① 不需要归一化（只用排名，不用原始分）
      ② 不需要调参（k=60 在各种数据集上验证过，几乎不需要改）
      ③ 对单一检索源的极端高分不敏感（排名 1 和排名 2 的 rrf 差距不大，
         防止向量路一个 0.99 分的噪音结果顶掉所有 BM25 的好结果）

    和加权求和的对比：
                    加权求和                    RRF
      需要归一化     ✅ Min-Max 归一化           ❌ 只用排名
      需要调参       ✅ alpha 要试                ❌ k=60 基本通用
      用到分数值     ✅                           ❌ 只用排名
      受极端值影响   ⚠️ 向量 0.99 分可能          ✅ 排名缓和了极端值
                      主导加权结果
      生产推荐       有标注数据时更好             快速上线首选
    """
    # 对两路结果按排名打分
    scores = defaultdict(float)
    doc_map = {}

    for rank, (_, doc) in enumerate(v_scored):
        key = doc["source"]
        scores[key] += 1.0 / (k + rank + 1)  # rank 从 0 开始，所以 +1 让它从 1 开始
        doc_map[key] = doc

    for rank, (_, doc) in enumerate(b_scored):
        key = doc["source"]
        scores[key] += 1.0 / (k + rank + 1)
        doc_map[key] = doc

    # 按 RRF 分降序取 top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(score, doc_map[key]) for key, score in ranked]


# ============================================================
# 4. 对比实验 —— 三栏并排：关键词 / 语义向量 / 混合RRF
# ============================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

    # ---- 一次性初始化（离线建库阶段） ----
    print("[1/4] 加载文档...")
    documents = load_data(DATA_DIR)
    print(f"      加载 {len(documents)} 条文档")

    print("[2/4] 向量化（BGE Embedding）...")
    documents = embed_documents(documents)
    print(f"      完成，向量维度 {len(documents[0]['embedding'])}")

    print("[3/4] 建 FAISS 向量索引...")
    v_idx = VectorIndex(documents)
    print(f"      完成，索引内 {v_idx.index.ntotal} 个向量")

    print("[4/4] 建 BM25 关键词索引...")
    bm25_idx = BM25(documents)
    print(f"      完成，{bm25_idx.N} 篇文档，平均长度 {bm25_idx.avgdl:.0f} 词")
    print(f"      词表大小 {len(bm25_idx.df)}")

    print(f"\n{'='*75}")
    print(f"[OK] FAISS + BM25 + 混合检索就绪")
    print(f"{'='*75}")

    # ---- 交互式对比实验 ----
    while True:
        try:
            query = input("\n>> 请输入查询（输入 q 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() == "q":
            break
        if not query:
            continue

        # ---- 三路检索 ----
        # 第一章：关键词检索
        kw_context = keyword_search(query, documents)

        # 第三章升级版：FAISS 向量检索（替代暴力 for 循环）
        vec_scored = v_idx.search(query, top_k=3)

        # 第四章：混合 RRF 检索
        # 两路各取 top_k*2 做候选池，给 RRF 更大的去重和重排空间
        vec_candidates = v_idx.search(query, top_k=6)
        bm25_candidates = bm25_idx.search(query, top_k=6)
        hyb_scored = hybrid_rrf(vec_candidates, bm25_candidates, k=60, top_k=3)

        # ---- 三栏并排展示 ----
        print(f"\n{' 来源 ':<22} {' 分数 ':<10} {' 内容预览 ':<40}")
        print("-" * 75)

        # 关键词检索结果
        print("[关键词检索 Top-3]")
        if kw_context:
            for block in kw_context.split("\n\n---\n\n"):
                lines = block.strip().split("\n")
                source = lines[0].replace("【来源：", "").replace("】", "")[:20] if lines else ""
                preview = "\n".join(lines[1:])[:38] if len(lines) > 1 else ""
                print(f"  {source:<20} {'-':<10} {preview}")
        else:
            print("  (无结果)")
        print()

        # 向量检索结果
        print("[向量检索 Top-3]")
        if vec_scored:
            for sim, doc in vec_scored:
                source = doc["source"][:20]
                preview = doc["content"][:38]
                print(f"  {source:<20} {sim:<10.4f} {preview}")
        else:
            print("  (无结果)")
        print()

        # 混合 RRF 结果
        print("[混合 RRF Top-3]")
        if hyb_scored:
            for score, doc in hyb_scored:
                source = doc["source"][:20]
                preview = doc["content"][:38]
                print(f"  {source:<20} {score:<10.4f} {preview}")
        else:
            print("  (无结果)")

        # ---- 可选：用混合结果调 LLM 回答 ----
        print(f"\n{'-'*75}")
        use_llm = input("是否用混合检索结果调用 LLM 回答？(y/n，默认 n)：").strip().lower()
        if use_llm == "y" and hyb_scored:
            # 把混合检索结果格式化成上下文
            context_parts = []
            for _, doc in hyb_scored:
                context_parts.append(
                    f"【来源：{doc['source']}】\n{doc['content']}"
                )
            context = "\n\n---\n\n".join(context_parts)

            messages = build_prompt(query, context)
            answer = ask_llm(messages)
            print(f"\n[AI] {answer}")
