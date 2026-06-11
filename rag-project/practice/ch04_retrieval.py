"""
第四章：FAISS 向量索引 + BM25 混合检索

解决第三章遗留的两个问题：
  痛点①：暴力遍历 O(n) → FAISS IndexFlatIP 索引
  痛点②：数字/专名匹配弱 → BM25 关键词检索补语义盲区
  最后：RRF 混合融合，取两路长项

你要实现的核心功能：
  1. FAISS 向量索引（建库 + 搜索）
  2. BM25 关键词检索（IDF × 饱和词频 × 长度归一化）
  3. RRF 混合融合（加权求和可选）
  4. 三路对比实验：关键词 vs 向量 vs 混合

依赖：pip install faiss-cpu jieba numpy
（复用第一章的 load_data / keyword_search，第三章的 _embed_model / embed_documents）
"""

import os
import sys
import math
import jieba
import numpy as np
import faiss
from collections import defaultdict

# 复用前两章的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# TODO: from ch01_naive_rag import load_data, search as keyword_search
# TODO: from ch03_embedding import _embed_model, embed_documents
# （练习时可以把三个文件放在同一个目录）


# ============================================================
# 1. FAISS 向量索引
# ============================================================

class VectorIndex:
    """
    把 embed_documents() 生成的向量灌进 FAISS，查询时一次 search() 调用
    替代第三章的 for 循环 + np.dot() 暴力遍历。

    关于 IndexFlatIP：
      - Flat：暴力计算，不压缩不近似，精度 100%
      - IP：Inner Product（内积），向量已归一化 → 内积 = 余弦相似度
      - 64 条文档用 Flat 完全够（0.1ms），千万级再换 IndexIVFFlat 或 IndexHNSW
    """

    def __init__(self, documents: list[dict]):
        """
        建索引三步：
          ① 把所有 embedding 堆成矩阵 (N, 768)，astype(np.float32)
          ② 创建 faiss.IndexFlatIP(dim)
          ③ index.add(embeddings)

        注意：
          - FAISS 只认 float32，显式 astype(np.float32) 最安全
          - embeddings 的 shape 是 (N, dim)，dim 从 documents[0]["embedding"] 拿到
          - add() 之后 index.ntotal 可以看向量总数
        """
        # TODO: ① np.stack + astype(np.float32)
        # TODO: ② 保存 self.embeddings 和 self.documents
        # TODO: ③ 创建并填充 index

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """
        一次 index.search() 替代第三章的：
            for doc in documents:
                sim = np.dot(q_vec, doc["embedding"])

        流程：
          ① query → _embed_model.encode([query], normalize_embeddings=True)
          ② reshape(1, -1) 成 (1, dim)，FAISS 要求二维输入
          ③ self.index.search(q_vec, top_k)，返回 scores 和 ids
          ④ 组装 [(相似度, doc_dict), ...]

        注意：
          - ids[0][i] 可能为 -1（搜不到足够结果时），跳过
          - scores 是 float32 → float() 转换
        """
        # TODO: 实现
        pass


# ============================================================
# 2. BM25 关键词检索
# ============================================================

class BM25:
    """
    BM25 = IDF(稀有词权重高) × 饱和词频 × 文档长度归一化

    和第一章 keyword_search 的区别：
      第一章：score = 命中次数（出现1次=1分，出现100次=100分）
      BM25：  score = IDF × 饱和词频（出现2次基本就饱和了，不会一直涨）
              + 长度归一化（长文档不会天然占优）

    k1 和 b 的含义（不需要改，但面试可能会问）：
      k1=1.5：词频饱和速度，越小饱和越快
      b=0.75：长度归一化强度，0=不管长度，1=完全按比例压缩
    """

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        """
        建 BM25 索引：
          ① 每篇文档用 jieba.cut() 分词 → self.doc_tokens
          ② 统计 DF：self.df[词] = 有多少篇文档包含这个词（用 set 去重）
          ③ 计算 self.avgdl = 所有文档的平均词数

        参数：
          k1：词频饱和参数，默认 1.5
          b：长度归一化参数，默认 0.75
        """
        # TODO: ① jieba 分词 → self.doc_tokens
        # TODO: ② self.N, self.avgdl
        # TODO: ③ DF 统计（用 set 去重每篇文档）
        # TODO: ④ 保存 self.raw_docs, self.k1, self.b

    def idf(self, term: str) -> float:
        """
        标准 BM25 IDF 公式：log(1 + (N - n + 0.5) / (n + 0.5))

        效果：
          - 稀有词（n 小）→ IDF 大，命中它贡献分高
          - 常见词（n 大）→ IDF 接近 0，基本没贡献
          - n=0（词不在任何文档里）→ IDF 最大

        提示：math.log 是自然对数
        """
        # TODO: 实现

    def _score_one(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """
        对一篇文档算 BM25 分。

        对 query 里的每个词 t：
          ① 查 f = doc_tokens.count(t)（该词在这篇文档出现几次）
          ② 算 tf_sat：f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
             — 分子：f × 2.5（线性增长但被分母压平）
             — 分母含长度归一：文档越长分母越大 → tf_sat 越小
          ③ 累加 idf(t) * tf_sat

        返回累积分数
        """
        # TODO: 实现

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """
        对 query 分词 → 逐文档算分 → 排序取 top_k。

        流程：
          ① jieba.cut(query) 分词
          ② 对所有文档调 _score_one()
          ③ sorted + enumerate 按分数降序
          ④ 取 top_k，过滤掉 0 分的

        返回 [(BM25分, doc_dict), ...]
        """
        # TODO: 实现


# ============================================================
# 3. 混合融合
# ============================================================

def hybrid_rrf(v_scored: list[tuple[float, dict]],
               b_scored: list[tuple[float, dict]],
               k: int = 60,
               top_k: int = 3) -> list[tuple[float, dict]]:
    """
    RRF (Reciprocal Rank Fusion) 融合。

    公式：RRF(doc) = Σ 1/(k + rank_i)
      其中 rank_i 是文档在第 i 路检索结果里的排名（从 1 开始）

    为什么 RRF 是生产首选？
      ① 不需要归一化（只用排名，不用原始分数值）
      ② 不需要调参（k=60 在各种数据集上验证过）
      ③ 对单一检索源的极端高分不敏感

    实现：
      ① 遍历 v_scored（已按分数降序），rank 从 0 开始
         → scores[doc_source] += 1.0 / (k + rank + 1)  # +1 因为 rank 从 0 开始
      ② 同样遍历 b_scored
      ③ 按 RRF 分降序排序，取 top_k

    用 doc["source"] 作为去重 key（同一篇文档可能同时出现在两路结果里）

    提示：from collections import defaultdict
          scores = defaultdict(float)  # 自动初始化为 0.0
    """
    # TODO: 实现
    pass


# ============================================================
# 4. 三路对比实验
# ============================================================

if __name__ == "__main__":
    """
    流程：
      ① 确定 data_dir 路径（practice/../data）
      ② load_data(data_dir) 加载文档
      ③ embed_documents(documents) 向量化
      ④ VectorIndex(documents) 建 FAISS 索引
      ⑤ BM25(documents) 建关键词索引
      ⑥ 打印就绪信息（文档数 / 向量数 / 词表大小）

      ⑦ while 循环：
          - 读取用户输入，q 退出
          - keyword_search(query, documents)   ← 第一章
          - v_idx.search(query, top_k=3)      ← 第三章(FAISS版)
          - 两路各取 top_k*2 → hybrid_rrf()    ← 第四章混合
          - 三栏并排打印 Top-3 来源 + 预览

    三栏展示格式：

      关键词                  向量语义                 混合RRF
      ------------------------------------------------------------
      极氪7X                  理想L8(0.6438)          理想L6(0.0251)
      比亚迪海狮08             AITO M7(0.6426)        比亚迪海狮08(0.0183)
      购车指南#sec_1          理想L6(0.6421)          极氪7X(0.0156)

    提示：
      - 来源名截断到 20 字：source[:20]
      - 内容预览截断到 40 字：content[:40]
      - 用 f-string 对齐列宽
    """
    # TODO: 实现
    pass
