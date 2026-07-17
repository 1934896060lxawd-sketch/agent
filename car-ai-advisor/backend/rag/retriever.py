"""混合检索引擎 — FAISS 向量检索 + BM25 关键词检索 + RRF 融合。

管线:
  用户 query → embed_query() → FAISS.search()  ─┐
              → jieba 分词 → BM25.search()  ───┤
                                                ├→ hybrid_rrf() → Top-K 文档
"""

import logging
import math
from collections import defaultdict

import faiss
import jieba
import numpy as np

from backend.rag.embeddings import get_embedding_model
from backend.rag.chunker import Document

logger = logging.getLogger(__name__)


# ============================================================
# FAISS 向量索引
# ============================================================
class VectorIndex:
    """FAISS IndexFlatIP — 内积搜索 = 归一化向量上的余弦相似度。

    当前知识库 ~100 条文档，Flat（暴力精确搜索）完全够用。
    扩展到 10 万+ 时换 IndexIVFFlat，百万+ 换 IndexHNSWFlat。
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents
        embeddings = np.stack([doc.embedding for doc in documents]).astype(np.float32)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info(f"FAISS 索引就绪: {self.index.ntotal} 个向量, 维度={dim}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Document]]:
        model = get_embedding_model()
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        q_vec = q_vec.reshape(1, -1)
        scores, ids = self.index.search(q_vec, top_k)
        results: list[tuple[float, Document]] = []
        for i in range(len(ids[0])):
            doc_id = ids[0][i]
            if doc_id < 0:
                continue
            results.append((float(scores[0][i]), self.documents[doc_id]))
        return results

    @classmethod
    def load(cls, embeddings_path: str, docs_path: str) -> "VectorIndex":
        """从磁盘加载 FAISS 索引和文档列表。"""
        import pickle
        index = faiss.read_index(embeddings_path)
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
        obj = cls.__new__(cls)
        obj.index = index
        obj.documents = documents
        logger.info(f"FAISS 索引从磁盘加载: {index.ntotal} 个向量")
        return obj

    def save(self, embeddings_path: str, docs_path: str) -> None:
        """保存 FAISS 索引和文档列表到磁盘。"""
        import pickle
        faiss.write_index(self.index, embeddings_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"FAISS 索引已保存: embeddings={embeddings_path}, docs={docs_path}")


# ============================================================
# BM25 关键词检索
# ============================================================
class BM25:
    """标准 BM25 实现 — IDF 加权 + 词频饱和 + 长度归一化。

    k1=1.5: 词频饱和速度，b=0.75: 长度归一化强度。
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_tokens = [list(jieba.cut(doc.content)) for doc in documents]
        self.N = len(self.doc_tokens)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / max(self.N, 1)

        self.df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1

        logger.info(f"BM25 就绪: {self.N} 篇文档, 平均长度={self.avgdl:.0f} 词, 词表={len(self.df)}")

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score_one(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        dl = len(doc_tokens)
        score = 0.0
        for t in query_tokens:
            if t not in doc_tokens:
                continue
            f = doc_tokens.count(t)
            tf_sat = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += self.idf(t) * tf_sat
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Document]]:
        tokens = list(jieba.cut(query))
        scores = [self._score_one(tokens, doc_tok) for doc_tok in self.doc_tokens]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(score, self.documents[idx]) for idx, score in ranked[:top_k] if score > 0]


# ============================================================
# RRF 混合融合
# ============================================================
def hybrid_rrf(
    vec_results: list[tuple[float, Document]],
    bm25_results: list[tuple[float, Document]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[float, Document]]:
    """Reciprocal Rank Fusion — 按排名融合，无需归一化。

    RRF(doc) = Σ 1/(k + rank_i)
    k=60 在所有数据集上通用，不需要调参。
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for rank, (_, doc) in enumerate(vec_results):
        scores[doc.chunk_id] += 1.0 / (k + rank + 1)
        doc_map[doc.chunk_id] = doc

    for rank, (_, doc) in enumerate(bm25_results):
        scores[doc.chunk_id] += 1.0 / (k + rank + 1)
        doc_map[doc.chunk_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(score, doc_map[key]) for key, score in ranked]
