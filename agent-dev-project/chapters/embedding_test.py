"""
第三章：向量嵌入（Embedding）语义检索 —— 对比实验

跑同一个 query，并排看「关键词检索」vs「语义检索」的结果差异。
"""
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# 把 ch01 的函数引进来
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_rag import load_data, search as keyword_search

# ============================================================
# 模块级初始化：优先从本地加载模型
# ============================================================
_LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models",
    "bge-base-zh-v1.5", "BAAI", "bge-base-zh-v1___5"
)
if os.path.isdir(_LOCAL_MODEL_DIR):
    _embed_model = SentenceTransformer(_LOCAL_MODEL_DIR, prompts={})
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    _embed_model = SentenceTransformer("BAAI/bge-base-zh-v1.5", prompts={})


def embed_documents(documents: list[dict]) -> list[dict]:
    """给每条文档的 content 生成 embedding 向量，直接写回 dict"""
    texts = [doc["content"] for doc in documents]
    embeddings = _embed_model.encode(texts, normalize_embeddings=True)
    for doc, vec in zip(documents, embeddings):
        doc["embedding"] = vec
    return documents


def semantic_search(query: str, documents: list[dict], top_k: int = 3) -> str:
    """用向量余弦相似度检索，替代关键词匹配"""
    q_vec = _embed_model.encode([query], normalize_embeddings=True)[0]

    # 计算 query 和每条文档的点积（归一化后等价于余弦相似度）
    scored = []
    for doc in documents:
        sim = float(np.dot(q_vec, doc["embedding"]))
        scored.append((sim, doc))

    # 按相似度降序，取 top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    parts = []
    for sim, doc in top:
        parts.append(
            f"【来源：{doc['source']}】(相似度：{sim:.4f})\n{doc['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ============================================================
# 对比实验
# ============================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

    # ① 加载 & 向量化（embed 跑一次就行）
    documents = load_data(DATA_DIR)
    documents = embed_documents(documents)
    print(f"[OK] 加载并向量化 {len(documents)} 条文档\n")

    # ② 交互对比
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

        # ---- 关键词检索 ----
        kw_context = keyword_search(query, documents)
        print("\n" + "=" * 60)
        print("【关键词检索 Top-3】")
        print("=" * 60)
        if kw_context:
            # 只打印来源行 + 前 150 字，方便对比
            for block in kw_context.split("\n\n---\n\n"):
                lines = block.strip().split("\n")
                source_line = lines[0] if lines else ""
                preview = "\n".join(lines[1:])[:150] if len(lines) > 1 else ""
                print(f"\n{source_line}")
                print(f"  {preview}...")
        else:
            print("  (无结果)")

        # ---- 语义检索 ----
        sem_context = semantic_search(query, documents)
        print("\n" + "=" * 60)
        print("【语义检索 Top-3】")
        print("=" * 60)
        if sem_context:
            for block in sem_context.split("\n\n---\n\n"):
                lines = block.strip().split("\n")
                source_line = lines[0] if lines else ""
                preview = "\n".join(lines[1:])[:150] if len(lines) > 1 else ""
                print(f"\n{source_line}")
                print(f"  {preview}...")
        else:
            print("  (无结果)")
