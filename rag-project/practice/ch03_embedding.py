import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# 复用第一章的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# TODO: from ch01_naive_rag import load_data, search as keyword_search
from ch01_naive_rag import load_data, search as keyword_search

# TODO: 1. 构造本地模型路径（practice/../models/bge-base-zh-v1.5/BAAI/bge-base-zh-v1___5）
_LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models",
    "bge-base-zh-v1.5", "BAAI", "bge-base-zh-v1___5"
)
if os.path.isdir(_LOCAL_MODEL_DIR):
    # 本地有了，直接加载
    _embed_model = SentenceTransformer(_LOCAL_MODEL_DIR, prompts={})
else:
    # 本地没有，从 HuggingFace 镜像下载（国内网络可能需要）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    _embed_model = SentenceTransformer("BAAI/bge-base-zh-v1.5", prompts={})

# 批量向量化
def embed_documents(documents: list[dict]) -> list[dict]:
    """
    给每条文档的 content 生成 embedding 向量，直接写回 dict 的 "embedding" 字段。

    流程：
      ① 提取所有文档的 content → texts 列表
      ② _embed_model.encode(texts, normalize_embeddings=True)
         → 得到 shape=(N, 768) 的 numpy 数组
      ③ 逐条写回 doc["embedding"] = vec

    为什么 normalize_embeddings=True？
      归一化后向量模长 = 1，两个向量的点积直接等于余弦相似度
      不需要每次检索时除以模长，省计算
    """
    # TODO: 实现
    texts = [doc["content"] for doc in documents]
    embeddings = _embed_model.encode(texts, normalize_embeddings=True)

    for doc, vec in zip(documents, embeddings):
        doc["embedding"] = vec.tolist()

    return documents

# 语义检索
def semantic_search(query: str, documents: list[dict], top_k: int = 3) -> str:
    """
    用向量余弦相似度检索，替代第一章的关键词匹配。

    流程：
      ① query 编码成向量（注意：encode() 必须传 list，v5.x 不接受单字符串）
         q_vec = _embed_model.encode([query], normalize_embeddings=True)[0]
      ② 遍历所有文档，算 q_vec 和 doc["embedding"] 的点积（=余弦相似度）
      ③ 按相似度降序排序，取 top_k
      ④ 拼成上下文文本返回

    提示：np.dot(vec_a, vec_b) 算两个向量的点积
    """
    # TODO: 实现
    # 修复1：取 [0] 降为一维数组 (768,)
    q_vec = _embed_model.encode([query], normalize_embeddings=True)[0]
    scored = []
    for doc in documents:
        # 修复2：list转numpy数组，保证同维度一维向量做点积
        doc_vec = np.array(doc["embedding"])
        sim = float(np.dot(q_vec, doc_vec))
        scored.append((sim,doc))

    scored.sort(key=lambda x:x[0], reverse=True)
    top = scored[:top_k]
    parts = []
    for sim, doc in top:
        parts.append(
            f"【来源：{doc['source']}】(相似度：{sim:.4f})\n{doc['content']}"
        )

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    """
    流程：
      ① 确定 data_dir 路径（practice/../data）
      ② 调用 load_data(data_dir) 加载文档
      ③ 调用 embed_documents(documents) 向量化（启动时跑一次）
      ④ 打印加载数量

      ⑤ while 循环：
          - 读取用户输入（EOFError/KeyboardInterrupt 处理退出）
          - q 退出，空输入跳过
          - 调用 keyword_search(query, documents) 获取关键词结果
          - 调用 semantic_search(query, documents) 获取语义结果
          - 两栏并排打印 Top-3（分别只打印来源 + 前 150 字预览）
    """
    # TODO: 实现
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    documents = load_data(DATA_DIR)
    documents = embed_documents(documents)
    print(f"[OK] 加载并向量化 {len(documents)} 条文档\n")
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

        # ---- 关键词检索（第一章）----
        kw_context = keyword_search(query, documents)
        print("\n" + "=" * 60)
        print("【关键词检索 Top-3】")
        print("=" * 60)
        if kw_context:
            for block in kw_context.split("\n\n---\n\n"):
                lines = block.strip().split("\n")
                source = lines[0] if lines else ""
                preview = "\n".join(lines[1:])[:150] if len(lines) > 1 else ""
                print(f"\n{source}")
                print(f"  {preview}...")
        else:
            print("  (无结果)")

        # ---- 语义检索（第三章）----
        sem_context = semantic_search(query, documents)
        print("\n" + "=" * 60)
        print("【语义检索 Top-3】")
        print("=" * 60)
        if sem_context:
            for block in sem_context.split("\n\n---\n\n"):
                lines = block.strip().split("\n")
                source = lines[0] if lines else ""
                preview = "\n".join(lines[1:])[:150] if len(lines) > 1 else ""
                print(f"\n{source}")
                print(f"  {preview}...")
        else:
            print("  (无结果)")