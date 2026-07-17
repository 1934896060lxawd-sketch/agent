"""知识库索引构建脚本 — 离线运行，生成 FAISS 索引和向量化文档。

用法:
    python knowledge_base/scripts/build_index.py

流程:
    load_data() → embed_documents() → build FAISS → save to processed/
"""

import logging
import pickle
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import faiss
import numpy as np

from backend.config import settings
from backend.rag.chunker import Document
from backend.rag.embeddings import get_embedding_model, embed_documents
from knowledge_base.scripts.load_data import load_data

logger = logging.getLogger(__name__)


def build_index(data_dir: str | None = None) -> tuple["faiss.IndexFlatIP", list[Document]]:
    """构建 FAISS 索引。

    Returns:
        (faiss_index, embedded_documents)
    """
    logger.info("=" * 50)
    logger.info("Step 1/3: 加载知识库文档...")
    docs = load_data(data_dir)
    logger.info(f"  加载 {len(docs)} 个文档")

    logger.info("Step 2/3: 向量化文档...")
    docs = embed_documents(docs)
    dim = docs[0].embedding.shape[0] if docs else 0
    logger.info(f"  向量维度: {dim}")

    logger.info("Step 3/3: 构建 FAISS 索引...")
    if not docs:
        index = faiss.IndexFlatIP(768)
    else:
        embeddings = np.stack([doc.embedding for doc in docs]).astype(np.float32)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    logger.info(f"  索引内 {index.ntotal} 个向量")

    return index, docs


def save_index(
    index: "faiss.IndexFlatIP",
    documents: list[Document],
    output_dir: str | None = None,
) -> str:
    """保存索引和文档到磁盘。"""
    if output_dir is None:
        output_dir = str(_project_root / settings.knowledge_base_dir / "processed")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = str(out / "faiss_index.bin")
    docs_path = str(out / "documents.pkl")

    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)

    logger.info(f"索引已保存: {index_path} ({index.ntotal} 向量)")
    logger.info(f"文档已保存: {docs_path} ({len(documents)} 条)")
    return str(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("预热嵌入模型...")
    model = get_embedding_model()
    logger.info(f"嵌入模型就绪: 维度={model.get_sentence_embedding_dimension()}")

    index, docs = build_index()
    output = save_index(index, docs)
    print(f"\n索引构建完成: {output}")
    print(f"向量数: {index.ntotal}, 文档数: {len(docs)}")
