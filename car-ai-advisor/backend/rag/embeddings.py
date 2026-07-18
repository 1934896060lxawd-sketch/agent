"""嵌入模型 — BGE-base-zh-v1.5 单例加载 + 向量化接口。

本地优先（models/bge-base-zh-v1.5/），不存在时从 HuggingFace 镜像拉取。
normalize_embeddings=True → 内积 = 余弦相似度。
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.rag.chunker import Document

logger = logging.getLogger(__name__)

_embed_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """获取嵌入模型单例。本地优先 → HuggingFace 镜像降级。"""
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    model_name = settings.embedding_model  # "BAAI/bge-base-zh-v1.5"

    # 路径推导：backend/rag/embeddings.py → backend/ → project_root/
    project_root = Path(__file__).resolve().parent.parent.parent
    local_path = project_root / "models" / "bge-base-zh-v1.5"

    if local_path.is_dir() and (local_path / "pytorch_model.bin").exists():
        logger.info(f"从本地加载嵌入模型: {local_path}")
        _embed_model = SentenceTransformer(str(local_path), local_files_only=True)
    else:
        logger.info(f"从 HuggingFace 加载嵌入模型: {model_name}")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        _embed_model = SentenceTransformer(model_name)

    dim = _embed_model.get_embedding_dimension()
    logger.info(f"嵌入模型就绪，向量维度: {dim}")
    return _embed_model


def embed_documents(docs: list[Document]) -> list[Document]:
    """批量向量化文档。为每个 Document.embedding 赋值 (768,) float32 数组。

    normalize_embeddings=True 保证向量单位长度，使内积等价于余弦相似度。
    """
    model = get_embedding_model()
    texts = [doc.content for doc in docs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    for doc, vec in zip(docs, embeddings):
        doc.embedding = vec.astype(np.float32)
    logger.info(f"已向量化 {len(docs)} 个文档")
    return docs


def embed_query(query: str) -> np.ndarray:
    """单条查询向量化。"""
    model = get_embedding_model()
    return model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
