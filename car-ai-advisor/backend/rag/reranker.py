"""BGE CrossEncoder 精排 — Query+Doc 联合编码，精度高于双塔模型。

管线位置: 粗排（Top-K×2）→ 精排（Reranker）→ Top-K 喂 LLM。
未安装模型时自动降级为跳过精排，不影响核心功能。
"""

import logging
import os
from pathlib import Path

from sentence_transformers import CrossEncoder

from backend.config import settings
from backend.rag.chunker import Document

logger = logging.getLogger(__name__)

_reranker_model: "CrossEncoder | None" = None
_load_attempted: bool = False


def get_reranker() -> "CrossEncoder | None":
    """获取精排模型单例。本地优先 → HuggingFace 降级 → None。"""
    global _reranker_model, _load_attempted
    if _load_attempted:
        return _reranker_model

    _load_attempted = True
    model_name = settings.reranker_model

    project_root = Path(__file__).resolve().parent.parent.parent
    local_path = project_root / "models" / "bge-reranker-base"
    if local_path.is_dir():
        logger.info(f"从本地加载精排模型: {local_path}")
        _reranker_model = CrossEncoder(str(local_path))
        return _reranker_model

    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        logger.info(f"从 HuggingFace 加载精排模型: {model_name}")
        _reranker_model = CrossEncoder(model_name)
    except Exception as e:
        logger.warning(f"精排模型加载失败: {e}。精排步骤将自动跳过。")
        _reranker_model = None

    return _reranker_model


def rerank(
    query: str,
    candidates: list[tuple[float, Document]],
    top_k: int = 5,
) -> list[tuple[float, Document]]:
    """对候选文档精排。模型不可用时直接返回原始候选（降级）。

    Args:
        query: 用户查询
        candidates: 粗排结果 [(score, Document), ...]
        top_k: 返回数量

    Returns:
        [(cross_score, Document), ...] 按分数降序
    """
    model = get_reranker()
    if model is None or not candidates:
        return candidates[:top_k]

    docs = [doc for _, doc in candidates]
    pairs = [[query, doc.content] for doc in docs]
    cross_scores = model.predict(pairs)

    ranked = sorted(zip(cross_scores, docs), key=lambda x: x[0], reverse=True)
    return [(float(s), doc) for s, doc in ranked[:top_k]]
