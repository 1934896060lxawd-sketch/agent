"""BGE CrossEncoder 精排 — Query+Doc 联合编码，精度高于双塔模型。

管线位置: 粗排（Top-K×2）→ 精排（Reranker）→ Top-K 喂 LLM。
未安装模型时自动降级为跳过精排，不影响核心功能。

注意：sentence_transformers（含 torch）导入需 20-40 秒，刻意放在函数内
延迟导入，让 uvicorn 启动与 /health 保持秒级响应（模型由启动预热任务加载）。
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from backend.config import settings
from backend.rag.chunker import Document

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

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

    from sentence_transformers import CrossEncoder  # 延迟导入（重）

    project_root = Path(__file__).resolve().parent.parent.parent
    local_path = project_root / "models" / "bge-reranker-base"
    if local_path.is_dir():
        try:
            logger.info(f"从本地加载精排模型: {local_path}")
            _reranker_model = CrossEncoder(str(local_path))
            return _reranker_model
        except Exception as e:
            # 本地模型损坏/缺文件时降级到 HF 分支，绝不能让异常击穿检索工具
            # （历史教训：此处未捕获时 search_car_knowledge 整体报错，
            # 模型拿不到数据就会凭记忆编造参数）
            logger.warning(f"本地精排模型加载失败: {e}，尝试 HuggingFace 降级")

    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        logger.info(f"从 HuggingFace 加载精排模型: {model_name}")

        # 线程超时下载：ThreadPoolExecutor 的 __exit__ 会 wait=True 等待线程完成，
        # 必须 shutdown(wait=False) 才能真正在超时后抛弃后台线程继续执行
        def _load():
            return CrossEncoder(model_name)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_load)
        try:
            _reranker_model = future.result(timeout=3.0)
        except concurrent.futures.TimeoutError:
            logger.warning("精排模型下载超时(>3s)，跳过精排。后续请求将不再重试。")
            _reranker_model = None
        pool.shutdown(wait=False)

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
