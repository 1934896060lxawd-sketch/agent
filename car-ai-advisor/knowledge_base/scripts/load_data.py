"""知识库数据加载器 — 遍历 raw/ 目录，自动检测格式并统一输出 Document 列表。

支持: .json / .md / .pdf / .docx
每个文件独立 try/except，一个文件损坏不影响其他文件加载。

用法:
    python knowledge_base/scripts/load_data.py    # 独立运行
    from knowledge_base.scripts.load_data import load_data  # 被 build_index.py 导入
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# 添加到项目根路径
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.config import settings
from backend.rag.chunker import chunk_file, Document, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def load_data(data_dir: Optional[str] = None) -> list[Document]:
    """遍历知识库目录，加载所有支持格式的文档。

    Args:
        data_dir: 知识库 raw/ 目录路径。默认从 settings.knowledge_base_dir 推导。

    Returns:
        统一 Document 列表。

    Raises:
        FileNotFoundError: 目录不存在时。
    """
    if data_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        data_dir = str(project_root / settings.knowledge_base_dir / "raw")
    else:
        data_dir = str(Path(data_dir))

    raw_path = Path(data_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"知识库目录不存在: {raw_path}")

    documents: list[Document] = []
    stats = {"total": 0, "loaded": 0, "skipped": 0, "by_type": {}}

    # 按文件名排序保证可复现
    for file_path in sorted(raw_path.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        stats["total"] += 1
        relative = file_path.relative_to(raw_path)
        logger.info(f"加载: {relative}")

        try:
            chunks = chunk_file(str(file_path))
            if chunks:
                documents.extend(chunks)
                stats["loaded"] += 1
                doc_type = chunks[0].doc_type
                stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + len(chunks)
                logger.info(f"  → {len(chunks)} 个分块 (类型={doc_type})")
            else:
                stats["skipped"] += 1
                logger.warning(f"  → 0 个分块，已跳过")
        except Exception as e:
            stats["skipped"] += 1
            logger.error(f"  → 加载失败 {file_path.name}: {e}", exc_info=True)

    # 汇总
    logger.info(
        f"加载完成: {stats['loaded']}/{stats['total']} 个文件成功, "
        f"{len(documents)} 个总分块, {stats['skipped']} 跳过"
    )
    for dtype, count in sorted(stats["by_type"].items()):
        logger.info(f"  {dtype}: {count} 分块")

    return documents


# ============================================================
# 独立运行入口
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    docs = load_data()
    print(f"\n总计 {len(docs)} 个文档已加载")

    if docs:
        sample = docs[0]
        print(f"示例: source={sample.source}, type={sample.doc_type}, "
              f"content_len={len(sample.content)}, metadata_keys={list(sample.metadata.keys())}")
