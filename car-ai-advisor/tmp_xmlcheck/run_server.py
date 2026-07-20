"""测试服务器：真实 FastAPI 全链路（与生产完全一致）。

历史说明：早期本机 venv 缺 torch，需要 stub sentence_transformers + 关闭稠密
检索。现在 venv 依赖齐全且模型已本地化（嵌入 + 精排），桩已全部移除——
e2e 走的就是真实生产路径（含启动预热）。仅端口隔离为 8123，避免与
开发服务器 8000 冲突；Redis 用 USE_FAKEREDIS=1 跳过本机连接探测。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_FAKEREDIS", "1")

import uvicorn  # noqa: E402

from backend.main import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8123, log_level="warning")
