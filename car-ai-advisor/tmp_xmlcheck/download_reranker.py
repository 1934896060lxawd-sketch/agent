"""从 hf-mirror 下载 BAAI/bge-reranker-base 到本地 models/ 目录。"""
import os, sys, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
target = ROOT / "models" / "bge-reranker-base"

from huggingface_hub import snapshot_download
t0 = time.time()
print(f"开始下载 → {target}", flush=True)
path = snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    local_dir=str(target),
    ignore_patterns=["onnx/*", "*.ot", "tf_model*", "rust_model*", "flax_model*", "*.tflite"],
)
print(f"下载完成 ({time.time()-t0:.0f}s): {path}", flush=True)
for f in sorted(target.iterdir()):
    print(f"  {f.name} {f.stat().st_size/1e6:.1f}MB", flush=True)
