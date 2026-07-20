import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = r"E:\coding\agent\.venv\Scripts\python.exe"
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
log = open(ROOT / "tmp_xmlcheck" / "download_reranker.log", "w", encoding="utf-8", errors="replace")
p = subprocess.Popen([PY, str(ROOT / "tmp_xmlcheck" / "download_reranker.py")],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=DETACHED)
(ROOT / "tmp_xmlcheck" / "download.pid").write_text(str(p.pid))
print(f"下载进程 pid={p.pid}，日志: tmp_xmlcheck/download_reranker.log")
