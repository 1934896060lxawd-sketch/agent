"""测量后端启动耗时与预热效果。"""
import subprocess, sys, time, urllib.request, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = r"E:\coding\agent\.venv\Scripts\python.exe"
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

log = open(ROOT / "tmp_xmlcheck" / "backend_warm.log", "w", encoding="utf-8", errors="replace")
import os
env = dict(os.environ, USE_FAKEREDIS="1")
p = subprocess.Popen([PY, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=DETACHED, env=env)
(ROOT / "tmp_xmlcheck" / "backend.pid").write_text(str(p.pid))
print(f"backend pid={p.pid}")

t0 = time.time()
while time.time() - t0 < 60:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
            if r.status == 200:
                print(f"✅ /health 就绪: {time.time()-t0:.1f}s")
                break
    except Exception:
        time.sleep(0.5)
else:
    print("❌ /health 超时"); sys.exit(1)
