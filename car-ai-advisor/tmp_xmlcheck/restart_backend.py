import os, subprocess, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = r"E:\coding\agent\.venv\Scripts\python.exe"
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
log = open(ROOT / "tmp_xmlcheck" / "pub_backend.log", "w", encoding="utf-8", errors="replace")
p = subprocess.Popen([PY, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=DETACHED,
                     env=dict(os.environ, USE_FAKEREDIS="1"))
(ROOT / "tmp_xmlcheck" / "pub_backend.pid").write_text(str(p.pid))
t0 = time.time()
while time.time() - t0 < 60:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        print(f"backend 重启就绪 pid={p.pid} ({time.time()-t0:.1f}s)"); break
    except Exception:
        time.sleep(1)
