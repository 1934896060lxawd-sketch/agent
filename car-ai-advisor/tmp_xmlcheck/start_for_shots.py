import os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = r"E:\coding\agent\.venv\Scripts\python.exe"
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

def spawn(name, cmd, logname, env=None):
    log = open(ROOT / "tmp_xmlcheck" / logname, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=DETACHED, env=env)
    (ROOT / "tmp_xmlcheck" / f"shot_{name}.pid").write_text(str(p.pid))

def wait(url, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2); return True
        except Exception:
            time.sleep(1)
    return False

spawn("backend", [PY, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
      "shot_backend.log", dict(os.environ, USE_FAKEREDIS="1"))
print("backend:", wait("http://127.0.0.1:8000/health", 60))
spawn("streamlit", [PY, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"],
      "shot_streamlit.log")
print("streamlit:", wait("http://127.0.0.1:8501", 60))
