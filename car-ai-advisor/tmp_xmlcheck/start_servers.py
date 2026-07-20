"""启动真实后端(8000) + Streamlit(8501) 用于UI验证，PID落盘便于清理。"""
import subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = r"E:\coding\agent\.venv\Scripts\python.exe"
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

def spawn(name, cmd, logname):
    log = open(ROOT / "tmp_xmlcheck" / logname, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=DETACHED)
    (ROOT / "tmp_xmlcheck" / f"{name}.pid").write_text(str(p.pid))
    print(f"{name} pid={p.pid}")
    return p

def wait_http(url, timeout, label):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    print(f"✅ {label} 就绪 ({time.time()-t0:.1f}s)")
                    return True
        except Exception:
            time.sleep(2)
    print(f"❌ {label} 超时未就绪")
    return False

spawn("backend", [PY, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"], "backend_ui.log")
ok_b = wait_http("http://127.0.0.1:8000/health", 120, "backend:8000")
spawn("streamlit", [PY, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"], "streamlit_ui.log")
ok_s = wait_http("http://127.0.0.1:8501", 90, "streamlit:8501")
sys.exit(0 if (ok_b and ok_s) else 1)
