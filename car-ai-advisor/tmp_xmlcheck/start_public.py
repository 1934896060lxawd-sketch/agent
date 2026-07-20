"""按 start_all.bat 同款方式拉起 后端+前端+隧道，输出公网地址。"""
import os, re, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = r"E:\coding\agent\.venv\Scripts\python.exe"
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

def spawn(name, cmd, logname, env=None):
    log = open(ROOT / "tmp_xmlcheck" / logname, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=DETACHED, env=env)
    (ROOT / "tmp_xmlcheck" / f"pub_{name}.pid").write_text(str(p.pid))
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
    print(f"❌ {label} 超时"); return False

env = dict(os.environ, USE_FAKEREDIS="1")
spawn("backend", [PY, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"], "pub_backend.log", env)
ok1 = wait_http("http://127.0.0.1:8000/health", 60, "backend")
spawn("streamlit", [PY, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"], "pub_streamlit.log")
ok2 = wait_http("http://127.0.0.1:8501", 60, "streamlit")
spawn("tunnel", [str(ROOT / "tools" / "cloudflared.exe"), "tunnel", "--url", "http://localhost:8501"], "pub_tunnel.log")

url = None
t0 = time.time()
while time.time() - t0 < 60:
    try:
        txt = open(ROOT / "tmp_xmlcheck" / "pub_tunnel.log", encoding="utf-8", errors="replace").read()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
        if m:
            url = m.group(0); break
    except FileNotFoundError:
        pass
    time.sleep(2)

if url:
    print(f"✅ 公网地址: {url} ({time.time()-t0:.1f}s)")
    (ROOT / "tmp_xmlcheck" / "tunnel_url.txt").write_text(url)
else:
    print("❌ 60秒内未获取到隧道地址")
sys.exit(0 if (ok1 and ok2 and url) else 1)
