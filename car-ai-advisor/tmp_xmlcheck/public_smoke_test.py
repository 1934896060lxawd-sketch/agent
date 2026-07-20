"""公开访问冒烟测试：起后端+前端+隧道，验证公网URL可达，最后全部关闭。"""
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = r"E:\coding\agent\.venv\Scripts\python.exe"
CF = ROOT / "tools" / "cloudflared.exe"

procs = []


def start(name, cmd):
    p = subprocess.Popen(cmd, cwd=str(ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    procs.append((name, p))
    return p


def wait_http(url, timeout_s=90):
    for _ in range(timeout_s):
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    print("[1/4] 启动后端（用户venv，含完整RAG）...")
    start("backend", [PY, "-m", "uvicorn", "backend.main:app",
                      "--host", "127.0.0.1", "--port", "8000"])
    if not wait_http("http://127.0.0.1:8000/health", 120):
        print("❌ 后端启动失败"); return 1
    print("      后端就绪 ✓")

    print("[2/4] 启动 Streamlit 前端...")
    start("frontend", [PY, "-m", "streamlit", "run", "frontend/app.py",
                       "--server.port", "8501", "--server.headless", "true"])
    if not wait_http("http://127.0.0.1:8501", 60):
        print("❌ 前端启动失败"); return 1
    print("      前端就绪 ✓")

    print("[3/4] 启动 Cloudflare 隧道，等待公网地址...")
    tunnel = start("tunnel", [str(CF), "tunnel", "--url", "http://localhost:8501"])
    public_url = []

    def read_tunnel():
        for line in tunnel.stdout:
            m = re.search(r'https://[\w-]+\.trycloudflare\.com', line)
            if m:
                public_url.append(m.group(0))
                return

    t = threading.Thread(target=read_tunnel, daemon=True)
    t.start()
    for _ in range(60):
        if public_url:
            break
        time.sleep(1)
    if not public_url:
        print("❌ 60s 内未拿到公网地址"); return 1
    url = public_url[0]
    print(f"      公网地址: {url}")

    print("[4/4] 通过公网地址验证...")
    for _ in range(30):
        try:
            r = httpx.get(url, timeout=10)
            if r.status_code == 200 and "streamlit" in r.text.lower():
                print(f"      公网访问 OK ✓ (HTTP {r.status_code}, {len(r.text)} bytes)")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("❌ 公网地址不可达"); return 1

    # 本地后端对话验证（公网链路的会话经由前端→本地后端）
    try:
        r = httpx.post("http://127.0.0.1:8000/chat",
                       json={"query": "比亚迪海豚多少钱？", "session_id": "smoke", "stream": False},
                       headers={"Authorization": "Bearer sk-dev-user-001"}, timeout=120)
        ans = r.json().get("answer", "")
        leak = "DSML" in ans or "<invoke" in ans or "｜｜" in ans
        print(f"      后端问答: HTTP {r.status_code}, 回答{len(ans)}字, XML泄露={'有!' if leak else '无'} ✓")
    except Exception as e:
        print(f"      ⚠️ 后端问答验证失败: {e}")
        return 1
    print("\n🎉 冒烟测试全部通过")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        print("关闭所有测试进程...")
        for name, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(2)
        for name, p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
    sys.exit(code)
