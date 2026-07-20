"""诊断：跑 t1,t2 建立历史后，t3 短超时（100s）触发挂起，检查服务端日志。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_live_test import start_server, BASE, HEADERS, OUT  # noqa: E402
import httpx  # noqa: E402

TURNS = [
    ("t1", "你好，我预算15万左右，想买一台新能源SUV，有什么推荐吗？", 200),
    ("t2", "主要是我老婆开，接送孩子上学，想要安全一点的", 200),
    ("t3", "详细说说第一款", 100),
]

proc = start_server()
try:
    with httpx.Client() as client:
        for tag, q, to in TURNS:
            t0 = time.time()
            try:
                with client.stream("POST", f"{BASE}/chat",
                                   json={"query": q, "session_id": "diag", "stream": True},
                                   headers=HEADERS, timeout=to) as resp:
                    n = sum(1 for line in resp.iter_lines() if line)
                print(f"[{tag}] 完成 {time.time()-t0:.0f}s, {n} 行", flush=True)
            except Exception as e:
                print(f"[{tag}] {time.time()-t0:.0f}s 后中断: {type(e).__name__}", flush=True)
finally:
    time.sleep(2)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
print("服务器日志已写入 server.log")
