"""会话 CRUD 全链路验证：创建/列表/详情/重命名/历史/删除/越权。"""
import httpx

BASE = "http://127.0.0.1:8000"
H = {"Authorization": "Bearer sk-dev-user-001"}
H_BAD = {"Authorization": "Bearer sk-wrong-key-999"}
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'✅' if ok else '❌'} {name} {detail}")

c = httpx.Client(base_url=BASE, headers=H, timeout=60)

# ── 创建 ──
r = c.post("/sessions", json={"title": "测试会话A"})
check("创建会话→201", r.status_code == 201, f"got {r.status_code}")
sid = r.json().get("session_id", "")
check("返回session_id", bool(sid), sid[:16])

r = c.post("/sessions", json={"title": "测试会话B"})
sid_b = r.json().get("session_id", "")
check("创建第二个会话", r.status_code == 201 and bool(sid_b))

# ── 列表 ──
r = c.get("/sessions")
titles = [s.get("title") for s in r.json().get("sessions", [])]
check("列表包含新会话", "测试会话A" in titles and "测试会话B" in titles, str(titles[:5]))

# ── 详情 ──
r = c.get(f"/sessions/{sid}")
check("详情查询", r.status_code == 200 and r.json().get("title") == "测试会话A")

# ── 重命名 ──
r = c.patch(f"/sessions/{sid}", json={"session_id": sid, "new_title": "改名后的会话"})
check("重命名", r.status_code == 200 and r.json().get("success"))
r = c.get(f"/sessions/{sid}")
check("重命名生效", r.json().get("title") == "改名后的会话")

# ── 在会话中聊天 → 历史 ──
r = c.post("/chat", json={"session_id": sid, "query": "比亚迪海豚多少钱", "stream": False})
check("会话内提问", r.status_code == 200 and len(r.json().get("answer", "")) > 0)
r = c.get(f"/sessions/{sid}/history")
msgs = r.json().get("messages", [])
check("历史含2条消息(问+答)", r.status_code == 200 and len(msgs) == 2, f"got {len(msgs)}")

# ── 越权/伪造 ──
r = httpx.Client(base_url=BASE, headers=H_BAD, timeout=30).get("/sessions")
check("伪造key访问→401/403", r.status_code in (401, 403), f"got {r.status_code}")

# ── 删除 ──
r = c.delete(f"/sessions/{sid_b}")
check("删除会话B→204", r.status_code == 204, f"got {r.status_code}")
r = c.get("/sessions")
titles = [s.get("title") for s in r.json().get("sessions", [])]
check("列表不再含B", "测试会话B" not in titles)
r = c.get(f"/sessions/{sid_b}")
check("删除后详情→404", r.status_code == 404, f"got {r.status_code}")
r = c.get(f"/sessions/{sid_b}/history")
check("删除后历史→404", r.status_code == 404, f"got {r.status_code}")
r = c.delete(f"/sessions/{sid_b}")
check("重复删除→404", r.status_code == 404, f"got {r.status_code}")

# ── 带消息的会话删除 → 消息应一并清理 ──
r = c.delete(f"/sessions/{sid}")
check("删除含消息的会话A→204", r.status_code == 204, f"got {r.status_code}")
r = c.get(f"/sessions/{sid}/history")
check("A的历史随之清除→404", r.status_code == 404, f"got {r.status_code}")

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'全部通过' if not failed else '失败: ' + str(failed)} ({len(results)-len(failed)}/{len(results)})")
