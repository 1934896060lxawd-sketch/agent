"""API 压力测试 — 会话CRUD + 20+次问答 + 历史验证

用法: python tests/api_stress_test.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx

BASE = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-dev-user-001", "Content-Type": "application/json"}

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


async def test():
    global passed, failed
    print("=" * 60)
    print("API 压力测试开始")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=180) as c:

        # ─── 1. 健康检查 ───
        print("\n── 1. 健康检查 ──")
        r = await c.get(f"{BASE}/health")
        check("GET /health 200", r.status_code == 200, str(r.status_code))

        # ─── 2. 会话 CRUD ───
        print("\n── 2. 会话创建 ──")
        sids = []
        for i in range(5):
            r = await c.post(f"{BASE}/sessions", json={"title": f"测试会话{i+1}"}, headers=HEADERS)
            ok = r.status_code == 201
            check(f"创建会话{i+1}", ok, str(r.status_code))
            if ok:
                data = r.json()
                sids.append(data["session_id"])

        print("\n── 3. 会话列表 ──")
        r = await c.get(f"{BASE}/sessions", headers=HEADERS)
        ok = r.status_code == 200
        data = r.json()
        check("GET /sessions 200", ok)
        check(f"会话数 >= 5", data.get("total", 0) >= 5, f"total={data.get('total')}")

        # ─── 4. 20+ 轮问答 ───
        print("\n── 4. 问答测试 (20轮) ──")
        queries = [
            "你好", "推荐一款15万的SUV", "小米SU7怎么样", "特斯拉Model 3价格",
            "比亚迪海豚续航多少", "理想L6和问界M7对比", "混动和纯电怎么选",
            "20万预算家庭用车", "极氪001值得买吗", "蔚来换电方便吗",
            "插混和增程的区别", "10万以内电动车推荐", "小鹏G6有什么优缺点",
            "家用第一辆车选轿车还是SUV", "比亚迪海豹适合年轻人吗",
            "30万预算大空间SUV", "新能源车保养贵吗", "买电动车还是燃油车",
            "问界M7值得买吗", "25万左右操控好的轿车",
        ]

        total_latency = 0.0
        success_count = 0
        sid = sids[0]  # 使用第一个会话
        for i, q in enumerate(queries):
            t0 = time.time()
            r = await c.post(f"{BASE}/chat",
                           json={"query": q, "session_id": sid, "stream": False},
                           headers=HEADERS)
            t1 = time.time()
            lat = (t1 - t0) * 1000
            ok = r.status_code == 200
            if ok:
                total_latency += lat
                success_count += 1
                data = r.json()
                ans_len = len(data.get("answer", ""))
                check(f"Q{i+1}: {q[:15]}...", ans_len > 0, f"{lat:.0f}ms, {ans_len}字")
            else:
                check(f"Q{i+1}: {q[:15]}...", False, f"status={r.status_code}")

        if success_count > 0:
            avg_lat = total_latency / success_count
            print(f"\n  [STATS] 平均延迟: {avg_lat:.0f}ms, 成功率: {success_count}/{len(queries)}")

        # ─── 5. 会话重命名 ───
        print("\n── 5. 会话重命名 ──")
        new_titles = ["SUV推荐", "价格咨询", "车型对比"]
        for i, (rename_sid, title) in enumerate(zip(sids[:3], new_titles)):
            r = await c.patch(f"{BASE}/sessions/{rename_sid}",
                            json={"session_id": rename_sid, "new_title": title},
                            headers=HEADERS)
            check(f"重命名 → {title}", r.status_code == 200, str(r.status_code))

        # 验证重命名生效
        r = await c.get(f"{BASE}/sessions", headers=HEADERS)
        sessions = r.json().get("sessions", [])
        renamed = [s for s in sessions if s["title"] in new_titles]
        check(f"重命名后列表含{len(new_titles)}个新名", len(renamed) == 3,
              f"找到{len(renamed)}个")

        # ─── 6. 历史消息验证 ───
        print("\n── 6. 历史消息验证 ──")
        r = await c.get(f"{BASE}/sessions/{sid}/history", headers=HEADERS)
        ok = r.status_code == 200
        data = r.json()
        check("GET history 200", ok)
        msg_count = data.get("count", 0)
        check(f"消息数={msg_count} (>=20)", msg_count >= 20,
              f"count={msg_count}")

        # ─── 7. 会话删除 ───
        print("\n── 7. 会话删除 ──")
        for i, sid in enumerate(sids[3:5]):  # 删除最后2个
            r = await c.delete(f"{BASE}/sessions/{sid}", headers=HEADERS)
            check(f"删除会话{sid[:8]}", r.status_code == 204, str(r.status_code))

        # 验证删除后数量
        r = await c.get(f"{BASE}/sessions", headers=HEADERS)
        final_total = r.json().get("total", 0)
        check("删除后会话数减少", final_total < data.get("total", 999),
              f"从{data.get('total')} → {final_total}")

        # ─── 8. 边界测试 ──
        print("\n── 8. 边界测试 ──")
        # 空query
        r = await c.post(f"{BASE}/chat",
                       json={"query": "", "session_id": "x", "stream": False},
                       headers=HEADERS)
        check("空query拒绝", r.status_code == 422, str(r.status_code))

        # 无认证
        r = await c.post(f"{BASE}/chat",
                       json={"query": "hi", "session_id": "x", "stream": False})
        check("无认证拒绝(401/403)", r.status_code in (401, 403), str(r.status_code))

        # 不存在的会话历史
        r = await c.get(f"{BASE}/sessions/nonexistent/history", headers=HEADERS)
        check("不存在会话返回404", r.status_code == 404, str(r.status_code))

    # ─── 结果汇总 ───
    print(f"\n{'=' * 60}")
    total = passed + failed
    print(f"结果: {passed}/{total} 通过", end="")
    if failed > 0:
        print(f", {failed} FAILED")
    else:
        print(" ALL PASSED!")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
