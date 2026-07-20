"""确定性测试：用用户真实录屏中的对话验证 _build_context_hint。

场景还原：
  turn1: 25万预算推荐什么SUV → AI推荐 Model Y / 宋L / 小鹏G6
  turn2: 详细说明比亚迪宋L的参数 → AI给宋L参数解析，结尾问"要不要算宋L的落地价和养车成本？😊"
  turn3: 用户说"需要" → 修复前答非所问（给了小鹏G6参数）
"""
import sys
import types

_stub = types.ModuleType("sentence_transformers")


class _Boom:
    def __init__(self, *a, **k):
        raise RuntimeError("stub")


_stub.SentenceTransformer = _Boom
_stub.CrossEncoder = _Boom
sys.modules["sentence_transformers"] = _stub
sys.path.insert(0, ".")

from backend.agent.advisor import (  # noqa: E402
    _build_context_hint, _extract_car_names, _final_sentence,
)

# ── 用户录屏中的真实 AI 回复（节选关键结构）──
SONGL_ANSWER = """好的，比亚迪宋L的详细数据都拿到了，给你整理一份完整的参数解析👇

比亚迪 宋L — 猎装SUV 全面解析
💰 价格与版本
版本 指导价
550km 后驱版 18.98万
662km 后驱版 22.98万 ← 最推荐，25万预算选这个
662km 四驱版 24.98万 ← 刚好卡预算线

🎯 和Model Y、小鹏G6怎么选？
对比项 宋L Model Y 小鹏G6
价格 18.98-24.98万 24.99万起 20.99-27.69万
续航 662km ✅ 最扎实 688km 755km
智驾 L2 ⚠️ 最弱 AP基础 XNGP最强

要不要我帮你算算宋L 662km后驱版的落地价和一年养车成本？😊"""

history = [
    {"role": "user", "content": "25万预算推荐什么SUV"},
    {"role": "assistant", "content": "推荐：特斯拉 Model Y、比亚迪 宋L、小鹏 G6……"},
    {"role": "user", "content": "详细说明比亚迪宋l的参数"},
    {"role": "assistant", "content": SONGL_ANSWER},
]

print("═══ 1. 结尾句提取（含😊尾巴）═══")
fl = _final_sentence(SONGL_ANSWER)
print(f"final_line: {fl!r}")
assert "养车成本" in fl, "❌ 结尾句被emoji截断！"

print("\n═══ 2. 车型识别（宋L不在价格库）═══")
cars = _extract_car_names(SONGL_ANSWER)
print(f"全篇车型: {cars}")
end_cars = _extract_car_names(fl)
print(f"结尾句车型: {end_cars}")
assert end_cars and "宋L" in end_cars[0], "❌ 结尾句车型识别失败"

print("\n═══ 3. 注入的上下文提示 ═══")
hint = _build_context_hint(history)
print(hint)

print("═══ 4. 关键断言 ═══")
checks = [
    ("意图指向用车成本", "用车成本" in hint or "落地价" in hint),
    ("意图绑定宋L", "宋L" in hint.split("；")[0] if "；" in hint else "宋L" in hint),
    ("不再误判为对比意图", "提议对比" not in hint),
    ("主角不是小鹏G6", "上轮结尾提到的车型：小鹏" not in hint),
    ("含行动指引", "calculate_ownership_cost" in hint),
]
ok = True
for name, passed in checks:
    print(f"  {'✅' if passed else '❌'} {name}")
    ok = ok and passed
print("\n" + ("全部通过 🎉" if ok else "存在失败项 ⚠️"))
sys.exit(0 if ok else 1)
