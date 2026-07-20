"""模拟外网访客完整流程（最终版）：从 session_state 精确取每轮回答。"""
import re, sys, tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

secrets = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8"))
PWD = secrets.get("ACCESS_PASSWORD", "")
LEAK = re.compile(r'<\s*/?\s*[\w-]*(?:invoke|parameter|tool[\w-]*calls?|function[\w-]*calls?)\b|<\||\|>|DSML|hy-(?:invoke|parameter)', re.I)

at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=180)
at.run()
md = " ".join(m.value for m in at.markdown)
print(f"1. 密码门拦截: {'✅' if '演示站访问验证' in md and len(at.chat_input) == 0 else '❌'}")

pwd_inputs = [t for t in at.text_input if "密码" in (t.label or "")]
pwd_inputs[0].set_value("wrong-password-123").run()
print(f"2. 错误密码被拒绝: {'✅' if '不正确' in ' '.join(e.value for e in at.error) else '❌'}")

pwd_inputs = [t for t in at.text_input if "密码" in (t.label or "")]
pwd_inputs[0].set_value(PWD).run()
md = " ".join(m.value for m in at.markdown)
print(f"3. 正确密码进入: {'✅' if '想看什么车？' in md and len(at.chat_input) == 1 else '❌'}")

def answers():
    store = at.session_state["messages_by_sid"]
    sid = at.session_state["current_sid"]
    return [m["content"] for m in store.get(sid, []) if m["role"] == "assistant"]

def ask(q):
    n = len(answers())
    at.chat_input[0].set_value(q).run()
    a = answers()
    return a[-1] if len(a) > n else ""

ans1 = ask("25万预算推荐什么SUV")
hits = [c for c in ["Model Y", "理想", "问界", "宋L", "G6", "银河"] if c in ans1]
print(f"4. 推荐问答: 命中车型={hits} {'✅' if len(hits) >= 2 else '❌'} | 泄露={LEAK.findall(ans1) or 'CLEAN'}")
print(f"   摘要: {ans1[:100]}")

ans2 = ask("详细说明比亚迪宋L的参数")
print(f"5. 参数问答: {'✅' if ('宋L' in ans2 and '662' in ans2) else '❌'} | 泄露={LEAK.findall(ans2) or 'CLEAN'}")

ans3 = ask("需要")
is_clarify = (0 < len(ans3) < 250 and "87kWh" not in ans3
              and any(k in ans3 for k in ["落地价", "养车", "对比", "还是想", "或者", "吗？"]))
print(f"6. 「需要」应答: 长度={len(ans3)} | 澄清式={'✅' if is_clarify else '❌'} | 泄露={LEAK.findall(ans3) or 'CLEAN'}")
print(f"   回答: {ans3[:130]}")

cap = [c.value for c in at.caption if "剩余提问次数" in c.value]
print(f"7. 提问计数: {'✅ ' + cap[0] if cap else '❌'}")
exc = [str(e.value)[:120] for e in at.exception]
print(f"8. 全程无异常: {'✅' if not exc else '❌ ' + str(exc)}")
