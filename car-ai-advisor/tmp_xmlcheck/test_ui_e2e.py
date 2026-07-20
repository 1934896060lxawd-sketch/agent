"""AppTest + 真实后端：从UI输入框发一个问题，验证回答渲染且无XML泄露。"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=120)
at.run()
assert not at.exception, [str(e.value) for e in at.exception]

at.chat_input[0].set_value("比亚迪海豚多少钱").run()

print("exception:", [str(e.value)[:200] for e in at.exception] or "无")
msgs = [m.value for m in at.markdown]
joined = "\n".join(msgs)
hit = [m for m in msgs if "9.98" in m]
print("用户消息已渲染:", any("比亚迪海豚多少钱" in m for m in msgs))
print("AI回答含真实价格(9.98-13.98):", bool(hit))
if hit:
    print("回答片段:", hit[0][:160].replace("\n", " "))
leak = re.findall(r'<\s*/?\s*[\w-]*(?:invoke|parameter|tool[\w-]*calls?|function[\w-]*calls?)\b|<\||\|>|DSML|hy-(?:invoke|parameter)', joined)
print("XML泄露检查:", leak or "CLEAN")
