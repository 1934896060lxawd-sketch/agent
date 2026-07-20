import sys, tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

secrets = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8"))
at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=180)
at.run()
pwd = [t for t in at.text_input if "密码" in (t.label or "")]
pwd[0].set_value(secrets.get("ACCESS_PASSWORD", "")).run()

for q in ["25万预算推荐什么SUV", "详细说明比亚迪宋L的参数", "需要"]:
    before = len(at.markdown)
    at.chat_input[0].set_value(q).run()
    new = [m.value for m in at.markdown[before:]]
    print(f"===== {q} =====")
    for m in new:
        if m.strip():
            print(m[:350])
    print()
