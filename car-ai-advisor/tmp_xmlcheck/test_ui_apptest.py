"""用 Streamlit AppTest 无头执行 app.py，验证改造后脚本无异常、关键元素在位。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=60)
at.run()

print("exception:", [str(e.value)[:200] for e in at.exception] or "无")
print("page_title 检查:", "汽车导购助手" in str(at.title) if at.title else "(title n/a)")

# 极简标题条 / 欢迎页 / 输入框是否存在
md_all = " ".join(m.value for m in at.markdown)
print("极简标题条 top-bar:", "top-bar" in md_all)
print("欢迎语:", "想看什么车？" in md_all or "演示站访问验证" in md_all)
print("chat_input 数量:", len(at.chat_input))
print("sidebar 元素数:", len(at.sidebar) if at.sidebar else 0)
