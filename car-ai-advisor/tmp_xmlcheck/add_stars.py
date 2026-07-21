# -*- coding: utf-8 -*-
"""给两份面试问题清单的每道题追加星级，并删除末尾的优先级总结表。"""
import re
from pathlib import Path

ROOT = Path(r"E:\coding\agent\car-ai-advisor")

# 题号 -> 星级（与原优先级总结表一致）
MAP1 = {**{n: "⭐⭐⭐" for n in (1, 4, 12, 13, 16, 23, 31, 40)},
        **{n: "⭐⭐" for n in (5, 6, 17, 18, 22, 27, 32, 35)}}
MAP2 = {**{n: "⭐⭐⭐" for n in (1, 4, 5, 9, 10, 17, 21, 26, 33, 42)},
        **{n: "⭐⭐" for n in (13, 15, 18, 23, 29, 31, 34, 36, 40, 45, 47)}}

HEADER = re.compile(r"^\*\*(\d+)\.\s*(.+?)\*\*\s*$")


def process(path: Path, mapping: dict, expected: int, cut_marker: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, tagged = [], 0
    for line in lines:
        if line.strip().startswith(cut_marker):
            break  # 删除附录（优先级总结表）及其后内容
        # 去掉附录前多余的分隔线
        m = HEADER.match(line)
        if m:
            n = int(m.group(1))
            stars = mapping.get(n, "⭐")
            line = f"**{n}. {m.group(2)}**　{stars}"
            tagged += 1
        out.append(line)
    # 清理结尾：去掉最后一个 "---" 分隔线（附录遗留）
    while out and out[-1].strip() in ("", "---"):
        out.pop()
    out.append("")
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"[OK] {path.name}: 标记 {tagged} 题（预期 {expected}）")


process(ROOT / "面试问题.md", MAP1, 45, "## 附：")
process(ROOT / "面试问题1.md", MAP2, 50, "## 附：高频题优先级")
