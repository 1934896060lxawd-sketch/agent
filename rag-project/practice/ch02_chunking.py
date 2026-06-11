"""
第二章：文本分块策略对比

回答一个问题：怎么把长文本切成合适的小块，让检索既精准又不丢上下文？

你要实现的三种策略：
  1. 固定大小切块（带重叠） — 最简单，可能把句子切碎
  2. 按段落切块          — 语义完整，但块大小不均
  3. 按章节标题切块       — 结构最合理，但依赖文本有规范标题

对比维度：块数 / 平均长度 / 最短最长 / 语义完整性

依赖：无（纯 Python 标准库）
"""

import re
import os


# ============================================================
# 策略一：固定大小切块（带重叠）
# ============================================================

def chunk_fixed_size(text: str, size: int = 400, overlap: int = 80) -> list[dict]:
    """
    每 size 个字符切一块，相邻块之间重叠 overlap 个字符。

    为什么要重叠？
      如果一句话刚好被切在中间："这台车的续航特别出色，/ 尤其是在冬天也不衰减"
      前半块和后半块都包含不完整的语义。重叠 80 字让被切断的句子在相邻块里完整出现。

    实现：
      ① start = 0
      ② 每轮取 text[start:start+size]
      ③ 下一轮 start += size - overlap
      ④ 直到 start >= len(text)

    返回：[{"content": "块内容", "chunk_id": "c_0"}, ...]
    """
    # TODO: 实现
    start = 0
    chunks = []
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append({"content": chunk.strip(), "chunk_id": f"{start}"})
        start += size - overlap
    return chunks

# ============================================================
# 策略二：按段落切块
# ============================================================

def chunk_paragraph(text: str) -> list[dict]:
    """
    按段落（\n\n 分隔）切块。

    优点：每个块是一个完整的段落，语义完整
    缺点：段落长度差异大——有的段落几百字，有的只有一行标题

    实现：
      ① text.split("\n\n") 分段
      ② 过滤掉空段落（strip() 后长度为 0）
      ③ 每段一个 chunk

    返回：[{"content": "段落内容", "chunk_id": "para_0"}, ...]
    """
    # TODO: 实现
    chunks = []
    paragraphs = text.split("\n\n")
    for i, p in enumerate(paragraphs):
        p = p.strip()
        if p:
            chunks.append({"content": p, "id": f"para_{i}"})
    return chunks


# ============================================================
# 策略三：按章节标题切块
# ============================================================

def chunk_sections(text: str) -> list[dict]:
    """
    按中文序号标题（一、二、三、...）切块。

    每个块 = 标题行 + 下面直到下一个标题之前的所有内容。

    实现思路：
      ① re.split(r'([一二三四五六七八九十]+、)', text)
         — 括号捕获分隔符，结果里交替出现：正文 / 标题 / 正文 / 标题 ...
      ② 处理开头——如果第一个元素不是标题，当作 "sec_preamble"
      ③ 两两配对：标题 + 正文 = 一个 chunk

    返回：[{"content": "一、市场概况\n...", "chunk_id": "sec_0"}, ...]

    提示：re.match(pattern, text) 返回 None 表示不匹配
    """
    # TODO: 实现
    pattern = r"([一二三四五六七八九十]+、)"
    parts = re.split(pattern, text)
    chunks = []
    i = 0
    
    if parts and not re.match(pattern, parts[0]):
        preamble = parts[0].strip()
        if preamble:
            chunks.append({"content": preamble, "chunk_id": "sec_preamble"})
        i = 1
    
    while i < len(parts):
        if re.match(pattern, parts[i]):
            title = parts[i]
            body = parts[i+1].strip() if i+1 < len(parts) else ""
            chunks.append({
                "content": title + "\n" + body,
                "chunk_id": f"sec_{len(chunks)}"
            })
            i += 2
        else:
            i += 1
            
    return chunks

# ============================================================
# 对比实验
# ============================================================

if __name__ == "__main__":
    """
    流程：
      ① 读取 data/industry_reports/ 下的第一份 .txt 报告作为测试文本
      ② 分别用三种策略切块
      ③ 打印对比表：

         策略               块数    平均长度    最短    最长
         --------------------------------------------------
         固定大小 400+80     XX       XXX       XX     XXX
         按段落              XX       XXX       XX     XXX
         按章节              XX       XXX       XX     XXX

      ④ 额外打印按章节切出的每个块的标题行（前 40 字）和长度

    提示：用 f-string 对齐：f"{label:<18} {len(chunks):<6} {avg:<8} {min_l:<6} {max_l:<6}"
    """
    # TODO: 实现
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REPORTS_DIR = os.path.join(BASE_DIR, "..", "data", "industry_reports")
    reports_path = os.path.join(REPORTS_DIR, "report_01_market_overview.txt")
    with open(reports_path, "r", encoding="utf-8") as f:
        text = f.read()
    test = chunk_sections(text)
    for t in test:
        print(t)
