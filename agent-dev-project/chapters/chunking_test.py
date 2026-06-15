import re

"""对比三种不同的切分策略"""

def chunk_size(text: str, size = 400, overlap = 80) -> list[dict]:
    """每 size 个字符切一块，相邻块之间重叠 overlap 个字符"""
    start = 0
    chunks = []
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append({"content": chunk.strip(), "chunk_id": f"c_{start}"})
        start += size - overlap  # 下一块起点 = 当前终点 - 重叠量
    return chunks

def chunk_paragraph(text):
    """按段落分割"""
    paragraphs = text.split("\n\n")
    chunks = []
    for i, p in enumerate(paragraphs):
        p = p.strip()
        if p:
            chunks.append({"content": p, "chunk_id": f"para_{i}"})
    return chunks


def chunk_sections(text):
    """按标题，章节分割"""
    pattern = r"([一二三四五六七八九十]+、)"
    parts = re.split(pattern, text)
    chunks = []
    i = 0

    # 处理开头一段（如果有内容且在第一个标题之前）
    if parts and not re.match(pattern, parts[0]):
        preamble = parts[0].strip()
        if preamble:
            chunks.append({"content": preamble, "chunk_id": "sec_preamble"})
        i = 1  # 从第二个元素开始

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



if __name__ == "__main__":
    import os
    import glob

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REPORTS_DIR = os.path.join(BASE_DIR, "..", "data", "industry_reports")

    # 拿第一份报告测试
    report_path = os.path.join(REPORTS_DIR, "report_01_market_overview.txt")
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"原文：{len(text)} 字符\n")

    # 三种策略
    size_chunks = chunk_size(text)
    para_chunks = chunk_paragraph(text)
    section_chunks = chunk_sections(text)

    # 对比表
    print(f"{'策略':<18} {'块数':<6} {'平均长度':<8} {'最短':<6} {'最长':<6}")
    print("-" * 50)

    for label, chunks in [
        ("固定大小 400+80", size_chunks),
        ("按段落", para_chunks),
        ("按章节", section_chunks),
    ]:
        lengths = [len(c["content"]) for c in chunks]
        print(f"{label:<18} {len(chunks):<6} {sum(lengths)//len(lengths):<8} {min(lengths):<6} {max(lengths):<6}")

    # 单独看按章节切出来什么
    print(f"\n{'='*50}")
    print("按章节切出的块标题：")
    for c in section_chunks:
        first_line = c["content"].split("\n")[0]
        print(f"  {c['chunk_id']}: {first_line[:40]}  ({len(c['content'])}字)")

