# -*- coding: utf-8 -*-
"""生成架构图 docs/images/architecture.png"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from daimon_runtime import setup_plot
setup_plot()

BLUE = "#4d6bfe"
LBLUE = "#eef1ff"
GRAY = "#666666"
DARK = "#1f2329"

fig, ax = plt.subplots(figsize=(12.4, 7.0), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, sub=None, fc=LBLUE, ec=BLUE, title_color=DARK, fs=13):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.6",
        linewidth=1.6, edgecolor=ec, facecolor=fc, mutation_aspect=1,
    )
    ax.add_patch(b)
    if sub:
        ax.text(x + w / 2, y + h / 2 + h * 0.16, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - h * 0.22, sub, ha="center", va="center",
                fontsize=9.5, color=GRAY)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=title_color)


def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, color=BLUE, style="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
        color=color, linestyle=style, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                ha="center", va="center", fontsize=9, color=GRAY)


# ---------- Layer 1: 访问链路 ----------
box(2, 82, 17, 11, "访客浏览器", "手机 / 电脑")
box(25, 82, 17, 11, "Cloudflare 隧道", "公网 HTTPS 分享")
box(48, 82, 20, 11, "Streamlit 前端", "端口 8501 · 访问密码")

arrow(19, 87.5, 25, 87.5)
arrow(42, 87.5, 48, 87.5)

# ---------- Layer 2: 后端 ----------
box(48, 61, 30, 12, "FastAPI 后端", "端口 8000 · 认证 / 滑动窗口限流 / 会话 CRUD / SSE 流式")
box(84, 61, 14, 12, "Redis", "多轮会话记忆\n(本地 fakeredis 兜底)", fs=11.5)

arrow(58, 82, 58, 73, "HTTP / SSE", lx=7.5)
arrow(78, 67, 84, 67, "读写会话", lx=0, ly=2.6, style="--")

# ---------- Layer 3: Agent ----------
box(30, 40, 40, 12, "ReAct Agent 决策核心", "首轮强制工具调用 · XML/协议泄露拦截 · 上下文锚定")
arrow(58, 61, 52, 52)

# ---------- Layer 4: LLM + 工具 ----------
box(2, 17, 24, 12, "LLM API", "GLM-4-Flash / DeepSeek\n一行配置即可切换", fs=12)
box(36, 17, 28, 12, "工具集 (5 个)", "知识库检索 · 车型对比 · 预算推荐\n参数查询 · 购车成本计算", fs=12)
box(74, 17, 24, 12, "RAG 混合检索", "FAISS 向量 + BM25 → RRF 融合\n→ BGE 精排（本地模型 · 启动预热）", fs=12)

arrow(44, 40, 38, 29)            # Agent -> 工具集
arrow(34, 40, 20, 29)            # Agent -> LLM
arrow(64, 23, 74, 23)            # 工具集 -> RAG（知识库检索）

# LLM <-> Agent 双向说明
ax.text(14, 34.5, "思考-行动-观察\n循环推理", ha="center", va="center", fontsize=8.5, color=GRAY)

# 底部说明
ax.text(50, 7.5, "数据流：用户提问 → 前端 SSE 流式渲染 ← 后端逐字符输出（工具调用标记不进入流）",
        ha="center", va="center", fontsize=10, color=DARK)
ax.text(50, 2.5, "知识库：汽车参数 / 评测 / 口碑文档切块入库，检索结果经 RRF 融合排序后由 BGE 精排取 Top-K",
        ha="center", va="center", fontsize=9, color=GRAY)

out = Path(r"E:\coding\agent\car-ai-advisor\docs\images\architecture.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"[OK] saved -> {out} ({out.stat().st_size} bytes)")
