"""对话气泡渲染 — 消息展示、检索来源引用。"""

import streamlit as st


def render_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            render_sources(sources)


def render_sources(sources: list[dict]) -> None:
    """渲染检索参考来源标签。"""
    if not sources:
        return

    with st.expander(f"📚 参考来源（共 {len(sources)} 条）", expanded=False):
        for doc in sources:
            score = doc.get("score", 0.0)
            source_name = doc.get("source", "未知来源")
            rank = doc.get("rank", "?")
            doc_type = doc.get("type", "")
            content_preview = doc.get("content", "")[:150]

            if score > 0.7:
                score_color = "green"
                score_label = "高相关"
            elif score > 0.4:
                score_color = "orange"
                score_label = "中等"
            else:
                score_color = "gray"
                score_label = "低相关"

            type_badge = _doc_type_label(doc_type)

            st.markdown(
                f"**#{rank}** `{source_name}` {type_badge}  "
                f":{score_color}[{score_label} · {score:.3f}]"
            )
            if content_preview:
                st.caption(content_preview)
            st.divider()


def _doc_type_label(doc_type: str) -> str:
    labels = {
        "vehicle": "🚗 车型",
        "review": "📝 评价",
        "glossary": "📖 术语",
        "guide": "📋 指南",
        "industry": "🏭 行业",
        "faq": "❓ 问答",
        "pdf": "📄 文档",
        "word": "📃 文档",
    }
    return labels.get(doc_type, "")
