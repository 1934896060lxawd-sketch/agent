"""Streamlit 主入口 — 智能汽车导购助手前端。"""

import asyncio
import logging
import sys
from pathlib import Path

# Streamlit 以脚本方式运行，需手动将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from frontend.api_client import APIClient, run_async
from frontend.components.chat import render_sources
from frontend.components.sidebar import render_sidebar, _current_msgs, _set_current_msgs

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="智能汽车导购助手",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局 CSS ──
st.markdown("""
<style>
/* 主色调：深蓝科技感 + 暖橙点缀 */
:root {
    --primary: #1a73e8;
    --accent: #ff6d00;
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #1e293b;
    --muted: #64748b;
    --border: #e2e8f0;
}

/* 页面背景 */
.stApp { background-color: var(--bg); }

/* 标题区域 */
.main-header {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    color: white;
}
.main-header h1 { color: white !important; margin: 0; font-size: 1.8rem; }
.main-header p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.95rem; }

/* 侧边栏美化 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f1f5f9 0%, #e8edf3 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px;
    transition: all 0.2s;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* 聊天输入 */
.stChatInput > div {
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.stChatInput > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 2px 16px rgba(26,115,232,0.12);
}

/* 聊天消息气泡 */
.stChatMessage {
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.stChatMessage[data-testid="stChatMessage"] {
    background: var(--card);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* 欢迎卡片 */
.welcome-card {
    text-align: center;
    padding: 3rem 2rem;
    background: var(--card);
    border-radius: 16px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin: 2rem auto;
    max-width: 650px;
}
.welcome-card h2 { color: var(--text); margin-bottom: 0.5rem; }
.welcome-card p { color: var(--muted); font-size: 0.95rem; }
.welcome-card .quick-actions {
    display: flex; gap: 0.5rem; justify-content: center;
    flex-wrap: wrap; margin-top: 1.5rem;
}
.quick-chip {
    background: var(--bg); border: 1px solid var(--border);
    padding: 0.4rem 1rem; border-radius: 20px;
    font-size: 0.85rem; color: var(--muted);
}

/* 状态提示 */
.stAlert {
    border-radius: 10px;
    border: none;
}

/* 分割线 */
hr { border-color: var(--border); margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ── 标题 ──
st.markdown("""
<div class="main-header">
    <h1>🚗 智能汽车导购助手</h1>
    <p>基于 DeepSeek 大模型 + RAG 知识库，为您提供专业的新能源汽车选购建议</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_api_client() -> APIClient:
    return APIClient(
        base_url=st.secrets.get("API_BASE_URL", "http://localhost:8000"),
        api_key=st.secrets.get("API_KEY", "sk-dev-user-001"),
    )


def init_session_state() -> None:
    if "current_sid" not in st.session_state:
        st.session_state.current_sid = "default"
    if "messages_by_sid" not in st.session_state:
        st.session_state.messages_by_sid = {}
    if "session_named" not in st.session_state:
        st.session_state.session_named = False


def main() -> None:
    init_session_state()
    client = get_api_client()

    render_sidebar(client)

    msgs = _current_msgs()

    # ── 欢迎页（无消息时）──
    if not msgs:
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 欢迎使用智能汽车导购</h2>
            <p>我了解 16 款主流新能源车型的详细参数、价格、车主评价</p>
            <p>可以帮您推荐车型、对比配置、计算用车成本</p>
            <div class="quick-actions">
                <span class="quick-chip">💰 预算推荐</span>
                <span class="quick-chip">⚡ 续航对比</span>
                <span class="quick-chip">🔍 车型查询</span>
                <span class="quick-chip">📊 配置对比</span>
                <span class="quick-chip">🧮 成本计算</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 渲染消息 ──
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    # ── 输入框 ──
    if prompt := st.chat_input('请输入您的问题，比如"25万预算推荐什么SUV？"'):

        if not st.session_state.session_named:
            st.session_state.session_named = True
            sid = st.session_state.current_sid
            if sid != "default":
                short_name = prompt[:15].replace("\n", " ")
                try:
                    run_async(client.rename_session(sid, short_name))
                except Exception:
                    pass

        msgs.append({"role": "user", "content": prompt})
        _set_current_msgs(msgs)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            text_placeholder = st.empty()

            full_text = ""
            sources: list[dict] = []
            has_first_token = False

            try:
                status_placeholder.info("⏳ 正在分析您的问题…")

                async def _stream():
                    nonlocal full_text, sources, has_first_token
                    async for event in client.chat_stream(
                        prompt, st.session_state.current_sid
                    ):
                        event_type = event.get("type", "")

                        if event_type == "token":
                            if not has_first_token:
                                has_first_token = True
                                status_placeholder.empty()
                            full_text += event.get("content", "")
                            text_placeholder.markdown(full_text + "▌")

                        elif event_type == "source":
                            if not has_first_token:
                                status_placeholder.info("🔍 正在检索知识库…")
                            docs = event.get("documents", [])
                            sources = [d for d in docs
                                       if not d.get("source", "").startswith("tool:")]

                        elif event_type == "done":
                            status_placeholder.empty()
                            text_placeholder.markdown(full_text)
                            if sources:
                                render_sources(sources)

                        elif event_type == "error":
                            status_placeholder.empty()
                            text_placeholder.error(f"请求失败: {event.get('message', '未知错误')}")
                            break

                asyncio.run(_stream())

            except Exception as e:
                status_placeholder.empty()
                text_placeholder.error(f"对话异常: {str(e)}")
                logger.error(f"对话失败: {e}", exc_info=True)

            if full_text:
                msgs = _current_msgs()
                msgs.append({
                    "role": "assistant",
                    "content": full_text,
                    "sources": sources,
                })
                _set_current_msgs(msgs)


if __name__ == "__main__":
    main()
