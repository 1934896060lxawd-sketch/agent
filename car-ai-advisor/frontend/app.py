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
    page_title="汽车导购助手",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 全局 CSS（DeepSeek 式极简风）──
st.markdown("""
<style>
/* 调色板：白底 + DeepSeek 蓝点缀，无渐变无阴影 */
:root {
    --primary: #4d6bfe;
    --bg: #ffffff;
    --sidebar-bg: #f9fafb;
    --text: #171717;
    --muted: #6b7280;
    --border: #e5e7eb;
    --user-bubble: #edf1fd;
}

/* 页面背景与内容列 */
.stApp { background-color: var(--bg); }
.block-container {
    max-width: 800px;
    padding-top: 1.5rem;
}

/* 隐藏 Streamlit 默认装饰（菜单/工具栏/页脚），保持界面干净 */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* 顶部极简标题条 */
.top-bar {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding: 0.2rem 0 0.8rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.top-bar .top-title { font-size: 1.05rem; font-weight: 600; color: var(--text); }
.top-bar .top-sub { font-size: 0.8rem; color: var(--muted); }

/* 侧边栏：浅灰底、扁平按钮 */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px;
    border: none;
    background: transparent;
    box-shadow: none;
    font-size: 0.88rem;
    text-align: left;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #eef0f3;
    box-shadow: none;
    transform: none;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--primary);
    color: #fff;
    text-align: center;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #3d5aee;
}

/* 聊天消息：扁平、无卡片阴影 */
[data-testid="stChatMessage"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0.35rem 0;
    margin: 0.15rem 0;
}

/* 用户消息：右侧浅蓝气泡（DeepSeek 同款），隐藏头像 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    background: var(--user-bubble);
    border-radius: 14px 14px 4px 14px;
    padding: 0.6rem 1rem;
    max-width: 85%;
}

/* AI 消息：纯文本流，左侧小头像，无气泡 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {
    background: transparent;
    padding: 0.2rem 0 0.2rem 0.2rem;
    width: 100%;
}

/* 聊天输入框：大圆角胶囊、细边框 */
[data-testid="stChatInput"] > div {
    border: 1px solid var(--border) !important;
    border-radius: 24px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    background: #fff;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 1px 8px rgba(77,107,254,0.15) !important;
}

/* 欢迎页：居中大留白，无卡片边框 */
.welcome-card {
    text-align: center;
    padding: 5rem 1rem 2rem 1rem;
}
.welcome-card .welcome-logo { font-size: 3rem; }
.welcome-card h2 {
    color: var(--text);
    font-weight: 600;
    margin: 1rem 0 0.4rem 0;
    font-size: 1.5rem;
}
.welcome-card p { color: var(--muted); font-size: 0.92rem; margin: 0.2rem 0; }
.welcome-card .quick-actions {
    display: flex; gap: 0.5rem; justify-content: center;
    flex-wrap: wrap; margin-top: 1.6rem;
}
.quick-chip {
    background: #fff;
    border: 1px solid var(--border);
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.82rem;
    color: var(--muted);
}

/* 参考来源折叠条：极简细边框 */
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: none;
}

/* 状态提示条 */
.stAlert { border-radius: 10px; border: none; }

/* 分割线 */
hr { border-color: var(--border); margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ── 极简标题条 ──
st.markdown("""
<div class="top-bar">
    <span class="top-title">🚗 汽车导购助手</span>
    <span class="top-sub">新能源车型推荐 · 对比 · 成本计算</span>
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


def check_access() -> None:
    """公开访问门禁 — 访问密码 + 单会话提问软上限。

    在 .streamlit/secrets.toml 配置：
      ACCESS_PASSWORD = "xxx"            # 未配置则放行（本地开发模式）
      MAX_QUESTIONS_PER_SESSION = 50     # 0 表示不限
    防止公开分享后陌生人刷爆 DeepSeek API 额度。
    """
    required = st.secrets.get("ACCESS_PASSWORD", "")
    if required and not st.session_state.get("access_granted"):
        st.markdown("## 🔒 演示站访问验证")
        pwd = st.text_input("请输入访问密码（向分享者索取）", type="password")
        if not pwd:
            st.stop()
        if pwd != required:
            st.error("密码不正确，请重试")
            st.stop()
        st.session_state.access_granted = True
        # 立即重跑：否则门禁组件（密码框）会在本次运行中残留显示在内容上方，
        # 要等下一次交互才消失，访客会以为密码没通过
        st.rerun()

    max_q = int(st.secrets.get("MAX_QUESTIONS_PER_SESSION", 0) or 0)
    if max_q > 0:
        asked = int(st.session_state.get("question_count", 0))
        remaining = max_q - asked
        if remaining <= 0:
            st.warning(f"😊 本次体验已用完 {max_q} 次提问额度，感谢试用！如需继续请联系分享者。")
            st.stop()
        st.caption(f"💬 本次体验剩余提问次数：{remaining}")


def main() -> None:
    init_session_state()
    check_access()
    client = get_api_client()

    render_sidebar(client)

    msgs = _current_msgs()

    # ── 欢迎页（无消息时）──
    if not msgs:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-logo">🚗</div>
            <h2>想看什么车？</h2>
            <p>我了解 16 款主流新能源车型的参数、价格与车主评价</p>
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
        st.session_state.question_count = int(st.session_state.get("question_count", 0)) + 1

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
