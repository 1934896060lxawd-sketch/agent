"""侧边栏 — 会话列表管理。每个会话消息独立存储，切换不丢失。"""

import json as _json
import logging

import streamlit as st

from frontend.api_client import run_async

logger = logging.getLogger(__name__)


def _current_msgs() -> list[dict]:
    sid = st.session_state.get("current_sid", "default")
    cache = st.session_state.get("messages_by_sid", {})
    if sid not in cache:
        cache[sid] = []
    return cache[sid]


def _set_current_msgs(messages: list[dict]) -> None:
    sid = st.session_state.get("current_sid", "default")
    st.session_state.messages_by_sid[sid] = messages


def render_sidebar(client) -> None:
    with st.sidebar:
        # ── 品牌区 ──
        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0 1rem 0;">
            <div style="font-size:2.2rem;">🚗</div>
            <div style="font-weight:700; font-size:1.1rem; color:#1e293b;">AI 导购助手</div>
            <div style="font-size:0.75rem; color:#94a3b8;">Powered by DeepSeek</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 新建会话 ──
        st.markdown("##### ➕ 新建会话")
        new_title = st.text_input(
            "名称", value="", placeholder="输入名称，如「SUV选购」",
            key="input_new_title", label_visibility="collapsed",
        )
        if st.button("创建会话", use_container_width=True, key="btn_create",
                      type="primary"):
            title = new_title.strip() if new_title.strip() else "新对话"
            try:
                result = run_async(client.create_session(title))
                new_id = result.get("session_id", "")
                if new_id:
                    _switch_to(client, new_id)
                    st.rerun()
            except Exception as e:
                st.error(f"创建失败: {e}")

        st.divider()

        # ── 会话列表 ──
        st.markdown("##### 💬 我的会话")
        try:
            sessions = run_async(client.list_sessions())
        except Exception:
            st.caption("⚠️ 加载失败")
            sessions = []

        if not sessions:
            st.caption("暂无会话，输入名称后点击创建")
        else:
            current_id = st.session_state.get("current_sid", "default")
            for i, sess in enumerate(sessions):
                sid = sess.get("session_id", "")
                title = sess.get("title", "未命名")
                msg_count = sess.get("message_count", 0)
                is_active = sid == current_id

                is_renaming = st.session_state.get("renaming_sid") == sid
                is_confirming = st.session_state.get("deleting_sid") == sid

                if is_renaming:
                    _render_rename_row(client, sid, title)
                elif is_confirming:
                    _render_delete_confirm(client, sid, title)
                else:
                    _render_session_row(client, sid, title, msg_count, is_active, i)

        st.divider()
        msgs = _current_msgs()
        st.caption(
            f"共 {len(sessions)} 个会话 · 当前 {len(msgs)} 条消息"
        )


def _switch_to(client, sid: str) -> None:
    old_sid = st.session_state.get("current_sid", "default")
    current_msgs = st.session_state.messages_by_sid.get(old_sid, [])
    st.session_state.messages_by_sid[old_sid] = current_msgs

    cache = st.session_state.messages_by_sid
    if sid in cache:
        st.session_state.current_sid = sid
        return

    try:
        hist = run_async(client.get_history(sid))
        messages = []
        for msg in hist.get("messages", []):
            try:
                data = _json.loads(msg) if isinstance(msg, str) else msg
                role = data.get("role", "")
                content = data.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            except Exception:
                continue
        cache[sid] = messages
    except Exception as e:
        logger.warning(f"加载历史失败 ({sid[:8]}): {e}")
        cache[sid] = []

    st.session_state.current_sid = sid


def _render_session_row(client, sid: str, title: str, msg_count: int,
                         is_active: bool, idx: int) -> None:
    c1, c2, c3 = st.columns([7, 1.5, 1.5])

    with c1:
        indicator = "●" if is_active else "○"
        suffix = f" · {msg_count}" if msg_count else ""
        btn_label = f"{indicator} {title}{suffix}"
        if st.button(
            btn_label,
            key=f"btn_sess_{sid}_{idx}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            _switch_to(client, sid)
            st.rerun()

    with c2:
        if st.button("✏️", key=f"btn_ren_{sid}_{idx}",
                     use_container_width=True, help="重命名"):
            st.session_state["renaming_sid"] = sid

    with c3:
        if st.button("🗑️", key=f"btn_del_{sid}_{idx}",
                     use_container_width=True, help="删除"):
            st.session_state["deleting_sid"] = sid


def _render_rename_row(client, sid: str, old_title: str) -> None:
    new_name = st.text_input(
        "新名称", value=old_title, key=f"input_ren_{sid}",
        label_visibility="collapsed",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✓ 确认", key=f"ok_ren_{sid}", use_container_width=True):
            try:
                run_async(client.rename_session(sid, new_name.strip() or old_title))
            except Exception as e:
                st.error(f"失败: {e}")
            st.session_state.pop("renaming_sid", None)
            st.rerun()
    with c2:
        if st.button("✗ 取消", key=f"cancel_ren_{sid}", use_container_width=True):
            st.session_state.pop("renaming_sid", None)
            st.rerun()


def _render_delete_confirm(client, sid: str, title: str) -> None:
    st.warning(f"删除「{title}」？此操作不可恢复")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✓ 确认删除", key=f"ok_del_{sid}", use_container_width=True,
                     type="primary"):
            try:
                success = run_async(client.delete_session(sid))
                if success:
                    st.session_state.messages_by_sid.pop(sid, None)
                    if sid == st.session_state.get("current_sid"):
                        st.session_state.current_sid = "default"
            except Exception as e:
                st.error(f"删除失败: {e}")
            st.session_state.pop("deleting_sid", None)
            st.rerun()
    with c2:
        if st.button("✗ 取消", key=f"cancel_del_{sid}", use_container_width=True):
            st.session_state.pop("deleting_sid", None)
            st.rerun()
