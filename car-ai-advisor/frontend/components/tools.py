"""工具调用可视化 — 展示 Agent ReAct 推理过程。"""

import streamlit as st

# 工具图标映射
TOOL_ICONS: dict[str, str] = {
    "search_car_knowledge": "🔍",
    "get_car_price": "💰",
    "compare_cars": "⚖️",
    "recommend_cars": "🎯",
    "calculate_ownership_cost": "🧮",
}

# 工具中文名映射
TOOL_NAMES_CN: dict[str, str] = {
    "search_car_knowledge": "搜索知识库",
    "get_car_price": "查询价格",
    "compare_cars": "对比车型",
    "recommend_cars": "智能推荐",
    "calculate_ownership_cost": "计算用车成本",
}


def render_tool_calls(events: list[dict]) -> None:
    """渲染 Agent 工具调用的折叠面板。

    每个事件:
        {"tool": "search_car_knowledge", "args": "{...}", "result": "..."}
    """
    if not events:
        return

    total = len(events)
    with st.expander(f"🔧 推理过程（共 {total} 步）", expanded=True):
        for i, evt in enumerate(events, 1):
            tool_name = evt.get("tool", "未知工具")
            args_str = evt.get("args", "")
            result_str = evt.get("result", "")

            icon = TOOL_ICONS.get(tool_name, "🔌")
            name_cn = TOOL_NAMES_CN.get(tool_name, tool_name)

            # 步骤标题
            st.markdown(f"**第 {i} 步**: {icon} {name_cn}")

            # 参数
            if args_str:
                with st.expander("📥 调用参数", expanded=False):
                    st.code(
                        args_str[:500] if len(args_str) > 500 else args_str,
                        language="json",
                    )

            # 结果
            if result_str:
                with st.expander("📤 返回结果", expanded=False):
                    st.caption(
                        result_str[:600] if len(result_str) > 600 else result_str
                    )

            if i < total:
                st.divider()


def render_tool_status(status_msg: str) -> None:
    """渲染 Agent 状态提示。"""
    st.info(f"⏳ {status_msg}")
