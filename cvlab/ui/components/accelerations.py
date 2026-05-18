"""加速选项面板 - 可复用 Streamlit 组件。"""

from __future__ import annotations

import streamlit as st

from cvlab.detect.probe import EnvironmentProbe, AccelerationPanel


def show_acceleration_panel(panel: AccelerationPanel) -> None:
    """显示加速选项面板。"""
    st.subheader("训练加速配置")
    for opt in panel.options:
        icon = "✅" if opt.supported else "❌"
        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric(opt.name, "✓" if opt.enabled else " ")
        with col2:
            st.caption(f"{icon} {opt.benefit}")
            if not opt.supported:
                st.caption(f"  需要: {opt.condition}")
            if opt.risk:
                st.caption(f"  ⚠️ {opt.risk}")


def show_dataloader_recommendations(panel: AccelerationPanel) -> None:
    """显示 DataLoader 配置推荐。"""
    st.subheader("DataLoader 配置推荐")
    col1, col2, col3 = st.columns(3)
    col1.info(f"num_workers = {panel.recommended_num_workers}")
    col2.info(f"prefetch_factor = {panel.recommended_prefetch}")
    col3.info(f"pin_memory = {'✓' if panel.recommended_pin_memory else '✗'}")
