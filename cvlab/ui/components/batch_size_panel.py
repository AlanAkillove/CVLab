"""Batch Size 探测展示组件。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cvlab.core.types import ProbeResult


def show_probe_result(result: ProbeResult) -> None:
    """显示 Batch Size 探测结果。

    Args:
        result: Batch Size 探测结果。
    """
    st.success(f"推荐 Batch Size: **{result.recommended_batch_size}**")

    col1, col2 = st.columns(2)
    col1.metric("探测模式", "AMP" if result.with_amp else "FP32")
    col2.metric("峰值显存", f"{result.peak_memory_gb:.2f} GB")

    # 候选值表格
    candidates = []
    for c in result.candidates:
        candidates.append({
            "Batch Size": c.batch_size,
            "显存 (GB)": f"{c.memory_gb:.2f}" if c.memory_gb else "N/A",
            "成功": "✓" if c.success else "✗",
        })

    st.dataframe(
        pd.DataFrame(candidates),
        width='stretch',
        hide_index=True,
    )
