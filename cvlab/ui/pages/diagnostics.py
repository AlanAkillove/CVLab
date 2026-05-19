"""Environment diagnostics page — Swiss Design, i18n-ready."""

from __future__ import annotations

import streamlit as st

from cvlab.detect.probe import EnvironmentProbe
from cvlab.i18n import _
from cvlab.ui.components.layout import (
    divider,
    inject_language_switcher,
    metric_card,
    section_header,
    sidebar_footer,
)


def show_diagnostics():
    inject_language_switcher()
    st.title(_("Diagnostics"))

    probe = EnvironmentProbe()

    if st.button(_("确认"), type="primary"):
        with st.spinner(_("正在诊断") + "..."):
            report = probe.probe()
            panel = probe.get_acceleration_panel(report)

        st.markdown(
            f'<div style="margin:1rem 0;color:var(--success);font-size:0.85rem;">'
            f'{_("诊断完成")}</div>',
            unsafe_allow_html=True,
        )

        # ── System ───────────────────────────────────────
        section_header(_("系统"))

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("OS", f"{report.os_type} {report.os_version}")
        with col2:
            metric_card("Python", report.python_version.split()[0] if report.python_version else "—")
        with col3:
            metric_card("PyTorch", report.torch_version)

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("CPU", f"{report.cpu_cores}C / {report.cpu_threads}T")
        with col2:
            metric_card(_("内存"), f"{report.total_ram_gb:.1f} GB")
        with col3:
            storage_label = report.storage_type.upper() if report.storage_type not in ("unknown", "") else "—"
            metric_card(_("存储类型"), storage_label)

        if report.is_wsl:
            st.markdown(
                f'<div class="metric-card" style="border-left-color:var(--warning);">'
                f'<div class="label">{_("警告")}</div>'
                f'<div class="value" style="font-size:0.85rem;">WSL2 detected — GPU via virtualization, ~10-15% overhead</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        divider()

        # ── GPU ──────────────────────────────────────────
        if report.gpus:
            section_header("GPU", badge=str(report.num_gpus))
            cols = st.columns(min(report.num_gpus, 4))
            for i, gpu in enumerate(report.gpus):
                with cols[i % len(cols)]:
                    cc = f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}" if gpu.compute_capability else "—"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="label">GPU {gpu.index}: {gpu.name}</div>'
                        f'<div class="value">{gpu.total_memory_gb:.1f} GB</div>'
                        f'<div class="sublabel">CC {cc} · TensorCore: {"✓" if gpu.supports_tensor_core else "✗"}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info(_("GPU 无（CPU 模式）"))

        divider()

        # ── Acceleration ─────────────────────────────────
        section_header(_("训练加速配置"))

        for opt in panel.options:
            icon = "✓" if opt.supported else "✗"
            # Build sublabel HTML outside f-string for Python 3.10 compat
            sublabel_parts = []
            if not opt.supported and opt.condition:
                sublabel_parts.append(f'<div class="sublabel">Requires: {opt.condition}</div>')
            if opt.risk:
                sublabel_parts.append(f'<div class="sublabel">\u26a0 {opt.risk}</div>')
            sublabel_html = "".join(sublabel_parts)
            border_color = "var(--success)" if opt.supported else "var(--border)"
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{border_color};">'
                f'<div class="label">{icon} {opt.name}</div>'
                f'<div class="value" style="font-size:0.85rem;">{opt.benefit}</div>'
                f'{sublabel_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        divider()

        # ── DataLoader ───────────────────────────────────
        section_header(_("DataLoader 配置"))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">num_workers</div>'
                f'<div class="value">{panel.recommended_num_workers}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">prefetch_factor</div>'
                f'<div class="value">{panel.recommended_prefetch}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col3:
            pin = "✓" if panel.recommended_pin_memory else "✗"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">pin_memory</div>'
                f'<div class="value">{pin}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            f'<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            f'font-size:0.9rem;">{_("点击")} "Run diagnostics" {_("探测环境")}</div>',
            unsafe_allow_html=True,
        )

    sidebar_footer()
