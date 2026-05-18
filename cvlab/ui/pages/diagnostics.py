"""环境诊断页 — Swiss Design。"""

from __future__ import annotations

import streamlit as st

from cvlab.detect.probe import EnvironmentProbe
from cvlab.ui.components.layout import section_header, metric_card, divider


def show_diagnostics():
    st.title("Diagnostics")

    probe = EnvironmentProbe()

    if st.button("Run diagnostics", type="primary"):
        with st.spinner("Probing environment..."):
            report = probe.probe()
            panel = probe.get_acceleration_panel(report)

        st.markdown(
            '<div style="margin:1rem 0;color:var(--success);font-size:0.85rem;">'
            'Diagnostics complete</div>',
            unsafe_allow_html=True,
        )

        # ── System ───────────────────────────────────────
        section_header("System")

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
            metric_card("Memory", f"{report.total_ram_gb:.1f} GB")
        with col3:
            storage_label = report.storage_type.upper() if report.storage_type not in ("unknown", "") else "—"
            metric_card("Storage", storage_label)

        if report.is_wsl:
            st.markdown(
                f'<div class="metric-card" style="border-left-color:var(--warning);">'
                f'<div class="label">Note</div>'
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
            st.info("No GPU detected — will use CPU mode")

        divider()

        # ── Acceleration ─────────────────────────────────
        section_header("Acceleration Options")

        for opt in panel.options:
            icon = "✓" if opt.supported else "✗"
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{"var(--success)" if opt.supported else "var(--border)"};">'
                f'<div class="label">{icon} {opt.name}</div>'
                f'<div class="value" style="font-size:0.85rem;">{opt.benefit}</div>'
                f'{"<div class=\"sublabel\">Requires: " + opt.condition + "</div>" if not opt.supported and hasattr(opt, "condition") and opt.condition else ""}'
                f'{"<div class=\"sublabel\">⚠ " + opt.risk + "</div>" if hasattr(opt, "risk") and opt.risk else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

        divider()

        # ── DataLoader ───────────────────────────────────
        section_header("DataLoader Recommendations")

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
            '<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            'font-size:0.9rem;">Click "Run diagnostics" to probe your environment.</div>',
            unsafe_allow_html=True,
        )
