"""Sweep management page — Swiss Design, i18n-ready."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cvlab.db.database import Database
from cvlab.i18n import _
from cvlab.ui.components.layout import (
    divider,
    inject_language_switcher,
    metric_card,
    section_header,
    sidebar_footer,
    status_badge,
)


def show_sweep():
    inject_language_switcher()
    st.title(_("Sweeps"))

    db = Database()
    sweeps = db.list_sweeps()

    if not sweeps:
        st.markdown(
            f'<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            f'font-size:0.9rem;">{_("暂无实验")}. '
            f'{_("启动训练")}: <code>cvlab sweep</code></div>',
            unsafe_allow_html=True,
        )
        with st.expander(_("超参扫描")):
            st.markdown(f"""
            {_("超参扫描")} — {_("超参扫描")}. {_("两种策略")}:

            - **{_("Grid 搜索")}**: Cartesian product of all parameter combinations
            - **{_("Random 搜索")}**: Random sampling from parameter space

            ```bash
            cvlab sweep --config config.yaml --params params.yaml
            ```
            """)
        sidebar_footer()
        return

    # ── Sweep list ───────────────────────────────────────
    section_header("All Sweeps", badge=str(len(sweeps)))

    sweep_data = []
    for s in sweeps:
        sweep_data.append({
            _("ID"): s["id"],
            _("策略"): s["strategy"],
            _("状态"): status_badge(s["status"]),
            _("创建时间"): s["created_at"][:19] if s.get("created_at") else "",
        })

    if sweep_data:
        df = pd.DataFrame(sweep_data)
        st.markdown(
            df.to_html(index=False, escape=False),
            unsafe_allow_html=True,
        )

    divider()

    # ── Sweep selector ───────────────────────────────────
    sweep_ids = [s["id"] for s in sweeps]
    selected_id = st.selectbox(
        _("选择实验"),
        sweep_ids,
        format_func=lambda x: x,
        key="sweep_selector",
    )

    if not selected_id:
        sidebar_footer()
        return

    sweep_info = db.get_sweep(selected_id)
    if not sweep_info:
        st.error(_("实验不存在"))
        sidebar_footer()
        return

    # ── Sweep info ───────────────────────────────────────
    section_header(selected_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(_("策略"), sweep_info["strategy"])
    with col2:
        metric_card(_("状态"), sweep_info["status"])
    with col3:
        metric_card(_("创建时间"), sweep_info["created_at"][:19])

    with st.expander(_("超参配置")):
        try:
            st.json(json.loads(sweep_info["config_json"]))
        except (json.JSONDecodeError, TypeError):
            st.text(sweep_info.get("config_json", ""))

    divider()

    # ── Trials ───────────────────────────────────────────
    trials = db.get_sweep_trials(selected_id)

    if not trials:
        st.info(_("暂无实验"))
        sidebar_footer()
        return

    section_header("Trials", badge=str(len(trials)))

    trial_rows = []
    for t in trials:
        exp = db.get_experiment(t["experiment_id"])
        trial_status = exp["status"] if exp else t["status"]
        trial_rows.append({
            _("ID"): t["trial_index"],
            "Experiment": t["experiment_id"],
            _("状态"): status_badge(trial_status),
        })

    if trial_rows:
        df_trials = pd.DataFrame(trial_rows)
        st.markdown(
            df_trials.to_html(index=False, escape=False),
            unsafe_allow_html=True,
        )

    divider()

    # ── Metric comparison ────────────────────────────────
    section_header(_("训练指标"))

    completed_trials = [
        t for t in trials
        if db.get_experiment(t["experiment_id"]) is not None
    ]

    if not completed_trials:
        st.info(_("暂无实验"))
        sidebar_footer()
        return

    metric_keys: set[str] = set()
    exp_metrics: dict[str, dict[str, float]] = {}
    for t in completed_trials:
        eid = t["experiment_id"]
        metrics = db.get_metrics(eid)
        if metrics:
            latest: dict[str, float] = {}
            for m in metrics:
                latest[m["key"]] = m["value"]
                metric_keys.add(m["key"])
            exp_metrics[eid] = latest

    if metric_keys:
        from cvlab.ui.components.charts import plot_trial_comparison

        for mk in sorted(metric_keys):
            trial_values = []
            for t in completed_trials:
                eid = t["experiment_id"]
                if eid in exp_metrics and mk in exp_metrics[eid]:
                    trial_values.append({
                        "Trial": t["trial_index"],
                        "Exp": eid[:12],
                        mk: exp_metrics[eid][mk],
                    })
            if trial_values:
                plot_trial_comparison(trial_values, mk)

    sidebar_footer()
