"""Sweep 管理页 — Swiss Design。"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cvlab.db.database import Database
from cvlab.ui.components.layout import (
    section_header,
    metric_card,
    status_badge,
    divider,
)


def show_sweep():
    st.title("Sweeps")

    db = Database()
    sweeps = db.list_sweeps()

    if not sweeps:
        st.markdown(
            '<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            'font-size:0.9rem;">No sweeps yet. Create one with <code>cvlab sweep</code>.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("What is a Sweep?"):
            st.markdown("""
            Sweep is hyperparameter search. Two strategies:

            - **Grid search**: Cartesian product of all parameter combinations
            - **Random search**: Random sampling from parameter space

            ```bash
            cvlab sweep --config config.yaml --params params.yaml
            ```
            """)
        return

    # ── Sweep list ───────────────────────────────────────
    section_header("All Sweeps", badge=str(len(sweeps)))

    sweep_data = []
    for s in sweeps:
        sweep_data.append({
            "ID": s["id"],
            "Strategy": s["strategy"],
            "Status": status_badge(s["status"]),
            "Created": s["created_at"][:19] if s.get("created_at") else "",
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
    selected_id = st.selectbox("Select sweep", sweep_ids, format_func=lambda x: x)

    if not selected_id:
        return

    sweep_info = db.get_sweep(selected_id)
    if not sweep_info:
        st.error("Sweep not found")
        return

    # ── Sweep info ───────────────────────────────────────
    section_header(selected_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Strategy", sweep_info["strategy"])
    with col2:
        metric_card("Status", sweep_info["status"])
    with col3:
        metric_card("Created", sweep_info["created_at"][:19])

    with st.expander("Sweep Configuration"):
        try:
            st.json(json.loads(sweep_info["config_json"]))
        except (json.JSONDecodeError, TypeError):
            st.text(sweep_info.get("config_json", ""))

    divider()

    # ── Trials ───────────────────────────────────────────
    trials = db.get_sweep_trials(selected_id)

    if not trials:
        st.info("No trials recorded")
        return

    section_header("Trials", badge=str(len(trials)))

    trial_rows = []
    for t in trials:
        exp = db.get_experiment(t["experiment_id"])
        trial_status = exp["status"] if exp else t["status"]
        trial_rows.append({
            "Trial": t["trial_index"],
            "Experiment": t["experiment_id"],
            "Status": status_badge(trial_status),
        })

    if trial_rows:
        df_trials = pd.DataFrame(trial_rows)
        st.markdown(
            df_trials.to_html(index=False, escape=False),
            unsafe_allow_html=True,
        )

    divider()

    # ── Metric comparison ────────────────────────────────
    section_header("Trial Metrics")

    completed_trials = [
        t for t in trials
        if db.get_experiment(t["experiment_id"]) is not None
    ]

    if not completed_trials:
        st.info("No completed trials yet")
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
