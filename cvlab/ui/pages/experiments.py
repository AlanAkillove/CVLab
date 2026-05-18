"""实验列表页 — Swiss Design。"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cvlab.db.database import Database
from cvlab.ui.components.layout import (
    section_header,
    metric_row,
    status_badge,
    divider,
)


def show_experiments():
    st.title("Experiments")

    db = Database()

    # ── Filter bar ───────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["all", "created", "running", "completed", "failed", "archived"],
            key="exp_status_filter",
        )
    with col2:
        tag_filter = st.text_input("Tag", placeholder="filter by tag...", key="exp_tag_filter")

    kwargs: dict = {"limit": 100}
    if status_filter != "all":
        kwargs["status"] = status_filter
    if tag_filter:
        kwargs["tag"] = tag_filter

    exps = db.list_experiments(**kwargs)

    if not exps:
        st.markdown(
            '<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            'font-size:0.9rem;">No experiments yet. Start one with <code>cvlab train</code></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Summary row ──────────────────────────────────────
    total = len(exps)
    running = sum(1 for e in exps if e["status"] == "running")
    completed = sum(1 for e in exps if e["status"] == "completed")
    failed = sum(1 for e in exps if e["status"] == "failed")

    metric_row([
        ("Total", str(total), None),
        ("Running", str(running), None),
        ("Completed", str(completed), None),
        ("Failed", str(failed), None),
    ])

    # ── Experiment table ─────────────────────────────────
    section_header("All Experiments", badge=str(total))

    rows = []
    for e in exps:
        config = e.get("config_json", "{}")
        try:
            cfg = json.loads(config) if isinstance(config, str) else config
        except (json.JSONDecodeError, TypeError):
            cfg = {}
        model_name = cfg.get("model", {}).get("name", "—") if isinstance(cfg, dict) else "—"

        rows.append({
            "ID": e["id"],
            "Name": e["name"],
            "Model": model_name,
            "Status": status_badge(e["status"]),
            "Created": e["created_at"][:19] if e.get("created_at") else "",
        })

    df = pd.DataFrame(rows)

    st.markdown(
        df.to_html(
            columns=["ID", "Name", "Model", "Status", "Created"],
            index=False,
            escape=False,
            classes="data-card",
        ),
        unsafe_allow_html=True,
    )

    divider()

    # ── Experiment selector ──────────────────────────────
    exp_ids = [e["id"] for e in exps]
    selected_id = st.selectbox(
        "View experiment",
        exp_ids,
        format_func=lambda x: f"{x} — {db.get_experiment(x)['name']}",
        key="exp_selector",
    )

    if not selected_id:
        return

    exp = db.get_experiment(selected_id)
    if not exp:
        st.error("Experiment not found")
        return

    # ── Quick detail panel ───────────────────────────────
    section_header(selected_id, badge=exp["status"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Status</div>'
            f'<div class="value">{status_badge(exp["status"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Seed</div>'
            f'<div class="value">{exp.get("seed", "—")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Created</div>'
            f'<div class="value" style="font-size:0.95rem;">{exp["created_at"][:19]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if exp.get("failure_reason"):
        st.markdown(
            f'<div class="metric-card" style="border-left-color:var(--accent);'
            f'background:var(--accent-subtle);">'
            f'<div class="label">Failure Reason</div>'
            f'<div class="value" style="font-size:0.9rem;">{exp["failure_reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Configuration"):
        try:
            cfg = json.loads(exp["config_json"]) if isinstance(exp["config_json"], str) else exp["config_json"]
            st.json(cfg)
        except (json.JSONDecodeError, TypeError):
            st.code(exp.get("config_json", ""), language="json")

    with st.expander("Environment"):
        try:
            env = json.loads(exp.get("env_json", "{}"))
            st.json(env)
        except (json.JSONDecodeError, TypeError):
            st.code(exp.get("env_json", ""), language="json")

    # ── Metrics ─────────────────────────────────────────
    metrics = db.get_metrics(selected_id)
    if metrics:
        section_header("Training Metrics")
        df_metrics = db.get_metrics_dataframe(selected_id)
        if df_metrics is not None and not df_metrics.empty:
            from cvlab.ui.components.charts import plot_single_metric
            for col in df_metrics.columns:
                plot_single_metric(df_metrics, col)
