"""实验详情页 — Swiss Design。"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cvlab.db.database import Database
from cvlab.ui.components.layout import (
    section_header,
    metric_card,
    status_badge,
    tag_chip,
    divider,
)


def show_experiment_detail():
    st.title("Experiment Detail")

    db = Database()
    exps = db.list_experiments(limit=50)

    if not exps:
        st.markdown(
            '<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            'font-size:0.9rem;">No experiments found.</div>',
            unsafe_allow_html=True,
        )
        return

    exp_ids = [e["id"] for e in exps]
    selected_id = st.selectbox(
        "Select experiment",
        exp_ids,
        format_func=lambda x: f"{x} — {db.get_experiment(x)['name']}",
        key="detail_selector",
    )

    if not selected_id:
        return

    exp = db.get_experiment(selected_id)
    if not exp:
        st.error("Experiment not found")
        return

    # ── Header ───────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;">'
        f'<span style="font-size:1.5rem;font-weight:400;">{selected_id}</span>'
        f'{status_badge(exp["status"])}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Info grid ────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Status", exp["status"])
    with col2:
        metric_card("Seed", str(exp.get("seed", "—")))
    with col3:
        metric_card("Created", exp["created_at"][:19])
    with col4:
        metric_card("Updated", exp["updated_at"][:19])

    if exp.get("failure_reason"):
        st.markdown(
            f'<div class="metric-card" style="border-left-color:var(--accent);'
            f'background:var(--accent-subtle);">'
            f'<div class="label">Failure Reason</div>'
            f'<div class="value" style="font-size:0.9rem;">{exp["failure_reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Tags ─────────────────────────────────────────────
    tags = db.get_tags(selected_id)
    if tags:
        tags_html = " ".join(tag_chip(t) for t in tags)
        st.markdown(f'<div style="margin:0.75rem 0;">{tags_html}</div>', unsafe_allow_html=True)

    divider()

    # ── Configuration ────────────────────────────────────
    section_header("Configuration")
    with st.expander("View config"):
        try:
            cfg = json.loads(exp["config_json"]) if isinstance(exp["config_json"], str) else exp["config_json"]
            st.json(cfg)
        except (json.JSONDecodeError, TypeError):
            st.code(exp.get("config_json", ""), language="json")

    # ── Environment ──────────────────────────────────────
    section_header("Environment")
    with st.expander("View environment"):
        try:
            env = json.loads(exp.get("env_json", "{}"))
            st.json(env)
        except (json.JSONDecodeError, TypeError):
            st.code(exp.get("env_json", ""), language="json")

    divider()

    # ── Datasets ──────────────────────────────────────────
    ds_path = exp.get("dataset_path")
    ds_files = exp.get("dataset_files")
    datasets_link = db.get_experiment_datasets(selected_id)

    if ds_path or datasets_link:
        section_header("Dataset")

    if ds_path:
        col1, col2 = st.columns(2)
        with col1:
            metric_card("Path", ds_path)
        with col2:
            metric_card("Files", str(ds_files) if ds_files is not None else "—")

    if datasets_link:
        for d in datasets_link:
            size_mb = d.get("total_size_bytes", 0) / (1024 * 1024)
            st.markdown(
                f'<div class="data-card">'
                f'<div class="card-title">{d.get("dataset_name", "Dataset")} '
                f'<span style="font-weight:400;">{d["version"]}</span></div>'
                f'<div style="font-size:0.85rem;color:var(--text-secondary);">'
                f'{d.get("dataset_path", "")}</div>'
                f'<div style="display:flex;gap:2rem;margin-top:0.5rem;font-size:0.8rem;">'
                f'<span>{d.get("total_files", 0)} files</span>'
                f'<span>{size_mb:.1f} MB</span>'
                f'<span style="font-family:var(--font-mono);font-size:0.75rem;'
                f'color:var(--text-tertiary);">hash: {d.get("root_hash", "")[:12]}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if ds_path or datasets_link:
        divider()

    # ── Training Metrics ─────────────────────────────────
    metrics = db.get_metrics(selected_id)
    if metrics:
        section_header("Training Metrics")
        df = db.get_metrics_dataframe(selected_id)
        if df is not None and not df.empty:
            from cvlab.ui.components.charts import plot_single_metric
            cols = st.columns(2)
            for i, col_name in enumerate(df.columns):
                with cols[i % 2]:
                    plot_single_metric(df, col_name)

    divider()

    # ── Checkpoints ──────────────────────────────────────
    ckpts = db.get_checkpoints(selected_id)
    if ckpts:
        section_header("Checkpoints", badge=str(len(ckpts)))
        ckpt_df = pd.DataFrame(ckpts)
        # Drop columns that are all NaN or uninteresting for display
        display_cols = [c for c in ["epoch", "metric_name", "metric_value", "is_best", "file_size"]
                        if c in ckpt_df.columns]
        if display_cols:
            st.dataframe(ckpt_df[display_cols], width='stretch', hide_index=True,
                         use_container_width=True)
        else:
            st.dataframe(ckpt_df, width='stretch', hide_index=True, use_container_width=True)

    divider()

    # ── Artifacts ────────────────────────────────────────
    artifacts = db.get_artifacts(selected_id)
    if artifacts:
        section_header("Artifacts", badge=str(len(artifacts)))
        for art in artifacts:
            label = f"{art['key']} — step {art['step']}"
            with st.expander(label):
                if art.get("file_path"):
                    try:
                        st.image(art["file_path"], width='stretch', use_container_width=True)
                    except Exception:
                        st.caption("(image unavailable)")

    # ── Reproduction ─────────────────────────────────────
    exp_command = exp.get("command") or ""
    if exp_command:
        divider()
        section_header("Reproduction")
        st.code(exp_command, language="bash")
