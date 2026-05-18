"""实验对比页 — Swiss Design。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from cvlab.db.database import Database
from cvlab.ui.components.layout import section_header, metric_card, divider


def show_compare():
    st.title("Compare")

    db = Database()
    exps = db.list_experiments(limit=100)

    if not exps or len(exps) < 2:
        st.markdown(
            '<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            'font-size:0.9rem;">Need at least 2 experiments to compare.</div>',
            unsafe_allow_html=True,
        )
        return

    exp_ids = [e["id"] for e in exps]
    selected = st.multiselect(
        "Select experiments (2–4)",
        exp_ids,
        default=exp_ids[: min(2, len(exp_ids))],
        format_func=lambda x: f"{x} — {db.get_experiment(x)['name']}",
        key="compare_selector",
    )

    if len(selected) < 2:
        st.info("Select at least 2 experiments")
        return

    if len(selected) > 4:
        st.warning("Max 4 experiments")
        selected = selected[:4]

    # ── Overview table ──────────────────────────────────
    section_header("Overview")
    info_rows = []
    for eid in selected:
        exp = db.get_experiment(eid)
        if exp:
            info_rows.append({
                "ID": eid,
                "Name": exp["name"],
                "Status": exp["status"],
                "Seed": exp.get("seed", "—"),
                "Created": exp["created_at"][:19] if exp.get("created_at") else "",
            })

    if info_rows:
        st.dataframe(pd.DataFrame(info_rows), width='stretch', hide_index=True,
                     use_container_width=True)

    divider()

    # ── Config comparison ────────────────────────────────
    section_header("Configuration")

    configs: list[dict[str, Any]] = []
    config_keys: set[str] = set()
    for eid in selected:
        exp = db.get_experiment(eid)
        if not exp or not exp.get("config_json"):
            continue
        try:
            cfg = json.loads(exp["config_json"]) if isinstance(exp["config_json"], str) else exp["config_json"]
        except (json.JSONDecodeError, TypeError):
            continue
        flat = _flatten_dict(cfg)
        configs.append(flat)
        config_keys.update(flat.keys())

    if configs:
        config_rows = []
        for key in sorted(config_keys):
            row: dict[str, str] = {"Parameter": key}
            for i, eid in enumerate(selected):
                row[eid[:16]] = str(configs[i].get(key, "—")) if i < len(configs) else "—"
            config_rows.append(row)
        st.dataframe(pd.DataFrame(config_rows), width='stretch', hide_index=True,
                     use_container_width=True)

    divider()

    # ── Metric overlay ──────────────────────────────────
    section_header("Metrics")

    metrics_by_exp: dict[str, pd.DataFrame] = {}
    all_metric_keys: set[str] = set()
    for eid in selected:
        df = db.get_metrics_dataframe(eid)
        if df is not None and not df.empty:
            metrics_by_exp[eid] = df
            all_metric_keys.update(df.columns.tolist())

    if not all_metric_keys:
        st.info("No metric data for selected experiments")
        return

    from cvlab.ui.components.charts import plot_metric_overlay

    for metric_key in sorted(all_metric_keys):
        plot_metric_overlay(metrics_by_exp, metric_key)

    divider()

    # ── Summary table ────────────────────────────────────
    section_header("Summary")

    summary_rows = []
    for eid in selected:
        if eid not in metrics_by_exp:
            continue
        df = metrics_by_exp[eid]
        row: dict[str, str | float] = {"Experiment": eid}
        for col in df.columns:
            if not df[col].empty:
                row[f"{col} (last)"] = round(float(df[col].iloc[-1]), 4)
                row[f"{col} (best)"] = round(float(df[col].min()), 4)
        summary_rows.append(row)

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True,
                     use_container_width=True)


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """展平嵌套字典为点号分隔键值对。"""
    result: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            result[key] = json.dumps(v)
        else:
            result[key] = v
    return result
