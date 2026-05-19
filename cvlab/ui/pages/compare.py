"""Experiment comparison page — Swiss Design, i18n-ready."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cvlab.core.utils import flatten_dict
from cvlab.db.database import Database
from cvlab.i18n import _
from cvlab.ui.components.layout import (
    divider,
    inject_language_switcher,
    section_header,
    sidebar_footer,
)


def show_compare():
    inject_language_switcher()
    st.title(_("Compare"))

    db = Database()
    exps = db.list_experiments(limit=100)

    if not exps or len(exps) < 2:
        st.markdown(
            f'<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            f'font-size:0.9rem;">{_("至少选择 2 个实验")}</div>',
            unsafe_allow_html=True,
        )
        sidebar_footer()
        return

    exp_ids = [e["id"] for e in exps]
    selected = st.multiselect(
        _("选择实验"),
        exp_ids,
        default=exp_ids[: min(2, len(exp_ids))],
        format_func=lambda x: f"{x} — {db.get_experiment(x)['name']}",
        key="compare_selector",
    )

    if len(selected) < 2:
        st.info(_("至少选择 2 个实验"))
        sidebar_footer()
        return

    if len(selected) > 4:
        st.warning(_("最多 4 个实验"))
        selected = selected[:4]

    # ── Overview table ──────────────────────────────────
    section_header(_("概览"))
    info_rows = []
    for eid in selected:
        exp = db.get_experiment(eid)
        if exp:
            info_rows.append({
                _("ID"): eid,
                _("名称"): exp["name"],
                _("状态"): exp["status"],
                "Seed": exp.get("seed", "—"),
                _("创建时间"): exp["created_at"][:19] if exp.get("created_at") else "",
            })

    if info_rows:
        st.dataframe(pd.DataFrame(info_rows), width='stretch', hide_index=True,
                     width='stretch')

    divider()

    # ── Config comparison ────────────────────────────────
    section_header(_("配置对比"))

    configs: list[dict[str, str]] = []
    config_keys: set[str] = set()
    for eid in selected:
        exp = db.get_experiment(eid)
        if not exp or not exp.get("config_json"):
            continue
        try:
            cfg = json.loads(exp["config_json"]) if isinstance(exp["config_json"], str) else exp["config_json"]
        except (json.JSONDecodeError, TypeError):
            continue
        flat = flatten_dict(cfg)
        configs.append(flat)
        config_keys.update(flat.keys())

    if configs:
        config_rows = []
        for key in sorted(config_keys):
            row: dict[str, str] = {_("参数"): key}
            for i, eid in enumerate(selected):
                row[eid[:16]] = str(configs[i].get(key, "—")) if i < len(configs) else "—"
            config_rows.append(row)
        st.dataframe(pd.DataFrame(config_rows), width='stretch', hide_index=True,
                     width='stretch')

    divider()

    # ── Metric overlay ──────────────────────────────────
    section_header(_("指标叠加"))

    metrics_by_exp: dict[str, pd.DataFrame] = {}
    all_metric_keys: set[str] = set()
    for eid in selected:
        df = db.get_metrics_dataframe(eid)
        if df is not None and not df.empty:
            metrics_by_exp[eid] = df
            all_metric_keys.update(df.columns.tolist())

    if not all_metric_keys:
        st.info(_("暂无实验"))
        sidebar_footer()
        return

    from cvlab.ui.components.charts import plot_metric_overlay

    for metric_key in sorted(all_metric_keys):
        plot_metric_overlay(metrics_by_exp, metric_key)

    divider()

    # ── Summary table ────────────────────────────────────
    section_header(_("汇总"))

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
                     width='stretch')

    sidebar_footer()
