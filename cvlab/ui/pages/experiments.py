"""Experiment list page — Swiss Design, i18n-ready."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cvlab.db.database import Database
from cvlab.i18n import _
from cvlab.ui.components.layout import (
    section_header,
    metric_row,
    status_badge,
    divider,
    inject_language_switcher,
    sidebar_footer,
)


def show_experiments():
    inject_language_switcher()
    st.title("Experiments")

    db = Database()

    # ── Filter bar ───────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            _("状态"),
            ["all", "created", "running", "completed", "failed", "archived"],
            key="exp_status_filter",
        )
    with col2:
        tag_filter = st.text_input(
            _("标签"),
            placeholder=_("过滤") + "...",
            key="exp_tag_filter",
        )

    kwargs: dict = {"limit": 100}
    if status_filter != "all":
        kwargs["status"] = status_filter
    if tag_filter:
        kwargs["tag"] = tag_filter

    exps = db.list_experiments(**kwargs)

    if not exps:
        st.markdown(
            f'<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
            f'font-size:0.9rem;">{_("暂无实验")}. '
            f'{_("启动训练")}: <code>cvlab train</code></div>',
            unsafe_allow_html=True,
        )
        sidebar_footer()
        return

    # ── Summary row ──────────────────────────────────────
    total = len(exps)
    running = sum(1 for e in exps if e["status"] == "running")
    completed = sum(1 for e in exps if e["status"] == "completed")
    failed = sum(1 for e in exps if e["status"] == "failed")

    metric_row([
        (_("全部"), str(total), None),
        (_("运行中"), str(running), None),
        (_("已完成"), str(completed), None),
        (_("已失败"), str(failed), None),
    ])

    # ── Experiment table ─────────────────────────────────
    section_header(_("实验列表"), badge=str(total))

    rows = []
    for e in exps:
        config = e.get("config_json", "{}")
        try:
            cfg = json.loads(config) if isinstance(config, str) else config
        except (json.JSONDecodeError, TypeError):
            cfg = {}
        model_name = cfg.get("model", {}).get("name", "—") if isinstance(cfg, dict) else "—"

        rows.append({
            _("ID"): e["id"],
            _("名称"): e["name"],
            _("模型"): model_name,
            _("状态"): status_badge(e["status"]),
            _("创建时间"): e["created_at"][:19] if e.get("created_at") else "",
        })

    df = pd.DataFrame(rows)

    st.markdown(
        df.to_html(
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
        _("选择实验"),
        exp_ids,
        format_func=lambda x: f"{x} — {db.get_experiment(x)['name']}",
        key="exp_selector",
    )

    if not selected_id:
        sidebar_footer()
        return

    exp = db.get_experiment(selected_id)
    if not exp:
        st.error(_("实验不存在"))
        sidebar_footer()
        return

    # ── Quick detail panel ───────────────────────────────
    section_header(selected_id, badge=exp["status"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{_("状态")}</div>'
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
            f'<div class="label">{_("创建时间")}</div>'
            f'<div class="value" style="font-size:0.95rem;">{exp["created_at"][:19]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if exp.get("failure_reason"):
        st.markdown(
            f'<div class="metric-card" style="border-left-color:var(--accent);'
            f'background:var(--accent-subtle);">'
            f'<div class="label">{_("失败原因")}</div>'
            f'<div class="value" style="font-size:0.9rem;">{exp["failure_reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander(_("超参配置")):
        try:
            cfg = json.loads(exp["config_json"]) if isinstance(exp["config_json"], str) else exp["config_json"]
            st.json(cfg)
        except (json.JSONDecodeError, TypeError):
            st.code(exp.get("config_json", ""), language="json")

    with st.expander(_("环境")):
        try:
            env = json.loads(exp.get("env_json", "{}"))
            st.json(env)
        except (json.JSONDecodeError, TypeError):
            st.code(exp.get("env_json", ""), language="json")

    # ── Metrics ─────────────────────────────────────────
    metrics = db.get_metrics(selected_id)
    if metrics:
        section_header(_("训练指标"))
        df_metrics = db.get_metrics_dataframe(selected_id)
        if df_metrics is not None and not df_metrics.empty:
            from cvlab.ui.components.charts import plot_single_metric
            for col in df_metrics.columns:
                plot_single_metric(df_metrics, col)

    sidebar_footer()
