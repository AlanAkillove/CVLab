"""Plotly 图表组件 —— Swiss Design 风格的训练指标可视化。"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st

# ── Swiss Design Plotly Template ─────────────────────────

pio.templates["swiss"] = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family='"Helvetica Neue", Helvetica, -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color="#1A1A1E",
        ),
        title=dict(
            font=dict(size=14, weight=500, color="#1A1A1E"),
            x=0,
            xanchor="left",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        hoverlabel=dict(
            font=dict(family='"SF Mono", "JetBrains Mono", monospace', size=11),
            bordercolor="#D6D6DA",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#EEEEF0",
            gridwidth=1,
            linecolor="#D6D6DA",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=11, color="#6B6B73"),
            title=dict(font=dict(size=11, color="#6B6B73")),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#EEEEF0",
            gridwidth=1,
            linecolor="#D6D6DA",
            linewidth=1,
            zeroline=False,
            tickfont=dict(size=11, color="#6B6B73"),
            title=dict(font=dict(size=11, color="#6B6B73")),
        ),
        colorway=[
            "#E4002B",  # Swiss Red
            "#1A1A1E",  # Near black
            "#6B6B73",  # Grey
            "#002FA7",  # Yves Klein Blue
            "#E65100",  # Orange
            "#9C9CA6",  # Light grey
        ],
        legend=dict(
            font=dict(size=11, color="#1A1A1E"),
            bordercolor="#E5E5E9",
            borderwidth=1,
            orientation="h",
            y=1.1,
            x=0,
            xanchor="left",
            yanchor="bottom",
        ),
        margin=dict(l=50, r=20, t=40, b=50),
        shapes=[
            # Hairline top border
            dict(
                type="line",
                x0=0, y0=1, x1=1, y1=1,
                xref="paper", yref="paper",
                line=dict(color="#E5E5E9", width=1),
            ),
        ],
    ),
)

pio.templates.default = "swiss"


def plot_metric_overlay(
    metrics_by_exp: dict[str, "pd.DataFrame"],
    metric_key: str,
) -> None:
    """叠加显示多个实验的同一指标曲线。

    Args:
        metrics_by_exp: {experiment_id: DataFrame}。
        metric_key: 指标名。
    """
    import pandas as pd  # noqa: F811

    fig = go.Figure()
    has_data = False

    for eid, df in metrics_by_exp.items():
        if metric_key in df.columns:
            fig.add_trace(go.Scatter(
                y=df[metric_key],
                name=eid[:12],
                mode="lines+markers",
                line=dict(width=1.5),
                marker=dict(size=4),
                connectgaps=False,
            ))
            has_data = True

    if has_data:
        fig.update_layout(
            title=dict(text=metric_key),
            xaxis_title="Step",
            yaxis_title="Value",
            template="swiss",
        )
        st.plotly_chart(fig, width='stretch', use_container_width=True)
    else:
        st.info(f"无 {metric_key} 数据")


def plot_single_metric(
    df: "pd.DataFrame",
    column: str,
    title: str | None = None,
) -> None:
    """绘制单条指标曲线。"""
    import pandas as pd  # noqa: F811

    if column not in df.columns or df[column].empty:
        st.info(f"无 {column} 数据")
        return

    fig = px.line(
        df, y=column,
        title=title or column,
        labels={"value": column, "index": "step"},
    )
    fig.update_traces(line=dict(width=1.5, color="#E4002B"))
    fig.update_layout(template="swiss")
    st.plotly_chart(fig, width='stretch', use_container_width=True)


def plot_confusion_matrix(
    cm: list[list[float]],
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> None:
    """绘制混淆矩阵热力图。"""
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#F0E0E4"], [1, "#E4002B"]],
        title=title,
        labels=dict(x="Predicted", y="True", color=""),
        aspect="auto",
    )
    fig.update_layout(
        template="swiss",
        coloraxis_colorbar=dict(
            tickfont=dict(size=10, color="#6B6B73"),
            thickness=12,
        ),
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickangle=0)
    st.plotly_chart(fig, width='stretch', use_container_width=True)


def plot_trial_comparison(
    trial_values: list[dict[str, Any]],
    metric_key: str,
) -> None:
    """绘制 Trial 指标柱状对比图。"""
    import pandas as pd  # noqa: F811

    df = pd.DataFrame(trial_values)
    fig = px.bar(
        df, x="Trial", y=metric_key,
        title=metric_key,
        color=metric_key,
        color_continuous_scale=[[0, "#EEEEF0"], [1, "#E4002B"]],
        text=metric_key,
    )
    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        marker=dict(line=dict(width=0)),
    )
    fig.update_layout(
        template="swiss",
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width='stretch', use_container_width=True)
