"""Plotly chart components — Swiss Design style training metric visualization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from cvlab.i18n import _

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
            "#E4002B",
            "#1A1A1E",
            "#6B6B73",
            "#002FA7",
            "#E65100",
            "#9C9CA6",
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
    metrics_by_exp: dict[str, pd.DataFrame],
    metric_key: str,
) -> None:
    """Overlay the same metric curve from multiple experiments."""
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
        st.plotly_chart(fig, width='stretch')
    else:
        st.info(f"{_('无')} {metric_key} {_('数据')}")


def plot_single_metric(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
) -> None:
    """Plot a single metric curve."""
    if column not in df.columns or df[column].empty:
        st.info(f"{_('无')} {column} {_('数据')}")
        return

    fig = px.line(
        df, y=column,
        title=title or column,
        labels={"value": column, "index": "step"},
    )
    fig.update_traces(line=dict(width=1.5, color="#E4002B"))
    fig.update_layout(template="swiss")
    st.plotly_chart(fig, width='stretch')


def plot_confusion_matrix(
    cm: list[list[float]],
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix heatmap."""
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
    st.plotly_chart(fig, width='stretch')


def plot_lr_loss_overlay(
    df: pd.DataFrame,
    loss_col: str = "train/loss",
    lr_col: str = "lr",
    title: str = "Loss & Learning Rate",
) -> None:
    """Plot Loss and LR on dual Y-axes."""
    if loss_col not in df.columns or lr_col not in df.columns:
        missing = [c for c in [loss_col, lr_col] if c not in df.columns]
        st.info(f"{_('无')} {', '.join(missing)} {_('数据')}")
        return

    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            y=df[loss_col],
            name=loss_col,
            mode="lines",
            line=dict(width=1.5, color="#E4002B"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            y=df[lr_col],
            name=lr_col,
            mode="lines",
            line=dict(width=1.5, color="#002FA7", dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=title),
        hovermode="x unified",
        template="swiss",
    )
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text=loss_col, secondary_y=False)
    fig.update_yaxes(title_text=lr_col, secondary_y=True, showgrid=False)

    # Detect LR decay events
    lr_series = df[lr_col].dropna()
    if len(lr_series) > 2:
        lr_drops = []
        for i in range(1, len(lr_series)):
            ratio = lr_series.iloc[i] / lr_series.iloc[i - 1]
            if ratio < 0.9:
                epoch = lr_series.index[i]
                lr_drops.append(epoch)
                fig.add_vline(x=epoch, line_dash="dash", line_color="#6B6B73", opacity=0.3)

    st.plotly_chart(fig, width='stretch')


# ── Plateau detection ─────────────────────────────────────


def detect_plateau(df: pd.DataFrame, col: str, window: int = 5,
                   threshold: float = 0.005) -> list[dict]:
    """检测指标平台期。"""
    if col not in df.columns or len(df) < window * 2:
        return []
    series = df[col].dropna()
    events = []
    for i in range(len(series) - window):
        segment = series.iloc[i:i + window]
        relative_change = abs(segment.iloc[-1] - segment.iloc[0]) / (abs(segment.iloc[0]) + 1e-8)
        if relative_change < threshold:
            start_epoch = series.index[i]
            end_epoch = series.index[i + window - 1]
            events.append({
                "start": start_epoch,
                "end": end_epoch,
                "relative_change": relative_change,
            })
    return events


def plot_trial_comparison(
    trial_values: list[dict[str, Any]],
    metric_key: str,
) -> None:
    """Plot trial metric bar chart."""
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
    st.plotly_chart(fig, width='stretch')
