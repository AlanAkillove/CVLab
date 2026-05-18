"""Swiss Design 布局组件 —— 可复用的 UI 元素。

所有组件使用 st.markdown() + CSS class 注入实现，
不依赖第三方 UI 库。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def load_css() -> None:
    """加载全局 CSS 样式。"""
    css_path = Path(__file__).parent.parent / "static" / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def section_header(label: str, badge: str | None = None) -> None:
    """渲染带 Swiss Red 下划线的章节标题。

    Args:
        label: 标题文字。
        badge: 可选的徽章文字（如计数）。
    """
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-header"><span class="label">{label}</span>{badge_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sublabel: str | None = None) -> None:
    """渲染带红色左侧线的指标卡片。

    Args:
        label: 指标名称（自动大写）。
        value: 指标值。
        sublabel: 可选的次要描述。
    """
    sub_html = f'<div class="sublabel">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_row(cards: list[tuple[str, str, str | None]]) -> None:
    """水平排列多个 metric_card。

    Args:
        cards: [(label, value, sublabel), ...]。
    """
    cols = st.columns(len(cards))
    for col, (label, value, sublabel) in zip(cols, cards):
        with col:
            metric_card(label, value, sublabel)


def data_card(title: str, content_html: str) -> None:
    """渲染通用数据卡片。

    Args:
        title: 卡片标题。
        content_html: 卡片内容的 HTML。
    """
    st.markdown(
        f'<div class="data-card">'
        f'<div class="card-title">{title}</div>'
        f'{content_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """生成状态徽章 HTML。

    Args:
        status: 实验状态（completed/running/failed/created/archived）。

    Returns:
        状态徽章的 HTML 字符串。
    """
    normalized = status.lower().replace(" ", "-")
    return f'<span class="status-badge {normalized}">{status}</span>'


def tag_chip(tag: str) -> str:
    """生成标签 Chip HTML。

    Args:
        tag: 标签文字。

    Returns:
        标签 Chip 的 HTML 字符串。
    """
    return f'<span class="tag-chip">{tag}</span>'


def specimen_slide(image_path: str, caption: str = "") -> None:
    """渲染标本幻灯片样式的图片展示。

    Args:
        image_path: 图片文件路径。
        caption: 图片说明文字。
    """
    st.markdown(
        f'<div class="specimen-slide">'
        f'<img src="file://{image_path}" style="width:100%;height:auto;display:block;" loading="lazy" />'
        f'{"<div class=\"slide-label\">" + caption + "</div>" if caption else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    """渲染细线分隔符。"""
    st.markdown("<hr />", unsafe_allow_html=True)
