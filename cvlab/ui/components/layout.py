"""Swiss Design layout components — reusable UI elements with i18n support.

All components use st.markdown() + CSS class injection,
no third-party UI libraries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from cvlab.i18n import _, current_language, set_language, get_available_languages


def load_css() -> None:
    """Load global CSS stylesheet."""
    css_path = Path(__file__).parent.parent / "static" / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def inject_language_switcher() -> None:
    """Inject language switcher and theme toggle into the UI.

    Renders a small fixed-position bar at the top-right corner.
    """
    current = current_language()
    available = get_available_languages()

    # Build language selector HTML
    options_html = ""
    for lang in available:
        code = lang["code"]
        # Use name_en for consistency
        display = lang["name_en"]
        selected = "selected" if code == current else ""
        options_html += f'<option value="{code}" {selected}>{display}</option>'

    # Detect current theme
    theme = "dark" if st.get_option("theme.base") == "dark" else "light"

    switcher_html = f"""
    <div class="lang-switcher">
        <select id="cvlab-lang-selector"
                onchange="changeLanguage(this.value)"
                title="Switch language">
            {options_html}
        </select>
        <button class="theme-toggle" id="cvlab-theme-toggle"
                onclick="toggleTheme()"
                title="Toggle dark/light mode">
            {'🌙' if theme == 'light' else '☀️'}
        </button>
    </div>
    <script>
    function changeLanguage(lang) {{
        const params = new URLSearchParams(window.location.search);
        params.set('lang', lang);
        window.location.search = params.toString();
    }}
    function toggleTheme() {{
        const html = document.documentElement;
        const current = html.getAttribute('data-theme');
        if (current === 'dark') {{
            html.removeAttribute('data-theme');
            html.setAttribute('data-theme', 'light');
        }} else {{
            html.removeAttribute('data-theme');
            html.setAttribute('data-theme', 'dark');
        }}
    }}
    </script>
    """

    st.markdown(switcher_html, unsafe_allow_html=True)


def sidebar_footer() -> None:
    """Render footer info in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-size:0.65rem;color:var(--text-tertiary);'
        f'text-align:center;padding:0.5rem 0;">'
        f'CVLab v0.1.0 &mdash; {_("CV实验管理平台")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(label: str, badge: str | None = None) -> None:
    """Render section header with Swiss Red underline.

    Args:
        label: Header text (will be translated via _()).
        badge: Optional badge text (e.g. count).
    """
    translated = _(label)
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-header"><span class="label">{translated}</span>{badge_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sublabel: str | None = None) -> None:
    """Render metric card with red left rule.

    Args:
        label: Metric label (auto-uppercased, translated).
        value: Metric value.
        sublabel: Optional secondary description.
    """
    sub_html = f'<div class="sublabel">{_(sublabel) if sublabel else ""}</div>' if sublabel else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{_(label)}</div>'
        f'<div class="value">{value}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_row(cards: list[tuple[str, str, str | None]]) -> None:
    """Horizontal arrangement of multiple metric_cards.

    Args:
        cards: [(label, value, sublabel), ...].
    """
    cols = st.columns(len(cards))
    for col, (label, value, sublabel) in zip(cols, cards):
        with col:
            metric_card(label, value, sublabel)


def data_card(title: str, content_html: str) -> None:
    """Render generic data card.

    Args:
        title: Card title (translated).
        content_html: Card content HTML.
    """
    st.markdown(
        f'<div class="data-card">'
        f'<div class="card-title">{_(title)}</div>'
        f'{content_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """Generate status badge HTML.

    Args:
        status: Experiment status (completed/running/failed/created/archived).

    Returns:
        Status badge HTML string.
    """
    normalized = status.lower().replace(" ", "-")
    display = _(status.capitalize())
    return f'<span class="status-badge {normalized}">{display}</span>'


def tag_chip(tag: str) -> str:
    """Generate tag chip HTML.

    Args:
        tag: Tag text.

    Returns:
        Tag chip HTML string.
    """
    return f'<span class="tag-chip">{tag}</span>'


def specimen_slide(image_path: str, caption: str = "") -> None:
    """Render specimen-slide style image display.

    Args:
        image_path: Image file path.
        caption: Image caption (translated).
    """
    caption_text = _(caption) if caption else ""
    st.markdown(
        f'<div class="specimen-slide">'
        f'<img src="file://{image_path}" style="width:100%;height:auto;display:block;" loading="lazy" />'
        f'{"<div class=\"slide-label\">" + caption_text + "</div>" if caption_text else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    """Render thin line divider."""
    st.markdown("<hr />", unsafe_allow_html=True)
