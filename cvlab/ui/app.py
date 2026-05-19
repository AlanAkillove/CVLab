"""CVLab Streamlit main entry — Swiss Design, with i18n support.

Language can be switched via:
    - Query parameter: ?lang=en or ?lang=zh
    - CVLAB_LANG environment variable
    - System locale detection (fallback)
"""

from __future__ import annotations

import streamlit as st

from cvlab.i18n import _, set_language
from cvlab.ui.components.layout import load_css

st.set_page_config(
    page_title="CVLab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Detect language from query params
    query_params = st.query_params
    lang = query_params.get("lang", None)
    if lang:
        set_language(lang)
    else:
        # Already auto-detected via cvlab.i18n.init() on import
        pass

    load_css()

    from cvlab.ui.pages.compare import show_compare
    from cvlab.ui.pages.datasets import show_datasets
    from cvlab.ui.pages.diagnostics import show_diagnostics
    from cvlab.ui.pages.experiment_detail import show_experiment_detail
    from cvlab.ui.pages.experiments import show_experiments
    from cvlab.ui.pages.sweep import show_sweep

    pages = {
        _("Experiments"): [
            st.Page(show_experiments, title=_("Experiments"), default=True),
            st.Page(show_experiment_detail, title=_("Experiment Detail")),
        ],
        _("Data"): [
            st.Page(show_datasets, title=_("Datasets")),
        ],
        _("Analysis"): [
            st.Page(show_compare, title=_("Compare")),
            st.Page(show_sweep, title=_("Sweeps")),
            st.Page(show_diagnostics, title=_("Diagnostics")),
        ],
    }

    pg = st.navigation(pages, position="sidebar")
    pg.run()


if __name__ == "__main__":
    main()
