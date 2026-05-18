"""CVLab Streamlit 主入口 — Swiss Design。"""

from __future__ import annotations

import streamlit as st

from cvlab.ui.components.layout import load_css

st.set_page_config(
    page_title="CVLab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    load_css()

    from cvlab.ui.pages.experiments import show_experiments
    from cvlab.ui.pages.experiment_detail import show_experiment_detail
    from cvlab.ui.pages.compare import show_compare
    from cvlab.ui.pages.sweep import show_sweep
    from cvlab.ui.pages.diagnostics import show_diagnostics
    from cvlab.ui.pages.datasets import show_datasets

    pages = {
        "Experiments": [
            st.Page(show_experiments, title="Experiments", default=True),
            st.Page(show_experiment_detail, title="Experiment Detail"),
        ],
        "Data": [
            st.Page(show_datasets, title="Datasets"),
        ],
        "Analysis": [
            st.Page(show_compare, title="Compare"),
            st.Page(show_sweep, title="Sweeps"),
            st.Page(show_diagnostics, title="Diagnostics"),
        ],
    }

    pg = st.navigation(pages, position="sidebar")
    pg.run()


if __name__ == "__main__":
    main()
