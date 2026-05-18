"""数据集版本管理页 — Swiss Design。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd

from cvlab.db.database import Database
from cvlab.ui.components.layout import (
    section_header,
    metric_card,
    status_badge,
    divider,
)
from cvlab.data.provenance import ProvenanceTracker


def show_datasets():
    st.title("Datasets")

    db = Database()

    tab1, tab2 = st.tabs(["Registered", "Register"])

    # ── Tab 2: Register new dataset ───────────────────────
    with tab2:
        section_header("Register Dataset")

        with st.form("register_dataset_form"):
            col1, col2 = st.columns(2)
            with col1:
                ds_name = st.text_input("Dataset name", placeholder="e.g. CIFAR-10")
            with col2:
                ds_path = st.text_input("Path", placeholder="e.g. ./data/cifar10")
            ds_desc = st.text_input("Description (optional)", placeholder="Brief description")

            submitted = st.form_submit_button("Register", type="primary")

            if submitted and ds_name and ds_path:
                path_obj = Path(ds_path)
                if not path_obj.exists():
                    st.error(f"Path does not exist: {ds_path}")
                else:
                    ds_id = db.register_dataset(ds_name, ds_path, ds_desc)

                    # 自动创建第一个版本快照
                    tracker = ProvenanceTracker()
                    prov = tracker.snapshot(ds_path, hash_annotations=True)

                    version_label = f"v1"
                    db.record_dataset_version(
                        dataset_id=ds_id,
                        version=version_label,
                        root_hash=prov.root_hash,
                        ann_hash=prov.ann_hash,
                        total_files=prov.total_files,
                        total_size_bytes=prov.total_size_bytes,
                        file_count_by_ext=prov.file_count_by_ext,
                    )

                    st.markdown(
                        f'<div class="metric-card" style="border-left-color:var(--success);">'
                        f'<div class="label">Registered</div>'
                        f'<div class="value" style="font-size:0.95rem;">{ds_id} — {ds_name}</div>'
                        f'<div class="sublabel">Auto-snapshot v1: {prov.total_files} files, '
                        f'{prov.total_size_bytes / 1024**2:.1f} MB</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            elif submitted:
                st.warning("Name and path are required")

    # ── Tab 1: Dataset list ───────────────────────────────
    with tab1:
        datasets = db.get_datasets()

        if not datasets:
            st.markdown(
                '<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
                'font-size:0.9rem;">No datasets registered yet. Use the Register tab to add one.</div>',
                unsafe_allow_html=True,
            )
            return

        section_header("All Datasets", badge=str(len(datasets)))

        # Summary row
        total_files = sum(d.get("latest_files", 0) or 0 for d in datasets)
        metric_card("Total Datasets", str(len(datasets)),
                    f"{total_files} files across all datasets")

        divider()

        # Dataset list
        for ds in datasets:
            _render_dataset_card(db, ds)


def _render_dataset_card(db: Database, ds: dict) -> None:
    """渲染单个数据集卡片。"""
    ds_id = ds["id"]
    versions = db.get_dataset_versions(ds_id)
    latest_ver = ds.get("latest_version", "—")

    with st.container():
        st.markdown(
            f'<div class="data-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<div class="card-title">{ds_id}</div>'
            f'<div style="font-size:1.1rem;font-weight:500;">{ds["name"]}</div>'
            f'<div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.25rem;">'
            f'{ds["path"]}</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-secondary);">'
            f'Latest</div>'
            f'<div style="font-size:1.1rem;font-weight:500;">{latest_ver}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if ds.get("description"):
            st.caption(ds["description"])

        # Quick stats
        if versions:
            v0 = versions[0]
            size_mb = v0.get("total_size_bytes", 0) / (1024 * 1024)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Files</div>'
                    f'<div class="value">{v0.get("total_files", 0)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Size</div>'
                    f'<div class="value">{size_mb:.1f} MB</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Versions</div>'
                    f'<div class="value">{len(versions)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Snapshot button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Snapshot", key=f"snap_{ds_id}"):
                tracker = ProvenanceTracker()
                prov = tracker.snapshot(ds["path"], hash_annotations=True)
                version_num = len(versions) + 1
                version_label = f"v{version_num}"
                db.record_dataset_version(
                    dataset_id=ds_id,
                    version=version_label,
                    root_hash=prov.root_hash,
                    ann_hash=prov.ann_hash,
                    total_files=prov.total_files,
                    total_size_bytes=prov.total_size_bytes,
                    file_count_by_ext=prov.file_count_by_ext,
                )
                st.rerun()

        # Version history
        if versions:
            with st.expander(f"Version history ({len(versions)} versions)"):
                version_rows = []
                for v in versions:
                    size_mb_v = v.get("total_size_bytes", 0) / (1024 * 1024)
                    ann = v.get("ann_hash", "")[:12] if v.get("ann_hash") else "—"
                    version_rows.append({
                        "Version": v["version"],
                        "Files": v.get("total_files", 0),
                        "Size": f"{size_mb_v:.1f} MB",
                        "Annotation Hash": ann,
                        "Recorded": v["recorded_at"][:19] if v.get("recorded_at") else "",
                        "Experiment": v.get("experiment_name") or "—",
                    })

                if version_rows:
                    df = pd.DataFrame(version_rows)
                    st.dataframe(df, width='stretch', hide_index=True,
                                 use_container_width=True)

                    # File type breakdown for latest version
                    if version_rows:
                        latest = versions[0]
                    ext_json = latest.get("file_count_by_ext", "{}")
                    if isinstance(ext_json, str):
                        try:
                            ext_data = json.loads(ext_json)
                        except (json.JSONDecodeError, TypeError):
                            ext_data = {}
                    else:
                        ext_data = ext_json

                    if ext_data:
                        st.markdown(
                            '<div style="font-size:0.75rem;font-weight:600;'
                            'text-transform:uppercase;letter-spacing:0.05em;'
                            'color:var(--text-secondary);margin:0.5rem 0 0.25rem;">'
                            'File type breakdown</div>',
                            unsafe_allow_html=True,
                        )
                        ext_df = pd.DataFrame([
                            {"Extension": ext, "Count": count}
                            for ext, count in sorted(ext_data.items())
                        ])
                        st.dataframe(ext_df, width='stretch', hide_index=True,
                                     use_container_width=True)

        divider()
