"""Dataset version management page — Swiss Design, i18n-ready."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from cvlab.data.provenance import ProvenanceTracker
from cvlab.db.database import Database
from cvlab.i18n import _
from cvlab.ui.components.layout import (
    divider,
    inject_language_switcher,
    metric_card,
    section_header,
    sidebar_footer,
)


def show_datasets():
    inject_language_switcher()
    st.title(_("Datasets"))

    db = Database()

    tab1, tab2, tab3 = st.tabs([_("数据集"), _("注册数据集"), _("增强预览")])

    # ── Tab 2: Register new dataset ───────────────────────
    with tab2:
        section_header(_("注册数据集"))

        with st.form("register_dataset_form"):
            col1, col2 = st.columns(2)
            with col1:
                ds_name = st.text_input(_("名称"), placeholder="e.g. CIFAR-10")
            with col2:
                ds_path = st.text_input(_("路径"), placeholder="e.g. ./data/cifar10")
            ds_desc = st.text_input(_("描述"), placeholder="Brief description")

            submitted = st.form_submit_button(_("确认"), type="primary")

            if submitted and ds_name and ds_path:
                path_obj = Path(ds_path)
                if not path_obj.exists():
                    st.error(f"{_('路径')} {_('无')}: {ds_path}")
                else:
                    ds_id = db.register_dataset(ds_name, ds_path, ds_desc)

                    # Auto-create first version snapshot
                    tracker = ProvenanceTracker()
                    prov = tracker.snapshot(ds_path, hash_annotations=True)

                    version_label = "v1"
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
                        f'<div class="label">{_("注册成功")}</div>'
                        f'<div class="value" style="font-size:0.95rem;">{ds_id} — {ds_name}</div>'
                        f'<div class="sublabel">{_("自动快照")} v1: {prov.total_files} {_("文件数")}, '
                        f'{prov.total_size_bytes / 1024**2:.1f} MB</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.rerun()
            elif submitted:
                st.warning(_("名称") + " " + _("路径") + " " + _("必填"))

    # ── Tab 1: Dataset list ───────────────────────────────
    with tab1:
        datasets = db.get_datasets()

        if not datasets:
            st.markdown(
                f'<div style="text-align:center;padding:4rem 0;color:var(--text-tertiary);'
                f'font-size:0.9rem;">{_("暂无数据集")}. {_("使用注册标签添加数据集")}</div>',
                unsafe_allow_html=True,
            )
            sidebar_footer()
            return

        section_header(_("所有数据集"), badge=str(len(datasets)))

        # Summary row
        total_files = sum(d.get("latest_files", 0) or 0 for d in datasets)
        metric_card(_("数据集总数"), str(len(datasets)),
                    f"{total_files} {_('文件数')}")

        divider()

        # Dataset list
        for ds in datasets:
            _render_dataset_card(db, ds)

    # ── Tab 3: Augmentation preview ───────────────────────
    with tab3:
        section_header(_("数据增强预览"))

        import numpy as np
        import torchvision.transforms as T
        from PIL import Image as PILImage


        col1, col2 = st.columns([1, 1])

        image_tensor = None
        source_image_path = None

        with col1:
            st.markdown(f'<div style="font-size:0.85rem;font-weight:500;'
                        f'margin-bottom:0.5rem;">{_("图片来源")}</div>',
                        unsafe_allow_html=True)

            # Option 1: Upload
            uploaded = st.file_uploader(
                _("上传图片"), type=["jpg", "jpeg", "png", "bmp"],
                label_visibility="collapsed",
            )
            if uploaded:
                pil_img = PILImage.open(uploaded).convert("RGB")
                image_tensor = T.ToTensor()(pil_img)

            # Option 2: pick from registered datasets
            if image_tensor is None:
                registered = [d for d in datasets if Path(d["path"]).is_dir()]
                if registered:
                    ds_names = [d["name"] for d in registered]
                    sel_name = st.selectbox(_("从已有数据集选取"), ["", *ds_names],
                                            key="aug_ds_select")
                    if sel_name:
                        ds = next(d for d in registered if d["name"] == sel_name)
                        ds_path = Path(ds["path"])
                        # Find first image
                        exts = {"*.jpg", "*.jpeg", "*.png", "*.bmp"}
                        found = []
                        for e in exts:
                            found.extend(list(ds_path.rglob(e))[:1])
                        if found:
                            source_image_path = found[0]
                            pil_img = PILImage.open(source_image_path).convert("RGB")
                            image_tensor = T.ToTensor()(pil_img)

            # Still no image? Use a blank canvas
            if image_tensor is None:
                blank = PILImage.new("RGB", (224, 224), color=(200, 200, 200))
                image_tensor = T.ToTensor()(blank)

            # Augmentation params
            st.markdown(f'<div style="font-size:0.85rem;font-weight:500;'
                        f'margin-top:1rem;margin-bottom:0.5rem;">{_("增强参数")}</div>',
                        unsafe_allow_html=True)

            hflip = st.checkbox("RandomHorizontalFlip", value=True, key="aug_hflip")
            vflip = st.checkbox("RandomVerticalFlip", value=False, key="aug_vflip")
            brightness = st.slider(_("亮度"), 0.0, 2.0, 1.2, 0.1, key="aug_bright")
            contrast = st.slider(_("对比度"), 0.0, 2.0, 1.0, 0.1, key="aug_contrast")
            saturation = st.slider(_("饱和度"), 0.0, 2.0, 1.0, 0.1, key="aug_sat")
            rotation = st.slider(_("旋转角度"), 0, 90, 15, 5, key="aug_rot")
            blur_kernel = st.selectbox(_("高斯模糊"), [0, 3, 5, 7], index=0, key="aug_blur")

            num_variants = st.slider(_("生成变体数"), 1, 6, 3, key="aug_n")

            regen = st.button(_("刷新预览"), type="primary", use_container_width=True)

        with col2:
            if regen or "aug_last_input" not in st.session_state:
                st.session_state["aug_last_input"] = image_tensor.clone()

            show_img = st.session_state["aug_last_input"]
            used_pil = T.ToPILImage()(show_img)

            # Build transforms
            aug_transforms = []
            if hflip:
                aug_transforms.append(T.RandomHorizontalFlip(p=1.0))
            if vflip:
                aug_transforms.append(T.RandomVerticalFlip(p=1.0))
            aug_transforms.extend([
                T.ColorJitter(brightness=brightness, contrast=contrast,
                              saturation=saturation, hue=0),
                T.RandomRotation(degrees=rotation),
            ])
            if blur_kernel > 0:
                aug_transforms.append(T.GaussianBlur(kernel_size=blur_kernel))

            composed = T.Compose(aug_transforms)

            st.markdown(
                f'<div style="font-size:0.85rem;font-weight:500;margin-bottom:0.5rem;">'
                f'{_("预览")}</div>',
                unsafe_allow_html=True,
            )

            # Show original
            st.markdown(
                f'<div style="font-size:0.75rem;color:var(--text-secondary);'
                f'margin-bottom:0.25rem;">{_("原图")}</div>',
                unsafe_allow_html=True,
            )
            st.image(np.array(used_pil), width=200)

            st.markdown(
                f'<div style="font-size:0.75rem;color:var(--text-secondary);'
                f'margin-top:0.75rem;margin-bottom:0.25rem;">{_("增强后")} ({num_variants} {_("变体")})</div>',
                unsafe_allow_html=True,
            )

            # Generate variants
            variant_cols = st.columns(min(num_variants, 3))
            for vi in range(num_variants):
                aug_img = composed(used_pil)
                with variant_cols[vi % len(variant_cols)]:
                    st.image(np.array(aug_img), width=180)

            # Show current transform sequence as code
            with st.expander(_("当前增强配置 (YAML)")):
                yaml_lines = ["transform_pipeline:"]
                if hflip:
                    yaml_lines.append("  - RandomHorizontalFlip:")
                if vflip:
                    yaml_lines.append("  - RandomVerticalFlip:")
                yaml_lines.append(f"  - ColorJitter: {{brightness: {brightness}, contrast: {contrast}, saturation: {saturation}}}")
                yaml_lines.append(f"  - RandomRotation: {{degrees: {rotation}}}")
                if blur_kernel > 0:
                    yaml_lines.append(f"  - GaussianBlur: {{kernel_size: {blur_kernel}}}")
                st.code("\n".join(yaml_lines), language="yaml")

    sidebar_footer()


def _render_dataset_card(db: Database, ds: dict) -> None:
    """Render a single dataset card."""
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
            f'{_("最新版本")}</div>'
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
                    f'<div class="label">{_("文件数")}</div>'
                    f'<div class="value">{v0.get("total_files", 0)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{_("大小")}</div>'
                    f'<div class="value">{size_mb:.1f} MB</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{_("版本")}</div>'
                    f'<div class="value">{len(versions)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Snapshot button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button(_("快照"), key=f"snap_{ds_id}"):
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
                        _("版本"): v["version"],
                        _("文件数"): v.get("total_files", 0),
                        _("大小"): f"{size_mb_v:.1f} MB",
                        _("标注哈希"): ann,
                        _("记录时间"): v["recorded_at"][:19] if v.get("recorded_at") else "",
                        _("实验"): v.get("experiment_name") or "—",
                    })

                if version_rows:
                    df = pd.DataFrame(version_rows)
                    st.dataframe(df, width='stretch', hide_index=True)

                    # File type breakdown for latest version
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
                            {"Extension": ext, _("数量"): count}
                            for ext, count in sorted(ext_data.items())
                        ])
                        st.dataframe(ext_df, width='stretch', hide_index=True)

        divider()
