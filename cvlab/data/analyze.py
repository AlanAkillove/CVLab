"""数据集分析 - 统计信息、类别分布、样本可视化。

支持格式:
- ImageFolder (分类目录结构)
- 平铺图片目录
- CIFAR-10/100 batch 格式
- VOC 格式 (JPEGImages + Annotations)
- YOLO 格式 (images + labels)
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetReport:
    name: str
    format_detected: str = "unknown"
    total_samples: int = 0
    total_classes: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    image_formats: dict[str, int] = field(default_factory=dict)
    min_dimensions: tuple[int, int] = (0, 0)
    max_dimensions: tuple[int, int] = (0, 0)
    avg_dimensions: tuple[float, float] = (0.0, 0.0)
    total_size_mb: float = 0.0
    has_annotation: bool = False
    annotation_format: str = ""
    warnings: list[str] = field(default_factory=list)


class DatasetAnalyzer:
    """数据集分析器，支持多种 CV 数据集格式。"""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def analyze(self) -> DatasetReport:
        """执行完整分析，自动检测数据集格式。"""
        report = DatasetReport(name=self.root.name)

        if not self.root.exists():
            report.warnings.append("路径不存在")
            return report

        # 按格式优先级检测
        if self._detect_cifar(report):
            return report
        if self._detect_voc(report):
            return report
        if self._detect_yolo(report):
            return report
        if self._detect_imagefolder(report):
            return report
        self._detect_flat(report)
        return report

    def _detect_cifar(self, report: DatasetReport) -> bool:
        """检测 CIFAR-10/100 batch 格式。"""
        root = self.root
        # CIFAR-10: data_batch_1..5 + test_batch + batches.meta
        cifar10_files = {"data_batch_1", "data_batch_2", "data_batch_3",
                         "data_batch_4", "data_batch_5", "test_batch", "batches.meta"}
        # CIFAR-100: train + test + meta
        cifar100_files = {"train", "test", "meta"}

        files = {f.name for f in root.iterdir() if f.is_file()}

        is_cifar10 = cifar10_files.issubset(files) or \
                     (any(f.startswith("data_batch_") for f in files) and "test_batch" in files)
        is_cifar100 = cifar100_files.issubset(files)

        if not is_cifar10 and not is_cifar100:
            return False

        report.format_detected = "CIFAR-10" if is_cifar10 else "CIFAR-100"
        try:
            import pickle
            batch_file = "data_batch_1" if is_cifar10 else "train"
            with open(root / batch_file, "rb") as f:
                batch = pickle.load(f, encoding="latin1")
            if isinstance(batch, dict):
                report.total_samples = len(batch.get(b"labels" if is_cifar10 else b"fine_labels",
                                         batch.get("labels", batch.get("data", []))))
                # 标签文件
                meta_file = "batches.meta" if is_cifar10 else "meta"
                meta_path = root / meta_file
                if meta_path.exists():
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f, encoding="latin1")
                    label_key = b"label_names" if is_cifar10 else b"fine_label_names"
                    names = meta.get(label_key, meta.get("label_names", []))
                    report.total_classes = len(names)
        except Exception:
            # fallback: file count
            report.total_samples = sum(1 for f in root.iterdir() if f.is_file()
                                       and "data_batch" in f.name)

        report.image_formats = {"pickle_batch": report.total_samples}
        report.total_size_mb = sum(f.stat().st_size for f in root.iterdir() if f.is_file()) / (1024 * 1024)
        report.has_annotation = True
        report.annotation_format = "CIFAR built-in labels"
        return True

    def _detect_voc(self, report: DatasetReport) -> bool:
        """检测 VOC 格式 (JPEGImages + Annotations + ImageSets)。"""
        root = self.root
        has_jpeg = (root / "JPEGImages").is_dir()
        has_ann = (root / "Annotations").is_dir()
        has_sets = (root / "ImageSets").is_dir()

        if not (has_jpeg and has_ann):
            return False

        report.format_detected = "VOC"
        jpeg_dir = root / "JPEGImages"
        ann_dir = root / "Annotations"
        extensions = {".jpg", ".jpeg", ".png"}
        images = [f for f in jpeg_dir.rglob("*") if f.suffix.lower() in extensions]
        annotations = [f for f in ann_dir.rglob("*.xml")]

        report.total_samples = len(images)
        report.total_size_mb = sum(f.stat().st_size for f in images) / (1024 * 1024)
        report.has_annotation = len(annotations) > 0
        report.annotation_format = "VOC XML"

        # 统计图片格式
        fmt_counter: Counter = Counter()
        for f in images:
            fmt_counter[f.suffix.lower()] += 1
        report.image_formats = dict(fmt_counter)

        # ImageSets 中的类别
        if has_sets:
            cls_file = root / "ImageSets" / "Main" / "classes.txt"
            if not cls_file.exists():
                # 尝试从分割文件推断类别数
                seg_root = root / "ImageSets" / "Main"
                txt_files = list(seg_root.glob("*.txt")) if seg_root.exists() else []
                report.total_classes = max(len(txt_files) - 1, 0)
        return True

    def _detect_yolo(self, report: DatasetReport) -> bool:
        """检测 YOLO 格式 (images + labels)。"""
        root = self.root
        # YOLO 通常有 images/ 和 labels/ 目录（可能在 train/val 子目录下）
        yolo_variants = [
            (root / "images", root / "labels"),
        ]
        # 也检查 train/val 子目录
        for split in ["train", "val", "test"]:
            yolo_variants.append((root / split / "images", root / split / "labels"))

        has_yolo = False
        total_images = 0
        total_labels = 0
        all_images: list[Path] = []
        all_labels: list[Path] = []
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for img_dir, lbl_dir in yolo_variants:
            if img_dir.is_dir():
                images = [f for f in img_dir.rglob("*") if f.suffix.lower() in extensions]
                all_images.extend(images)
                total_images += len(images)
                if lbl_dir.is_dir():
                    labels = [f for f in lbl_dir.rglob("*.txt")]
                    all_labels.extend(labels)
                    total_labels += len(labels)
                has_yolo = True

        if not has_yolo or total_images == 0:
            return False

        report.format_detected = "YOLO"
        report.total_samples = total_images
        report.has_annotation = total_labels > 0
        report.annotation_format = "YOLO txt"

        fmt_counter: Counter = Counter()
        for f in all_images:
            fmt_counter[f.suffix.lower()] += 1
        report.image_formats = dict(fmt_counter)
        report.total_size_mb = sum(f.stat().st_size for f in all_images) / (1024 * 1024)

        # 从标签文件推断类别数（读取所有标签文件中的最大 class_id）
        max_class_id = -1
        for lbl in all_labels:
            try:
                text = lbl.read_text(encoding="utf-8").strip()
                for line in text.split("\n"):
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        max_class_id = max(max_class_id, class_id)
            except Exception:
                pass
        if max_class_id >= 0:
            report.total_classes = max_class_id + 1
        return True

    def _detect_imagefolder(self, report: DatasetReport) -> bool:
        """检测 ImageFolder 格式（类别子目录）。"""
        root = self.root
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        class_samples: dict[str, list[Path]] = {}
        format_counter: Counter = Counter()
        total_size = 0

        subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not subdirs:
            return False

        for cls_dir in subdirs:
            class_name = cls_dir.name
            files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in extensions]
            if not files:
                continue
            class_samples[class_name] = files
            for f in files:
                format_counter[f.suffix.lower()] += 1
                total_size += f.stat().st_size

        if not class_samples:
            return False

        report.format_detected = "ImageFolder"
        report.total_classes = len(class_samples)
        report.total_samples = sum(len(v) for v in class_samples.values())
        report.class_distribution = {k: len(v) for k, v in class_samples.items()}
        report.image_formats = dict(format_counter)
        report.total_size_mb = total_size / (1024 * 1024)
        # 检查 annotations
        ann_extensions = {".xml", ".json", ".txt"}
        for cls_dir in subdirs:
            ann_files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in ann_extensions]
            if ann_files:
                report.has_annotation = True
                report.annotation_format = "detected: " + ", ".join(
                    sorted({f.suffix for f in ann_files}))
                break
        return True

    def _detect_flat(self, report: DatasetReport) -> bool:
        """检测平铺图片目录（所有图片在根目录）。"""
        root = self.root
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        format_counter: Counter = Counter()
        total_size = 0

        files = [f for f in root.glob("*") if f.suffix.lower() in extensions]
        # 也检查子目录（递归一层）
        if not files:
            files = [f for f in root.rglob("*") if f.suffix.lower() in extensions]

        if not files:
            report.warnings.append("未找到图片文件（支持 .jpg/.png/.bmp/.tiff/.webp）")
            report.format_detected = "empty"
            return True

        report.format_detected = "flat"
        report.total_samples = len(files)
        for f in files:
            format_counter[f.suffix.lower()] += 1
            total_size += f.stat().st_size
        report.image_formats = dict(format_counter)
        report.total_size_mb = total_size / (1024 * 1024)
        return True
