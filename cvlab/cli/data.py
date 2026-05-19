"""cvlab data - 数据集管理工具。"""

from __future__ import annotations

import argparse
from pathlib import Path

from cvlab.cli.console import console, error, header, info, result, success, table
from cvlab.i18n import _


def cmd_data(args: argparse.Namespace) -> int:
    if args.data_command == "analyze":
        return _cmd_analyze(args)
    elif args.data_command == "augment":
        return _cmd_augment(args)
    elif args.data_command == "check":
        return _cmd_check(args)
    elif args.data_command == "history":
        return _cmd_history(args)
    else:
        error(_("未知 data 子命令: {}").format(args.data_command))
        return 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    """分析数据集统计信息。"""
    from cvlab.data.analyze import DatasetAnalyzer

    path = Path(args.path)
    if not path.exists():
        error(_("路径不存在: {}").format(path))
        return 1

    header(_("数据集分析: {}").format(path))
    analyzer = DatasetAnalyzer(str(path))
    report = analyzer.analyze()

    result(_("名称"), report.name)
    result(_("格式"), report.format_detected)
    result(_("样本总数"), str(report.total_samples))
    result(_("类别数"), str(report.total_classes))
    result(_("总大小"), f"{report.total_size_mb:.1f} MB")
    result(_("标注文件"), _("有") if report.has_annotation else _("无"))
    if report.annotation_format:
        result(_("标注格式"), report.annotation_format)

    if report.image_formats:
        header(_("图片格式分布"))
        table("", [_("格式"), _("数量")], list(report.image_formats.items()))

    if report.class_distribution:
        header(_("类别分布 (前 20)"))
        rows = sorted(report.class_distribution.items(), key=lambda x: -x[1])[:20]
        table("", [_("类别"), _("样本数")], rows)
        balance = DatasetAnalyzer.class_balance_score(report.class_distribution)
        result(_("类别平衡度"), f"{balance:.2f}")

    if report.avg_dimensions != (0.0, 0.0):
        header(_("图片尺寸"))
        result(_("最小"), f"{report.min_dimensions[0]}x{report.min_dimensions[1]}")
        result(_("最大"), f"{report.max_dimensions[0]}x{report.max_dimensions[1]}")
        result(_("平均"), f"{report.avg_dimensions[0]:.0f}x{report.avg_dimensions[1]:.0f}")

    if report.warnings:
        for w in report.warnings:
            console.print(f"  [yellow][WARN][/yellow] {w}")

    return 0


def _cmd_augment(args: argparse.Namespace) -> int:
    """预览数据增强效果。"""
    from cvlab.data.augment import TRANSFORM_REGISTRY, AugmentPreview

    image_path = Path(args.image)
    if not image_path.exists():
        error(_("图片不存在: {}").format(image_path))
        return 1

    import torchvision.transforms as T
    from PIL import Image

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        error(_("无法打开图片: {}").format(e))
        return 1

    tensor = T.ToTensor()(pil_img).unsqueeze(0)

    header(_("增强预览: {}").format(image_path.name))
    info(_("原始尺寸: {}x{}").format(pil_img.size[0], pil_img.size[1]))

    # 使用所有可用增强
    transform_specs = [{"name": name} for name in TRANSFORM_REGISTRY]
    results = AugmentPreview.apply_transforms(tensor[0], transform_specs)

    grid = AugmentPreview.make_grid(results, ncol=args.columns)
    info(_("应用了 {} 种增强").format(len(results) - 1))
    info(_("网格尺寸: {}").format(grid.shape))

    # 列出所有可用的增强
    header(_("可用增强"))
    names = sorted(TRANSFORM_REGISTRY.keys())
    table("", [_("#"), _("增强名称")], [[str(i + 1), n] for i, n in enumerate(names)])

    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """检查数据集血缘状态。"""
    from cvlab.data.provenance import ProvenanceTracker

    path = Path(args.path)
    if not path.exists():
        error(_("路径不存在: {}").format(path))
        return 1

    header(_("数据血缘检查: {}").format(path))

    tracker = ProvenanceTracker()
    changed = tracker.has_changed(str(path))

    if changed:
        console.print(f"  [yellow][WARN][/yellow] {_('数据集自上次记录后已发生变化')}")
    else:
        success(_("数据集自上次记录后未发生变化"))

    # 创建新快照
    prov = tracker.snapshot(str(path), hash_annotations=args.hash_annotations)
    result(_("根哈希"), prov.root_hash[:16] + "...")
    if prov.ann_hash:
        result(_("标注哈希"), prov.ann_hash[:16] + "...")
    result(_("文件数"), str(prov.total_files))
    result(_("总大小"), f"{prov.total_size_bytes / (1024 * 1024):.1f} MB")
    result(_("记录时间"), prov.recorded_at)

    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    """列出所有数据集快照。"""
    from cvlab.data.provenance import ProvenanceTracker

    header(_("数据血缘快照历史"))
    tracker = ProvenanceTracker()
    snapshots = tracker.list_snapshots()

    if not snapshots:
        info(_("暂无记录"))
        return 0

    rows = [[s["path"][:40], str(s["total_files"]), str(s["total_size_mb"]), s["recorded_at"]]
            for s in snapshots]
    table("", [_("数据集路径"), _("文件数"), _("大小 (MB)"), _("记录时间")], rows)
    result(_("总计"), _("{} 条记录").format(len(snapshots)))

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("data", help=_("数据集管理: 分析/增强/血缘检查"))
    sp = p.add_subparsers(dest="data_command")

    analyze_p = sp.add_parser("analyze", help=_("分析数据集统计信息"))
    analyze_p.add_argument("path", help=_("数据集路径"))
    analyze_p.set_defaults(data_command="analyze")

    augment_p = sp.add_parser("augment", help=_("预览数据增强效果"))
    augment_p.add_argument("image", help=_("图片路径"))
    augment_p.add_argument("--columns", "-c", type=int, default=4, help=_("网格列数"))
    augment_p.set_defaults(data_command="augment")

    check_p = sp.add_parser("check", help=_("检查数据血缘状态"))
    check_p.add_argument("path", help=_("数据集路径"))
    check_p.add_argument("--hash-annotations", action="store_true", help=_("计算标注文件哈希"))
    check_p.set_defaults(data_command="check")

    history_p = sp.add_parser("history", help=_("列出数据集快照历史"))
    history_p.set_defaults(data_command="history")

    p.set_defaults(func=cmd_data)
