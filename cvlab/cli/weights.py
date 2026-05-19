"""cvlab weights - 预训练权重管理。"""

from __future__ import annotations

import argparse

from cvlab.i18n import _
from cvlab.cli.console import header, info, result, table
from cvlab.weights.manager import WeightManager


def cmd_weights(args: argparse.Namespace) -> int:
    mgr = WeightManager()

    if args.weights_command == "list":
        header(_("本地权重缓存"))
        cache = mgr.cache_info()
        if cache["total_files"] == 0:
            info(_("无缓存的预训练权重"))
            return 0

        rows = [[f["name"], f"{f['size_mb']:.1f} MB"] for f in cache["files"]]
        table("", [_("名称"), _("大小")], rows)
        result(_("总计"), _("{} 文件, {} MB").format(cache['total_files'], f"{cache['total_size_mb']:.1f}"))
        return 0

    elif args.weights_command == "download":
        return _cmd_download(args, mgr)

    elif args.weights_command == "info":
        return _cmd_info(args, mgr)

    else:
        from cvlab.cli.console import error
        error(_("未知子命令: {}").format(args.weights_command))
        return 1


def _cmd_download(args: argparse.Namespace, mgr: WeightManager) -> int:
    from cvlab.cli.console import error, warning
    name = args.name

    # 尝试从 torchvision hub 下载
    header(_("下载权重: {}").format(name))
    try:
        import torch
        import torchvision.models as models

        builder = getattr(models, name, None)
        if builder is None:
            error(_("未知模型: {}").format(name))
            return 1

        warning(_("下载中（torchvision hub）..."))
        model = builder(weights="DEFAULT")

        # 保存到缓存
        cache_path = mgr.cache_dir / f"{name}.pth"
        torch.save(model.state_dict(), cache_path)
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        result(_("已缓存"), str(cache_path))
        result(_("大小"), f"{size_mb:.1f} MB")
        return 0
    except Exception as e:
        error(_("下载失败: {}").format(e))
        return 1


def _cmd_info(args: argparse.Namespace, mgr: WeightManager) -> int:
    """显示权重文件的诊断信息。"""
    from cvlab.cli.console import error

    try:
        import torch
        import torchvision.models as models

        builder = getattr(models, args.name, None)
        if builder is None:
            error(_("未知模型: {}").format(args.name))
            return 1

        header(_("权重信息: {}").format(args.name))
        model = builder(weights=None)
        total = sum(p.numel() for p in model.parameters())
        info(_("参数量: {}M").format(f"{total / 1e6:.2f}"))
        info(_("预期权重大小: {} MB (FP32)").format(f"{total * 4 / 1024 / 1024:.1f}"))

        # 检查缓存
        cache_path = mgr.cache_dir / f"{args.name}.pth"
        if cache_path.exists():
            info(_("缓存状态: 已缓存 ({} MB)").format(f"{cache_path.stat().st_size / 1024 / 1024:.1f}"))
        else:
            info(_("缓存状态: 未缓存（使用 cvlab weights download 下载）"))

        return 0
    except Exception as e:
        error(_("获取信息失败: {}").format(e))
        return 1


def add_subparser(sub) -> None:
    p = sub.add_parser("weights", help=_("预训练权重管理"))
    sp = p.add_subparsers(dest="weights_command")

    sp.add_parser("list", help=_("列出本地缓存的权重"))
    sp.add_parser("info", help=_("显示权重信息")).add_argument("name", help=_("模型名称"))
    sp.add_parser("download", help=_("下载预训练权重")).add_argument("name", help=_("模型名称"))
    p.set_defaults(func=cmd_weights)
