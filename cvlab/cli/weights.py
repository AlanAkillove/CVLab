"""cvlab weights - 预训练权重管理。"""

from __future__ import annotations

import argparse

from cvlab.cli.console import header, info, result, table
from cvlab.weights.manager import WeightManager


def cmd_weights(args: argparse.Namespace) -> int:
    mgr = WeightManager()

    if args.weights_command == "list":
        header("本地权重缓存")
        cache = mgr.cache_info()
        if cache["total_files"] == 0:
            info("无缓存的预训练权重")
            return 0

        rows = [[f["name"], f"{f['size_mb']:.1f} MB"] for f in cache["files"]]
        table("", ["名称", "大小"], rows)
        result("总计", f"{cache['total_files']} 文件, {cache['total_size_mb']:.1f} MB")
        return 0

    elif args.weights_command == "download":
        return _cmd_download(args, mgr)

    elif args.weights_command == "info":
        return _cmd_info(args, mgr)

    else:
        from cvlab.cli.console import error
        error(f"未知子命令: {args.weights_command}")
        return 1


def _cmd_download(args: argparse.Namespace, mgr: WeightManager) -> int:
    from cvlab.cli.console import error, warning
    name = args.name

    # 尝试从 torchvision hub 下载
    header(f"下载权重: {name}")
    try:
        import torch
        import torchvision.models as models

        builder = getattr(models, name, None)
        if builder is None:
            error(f"未知模型: {name}")
            return 1

        warning("下载中（torchvision hub）...")
        model = builder(weights="DEFAULT")

        # 保存到缓存
        cache_path = mgr.cache_dir / f"{name}.pth"
        torch.save(model.state_dict(), cache_path)
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        result("已缓存", str(cache_path))
        result("大小", f"{size_mb:.1f} MB")
        return 0
    except Exception as e:
        error(f"下载失败: {e}")
        return 1


def _cmd_info(args: argparse.Namespace, mgr: WeightManager) -> int:
    """显示权重文件的诊断信息。"""
    from cvlab.cli.console import error

    try:
        import torch
        import torchvision.models as models

        builder = getattr(models, args.name, None)
        if builder is None:
            error(f"未知模型: {args.name}")
            return 1

        header(f"权重信息: {args.name}")
        model = builder(weights=None)
        total = sum(p.numel() for p in model.parameters())
        info(f"参数量: {total / 1e6:.2f}M")
        info(f"预期权重大小: {total * 4 / 1024 / 1024:.1f} MB (FP32)")

        # 检查缓存
        cache_path = mgr.cache_dir / f"{args.name}.pth"
        if cache_path.exists():
            info(f"缓存状态: 已缓存 ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            info("缓存状态: 未缓存（使用 cvlab weights download 下载）")

        return 0
    except Exception as e:
        error(f"获取信息失败: {e}")
        return 1


def add_subparser(sub) -> None:
    p = sub.add_parser("weights", help="预训练权重管理")
    sp = p.add_subparsers(dest="weights_command")

    sp.add_parser("list", help="列出本地缓存的权重")
    sp.add_parser("info", help="显示权重信息").add_argument("name", help="模型名称")
    sp.add_parser("download", help="下载预训练权重").add_argument("name", help="模型名称")
    p.set_defaults(func=cmd_weights)
