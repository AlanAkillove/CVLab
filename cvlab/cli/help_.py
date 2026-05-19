"""cvlab help - 命令帮助概览。"""

from __future__ import annotations

import argparse

from cvlab.cli.console import console, header, table
from cvlab.i18n import _


def cmd_help(args: argparse.Namespace) -> int:
    """显示格式化的命令帮助概览。"""
    header(_("CVLab 命令概览"))

    commands = [
        ("train", _("训练模型"), "cvlab train --config <config.yaml> [--resume <exp_id>]"),
        ("sweep", _("超参扫描"), "cvlab sweep create --config <sweep.yaml>\ncvlab sweep analyze <sweep_id>"),
        ("diagnose", _("训练诊断"), "cvlab diagnose loss <exp_id>\ncvlab diagnose gradient <exp_id>\ncvlab diagnose dataloader <config.yaml>\ncvlab diagnose lr-loss <exp_id>\ncvlab diagnose experiment <exp_id>"),
        ("data", _("数据集管理"), "cvlab data analyze <path>\ncvlab data augment <image>\ncvlab data check <path>\ncvlab data history"),
        ("profile", _("模型画像"), "cvlab profile --model <name> [--device <cpu|cuda>]"),
        ("weights", _("权重管理"), "cvlab weights check <model>\ncvlab weights list"),
        ("list", _("列出实验"), "cvlab list [--status <status>] [--tag <tag>] [--limit <n>]"),
        ("show", _("查看实验详情"), "cvlab show <experiment_id>"),
        ("init", _("初始化 CVLab"), "cvlab init"),
        ("help", _("显示此帮助"), "cvlab help [command]"),
    ]

    rows = [[cmd, desc[:32], usage.replace("\n", "\n     ")]
            for cmd, desc, usage in commands]
    table("", [_("命令"), _("说明"), _("用法")], rows)

    console.print(f"\n  [dim]{_('更多信息: cvlab <command> --help')}[/dim]")

    # 如果指定了具体命令，显示详细帮助
    if args.command_name:
        console.print()
        header(_("详细帮助: {}").format(args.command_name))
        console.print(f"  [dim]{_('运行 cvlab {} --help 查看完整选项').format(args.command_name)}[/dim]")

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("help", help=_("显示命令帮助概览"))
    p.add_argument("command_name", nargs="?", help=_("具体命令名称"))
    p.set_defaults(func=cmd_help)
