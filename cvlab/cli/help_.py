"""cvlab help - 命令帮助概览。"""

from __future__ import annotations

import argparse

from cvlab.cli.console import console, header, table


def cmd_help(args: argparse.Namespace) -> int:
    """显示格式化的命令帮助概览。"""
    header("CVLab 命令概览")

    commands = [
        ("train", "训练模型", "cvlab train --config <config.yaml> [--resume <exp_id>]"),
        ("sweep", "超参扫描", "cvlab sweep create --config <sweep.yaml>\ncvlab sweep analyze <sweep_id>"),
        ("diagnose", "训练诊断", "cvlab diagnose loss <exp_id>\ncvlab diagnose gradient <exp_id>\ncvlab diagnose dataloader <config.yaml>\ncvlab diagnose lr-loss <exp_id>\ncvlab diagnose experiment <exp_id>"),
        ("data", "数据集管理", "cvlab data analyze <path>\ncvlab data augment <image>\ncvlab data check <path>\ncvlab data history"),
        ("profile", "模型画像", "cvlab profile --model <name> [--device <cpu|cuda>]"),
        ("weights", "权重管理", "cvlab weights check <model>\ncvlab weights list"),
        ("list", "列出实验", "cvlab list [--status <status>] [--tag <tag>] [--limit <n>]"),
        ("show", "查看实验详情", "cvlab show <experiment_id>"),
        ("init", "初始化 CVLab", "cvlab init"),
        ("help", "显示此帮助", "cvlab help [command]"),
    ]

    rows = [[cmd, desc[:32], usage.replace("\n", "\n     ")]
            for cmd, desc, usage in commands]
    table("", ["命令", "说明", "用法"], rows)

    console.print("\n  [dim]更多信息: cvlab <command> --help[/dim]")

    # 如果指定了具体命令，显示详细帮助
    if args.command_name:
        console.print()
        header(f"详细帮助: {args.command_name}")
        console.print(f"  [dim]运行 cvlab {args.command_name} --help 查看完整选项[/dim]")

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("help", help="显示命令帮助概览")
    p.add_argument("command_name", nargs="?", help="具体命令名称")
    p.set_defaults(func=cmd_help)
