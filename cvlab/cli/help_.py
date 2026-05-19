"""cvlab help - 命令帮助概览（分组显示）。"""

from __future__ import annotations

import argparse

from rich import box
from rich.table import Table

from cvlab.cli.console import console, header
from cvlab.i18n import _


def cmd_help(args: argparse.Namespace) -> int:
    """显示分组的命令帮助概览。"""
    if args.command_name:
        header(_("详细帮助: {}").format(args.command_name))
        console.print(f"  [dim]{_('运行 cvlab {} --help 查看完整选项').format(args.command_name)}[/dim]")
        return 0

    _show_grouped_help()
    return 0


def _show_grouped_help() -> None:
    """显示分组帮助信息。"""
    console.print()
    console.print(_("[bold]CVLab 命令分组[/bold]"))
    console.print()

    groups = [
        (_("训练"), [
            ("train", _("执行分类训练")),
            ("sweep", _("超参扫描")),
        ]),
        (_("诊断与对比"), [
            ("diagnose", _("训练诊断 (loss/梯度/IO)")),
            ("compare", _("多实验指标对比")),
            ("profile", _("模型性能画像")),
        ]),
        (_("数据"), [
            ("data", _("数据集分析/增强/血缘")),
            ("export", _("导出模型 (ONNX/TorchScript)")),
        ]),
        (_("管理"), [
            ("init", _("初始化 CVLab")),
            ("list", _("列出实验")),
            ("show", _("查看实验详情")),
            ("tag", _("实验标签管理")),
            ("note", _("实验备注")),
            ("weights", _("预训练权重管理")),
            ("ui", _("启动 Web 界面")),
        ]),
    ]

    for group_name, cmds in groups:
        table = Table(box=box.SIMPLE, header_style="bold cyan", show_header=False)
        table.add_column(_("命令"), style="bold")
        table.add_column(_("说明"))
        for cmd, desc in cmds:
            table.add_row(f"  cvlab {cmd}", desc)
        console.print(f"[bold]{group_name}[/bold]")
        console.print(table)
        console.print()

    console.print(_("[dim]查看命令详情: cvlab <command> --help[/dim]"))
    console.print(_("[dim]完整文档:  README.md | USAGE.md[/dim]"))


def add_subparser(sub) -> None:
    p = sub.add_parser("help", help=_("显示命令帮助概览"))
    p.add_argument("command_name", nargs="?", help=_("具体命令名称"))
    p.set_defaults(func=cmd_help)
