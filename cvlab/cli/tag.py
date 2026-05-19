"""cvlab tag - 实验标签管理。

用法:
    cvlab tag add <experiment_id> <tag>        # 添加标签
    cvlab tag remove <experiment_id> <tag>     # 移除标签
    cvlab tag list [<experiment_id>]            # 列出标签（可指定实验）
    cvlab tag search <tag>                      # 按标签搜索实验
"""

from __future__ import annotations

import argparse

from rich import box
from rich.table import Table

from cvlab.cli.console import console, error, header, info, result, success
from cvlab.db.database import Database
from cvlab.i18n import _


def cmd_tag(args: argparse.Namespace) -> int:
    db = Database()

    if args.action == "add":
        return _tag_add(db, args)
    elif args.action == "remove":
        return _tag_remove(db, args)
    elif args.action == "list":
        return _tag_list(db, args)
    elif args.action == "search":
        return _tag_search(db, args)
    else:
        error(_("未知操作: {}").format(args.action))
        return 1


def _tag_add(db: Database, args: argparse.Namespace) -> int:
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(_("实验 {} 不存在").format(args.experiment_id))
        return 1
    db.add_tag(args.experiment_id, args.tag)
    success(_("标签 '{}' 已添加到实验 {}").format(args.tag, args.experiment_id))
    return 0


def _tag_remove(db: Database, args: argparse.Namespace) -> int:
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(_("实验 {} 不存在").format(args.experiment_id))
        return 1
    db.remove_tag(args.experiment_id, args.tag)
    success(_("标签 '{}' 已从实验 {} 移除").format(args.tag, args.experiment_id))
    return 0


def _tag_list(db: Database, args: argparse.Namespace) -> int:
    if args.experiment_id:
        exp = db.get_experiment(args.experiment_id)
        if not exp:
            error(_("实验 {} 不存在").format(args.experiment_id))
            return 1
        tags = db.get_tags(args.experiment_id)
        if not tags:
            info(_("实验 {} 无标签").format(args.experiment_id))
            return 0
        header(_("实验 {} 的标签").format(args.experiment_id))
        for t in tags:
            console.print(f"  [cyan]#{t}[/cyan]")
        result(_("总计"), str(len(tags)))
    else:
        # 列出所有实验及其标签
        exps = db.list_experiments(limit=100)
        if not exps:
            info(_("暂无实验"))
            return 0
        table = Table(box=box.ROUNDED, header_style="bold cyan")
        table.add_column(_("实验 ID"))
        table.add_column(_("标签"))
        has_tags = False
        for exp in exps:
            tags = db.get_tags(exp["id"])
            if tags:
                has_tags = True
                table.add_row(exp["id"][:20], ", ".join(f"#{t}" for t in tags))
        if has_tags:
            console.print(table)
        else:
            info(_("暂无带标签的实验"))
    return 0


def _tag_search(db: Database, args: argparse.Namespace) -> int:
    exps = db.list_experiments(tag=args.tag, limit=100)
    if not exps:
        info(_("未找到带标签 '{}' 的实验").format(args.tag))
        return 0
    header(_("标签 '{}' 的实验列表").format(args.tag))
    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column(_("实验 ID"))
    table.add_column(_("名称"))
    table.add_column(_("状态"))
    table.add_column(_("创建时间"))
    for exp in exps:
        table.add_row(
            exp["id"][:20],
            exp["name"][:24],
            exp["status"],
            exp["created_at"][:19] if exp.get("created_at") else "",
        )
    console.print(table)
    result(_("总计"), str(len(exps)))
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("tag", help=_("实验标签管理"))
    subp = p.add_subparsers(dest="action", help=_("操作"))

    # tag add
    add_p = subp.add_parser("add", help=_("添加标签"))
    add_p.add_argument("experiment_id", help=_("实验 ID"))
    add_p.add_argument("tag", help=_("标签名"))
    add_p.set_defaults(func=cmd_tag)

    # tag remove
    rm_p = subp.add_parser("remove", help=_("移除标签"))
    rm_p.add_argument("experiment_id", help=_("实验 ID"))
    rm_p.add_argument("tag", help=_("标签名"))
    rm_p.set_defaults(func=cmd_tag)

    # tag list
    list_p = subp.add_parser("list", help=_("列出标签"))
    list_p.add_argument("experiment_id", nargs="?", default=None, help=_("实验 ID（可选）"))
    list_p.set_defaults(func=cmd_tag)

    # tag search
    search_p = subp.add_parser("search", help=_("按标签搜索实验"))
    search_p.add_argument("tag", help=_("标签名"))
    search_p.set_defaults(func=cmd_tag)
