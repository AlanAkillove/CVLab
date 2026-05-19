"""cvlab note - 实验备注管理。

用法:
    cvlab note <experiment_id>                # 查看备注
    cvlab note <experiment_id> "备注内容"      # 设置/更新备注
    cvlab note <experiment_id> --clear        # 清除备注
"""

from __future__ import annotations

import argparse

from cvlab.cli.console import error, info, panel, result, success
from cvlab.db.database import Database
from cvlab.i18n import _


def cmd_note(args: argparse.Namespace) -> int:
    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(_("实验 {} 不存在").format(args.experiment_id))
        return 1

    if args.clear:
        db.update_experiment(args.experiment_id, notes="")
        success(_("实验 {} 的备注已清除").format(args.experiment_id))
        return 0

    if args.text:
        # 设置备注
        db.update_experiment(args.experiment_id, notes=args.text)
        exp_name = exp.get("name", args.experiment_id)
        success(_("实验 {} 的备注已更新").format(exp_name))
        return 0

    # 查看备注
    existing_notes = exp.get("notes", "")
    if existing_notes:
        exp_name = exp.get("name", args.experiment_id)
        result(_("实验"), f"{args.experiment_id} — {exp_name}")
        panel(existing_notes, title=_("备注"))
    else:
        info(_("实验 {} 暂无备注").format(args.experiment_id))
        info(_("使用: cvlab note {} \"备注内容\"").format(args.experiment_id))

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("note", help=_("实验备注管理"))
    p.add_argument("experiment_id", help=_("实验 ID"))
    p.add_argument("text", nargs="?", default=None, help=_("备注内容（省略则查看现有备注）"))
    p.add_argument("--clear", action="store_true", help=_("清除备注"))
    p.set_defaults(func=cmd_note)
