"""cvlab compare - 多实验指标对比。

用法:
    cvlab compare exp_001 exp_002 [exp_003 ...]
    cvlab compare exp_001 exp_002 --metric val/acc
    cvlab compare exp_001 exp_002 --output table

输出 Rich 高亮对比表格，支持终端内快速查看实验差异。
"""

from __future__ import annotations

import argparse
import json

from rich import box
from rich.table import Table

from cvlab.cli.console import console, header, info, result
from cvlab.db.database import Database
from cvlab.i18n import _


def cmd_compare(args: argparse.Namespace) -> int:
    db = Database()

    if len(args.experiments) < 2:
        console.print(f"  [red][FAIL][/red] {_('至少选择 2 个实验')}")
        return 1

    # 验证所有实验存在
    experiments = []
    for eid in args.experiments:
        exp = db.get_experiment(eid)
        if not exp:
            console.print(f"  [red][FAIL][/red] {_('实验 {} 不存在').format(eid)}")
            return 1
        experiments.append(exp)

    header(_("实验对比"))

    # 基础信息对比表格
    info_table = Table(box=box.ROUNDED, header_style="bold cyan")
    info_table.add_column(_("属性"), style="bold")
    for exp in experiments:
        info_table.add_column(exp["id"][:16])

    info_table.add_row(
        _("名称"),
        *[e["name"][:24] for e in experiments],
    )
    info_table.add_row(
        _("状态"),
        *[e["status"] for e in experiments],
    )
    info_table.add_row(
        _("创建时间"),
        *[e["created_at"][:19] for e in experiments],
    )
    info_table.add_row(
        "Seed",
        *[str(e.get("seed", "—")) for e in experiments],
    )
    console.print(info_table)

    # 配置对比
    header(_("配置对比"))
    config_keys: set[str] = set()
    configs: list[dict[str, str]] = []
    for exp in experiments:
        flat = _flatten_config_json(exp.get("config_json", "{}"))
        configs.append(flat)
        config_keys.update(flat.keys())

    if config_keys:
        cfg_table = Table(box=box.ROUNDED, header_style="bold cyan")
        cfg_table.add_column(_("参数"), style="bold")
        for exp in experiments:
            cfg_table.add_column(exp["id"][:16])
        for key in sorted(config_keys):
            values = []
            for i, exp in enumerate(experiments):
                val = configs[i].get(key, "—")
                # 高亮不同的值
                if i > 0 and val != configs[0].get(key, "—"):
                    values.append(f"[yellow]{val}[/yellow]")
                else:
                    values.append(str(val))
            cfg_table.add_row(key, *values)
        console.print(cfg_table)

    # 指标对比
    if args.metric:
        metric_keys = [args.metric]
    else:
        metric_keys = sorted(_get_common_metrics(db, [e["id"] for e in experiments]))

    if metric_keys:
        header(_("指标对比"))

        metrics_table = Table(box=box.ROUNDED, header_style="bold cyan")
        metrics_table.add_column(_("指标"), style="bold")
        for exp in experiments:
            metrics_table.add_column(f"{exp['id'][:16]} ({_('最新')})")
            metrics_table.add_column(f"{exp['id'][:16]} ({_('最优')})")

        for mk in metric_keys:
            row: list[str] = [mk]
            for exp in experiments:
                all_vals = db.get_metrics(exp["id"])
                vals = [m["value"] for m in all_vals if m["key"] == mk]
                if vals:
                    last_val = f"{vals[-1]:.4f}"
                    best_val = (
                        f"{max(vals):.4f}" if "acc" in mk or "f1" in mk
                        else f"{min(vals):.4f}"
                    )
                else:
                    last_val = "—"
                    best_val = "—"
                row.append(last_val)
                row.append(best_val)
            metrics_table.add_row(*row)

        console.print(metrics_table)

    result(_("实验数"), str(len(experiments)))
    return 0


def _flatten_config_json(config_json: str) -> dict[str, str]:
    """Flatten config JSON to dot-separated key-value pairs."""
    from cvlab.core.utils import flatten_dict

    try:
        config = json.loads(config_json) if isinstance(config_json, str) else config_json
        flat = flatten_dict(config)
        return {k: str(v) for k, v in flat.items()}
    except (json.JSONDecodeError, TypeError):
        return {}


def _get_common_metrics(db: Database, exp_ids: list[str]) -> set[str]:
    """Get metric keys common to all specified experiments."""
    common: set[str] | None = None
    for eid in exp_ids:
        metrics = db.get_metrics(eid)
        keys = {m["key"] for m in metrics}
        if common is None:
            common = keys
        else:
            common &= keys
    return common or set()


def add_subparser(sub) -> None:
    p = sub.add_parser("compare", help=_("多实验指标对比"))
    p.add_argument(
        "experiments",
        nargs="+",
        help=_("实验 ID 列表（至少 2 个）"),
    )
    p.add_argument(
        "--metric", "-m",
        help=_("指定对比的指标（默认：所有共同指标）"),
    )
    p.add_argument(
        "--output",
        choices=["table"],
        default="table",
        help=_("输出格式（目前仅支持 table）"),
    )
    p.set_defaults(func=cmd_compare)
