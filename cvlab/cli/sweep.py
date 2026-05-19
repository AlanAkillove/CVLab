"""cvlab sweep - 超参扫描。"""

from __future__ import annotations

import argparse
from pathlib import Path

from cvlab.cli.console import header, info, result, table
from cvlab.config.config import load_config
from cvlab.i18n import _
from cvlab.sweep.sweeper import Sweeper


def cmd_sweep(args: argparse.Namespace) -> int:
    if args.sweep_command == "create":
        return _cmd_create(args)
    elif args.sweep_command == "analyze":
        return _cmd_analyze(args)
    elif args.sweep_command == "top":
        return _cmd_top(args)
    else:
        from cvlab.cli.console import error
        error(_("请指定 sweep 子命令: create / analyze / top"))
        return 1


def _cmd_create(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        from cvlab.cli.console import error
        error(_("Sweep 配置文件不存在: {}").format(config_path))
        return 1

    header(_("加载 Sweep 配置"))
    config = load_config(str(config_path))
    sweep_cfg = config.get("_sweep", {})

    if not sweep_cfg:
        params = config.get("params", {})
        strategy = config.get("strategy", "grid")
        base_config = config.get("base_config", config)
    else:
        params = sweep_cfg.get("params", {})
        strategy = sweep_cfg.get("strategy", "grid")
        base_config = config

    info(_("策略: {}").format(strategy))
    info(_("参数: {}").format(list(params.keys())))

    sweeper = Sweeper()
    seed = args.seed or config.get("seed")
    max_trials = args.max_trials

    sweep_id = sweeper.create_sweep(
        base_config=base_config,
        strategy=strategy,
        params=params,
        name=args.name or config.get("name"),
        seed=seed,
        max_trials=max_trials,
    )
    result("Sweep ID", sweep_id)

    header(_("Trial 列表"))
    trials = sweeper.get_trials(sweep_id)
    rows = [[t["trial_index"], t["experiment_id"][:16], t.get("status", "pending")]
            for t in trials]
    table("", ["Trial", _("实验 ID"), _("状态")], rows)
    result(_("总计"), _("{} trials").format(len(trials)))

    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """分析 Sweep 超参重要性。"""
    from cvlab.cli.console import error, info
    from cvlab.sweep.importance import analyze_importance

    try:
        imp = analyze_importance(args.sweep_id, target_metric=args.metric)
    except ValueError as e:
        error(_("分析失败: {}").format(e))
        return 1

    header(_("超参重要性分析: {}").format(args.sweep_id))
    result(_("目标指标"), imp.target_metric)
    result(_("分析实验数"), str(imp.total_trials))

    header(_("重要性排序"))
    bar_width = 30
    rows = []
    for i, (name, score) in enumerate(imp.importances.items()):
        filled = int(score * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        rows.append([str(i + 1), name[:40], bar, f"{score * 100:.0f}%"])
    table("", ["#", _("超参"), _("重要性"), _("分数")], rows)

    for s in imp.suggestions:
        info(_("结论: {}").format(s))

    return 0


def _cmd_top(args: argparse.Namespace) -> int:
    """显示 Top N trial。"""
    from rich import box
    from rich.table import Table

    from cvlab.cli.console import console, header, info, result

    sweeper = Sweeper()
    top = sweeper.get_top_trials(args.sweep_id, metric_key=args.metric, n=args.n)

    if not top:
        info(_("没有完成的 trial 或指标数据"))
        info(_("提示: 确保有 status=completed 的 trial 且记录了指标准备"))
        return 0

    header(_("Top {} Trials: {}").format(args.n, args.sweep_id))
    result(_("目标指标"), args.metric)

    t = Table(box=box.ROUNDED, header_style="bold cyan")
    t.add_column(_("排名"), justify="right")
    t.add_column("Trial")
    t.add_column(_("实验 ID"))
    t.add_column(args.metric, justify="right")
    t.add_column(_("状态"))

    maximize = "acc" in args.metric or "f1" in args.metric
    for i, tr in enumerate(top):
        val = tr["metric_value"]
        val_str = f"{val:.4f}"
        rank_str = f"[bold]{'#' + str(i + 1)}[/bold]"
        t.add_row(
            rank_str,
            str(tr["trial_index"]),
            tr["experiment_id"][:24],
            f"[green]{val_str}[/green]" if (i == 0) else val_str,
            "completed",
        )
    console.print(t)
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("sweep", help=_("超参扫描"))
    sp = p.add_subparsers(dest="sweep_command")

    create_p = sp.add_parser("create", help=_("创建并启动 Sweep"))
    create_p.add_argument("--config", "-c", required=True, help=_("Sweep 配置文件路径"))
    create_p.add_argument("--name", help=_("Sweep 名称"))
    create_p.add_argument("--seed", type=int, help=_("随机种子"))
    create_p.add_argument("--max-trials", type=int, help=_("最大 trial 数（random 模式）"))

    analyze_p = sp.add_parser("analyze", help=_("分析 Sweep 超参重要性"))
    analyze_p.add_argument("sweep_id", help="Sweep ID")
    analyze_p.add_argument("--metric", default="val/acc", help=_("目标指标名"))

    top_p = sp.add_parser("top", help=_("显示 Top N Trial"))
    top_p.add_argument("sweep_id", help="Sweep ID")
    top_p.add_argument("--metric", default="val/acc", help=_("目标指标名"))
    top_p.add_argument("-n", type=int, default=5, help=_("返回条数 (默认 5)"))

    p.set_defaults(func=cmd_sweep)
