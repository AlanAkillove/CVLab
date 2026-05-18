"""cvlab sweep - 超参扫描。"""

from __future__ import annotations

import argparse
from pathlib import Path

from cvlab.cli.console import header, info, result, table
from cvlab.config.config import load_config
from cvlab.sweep.sweeper import Sweeper


def cmd_sweep(args: argparse.Namespace) -> int:
    if args.sweep_command == "create":
        return _cmd_create(args)
    elif args.sweep_command == "analyze":
        return _cmd_analyze(args)
    else:
        from cvlab.cli.console import error
        error("请指定 sweep 子命令: create 或 analyze")
        return 1


def _cmd_create(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        from cvlab.cli.console import error
        error(f"Sweep 配置文件不存在: {config_path}")
        return 1

    header("加载 Sweep 配置")
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

    info(f"策略: {strategy}")
    info(f"参数: {list(params.keys())}")

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

    header("Trial 列表")
    trials = sweeper.get_trials(sweep_id)
    rows = [[t["trial_index"], t["experiment_id"][:16], t.get("status", "pending")]
            for t in trials]
    table("", ["Trial", "实验 ID", "状态"], rows)
    result("总计", f"{len(trials)} trials")

    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """分析 Sweep 超参重要性。"""
    from cvlab.cli.console import error, info
    from cvlab.sweep.importance import analyze_importance

    try:
        imp = analyze_importance(args.sweep_id, target_metric=args.metric)
    except ValueError as e:
        error(f"分析失败: {e}")
        return 1

    header(f"超参重要性分析: {args.sweep_id}")
    result("目标指标", imp.target_metric)
    result("分析实验数", f"{imp.total_trials}")

    header("重要性排序")
    bar_width = 30
    rows = []
    for i, (name, score) in enumerate(imp.importances.items()):
        filled = int(score * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        rows.append([str(i + 1), name[:40], bar, f"{score * 100:.0f}%"])
    table("", ["#", "超参", "重要性", "分数"], rows)

    for s in imp.suggestions:
        info(f"结论: {s}")

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("sweep", help="超参扫描")
    sp = p.add_subparsers(dest="sweep_command")

    create_p = sp.add_parser("create", help="创建并启动 Sweep")
    create_p.add_argument("--config", "-c", required=True, help="Sweep 配置文件路径")
    create_p.add_argument("--name", help="Sweep 名称")
    create_p.add_argument("--seed", type=int, help="随机种子")
    create_p.add_argument("--max-trials", type=int, help="最大 trial 数（random 模式）")

    analyze_p = sp.add_parser("analyze", help="分析 Sweep 超参重要性")
    analyze_p.add_argument("sweep_id", help="Sweep ID")
    analyze_p.add_argument("--metric", default="val/acc", help="目标指标名")

    p.set_defaults(func=cmd_sweep)
