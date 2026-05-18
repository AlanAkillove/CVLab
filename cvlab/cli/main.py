"""CLI 主入口 - argparse 分发器。"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cvlab",
        description="CVLab - CV 实验管理平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="显示版本号")

    sub = parser.add_subparsers(dest="command", help="可用命令")

    # 注入各命令的子解析器
    from cvlab.cli.train import add_subparser as add_train
    from cvlab.cli.sweep import add_subparser as add_sweep
    from cvlab.cli.profile import add_subparser as add_profile
    from cvlab.cli.weights import add_subparser as add_weights
    from cvlab.cli.diagnose import add_subparser as add_diagnose
    from cvlab.cli.data import add_subparser as add_data
    from cvlab.cli.help_ import add_subparser as add_help

    add_train(sub)
    add_sweep(sub)
    add_profile(sub)
    add_weights(sub)
    add_diagnose(sub)
    add_data(sub)
    add_help(sub)

    # cvlab init
    init_p = sub.add_parser("init", help="在当前目录初始化 CVLab")
    init_p.set_defaults(func=_cmd_init)

    # cvlab list
    list_p = sub.add_parser("list", help="列出实验")
    list_p.add_argument("--status", help="按状态筛选")
    list_p.add_argument("--tag", help="按标签筛选")
    list_p.add_argument("--limit", type=int, default=20, help="最大条数")
    list_p.set_defaults(func=_cmd_list)

    # cvlab show
    show_p = sub.add_parser("show", help="查看实验详情")
    show_p.add_argument("experiment_id", help="实验 ID")
    show_p.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)

    if args.version:
        from cvlab import __version__
        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


def _cmd_init(args: argparse.Namespace) -> int:
    from pathlib import Path
    from cvlab.cli.console import console, result

    cvlab_dir = Path(".cvlab")
    cvlab_dir.mkdir(exist_ok=True)
    result("初始化完成", str(cvlab_dir.resolve()))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from cvlab.cli.console import header, info, result, table
    from cvlab.db.database import Database

    db = Database()
    exps = db.list_experiments(status=args.status, tag=args.tag, limit=args.limit)

    if not exps:
        info("暂无实验")
        return 0

    header("实验列表")
    rows = [[e["id"][:20], e["name"][:20], e["status"], e["created_at"][:19]]
            for e in exps]
    table("", ["ID", "名称", "状态", "创建时间"], rows)
    result("总计", f"{len(exps)} 实验")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    import json
    from cvlab.cli.console import console, header, info, panel, result
    from cvlab.db.database import Database

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        console.print(f"  [red][FAIL][/red] 实验 {args.experiment_id} 不存在")
        return 1

    header(f"实验 {args.experiment_id}")
    result("名称", exp["name"])
    result("状态", exp["status"])
    result("创建时间", exp["created_at"])
    result("Seed", str(exp.get("seed", "N/A")))

    if exp.get("failure_reason"):
        console.print(f"  [red]失败原因: {exp['failure_reason']}[/red]")

    if exp.get("config_json"):
        try:
            config = json.loads(exp["config_json"])
            panel("", title="超参配置")
            console.print(json.dumps(config, indent=2, default=str))
        except json.JSONDecodeError:
            info(f"配置: {exp['config_json'][:200]}...")

    if exp.get("command"):
        info(f"命令: {exp['command']}")

    # 显示指标汇总
    metrics = db.get_metrics(args.experiment_id)
    if metrics:
        header("指标")
        latest: dict[str, float] = {}
        for m in metrics:
            latest[m["key"]] = m["value"]
        for key, val in latest.items():
            result(key, f"{val:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
