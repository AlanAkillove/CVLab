"""CLI main entry - argparse dispatcher with i18n support."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cvlab",
        description="CVLab - CV Experiment Management Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument(
        "--lang", choices=["zh", "en"], default=None,
        help="Display language (zh/en)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed logs and tracebacks",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    from cvlab.cli.compare import add_subparser as add_compare
    from cvlab.cli.data import add_subparser as add_data
    from cvlab.cli.diagnose import add_subparser as add_diagnose
    from cvlab.cli.export import add_subparser as add_export
    from cvlab.cli.help_ import add_subparser as add_help
    from cvlab.cli.note import add_subparser as add_note
    from cvlab.cli.profile import add_subparser as add_profile
    from cvlab.cli.sweep import add_subparser as add_sweep
    from cvlab.cli.tag import add_subparser as add_tag
    from cvlab.cli.train import add_subparser as add_train
    from cvlab.cli.ui import add_subparser as add_ui
    from cvlab.cli.weights import add_subparser as add_weights

    add_train(sub)
    add_compare(sub)
    add_sweep(sub)
    add_ui(sub)
    add_profile(sub)
    add_weights(sub)
    add_diagnose(sub)
    add_data(sub)
    add_export(sub)
    add_tag(sub)
    add_note(sub)
    add_help(sub)

    # cvlab init
    init_p = sub.add_parser("init", help="Initialize CVLab in current directory")
    init_p.set_defaults(func=_cmd_init)

    # cvlab list
    list_p = sub.add_parser("list", help="List experiments")
    list_p.add_argument("--status", help="Filter by status")
    list_p.add_argument("--tag", help="Filter by tag")
    list_p.add_argument("--limit", type=int, default=20, help="Maximum count")
    list_p.set_defaults(func=_cmd_list)

    # cvlab show
    show_p = sub.add_parser("show", help="Show experiment details")
    show_p.add_argument("experiment_id", help="Experiment ID")
    show_p.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)

    # Initialize i18n from CLI arg or env
    from cvlab.i18n import init_from_args
    init_from_args(getattr(args, "lang", None))

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

    from cvlab.cli.console import console
    from cvlab.i18n import _

    cvlab_dir = Path(".cvlab")
    cvlab_dir.mkdir(exist_ok=True)

    # 生成默认配置模板
    config_path = cvlab_dir / "config.yaml"
    _generate_config_template(config_path)

    # 欢迎面板
    console.print()
    console.print("[bold cyan]" + "-" * 40 + "[/bold cyan]")
    console.print(_("[bold]  CVLab {} Ready![/bold]").format("v0.2.5"))
    console.print("[bold cyan]" + "-" * 40 + "[/bold cyan]")
    console.print()
    console.print(_("  [OK] {}").format(str(cvlab_dir.resolve())))
    console.print(_("  [OK] {}").format(str(config_path.resolve())))
    console.print()
    console.print(_("  [bold]Next steps:[/bold]"))
    console.print(_("    1. Edit config:  {}").format("edit .cvlab/config.yaml"))
    console.print(_("    2. Start training: {}").format("cvlab train --config .cvlab/config.yaml"))
    console.print(_("    3. View experiments: {}").format("cvlab list"))
    console.print(_("    4. Web UI: {}").format("cvlab ui"))
    console.print()
    console.print(_("  [dim]More: cvlab help | README.md[/dim]"))
    console.print()
    return 0


def _generate_config_template(path: Path) -> None:
    """生成默认最小可用的训练配置模板。"""
    template = """# CVLab 训练配置
# 完整配置示例见 examples/cifar10_full.yaml

model:
  name: resnet18            # torchvision 分类模型名
  pretrained: false

training:
  epochs: 10                # 训练轮数（设为 null 自动探测）
  batch_size: null          # null = 自动探测最大 batch size
  optimizer: adam           # sgd | adam | adamw
  lr: 0.001                 # 学习率
  scheduler: cosine         # cosine | step | plateau | none

data:
  dataset_name: CIFAR10     # torchvision 内置数据集
  input_size: [3, 32, 32]   # [C, H, W]
  num_workers: 2
  val_split: 0.1

seed: 42
"""
    # 只在文件不存在时创建，避免覆盖用户修改
    if not path.exists():
        path.write_text(template, encoding="utf-8")


def _cmd_list(args: argparse.Namespace) -> int:
    import json

    from cvlab.cli.console import console, header, info, result
    from cvlab.db.database import Database
    from cvlab.i18n import _

    db = Database()
    exps = db.list_experiments(status=args.status, tag=args.tag, limit=args.limit)

    if not exps:
        info(_("暂无实验"))
        return 0

    header(_("实验列表"))
    # 每行一条实验，简单对齐格式（避免 Rich 在窄终端上截断 ID）
    for e in exps:
        cfg = {}
        from contextlib import suppress
        with suppress(json.JSONDecodeError, TypeError):
            cfg = json.loads(e.get("config_json", "{}")) if isinstance(e.get("config_json"), str) else e.get("config_json", {})
        model = cfg.get("model", {}).get("name", "")[:12] if isinstance(cfg, dict) else ""
        status_icon = {
            "completed": "[green]*[/green]",
            "failed": "[red]![/red]",
            "running": "[yellow]~[/yellow]",
            "created": "[dim]-[/dim]",
        }.get(e["status"], "?")
        console.print(f"  {status_icon}  {e['id']}  [dim]{e['name'][:24]}[/dim]  [cyan]{model}[/cyan]")
    result(_("总计"), f"{len(exps)} {_('实验')}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    import json

    from cvlab.cli.console import console, header, info, panel, result
    from cvlab.db.database import Database
    from cvlab.i18n import _

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        console.print(f"  [red][FAIL][/red] {_('实验 {} 不存在').format(args.experiment_id)}")
        return 1

    header(f"{exp.get('name', '?')[:40]}")
    result(_("实验 ID"), args.experiment_id)
    result(_("状态"), exp["status"])
    result(_("创建时间"), exp["created_at"])
    result("Seed", str(exp.get("seed", "N/A")))

    if exp.get("failure_reason"):
        console.print(f"  [red]{_('失败原因')}: {exp['failure_reason']}[/red]")

    if exp.get("config_json"):
        try:
            config_raw = json.loads(exp["config_json"])
            config_str = json.dumps(config_raw, indent=2, default=str)
            panel(config_str, title=_("超参配置"))
        except json.JSONDecodeError:
            info(f"{_('配置')}: {exp['config_json'][:200]}...")

    if exp.get("command"):
        info(f"{_('命令')}: {exp['command']}")

    # Show metrics summary
    metrics = db.get_metrics(args.experiment_id)
    if metrics:
        header(_("指标"))
        latest: dict[str, float] = {}
        for m in metrics:
            latest[m["key"]] = m["value"]
        for key, val in latest.items():
            result(key, f"{val:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
