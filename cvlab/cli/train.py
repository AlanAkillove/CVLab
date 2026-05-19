"""cvlab train - 执行分类训练（子进程 + OOM 恢复）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cvlab.cli.console import console, error, header, info, result, warning
from cvlab.config.config import load_config, validate_config
from cvlab.core.tracker import Tracker
from cvlab.i18n import _
from cvlab.train.run import train_classification


def cmd_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        error(_("配置文件不存在: {}").format(config_path))
        return 1

    # --resume 模式：直接运行，不经过子进程
    if args.resume:
        try:
            exp_id = train_classification(str(config_path), experiment_id=args.resume)
            result(_("完成"), _("实验: {}").format(exp_id))
            return 0
        except Exception as e:
            error(_("训练失败: {}").format(e))
            import traceback
            console.print(traceback.format_exc())
            return 1

    return _run_training_subprocess(args, config_path)


def _run_training_subprocess(args: argparse.Namespace, config_path: Path) -> int:
    # 验证配置
    config = load_config(str(config_path))
    # CLI 参数覆盖配置
    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config.setdefault("training", {})["lr"] = args.lr
    if args.name is not None:
        config["name"] = args.name
    val_errors = validate_config(config)
    if val_errors:
        for e in val_errors:
            console.print(f"  [red][FAIL][/red] {e}")
        return 1

    # 在主进程中创建实验记录，获取实验 ID
    header(_("创建实验"))
    tracker = Tracker(config=config)
    exp_id = tracker.experiment_id
    result(_("实验 ID"), exp_id)

    batch_size = config.get("training", {}).get("batch_size", 64)
    max_retries = 2
    oom_reduction_factor = 0.8  # 每次 OOM batch size 减小 20%

    for attempt in range(max_retries + 1):
        if attempt == 0:
            current_bs = batch_size
        else:
            # 每次减小 20% 而非减半，避免过度激进
            from math import ceil
            current_bs = max(1, ceil(batch_size * (oom_reduction_factor ** attempt)))

        if attempt > 0:
            warning(_("OOM 恢复: 第 {} 次重试, batch_size={}").format(attempt, current_bs))
            tracker.db.update_experiment_status(
                exp_id, "running",
                failure_reason=f"OOM retry #{attempt}, batch_size={current_bs}",
            )

        # 子进程继承父进程 stdout/stderr，保证 Rich 输出正常显示
        info(_("启动训练子进程 (attempt {}/{})").format(attempt + 1, max_retries + 1))
        proc = subprocess.run(
            [
                sys.executable, "-m", "cvlab.train.subprocess_worker",
                "--experiment-id", exp_id,
                "--config", str(config_path.resolve()),
                "--batch-size", str(current_bs),
            ],
        )

        if proc.returncode == 0:
            result(_("完成"), _("实验: {}").format(exp_id))
            return 0

        # OOM 检测：exit code 137 (CUDA OOM / RuntimeError / system OOM killing)
        is_oom = proc.returncode == 137
        if is_oom and attempt < max_retries and current_bs > 1:
            continue

        # 不可恢复的错误
        if is_oom:
            msg = _("训练失败 (OOM, 已重试 {} 次)").format(max_retries)
            error(msg)
            tracker.db.update_experiment_status(
                exp_id, "failed", failure_reason="OOM after retries",
            )
        else:
            msg = _("训练失败 (exit code {})").format(proc.returncode)
            error(msg)
            tracker.db.update_experiment_status(
                exp_id, "failed", failure_reason=f"exit code {proc.returncode}",
            )
        return 1

    return 1  # 不应到达这里


def add_subparser(sub) -> None:
    p = sub.add_parser("train", help=_("执行分类训练"))
    p.add_argument("--config", "-c", required=True, help=_("配置文件路径"))
    p.add_argument("--resume", help=_("从指定实验恢复训练"))
    p.add_argument("--seed", type=int, help=_("覆盖随机种子"))
    p.add_argument("--batch-size", type=int, help=_("覆盖 Batch Size"))
    p.add_argument("--epochs", type=int, help=_("覆盖训练轮数"))
    p.add_argument("--lr", type=float, help=_("覆盖学习率"))
    p.add_argument("--name", help=_("覆盖实验名称"))
    p.set_defaults(func=cmd_train)
