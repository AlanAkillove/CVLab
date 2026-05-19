"""cvlab diagnose - 训练诊断工具。"""

from __future__ import annotations

import argparse

from cvlab.i18n import _
from cvlab.cli.console import console, error, header, info, result, table, warning


def cmd_diagnose(args: argparse.Namespace) -> int:
    if args.diag_command == "loss":
        return _diag_loss(args)
    elif args.diag_command == "dataloader":
        return _diag_dataloader(args)
    elif args.diag_command == "gradient":
        return _diag_gradient(args)
    elif args.diag_command == "lr-loss":
        return _diag_lr_loss(args)
    elif args.diag_command == "experiment":
        return _diag_experiment(args)
    else:
        error(_("未知诊断命令: {}").format(args.diag_command))
        return 1


def _diag_loss(args: argparse.Namespace) -> int:
    """分析实验的 loss 指标。"""
    from cvlab.db.database import Database
    from cvlab.diagnose.loss import LossDetector

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(_("实验 {} 不存在").format(args.experiment_id))
        return 1

    header(_("Loss 分析: {}").format(args.experiment_id))

    metrics = db.get_metrics(args.experiment_id, keys=["train/loss", "val/loss"])
    if not metrics:
        info(_("无 loss 指标数据"))
        return 0

    train_losses = [(m["step"], m["value"]) for m in metrics if m["key"] == "train/loss"]
    val_losses = [(m["step"], m["value"]) for m in metrics if m["key"] == "val/loss"]

    if not train_losses and not val_losses:
        info(_("无 loss 指标数据"))
        return 0

    total_anomalies = 0
    for label, loss_seq in [("train", train_losses), ("val", val_losses)]:
        if not loss_seq:
            continue
        header(_("{}/loss 序列 ({} 个点)").format(label, len(loss_seq)))
        detector = LossDetector(window_size=min(10, len(loss_seq)))
        anomalies = 0
        for step, loss in loss_seq:
            report = detector.step(loss, step)
            if report.has_anomaly:
                anomalies += 1
                for w in report.warnings:
                    warning(w)
                for s in report.suggestions:
                    info(_("  建议: {}").format(s))

        if anomalies == 0:
            info(_("未检测到异常"))
        total_anomalies += anomalies

    if total_anomalies > 0:
        info(_("共检测到 {} 个异常").format(total_anomalies))
    else:
        info(_("loss 曲线表现正常"))
    return 0


def _diag_dataloader(args: argparse.Namespace) -> int:
    """诊断 DataLoader 性能。"""
    from pathlib import Path

    from cvlab.config.config import load_config
    from cvlab.diagnose.io_bottleneck import DataLoaderProfiler
    from cvlab.train.run import _load_data

    config_path = Path(args.config)
    if not config_path.exists():
        error(_("配置文件不存在: {}").format(config_path))
        return 1

    header(_("DataLoader 性能诊断"))
    config = load_config(str(config_path))
    info(_("加载数据集..."))
    try:
        train_loader, val_loader, class_names = _load_data(config)
    except Exception as e:
        error(_("数据加载失败: {}").format(e))
        return 1

    header(_("训练集 DataLoader"))
    info(_("num_workers={}, batch_size={}, pin_memory={}").format(
        train_loader.num_workers, train_loader.batch_size, train_loader.pin_memory))
    info(_("总样本数: {}, batches: {}").format(len(train_loader.dataset), len(train_loader)))

    profiler = DataLoaderProfiler()
    train_stats = profiler.profile_dataloader(train_loader, num_batches=args.num_batches)

    table(_("训练集加载性能"), [_("指标"), _("值")], [
        [_("采样批次数"), str(train_stats["batches"])],
        [_("平均加载时间"), _("{} ms").format(f"{train_stats['mean']:.2f}")],
        [_("标准差"), _("{} ms").format(f"{train_stats['std']:.2f}")],
        [_("最慢"), _("{} ms").format(f"{train_stats['max']:.2f}")],
        [_("最快"), _("{} ms").format(f"{train_stats['min']:.2f}")],
    ])

    avg = train_stats["mean"]
    if avg > 50:
        warning(_("数据加载偏慢 ({}ms/batch)").format(f"{avg:.0f}"))
        info(_("建议增加 num_workers 或启用 pin_memory"))
    elif avg < 10:
        info(_("数据加载速度良好"))

    header(_("验证集 DataLoader"))
    val_stats = profiler.profile_dataloader(val_loader, num_batches=args.num_batches)
    table(_("验证集加载性能"), [_("指标"), _("值")], [
        [_("采样批次数"), str(val_stats["batches"])],
        [_("平均加载时间"), _("{} ms").format(f"{val_stats['mean']:.2f}")],
        [_("标准差"), _("{} ms").format(f"{val_stats['std']:.2f}")],
        [_("最慢"), _("{} ms").format(f"{val_stats['max']:.2f}")],
        [_("最快"), _("{} ms").format(f"{val_stats['min']:.2f}")],
    ])
    return 0


def _diag_gradient(args: argparse.Namespace) -> int:
    """分析实验的梯度健康状态。"""
    from cvlab.db.database import Database
    from cvlab.diagnose.gradient import GradientDiagnosis

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(_("实验 {} 不存在").format(args.experiment_id))
        return 1

    header(_("梯度诊断: {}").format(args.experiment_id))
    diagnosis = GradientDiagnosis().diagnose(args.experiment_id, db)
    icons = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
    result(_("总体状态"), f"{icons.get(diagnosis.overall_status, '?')} {diagnosis.overall_status}")

    if diagnosis.total_layers == 0:
        info(diagnosis.suggestions[0] if diagnosis.suggestions else _("无梯度数据"))
        return 0

    result(_("监控层数"), str(diagnosis.total_layers))
    result(_("健康"), str(diagnosis.healthy_count))
    result(_("警告"), str(diagnosis.warning_count))
    result(_("严重"), str(diagnosis.critical_count))

    if diagnosis.layer_statuses:
        header(_("逐层梯度"))
        rows = []
        for s in diagnosis.layer_statuses:
            icon = {"vanishing": "❌", "exploding": "❌", "low": "⚠️", "high": "⚠️", "normal": "✅"}
            rows.append([icon.get(s.status, "?"), s.name[:40], f"{s.mean_norm:.6f}", s.status])
        table("", ["", _("层"), _("平均范数"), _("状态")], rows)

    for s in diagnosis.suggestions:
        info(_("建议: {}").format(s))
    return 0


def _diag_lr_loss(args: argparse.Namespace) -> int:
    """学习率与 Loss 联动分析。"""
    from cvlab.db.database import Database

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(_("实验 {} 不存在").format(args.experiment_id))
        return 1

    header(_("LR-Loss 联动分析: {}").format(args.experiment_id))

    metrics = db.get_metrics(args.experiment_id, keys=["lr", "train/loss", "val/loss"])
    if not metrics:
        info(_("无相关指标数据（需要 lr + loss 指标）"))
        return 0

    lr_points = {m["step"]: m["value"] for m in metrics if m["key"] == "lr"}
    train_loss = {m["step"]: m["value"] for m in metrics if m["key"] == "train/loss"}
    val_loss = {m["step"]: m["value"] for m in metrics if m["key"] == "val/loss"}

    if not lr_points:
        info(_("无学习率记录（Tracker 会自动记录 lr 指标）"))
        return 0

    steps = sorted(set(lr_points.keys()) | set(train_loss.keys()) | set(val_loss.keys()))
    info(_("共 {} 个步点，{} 个 lr 变更点").format(len(steps), len(lr_points)))

    # 检测 lr 调度阶段
    lr_values = sorted(set(lr_points.values()))
    result(_("LR 范围"), f"{min(lr_values):.2e} ~ {max(lr_values):.2e}")
    result(_("LR 阶段数"), str(len(lr_values)))

    # 检测 lr 下降后 loss 响应
    if len(lr_values) >= 2 and len(train_loss) >= 3:
        header(_("LR 下降响应分析"))
        for step in sorted(lr_points.keys()):
            current_lr = lr_points[step]
            # 找这个 step 附近的 loss
            if step in train_loss:
                loss_before = train_loss[step]
                # 看后续 2 步的 loss
                future = [train_loss[s] for s in sorted(train_loss.keys()) if s >= step][:3]
                if len(future) >= 2:
                    loss_after = future[-1]
                    if loss_after < loss_before * 0.95:
                        info(_("  Step {}: lr→{}, loss {}→{} ✅ 有效").format(
                            step, f"{current_lr:.2e}", f"{loss_before:.4f}", f"{loss_after:.4f}"))
                    elif loss_after > loss_before * 1.05:
                        warning(_("  Step {}: lr→{}, loss {}→{} ⚠️ loss 反弹").format(
                            step, f"{current_lr:.2e}", f"{loss_before:.4f}", f"{loss_after:.4f}"))
                    else:
                        info(_("  Step {}: lr→{}, loss {}→{} → 平稳").format(
                            step, f"{current_lr:.2e}", f"{loss_before:.4f}", f"{loss_after:.4f}"))

    # 检测是否过拟合 (val loss 上升但 train loss 下降)
    if len(val_loss) >= 5 and len(train_loss) >= 5:
        val_steps_sorted = sorted(val_loss.keys())
        train_steps_sorted = sorted(train_loss.keys())
        recent_val = [val_loss[s] for s in val_steps_sorted[-3:]]
        recent_train = [train_loss[s] for s in train_steps_sorted[-3:]]
        if all(recent_val[i] > recent_val[0] for i in range(1, len(recent_val))) and \
           all(recent_train[i] < recent_train[0] for i in range(1, len(recent_train))):
            warning(_("过拟合迹象: val loss 持续上升但 train loss 持续下降"))
            info(_("建议: 增加正则化、增强数据增强、早停"))

    return 0


def _diag_experiment(args: argparse.Namespace) -> int:
    """全面诊断实验。"""
    from cvlab.db.database import Database
    from cvlab.diagnose.loss import LossDetector

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(_("实验 {} 不存在").format(args.experiment_id))
        return 1

    header(_("全面诊断: {}").format(args.experiment_id))
    result(_("名称"), exp["name"])
    result(_("状态"), exp["status"])

    # Loss 分析
    metrics = db.get_metrics(args.experiment_id, keys=["train/loss", "val/loss"])
    if metrics:
        train_losses = [(m["step"], m["value"]) for m in metrics if m["key"] == "train/loss"]
        if train_losses:
            detector = LossDetector(window_size=min(10, len(train_losses)))
            anomalies = 0
            for step, loss in train_losses:
                report = detector.step(loss, step)
                if report.has_anomaly:
                    anomalies += 1
                    for w in report.warnings:
                        warning(w)
            info(_("Loss 分析: {} 个点, {} 个异常").format(len(train_losses), anomalies) if anomalies else
                 _("Loss 分析: {} 个点, 无异常").format(len(train_losses)))

    # 梯度分析
    from cvlab.diagnose.gradient import GradientDiagnosis
    grad = GradientDiagnosis().diagnose(args.experiment_id, db)
    if grad.total_layers > 0:
        icons = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
        info(_("梯度: {} {} ({}健康/{}警告/{}严重)").format(
            icons.get(grad.overall_status, '?'), grad.overall_status,
            grad.healthy_count, grad.warning_count, grad.critical_count))

    # LR-Loss 联动
    lr_points = {m["step"]: m["value"] for m in metrics if m["key"] == "lr"} if metrics else {}
    if lr_points:
        info(_("LR 阶段: {} 个, 范围 {}~{}").format(
            len(set(lr_points.values())),
            f"{min(lr_points.values()):.2e}",
            f"{max(lr_points.values()):.2e}"))

    # 最佳指标
    best_acc = db.get_metrics(args.experiment_id, keys=["val/acc"])
    if best_acc:
        best_val = max(best_acc, key=lambda m: m["value"])
        result(_("最佳 val_acc"), _("{}% (step {})").format(f"{best_val['value']*100:.2f}", best_val['step']))

    ckpts = db.get_checkpoints(args.experiment_id)
    if ckpts:
        result(_("Checkpoints"), _("{} 个").format(len(ckpts)))
        best = next((c for c in ckpts if c["is_best"]), None)
        if best:
            result(_("最优 Checkpoint"), _("epoch {} ({})").format(best['epoch'], best.get('metric_value', 'N/A')))

    if exp.get("env_json") and exp["env_json"] != "{}":
        import json
        env = json.loads(exp["env_json"])
        if env.get("torch_version"):
            info(_("PyTorch: {}").format(env['torch_version']))
        if env.get("cuda_version"):
            info(_("CUDA: {}").format(env['cuda_version']))
        if env.get("num_gpus"):
            info(_("GPU 数: {}").format(env['num_gpus']))

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("diagnose", help=_("训练诊断工具"))
    sp = p.add_subparsers(dest="diag_command")

    sp.add_parser("loss", help=_("分析实验 loss 指标")).add_argument("experiment_id", help=_("实验 ID"))

    dl_p = sp.add_parser("dataloader", help=_("DataLoader 性能诊断"))
    dl_p.add_argument("config", help=_("配置文件路径"))
    dl_p.add_argument("--num-batches", type=int, default=50, help=_("采样批次数"))

    sp.add_parser("gradient", help=_("分析实验梯度健康")).add_argument("experiment_id", help=_("实验 ID"))

    sp.add_parser("lr-loss", help=_("学习率与 Loss 联动分析")).add_argument("experiment_id", help=_("实验 ID"))

    sp.add_parser("experiment", help=_("全面诊断实验")).add_argument("experiment_id", help=_("实验 ID"))

    p.set_defaults(func=cmd_diagnose)
