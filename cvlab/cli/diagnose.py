"""cvlab diagnose - 训练诊断工具。"""

from __future__ import annotations

import argparse

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
        error(f"未知诊断命令: {args.diag_command}")
        return 1


def _diag_loss(args: argparse.Namespace) -> int:
    """分析实验的 loss 指标。"""
    from cvlab.db.database import Database
    from cvlab.diagnose.loss import LossDetector

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(f"实验 {args.experiment_id} 不存在")
        return 1

    header(f"Loss 分析: {args.experiment_id}")

    metrics = db.get_metrics(args.experiment_id, keys=["train/loss", "val/loss"])
    if not metrics:
        info("无 loss 指标数据")
        return 0

    train_losses = [(m["step"], m["value"]) for m in metrics if m["key"] == "train/loss"]
    val_losses = [(m["step"], m["value"]) for m in metrics if m["key"] == "val/loss"]

    if not train_losses and not val_losses:
        info("无 loss 指标数据")
        return 0

    total_anomalies = 0
    for label, loss_seq in [("train", train_losses), ("val", val_losses)]:
        if not loss_seq:
            continue
        header(f"{label}/loss 序列 ({len(loss_seq)} 个点)")
        detector = LossDetector(window_size=min(10, len(loss_seq)))
        anomalies = 0
        for step, loss in loss_seq:
            report = detector.step(loss, step)
            if report.has_anomaly:
                anomalies += 1
                for w in report.warnings:
                    warning(w)
                for s in report.suggestions:
                    info(f"  建议: {s}")

        if anomalies == 0:
            info("未检测到异常")
        total_anomalies += anomalies

    if total_anomalies > 0:
        info(f"共检测到 {total_anomalies} 个异常")
    else:
        info("loss 曲线表现正常")
    return 0


def _diag_dataloader(args: argparse.Namespace) -> int:
    """诊断 DataLoader 性能。"""
    from pathlib import Path

    from cvlab.config.config import load_config
    from cvlab.diagnose.io_bottleneck import DataLoaderProfiler
    from cvlab.train.run import _load_data

    config_path = Path(args.config)
    if not config_path.exists():
        error(f"配置文件不存在: {config_path}")
        return 1

    header("DataLoader 性能诊断")
    config = load_config(str(config_path))
    info(f"加载数据集...")
    try:
        train_loader, val_loader, class_names = _load_data(config)
    except Exception as e:
        error(f"数据加载失败: {e}")
        return 1

    header("训练集 DataLoader")
    info(f"num_workers={train_loader.num_workers}, "
         f"batch_size={train_loader.batch_size}, "
         f"pin_memory={train_loader.pin_memory}")
    info(f"总样本数: {len(train_loader.dataset)}, batches: {len(train_loader)}")

    profiler = DataLoaderProfiler()
    train_stats = profiler.profile_dataloader(train_loader, num_batches=args.num_batches)

    table("训练集加载性能", ["指标", "值"], [
        ["采样批次数", str(train_stats["batches"])],
        ["平均加载时间", f"{train_stats['mean']:.2f} ms"],
        ["标准差", f"{train_stats['std']:.2f} ms"],
        ["最慢", f"{train_stats['max']:.2f} ms"],
        ["最快", f"{train_stats['min']:.2f} ms"],
    ])

    avg = train_stats["mean"]
    if avg > 50:
        warning(f"数据加载偏慢 ({avg:.0f}ms/batch)")
        info("建议增加 num_workers 或启用 pin_memory")
    elif avg < 10:
        info("数据加载速度良好")

    header("验证集 DataLoader")
    val_stats = profiler.profile_dataloader(val_loader, num_batches=args.num_batches)
    table("验证集加载性能", ["指标", "值"], [
        ["采样批次数", str(val_stats["batches"])],
        ["平均加载时间", f"{val_stats['mean']:.2f} ms"],
        ["标准差", f"{val_stats['std']:.2f} ms"],
        ["最慢", f"{val_stats['max']:.2f} ms"],
        ["最快", f"{val_stats['min']:.2f} ms"],
    ])
    return 0


def _diag_gradient(args: argparse.Namespace) -> int:
    """分析实验的梯度健康状态。"""
    from cvlab.db.database import Database
    from cvlab.diagnose.gradient import GradientDiagnosis

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(f"实验 {args.experiment_id} 不存在")
        return 1

    header(f"梯度诊断: {args.experiment_id}")
    diagnosis = GradientDiagnosis().diagnose(args.experiment_id, db)
    icons = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
    result("总体状态", f"{icons.get(diagnosis.overall_status, '?')} {diagnosis.overall_status}")

    if diagnosis.total_layers == 0:
        info(diagnosis.suggestions[0] if diagnosis.suggestions else "无梯度数据")
        return 0

    result("监控层数", str(diagnosis.total_layers))
    result("健康", str(diagnosis.healthy_count))
    result("警告", str(diagnosis.warning_count))
    result("严重", str(diagnosis.critical_count))

    if diagnosis.layer_statuses:
        header("逐层梯度")
        rows = []
        for s in diagnosis.layer_statuses:
            icon = {"vanishing": "❌", "exploding": "❌", "low": "⚠️", "high": "⚠️", "normal": "✅"}
            rows.append([icon.get(s.status, "?"), s.name[:40], f"{s.mean_norm:.6f}", s.status])
        table("", ["", "层", "平均范数", "状态"], rows)

    for s in diagnosis.suggestions:
        info(f"建议: {s}")
    return 0


def _diag_lr_loss(args: argparse.Namespace) -> int:
    """学习率与 Loss 联动分析。"""
    from cvlab.db.database import Database

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(f"实验 {args.experiment_id} 不存在")
        return 1

    header(f"LR-Loss 联动分析: {args.experiment_id}")

    metrics = db.get_metrics(args.experiment_id, keys=["lr", "train/loss", "val/loss"])
    if not metrics:
        info("无相关指标数据（需要 lr + loss 指标）")
        return 0

    lr_points = {m["step"]: m["value"] for m in metrics if m["key"] == "lr"}
    train_loss = {m["step"]: m["value"] for m in metrics if m["key"] == "train/loss"}
    val_loss = {m["step"]: m["value"] for m in metrics if m["key"] == "val/loss"}

    if not lr_points:
        info("无学习率记录（Tracker 会自动记录 lr 指标）")
        return 0

    steps = sorted(set(lr_points.keys()) | set(train_loss.keys()) | set(val_loss.keys()))
    info(f"共 {len(steps)} 个步点，{len(lr_points)} 个 lr 变更点")

    # 检测 lr 调度阶段
    lr_values = sorted(set(lr_points.values()))
    result("LR 范围", f"{min(lr_values):.2e} ~ {max(lr_values):.2e}")
    result("LR 阶段数", str(len(lr_values)))

    # 检测 lr 下降后 loss 响应
    if len(lr_values) >= 2 and len(train_loss) >= 3:
        header("LR 下降响应分析")
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
                        info(f"  Step {step}: lr→{current_lr:.2e}, loss {loss_before:.4f}→{loss_after:.4f} ✅ 有效")
                    elif loss_after > loss_before * 1.05:
                        warning(f"  Step {step}: lr→{current_lr:.2e}, loss {loss_before:.4f}→{loss_after:.4f} ⚠️ loss 反弹")
                    else:
                        info(f"  Step {step}: lr→{current_lr:.2e}, loss {loss_before:.4f}→{loss_after:.4f} → 平稳")

    # 检测是否过拟合 (val loss 上升但 train loss 下降)
    if len(val_loss) >= 5 and len(train_loss) >= 5:
        val_steps_sorted = sorted(val_loss.keys())
        train_steps_sorted = sorted(train_loss.keys())
        recent_val = [val_loss[s] for s in val_steps_sorted[-3:]]
        recent_train = [train_loss[s] for s in train_steps_sorted[-3:]]
        if all(recent_val[i] > recent_val[0] for i in range(1, len(recent_val))) and \
           all(recent_train[i] < recent_train[0] for i in range(1, len(recent_train))):
            warning("过拟合迹象: val loss 持续上升但 train loss 持续下降")
            info("建议: 增加正则化、增强数据增强、早停")

    return 0


def _diag_experiment(args: argparse.Namespace) -> int:
    """全面诊断实验。"""
    from cvlab.db.database import Database
    from cvlab.diagnose.loss import LossDetector

    db = Database()
    exp = db.get_experiment(args.experiment_id)
    if not exp:
        error(f"实验 {args.experiment_id} 不存在")
        return 1

    header(f"全面诊断: {args.experiment_id}")
    result("名称", exp["name"])
    result("状态", exp["status"])

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
            info(f"Loss 分析: {len(train_losses)} 个点, {anomalies} 个异常" if anomalies else
                 f"Loss 分析: {len(train_losses)} 个点, 无异常")

    # 梯度分析
    from cvlab.diagnose.gradient import GradientDiagnosis
    grad = GradientDiagnosis().diagnose(args.experiment_id, db)
    if grad.total_layers > 0:
        icons = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
        info(f"梯度: {icons.get(grad.overall_status, '?')} {grad.overall_status} "
             f"({grad.healthy_count}健康/{grad.warning_count}警告/{grad.critical_count}严重)")

    # LR-Loss 联动
    lr_points = {m["step"]: m["value"] for m in metrics if m["key"] == "lr"} if metrics else {}
    if lr_points:
        info(f"LR 阶段: {len(set(lr_points.values()))} 个, 范围 {min(lr_points.values()):.2e}~{max(lr_points.values()):.2e}")

    # 最佳指标
    best_acc = db.get_metrics(args.experiment_id, keys=["val/acc"])
    if best_acc:
        best_val = max(best_acc, key=lambda m: m["value"])
        result("最佳 val_acc", f"{best_val['value']*100:.2f}% (step {best_val['step']})")

    ckpts = db.get_checkpoints(args.experiment_id)
    if ckpts:
        result("Checkpoints", f"{len(ckpts)} 个")
        best = next((c for c in ckpts if c["is_best"]), None)
        if best:
            result("最优 Checkpoint", f"epoch {best['epoch']} ({best.get('metric_value', 'N/A')})")

    if exp.get("env_json") and exp["env_json"] != "{}":
        import json
        env = json.loads(exp["env_json"])
        if env.get("torch_version"):
            info(f"PyTorch: {env['torch_version']}")
        if env.get("cuda_version"):
            info(f"CUDA: {env['cuda_version']}")
        if env.get("num_gpus"):
            info(f"GPU 数: {env['num_gpus']}")

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("diagnose", help="训练诊断工具")
    sp = p.add_subparsers(dest="diag_command")

    sp.add_parser("loss", help="分析实验 loss 指标").add_argument("experiment_id", help="实验 ID")

    dl_p = sp.add_parser("dataloader", help="DataLoader 性能诊断")
    dl_p.add_argument("config", help="配置文件路径")
    dl_p.add_argument("--num-batches", type=int, default=50, help="采样批次数")

    sp.add_parser("gradient", help="分析实验梯度健康").add_argument("experiment_id", help="实验 ID")

    sp.add_parser("lr-loss", help="学习率与 Loss 联动分析").add_argument("experiment_id", help="实验 ID")

    sp.add_parser("experiment", help="全面诊断实验").add_argument("experiment_id", help="实验 ID")

    p.set_defaults(func=cmd_diagnose)
