"""cvlab estimate - 预计算训练耗时。

在正式训练前运行少量 batch，估算每 epoch 时间和总耗时，
帮助决定当前配置是否值得跑完。

用法:
    cvlab estimate --config config.yaml
    cvlab estimate --config config.yaml --batches 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from cvlab.cli.console import console, error, header, info, result, warning
from cvlab.config.config import load_config
from cvlab.i18n import _
from cvlab.train.run import _create_model, _create_optimizer, _load_data


def cmd_estimate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        error(_("配置文件不存在: {}").format(config_path))
        return 1

    header(_("训练耗时估算"))
    result(_("配置文件"), str(config_path.resolve()))

    config = load_config(str(config_path))
    n_batches = args.batches

    # 覆盖关键配置
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    epochs = config["training"].get("epochs", 50)
    batch_size = config["training"].get("batch_size", 64)

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    result(_("设备"), str(device))
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        result(_("GPU"), f"{gpu_name} ({gpu_mem:.1f} GB)")

    # 数据加载
    header(_("数据加载"))
    try:
        train_loader, val_loader, class_names = _load_data(config)
    except Exception as e:
        error(_("数据加载失败: {}").format(e))
        return 1
    result(_("训练集"), _("{} 样本, {} batches/epoch").format(len(train_loader.dataset), len(train_loader)))
    result(_("验证集"), _("{} 样本, {} batches/epoch").format(len(val_loader.dataset), len(val_loader)))
    num_classes = len(class_names)

    # 模型创建
    header(_("模型创建"))
    try:
        model = _create_model(config, num_classes, device)
    except Exception as e:
        error(_("模型创建失败: {}").format(e))
        return 1
    total_params = sum(p.numel() for p in model.parameters())
    result(_("模型"), f"{config['model']['name']} ({total_params/1e6:.2f}M params)")

    # 优化器 + 损失函数
    optimizer = _create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # ── 预热 ────────────────────────────────────────────
    header(_("性能采样 ({} batches)").format(n_batches))
    info(_("运行 {} 个 batch 进行预热和测量...").format(n_batches))

    model.train()
    warmup_batches = max(1, n_batches // 3)  # 前 1/3 用于预热

    train_iter = iter(train_loader)
    batch_times: list[float] = []
    data_times: list[float] = []
    mem_samples: list[float] = []
    throughputs: list[float] = []

    for i in range(n_batches + warmup_batches):
        # 数据加载计时
        t_data_start = time.perf_counter()
        try:
            inputs, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, labels = next(train_iter)
        t_data_end = time.perf_counter()
        data_time = t_data_end - t_data_start

        inputs, labels = inputs.to(device), labels.to(device)

        # 前向 + 反向计时
        t_start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        # 只记录 warmup 之后的采样
        if i >= warmup_batches:
            batch_times.append(t_end - t_start)
            data_times.append(data_time)
            throughputs.append(batch_size / (t_end - t_start))
            if device.type == "cuda":
                mem_samples.append(torch.cuda.memory_allocated(device) / (1024**3))

    # ── 结果汇总 ────────────────────────────────────────
    if not batch_times:
        error(_("采样失败"))
        return 1

    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_data_time = sum(data_times) / len(data_times)
    avg_throughput = sum(throughputs) / len(throughputs)
    batches_per_epoch = len(train_loader)
    time_per_epoch = avg_batch_time * batches_per_epoch

    console.print()
    result(_("平均每 batch"), f"{avg_batch_time*1000:.0f} ms  (数据加载 {avg_data_time*1000:.0f} ms + 计算 {(avg_batch_time-avg_data_time)*1000:.0f} ms)")
    result(_("吞吐量"), f"{avg_throughput:.0f} samples/sec")
    result(_("每 epoch"), f"{time_per_epoch:.1f} s  ({batches_per_epoch} batches)")
    result(_("总耗时估算 ({})").format(_("{} epochs").format(epochs)),
           _("{:.0f} min ({:.1f} hours)").format(time_per_epoch * epochs / 60, time_per_epoch * epochs / 3600))

    # DataLoader 瓶颈检测
    data_ratio = avg_data_time / avg_batch_time
    if data_ratio > 0.5:
        warning(_("DataLoader 可能是瓶颈 (数据加载占比 {:.0f}%)").format(data_ratio * 100))
        info(_("建议: 增大 num_workers 或将数据移到 SSD"))

    # GPU 显存
    if device.type == "cuda" and mem_samples:
        peak_mem = max(mem_samples)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        result(_("峰值显存"), f"{peak_mem:.1f} GB / {total_mem:.1f} GB ({peak_mem / total_mem * 100:.0f}%)")
        if peak_mem / total_mem > 0.9:
            warning(_("显存接近上限，可能 OOM"))

    # 训练建议
    console.print()
    if time_per_epoch * epochs < 300:  # <5 min
        info(_("提示: 训练很快，建议直接运行 cvlab train"))
    elif time_per_epoch * epochs < 3600:  # <1 hour
        info(_("提示: 训练时间适中，可运行 cvlab train 或调整 epochs"))
    else:
        info(_("提示: 训练时间较长，建议先用少量 epochs 测试"))

    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("estimate", help=_("预计算训练耗时（跑少量 batch 估算）"))
    p.add_argument("--config", "-c", required=True, help=_("配置文件路径"))
    p.add_argument("--batches", type=int, default=5, help=_("采样 batch 数 (默认 5)"))
    p.add_argument("--batch-size", type=int, help=_("覆盖 Batch Size"))
    p.add_argument("--epochs", type=int, help=_("覆盖训练轮数"))
    p.add_argument("--cpu", action="store_true", help=_("强制使用 CPU"))
    p.set_defaults(func=cmd_estimate)
