"""标准训练循环 - 分类任务。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cvlab.i18n import _
from cvlab.cli.console import console, header, info, progress, result, success, warning
from cvlab.config.config import load_config, validate_config
from cvlab.core.seed import seed_everything
from cvlab.core.tracker import Tracker


def train_classification(config_path: str,
                         experiment_id: str | None = None,
                         batch_size: int | None = None) -> str:
    """执行分类训练。

    Args:
        config_path: YAML 配置文件路径。
        experiment_id: 实验 ID（子进程模式，复用已有实验）。
        batch_size: 覆盖配置中的 batch_size（OOM 恢复时使用）。

    Returns:
        实验 ID。
    """
    # 配置加载
    header(_("加载配置"))
    config = load_config(config_path)
    if batch_size is not None:
        config.setdefault("training", {})["batch_size"] = batch_size
    errors = validate_config(config)
    if errors:
        for e in errors:
            console.print(f"  [red][FAIL][/red] {e}")
        raise ValueError(_("配置验证失败: {}").format(errors))
    info(_("模型: {}").format(config['model']['name']))
    info(_("Epochs: {}").format(config['training']['epochs']))

    # 种子
    seed = config.get("seed", 42)
    seed_everything(seed)

    # 设备检测
    header(_("设备检测"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_properties(0).name
        info(_("GPU: {} ({:.1f} GB)").format(gpu_name, torch.cuda.get_device_properties(0).total_memory / 1024**3))
    else:
        info(_("CPU 模式"))
    result(_("设备"), str(device))

    # 创建或加载实验
    header(_("实验"))
    if experiment_id:
        tracker = Tracker(experiment_id=experiment_id, resume=True)
        tracker.db.update_experiment_status(experiment_id, "running")
        info(_("恢复实验 {}").format(experiment_id))
        info(_("目录: {}").format(tracker.exp_dir))
    else:
        tracker = Tracker(config=config)
        tracker.db.update_experiment(tracker.experiment_id,
                                     command=f"cvlab train --config {config_path}")
        result(_("实验 ID"), tracker.experiment_id)
        info(_("目录: {}").format(tracker.exp_dir))

    # 数据加载
    header(_("数据加载"))
    train_loader, val_loader, class_names = _load_data(config)
    result(_("训练集"), _("{} 样本, {} batches/epoch").format(len(train_loader.dataset), len(train_loader)))
    result(_("验证集"), _("{} 样本, {} batches/epoch").format(len(val_loader.dataset), len(val_loader)))
    num_classes = len(class_names)

    # 模型创建
    header(_("模型创建"))
    model = _create_model(config, num_classes, device)
    result(_("模型"), config["model"]["name"])
    total_params = sum(p.numel() for p in model.parameters())
    result(_("参数量"), f"{total_params / 1e6:.2f}M")

    # Batch Size 自动探测（配置中未指定时）
    if config.get("training", {}).get("batch_size") is None:
        input_size = config.get("data", {}).get("input_size", [3, 224, 224])
        num_gpus = max(1, torch.cuda.device_count())
        from cvlab.probe.batch_size import BatchSizeProbe
        probe = BatchSizeProbe(model, tuple(input_size), config, num_gpus=num_gpus)
        probe_result = probe.probe()
        config["training"]["batch_size"] = probe_result.recommended_batch_size
        result(_("推荐 Batch Size"), probe_result.recommended_batch_size)
    else:
        result(_("Batch Size"), config["training"]["batch_size"])

    # Hook 注入
    watch_cfg = config.get("watch", {})
    if watch_cfg.get("log_gradients", True):
        tracker.watch(
            model,
            log_gradients=True,
            log_activations=watch_cfg.get("log_activations", False),
            watch_layers=watch_cfg.get("watch_layers"),
            log_freq=watch_cfg.get("log_freq", 50),
        )
        info(_("梯度监控已注入"))

    # 优化器
    optimizer = _create_optimizer(model, config)
    scheduler = _create_scheduler(optimizer, config, len(train_loader))

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    header(_("训练开始"))
    pbar = progress()
    epoch_task = pbar.add_task(_("Epochs"), total=config["training"]["epochs"])
    best_acc = 0.0

    with pbar:
        for epoch in range(config["training"]["epochs"]):
            epoch_start = time.perf_counter()

            train_loss, train_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler=scheduler, epoch=epoch,
                pbar=pbar, batch_desc=_("  Epoch {}").format(epoch+1),
            )

            val_loss, val_acc = _validate(model, val_loader, criterion, device)

            epoch_time = time.perf_counter() - epoch_start

            # 更新 LR scheduler (plateau)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            # 记录指标
            tracker.log({
                "train/loss": train_loss,
                "train/acc": train_acc / 100.0,
                "val/loss": val_loss,
                "val/acc": val_acc / 100.0,
                "lr": optimizer.param_groups[0]["lr"],
            }, epoch)

            # 保存 checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            tracker.save_checkpoint(model, optimizer, epoch, {
                "val_acc": val_acc / 100.0,
                "train_loss": train_loss,
            }, is_best=is_best)

            pbar.update(epoch_task, advance=1)
            console.print(
                _("  Epoch {:3d}/{:3d} | train loss: {:.4f} | train acc: {:.2f}% | val acc: {:.2f}% | lr: {:.2e} | {:.1f}s").format(
                    epoch + 1, config['training']['epochs'],
                    train_loss, train_acc, val_acc,
                    optimizer.param_groups[0]['lr'],
                    epoch_time)
            )

    tracker.finish("completed")

    # 生成报告
    header(_("报告生成"))
    try:
        from cvlab.report.html_report import HtmlReportGenerator
        gen = HtmlReportGenerator(db=tracker.db)
        report_path = Path.cwd() / f"{tracker.experiment_id}.html"
        gen.save(tracker.experiment_id, str(report_path))
        result(_("报告"), str(report_path))
    except Exception as e:
        warning(_("报告生成失败: {}").format(e))

    # 展示最终结果
    header(_("训练完成"))
    result(_("实验"), tracker.experiment_id)
    result(_("最佳验证准确率"), f"{best_acc:.2f}%")
    result(_("状态"), "completed")

    return tracker.experiment_id


def _load_data(config: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    """加载训练/验证数据。"""
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "")
    dataset_path = data_cfg.get("dataset", "")
    input_size = data_cfg.get("input_size", [3, 224, 224])
    batch_size = config["training"].get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 2)
    val_split = data_cfg.get("val_split", 0.2)
    pin_memory = data_cfg.get("pin_memory", True) and torch.cuda.is_available()

    # 标准预处理
    from torchvision import transforms as T
    h, w = input_size[1], input_size[2] if len(input_size) >= 3 else input_size[0]
    img_size = max(h, w)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])

    train_transform = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])

    # 尝试 torchvision 内置数据集
    if dataset_name:
        try:
            import torchvision.datasets as datasets
            ds_cls = getattr(datasets, dataset_name, None)
            if ds_cls is None:
                raise ValueError(_("未知数据集: {}").format(dataset_name))

            root = Path(dataset_path) if dataset_path else Path(".cvlab/data")
            root.mkdir(parents=True, exist_ok=True)

            full_train = ds_cls(root=str(root), train=True, download=True, transform=train_transform)
            full_val = ds_cls(root=str(root), train=False, download=True, transform=val_transform)
            class_names = full_train.classes

            train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(full_val, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory)
            return train_loader, val_loader, class_names
        except Exception as e:
            raise RuntimeError(_("内置数据集 '{}' 加载失败: {}").format(dataset_name, e))

    # ImageFolder 格式
    if dataset_path:
        from torchvision.datasets import ImageFolder
        from torch.utils.data import random_split

        root = Path(dataset_path)
        if not root.exists():
            raise FileNotFoundError(_("数据集路径不存在: {}").format(root))

        full_dataset = ImageFolder(str(root), transform=train_transform)
        class_names = full_dataset.classes

        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        val_ds.dataset.transform = val_transform

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, class_names

    raise ValueError(_("配置中需要指定 data.dataset 路径或 data.dataset_name"))


def _create_model(config: dict, num_classes: int, device: torch.device) -> nn.Module:
    """创建模型。"""
    import torchvision.models as models

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "resnet18")
    pretrained = model_cfg.get("pretrained", False)
    weights = "DEFAULT" if pretrained else None

    # 构建模型
    try:
        builder = getattr(models, model_name, None)
        if builder is None:
            raise ValueError(_("不支持的模型: {}").format(model_name))
        model = builder(weights=weights)
    except Exception as e:
        raise ValueError(_("模型 '{}' 创建失败: {}").format(model_name, e))

    # 替换分类头
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


def _create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """创建优化器。"""
    train_cfg = config.get("training", {})
    lr = train_cfg.get("lr", 0.001)
    wd = train_cfg.get("weight_decay", 0.0001)
    opt_name = train_cfg.get("optimizer", "adam").lower()

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(_("不支持的优化器: {}").format(opt_name))


def _create_scheduler(optimizer: torch.optim.Optimizer,
                      config: dict,
                      steps_per_epoch: int) -> Any:
    """创建学习率调度器。"""
    train_cfg = config.get("training", {})
    sched_name = train_cfg.get("scheduler", "cosine").lower()
    epochs = train_cfg.get("epochs", 50)

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        step_size = train_cfg.get("step_size", 10)
        gamma = train_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif sched_name == "none":
        return None
    else:
        raise ValueError(_("不支持的 scheduler: {}").format(sched_name))


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any = None,
    epoch: int = 0,
    pbar=None,
    batch_desc: str = "",
) -> tuple[float, float]:
    """训练一个 epoch。

    Args:
        model: 模型。
        train_loader: 训练数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        device: 设备。
        scheduler: 学习率调度器（per-batch 更新）。
        epoch: 当前 epoch 编号。
        pbar: Rich 进度条实例（可选）。
        batch_desc: 进度条描述文字。

    Returns:
        (平均 loss, 准确率 %) 元组。
    """
    if len(train_loader) == 0:
        return 0.0, 0.0

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    batch_pbar = None
    if pbar is not None:
        batch_pbar = pbar.add_task(batch_desc or f"Epoch {epoch+1}", total=len(train_loader))

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step(epoch + len(train_loader))
            else:
                scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if pbar is not None and batch_pbar is not None:
            pbar.update(batch_pbar, advance=1)

    if pbar is not None and batch_pbar is not None:
        pbar.remove_task(batch_pbar)

    avg_loss = train_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """验证一个 epoch。

    Args:
        model: 模型。
        val_loader: 验证数据加载器。
        criterion: 损失函数。
        device: 设备。

    Returns:
        (平均 loss, 准确率 %) 元组。
    """
    if len(val_loader) == 0:
        return 0.0, 0.0

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100.0 * val_correct / val_total
    return avg_loss, accuracy
