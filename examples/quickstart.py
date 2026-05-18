"""
CVLab Quickstart - 完整训练流程示例
====================================

在 Cifar10 上训练一个简单的 CNN，展示 CVLab 的核心 API。

运行方式：
    python examples/quickstart.py

前提：
    pip install torch torchvision cvlab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

import cvlab
from cvlab.core.tracker import Tracker


# ── 1. 定义模型 ──────────────────────────────────────────
# 完全原生的 nn.Module，不需要继承任何 CVLab 基类

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    # ── 2. 创建实验 ──────────────────────────────────────
    # Tracker 自动创建实验、固定随机种子、保存环境快照
    tracker = Tracker(config={
        "name": "cifar10_quickstart",
        "model": {"name": "SimpleCNN"},
        "training": {
            "epochs": 5,
            "batch_size": 64,
            "lr": 0.001,
            "optimizer": "adam",
        },
    })

    print(f"实验 ID: {tracker.experiment_id}")
    print(f"配置: {tracker.config}")

    # ── 3. 准备数据 ──────────────────────────────────────
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # 记录数据集信息
    tracker.snapshot_dataset("./data")

    # ── 4. 初始化模型 ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ── 5. Hook 注入（非侵入式） ─────────────────────────
    # 监控 conv1 和 conv2 的梯度，每 10 step 采集一次
    monitor = tracker.watch(
        model,
        log_gradients=True,
        watch_layers=["conv1", "conv2"],
        log_freq=10,
    )

    # ── 6. 训练循环 ──────────────────────────────────────
    global_step = 0
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # 梯度监控（采样模式，每 10 step 采集）
            grad_report = monitor.step(global_step)
            if grad_report and grad_report.warnings:
                print(f"  [WARN] Step {global_step} 梯度告警: {grad_report.warnings}")

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            global_step += 1

        # 记录指标到实验
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total
        tracker.log({
            "train/loss": epoch_loss,
            "train/acc": epoch_acc,
        }, step=epoch)

        # 保存 Checkpoint
        if epoch % 2 == 0:
            tracker.save_checkpoint(model, optimizer, epoch=epoch,
                                     metrics={"val_acc": epoch_acc})

        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    # 保存最终模型
    tracker.save_checkpoint(
        model, optimizer, epoch=4,
        metrics={"val_acc": epoch_acc}, is_best=True
    )

    # ── 7. 完成实验 ──────────────────────────────────────
    tracker.finish("completed")
    print(f"\n[OK] 实验 {tracker.experiment_id} 完成")
    print(f"复现命令:\n{tracker.get_reproduce_command()}")

    # 查看已保存的 Checkpoints
    ckpts = tracker.db.get_checkpoints(tracker.experiment_id)
    print(f"\nCheckpoints:")
    for c in ckpts:
        print(f"  epoch {c['epoch']}: {c['metric_value']}")


if __name__ == "__main__":
    main()
