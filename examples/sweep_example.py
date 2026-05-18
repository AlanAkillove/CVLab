"""
CVLab Sweep Example - 超参扫描示例
===================================

演示如何使用 CVLab 的 Sweep 功能进行超参搜索。

运行方式：
    python examples/sweep_example.py

前提：
    pip install torch torchvision cvlab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvlab import Tracker
from cvlab.sweep.sweeper import Sweeper


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


def train_trial(config):
    """使用给定配置训练一个 trial 并返回指标。"""
    tracker = Tracker(config=config)
    model = ToyModel()
    optimizer_class = getattr(torch.optim, config["training"]["optimizer"], torch.optim.Adam)
    optimizer = optimizer_class(model.parameters(), lr=config["training"]["lr"])
    monitor = tracker.watch(model, log_gradients=False, log_freq=100)

    inputs = torch.randn(100, 10)
    labels = torch.randint(0, 3, (100,))

    for epoch in range(config["training"]["epochs"]):
        perm = torch.randperm(100)
        running_loss = 0.0
        correct = 0

        for idx in range(0, 100, config["training"]["batch_size"]):
            batch_idx = perm[idx: idx + config["training"]["batch_size"]]
            optimizer.zero_grad()
            outputs = model(inputs[batch_idx])
            loss = F.cross_entropy(outputs, labels[batch_idx])
            loss.backward()
            monitor.step(epoch * (100 // config["training"]["batch_size"]) + idx)
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels[batch_idx]).sum().item()

        loss_val = running_loss / (100 // config["training"]["batch_size"])
        acc_val = correct / 100
        tracker.log({"train/loss": loss_val, "train/acc": acc_val}, step=epoch)

    best_acc = max(
        m["value"] for m in tracker.db.get_metrics(tracker.experiment_id)
        if m["key"] == "train/acc"
    )
    tracker.finish("completed")
    return best_acc, tracker.experiment_id


def main():
    # ── Grid 搜索 ────────────────────────────────────────────
    print("[Grid] 搜索 lr 和 batch_size 的最佳组合...")
    sweeper = Sweeper()

    sweep_id = sweeper.create_sweep(
        base_config={
            "name": "sweep_demo",
            "model": {"name": "ToyModel"},
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "lr": 0.01,
                "optimizer": "adam",
            },
        },
        strategy="grid",
        params={
            "training.lr": [0.1, 0.01, 0.001],
            "training.batch_size": [8, 16],
        },
        name="lr_bs_grid",
    )
    print(f"  Sweep ID: {sweep_id}")

    # ── 执行 Trial ──────────────────────────────────────────
    for trial in sweeper.get_trials(sweep_id):
        config = _build_config(trial)
        print(f"  Trial {trial['trial_index']}: lr={config['training']['lr']}, "
              f"bs={config['training']['batch_size']}...")
        best_acc, exp_id = train_trial(config)
        print(f"    best_acc={best_acc:.4f}, exp_id={exp_id}")

    best = sweeper.get_best_trial(sweep_id, metric_key="train/acc")
    if best:
        print(f"\n  最佳 Trial: {best['trial_index']}, acc={best['metric_value']:.4f}")

    # ── Random 搜索 ─────────────────────────────────────────
    print("\n[Random] 随机搜索 5 组超参...")
    random_sweep_id = sweeper.create_sweep(
        base_config={
            "name": "random_sweep_demo",
            "model": {"name": "ToyModel"},
            "training": {"epochs": 3, "batch_size": 16, "optimizer": "adam"},
        },
        strategy="random",
        params={
            "training.lr": {"type": "loguniform", "min": 1e-4, "max": 1e-1},
            "training.batch_size": {"type": "choice", "values": [8, 16, 32]},
        },
        max_trials=5,
        seed=42,
    )
    print(f"  Sweep ID: {random_sweep_id}")

    for trial in sweeper.get_trials(random_sweep_id):
        config = _build_config(trial)
        print(f"  Trial {trial['trial_index']}: lr={config['training']['lr']:.5f}, "
              f"bs={config['training']['batch_size']}...")
        best_acc, exp_id = train_trial(config)
        print(f"    best_acc={best_acc:.4f}, exp_id={exp_id}")

    print("\n[OK] Sweep 完成，在 UI 中查看结果: cvlab ui")


def _build_config(trial):
    """从 trial 记录重建完整配置。"""
    import json
    config = json.loads(trial["exp_config"])
    return config


if __name__ == "__main__":
    main()
