"""
CVLab Full Pipeline - 完整训练流程示例
=======================================

展示 Tracker 的全部核心功能：指标记录、梯度监控、Checkpoint、
混淆矩阵、检测可视化、分割可视化、数据血缘追踪。

运行方式：
    python examples/full_pipeline.py

注意：本示例生成模拟数据来展示可视化功能，不需真实数据集。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvlab import Tracker


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def generate_synthetic_data(num_samples=50, num_classes=5):
    """生成模拟图片和标签。"""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


def main():
    tracker = Tracker(config={
        "name": "full_pipeline_demo",
        "model": {"name": "ToyModel"},
        "training": {
            "epochs": 5,
            "batch_size": 16,
            "lr": 0.01,
            "optimizer": "sgd",
            "momentum": 0.9,
        },
        "data": {
            "dataset": "synthetic",
            "num_classes": 5,
        },
    })
    print(f"实验 ID: {tracker.experiment_id}")

    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    monitor = tracker.watch(model, log_gradients=True, log_freq=10)

    images, labels = generate_synthetic_data()
    class_names = ["cat", "dog", "bird", "fish", "fox"]

    global_step = 0
    for epoch in range(5):
        perm = torch.randperm(len(images))
        running_loss = 0.0
        correct = 0
        total = 0

        for idx in range(0, len(images), 16):
            batch_idx = perm[idx: idx + 16]
            inputs = images[batch_idx]
            targets = labels[batch_idx]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            grad_report = monitor.step(global_step)
            if grad_report and grad_report.warnings:
                print(f"  Step {global_step} 梯度告警: {grad_report.warnings}")

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_step += 1

        epoch_loss = running_loss / ((len(images) + 15) // 16)
        epoch_acc = correct / total
        tracker.log({"train/loss": epoch_loss, "train/acc": epoch_acc}, step=epoch)
        tracker.save_checkpoint(model, optimizer, epoch=epoch,
                                 metrics={"val_acc": epoch_acc})
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    # 混淆矩阵
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for idx in range(0, len(images), 16):
            outputs = model(images[idx: idx + 16])
            all_preds.extend(outputs.argmax(1).tolist())
            all_targets.extend(labels[idx: idx + 16].tolist())
    tracker.log_confusion_matrix(
        all_targets, all_preds, class_names, step=4, normalize=False,
    )

    # 模拟检测可视化
    fake_image = (images[0].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    fake_image = (fake_image * 255).astype(np.uint8)
    fake_boxes = np.array([[5, 5, 20, 20], [15, 15, 30, 30]], dtype=np.float32)
    fake_scores = np.array([0.9, 0.7])
    fake_labels = np.array([0, 2])
    tracker.log_detection(
        "detection_sample", fake_image, fake_boxes, fake_scores,
        fake_labels, class_names, step=4, score_threshold=0.5,
    )

    # 模拟分割可视化
    h, w = fake_image.shape[:2]
    fake_pred_mask = np.random.randint(0, 5, (h, w), dtype=np.int32)
    fake_gt_mask = np.random.randint(0, 5, (h, w), dtype=np.int32)
    tracker.log_segmentation(
        "seg_sample", fake_image, fake_pred_mask, step=4,
        gt_mask=fake_gt_mask, alpha=0.5,
    )

    # 数据血缘追踪
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "sample.jpg").write_text("mock")
        tracker.snapshot_dataset(td)

    from pathlib import Path
    tracker.finish("completed")
    print(f"\n[OK] 实验 {tracker.experiment_id} 完成")
    print(f"复现命令:\n{tracker.get_reproduce_command()}")


if __name__ == "__main__":
    main()
