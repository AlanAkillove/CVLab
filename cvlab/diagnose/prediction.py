"""预测结果可视化 - 分类/检测/分割的预测采样展示。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def classification_samples(
    images: torch.Tensor,
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    max_samples: int = 16,
    ncol: int = 4,
) -> np.ndarray:
    """生成分类预测采样图。

    Args:
        images: 图片张量 (N, C, H, W)，值范围 [0, 1]。
        true_labels: 真实标签列表。
        pred_labels: 预测标签列表。
        class_names: 类别名称列表。
        max_samples: 最多显示的样本数。
        ncol: 每行显示数。

    Returns:
        RGB 图像数组 (H, W, 3)，值范围 [0, 255]，uint8。
    """
    import torchvision.utils as vutils

    n = min(len(images), max_samples)
    images = images[:n].cpu()
    true_labels = true_labels[:n]
    pred_labels = pred_labels[:n]

    # 给每张图片底部添加标签
    labeled: list[torch.Tensor] = []
    for i in range(n):
        img = images[i]
        label_true = class_names[true_labels[i]] if true_labels[i] < len(class_names) else str(true_labels[i])
        label_pred = class_names[pred_labels[i]] if pred_labels[i] < len(class_names) else str(pred_labels[i])
        correct = true_labels[i] == pred_labels[i]
        marker = "[OK]" if correct else "[X]"
        text = f"{marker} T:{label_true} P:{label_pred}"

        # 把文字绘制到图片底部
        img_with_label = _add_text_to_image(img, text)
        labeled.append(img_with_label)

    if not labeled:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    grid = vutils.make_grid(
        torch.stack(labeled),
        nrow=ncol,
        padding=4,
    )
    grid_np = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return grid_np


def detection_overlay(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    score_threshold: float = 0.5,
) -> np.ndarray:
    """在图片上绘制检测框。

    Args:
        image: 输入图片 (H, W, 3)，值范围 [0, 255]。
        boxes: 检测框数组 (N, 4)，格式 [x1, y1, x2, y2]。
        scores: 置信度数组 (N,)。
        labels: 标签索引数组 (N,)。
        class_names: 类别名称列表。
        score_threshold: 置信度阈值。

    Returns:
        绘制了检测框的图片 (H, W, 3)，uint8。
    """
    import cv2

    img = image.copy()
    h, w = img.shape[:2]

    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        label = class_names[labels[i]] if labels[i] < len(class_names) else str(labels[i])
        text = f"{label} {scores[i]:.2f}"

        # 颜色基于类别索引
        color = _get_color(labels[i])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 文字背景
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def segmentation_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    alpha: float = 0.5,
    num_classes: int = 21,
) -> np.ndarray:
    """在图片上叠加分割掩码。

    Args:
        image: 输入图片 (H, W, 3)，值范围 [0, 255]。
        pred_mask: 预测分割掩码 (H, W)，整数标签。
        gt_mask: 真实分割掩码 (H, W)，可选。
        alpha: 叠加透明度。
        num_classes: 分割类别数。

    Returns:
        叠加了掩码的图片 (H, W, 3)，uint8。
    """
    import cv2

    img = image.copy()

    _mask_to_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for c in range(num_classes):
        color = _get_color(c)
        _mask_to_color[pred_mask == c] = color

    overlay = cv2.addWeighted(img, 1 - alpha, _mask_to_color, alpha, 0)

    if gt_mask is not None:
        # 在右下角显示 GT 叠加
        h, w = img.shape[:2]
        gt_overlay = np.zeros_like(img)
        for c in range(num_classes):
            color = _get_color(c)
            gt_overlay[gt_mask == c] = color
        gt_combined = cv2.addWeighted(img, 1 - alpha, gt_overlay, alpha, 0)
        overlay = np.concatenate([overlay, gt_combined], axis=1)

    return overlay


def _add_text_to_image(img: torch.Tensor, text: str) -> torch.Tensor:
    """在图片底部添加文字标签。

    Args:
        img: (C, H, W)，值范围 [0, 1]。
        text: 要绘制的文字。

    Returns:
        添加了文字的图片张量 (C, H+20, W)。
    """
    import cv2

    c, h, w = img.shape
    img_np = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    # 在底部添加文字区域
    canvas = np.ones((h + 20, w, 3), dtype=np.uint8) * 255
    canvas[:h] = img_np

    cv2.putText(
        canvas, text, (4, h + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
    )

    result = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    return result


_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (192, 192, 192),
    (255, 165, 0), (128, 64, 64), (64, 128, 64), (64, 64, 128),
    (255, 192, 203), (0, 0, 0),  # 共 22 色
]


def _get_color(index: int) -> tuple[int, int, int]:
    """根据索引返回循环颜色。"""
    return _COLORS[index % len(_COLORS)]
