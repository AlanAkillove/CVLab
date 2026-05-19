"""随机种子管理 - 实验可复现性的基础设施。"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """固定所有随机源，确保实验可复现。

    Args:
        seed: 随机种子
        deterministic: 是否启用完全确定性模式。
            开启后 cuDNN 算法变为确定，保证完全复现，但速度损失约 10%。
            关闭时 cuDNN Benchmark 可能使结果不可完全复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 非 deterministic 模式下，cudnn.benchmark 由用户通过加速面板控制


def get_deterministic_warning() -> str | None:
    """如果当前处于非确定性模式，返回提示文字。"""
    if not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark:
        return (
            "cuDNN Benchmark 已启用 (benchmark=True)，每次运行时算法选择可能不同，"
            "无法保证完全复现。如需严格复现，请关闭 Benchmark 并设置 deterministic=True "
            "（速度损失约 10%）。"
        )
    return None
