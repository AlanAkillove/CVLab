"""Grid 搜索 - 笛卡尔积参数组合生成。"""

from __future__ import annotations

import itertools
from typing import Any


def generate_grid(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """生成所有超参组合的笛卡尔积。

    Args:
        params: 超参名 -> 候选值列表的映射。
                嵌套超参使用点号分隔，如 "training.lr": [0.001, 0.0001]。

    Returns:
        所有组合的列表，每个组合是展平的 {key: value} 字典。
    """
    keys = list(params.keys())
    values = list(params.values())
    combinations: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def count_grid(params: dict[str, list[Any]]) -> int:
    """计算 Grid 搜索的总组合数。"""
    total = 1
    for v in params.values():
        total *= len(v)
    return total
