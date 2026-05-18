"""Random 搜索 - 随机采样超参组合。"""

from __future__ import annotations

import random
from typing import Any


def sample_random(
    params: dict[str, dict[str, Any]],
    n_trials: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """从参数空间中随机采样 n_trials 组超参。

    Args:
        params: 超参定义，格式为 {name: {"type": "choice"|"uniform"|"loguniform", ...}}。
            choice: {"type": "choice", "values": [1, 2, 3]}
            uniform: {"type": "uniform", "min": 0.0, "max": 1.0}
            loguniform: {"type": "loguniform", "min": 1e-5, "max": 1e-1}
        n_trials: 采样组数。
        seed: 随机种子。

    Returns:
        n_trials 组超参组合。
    """
    if seed is not None:
        random.seed(seed)

    samples: list[dict[str, Any]] = []
    for _ in range(n_trials):
        sample: dict[str, Any] = {}
        for key, spec in params.items():
            sample[key] = _sample_one(spec)
        samples.append(sample)

    return samples


def _sample_one(spec: dict[str, Any]) -> Any:
    """从单个参数定义中采样一个值。"""
    t = spec.get("type", "choice")
    if t == "choice":
        values = spec.get("values", [])
        return random.choice(values) if values else None
    elif t == "uniform":
        return random.uniform(spec.get("min", 0.0), spec.get("max", 1.0))
    elif t == "loguniform":
        import math
        log_min = math.log(spec.get("min", 1e-5))
        log_max = math.log(spec.get("max", 1.0))
        return math.exp(random.uniform(log_min, log_max))
    elif t == "int":
        return random.randint(spec.get("min", 0), spec.get("max", 100))
    else:
        raise ValueError(f"不支持的采样类型: {t}")
