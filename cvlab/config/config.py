"""配置加载与验证。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

# 默认实验配置
DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "resnet18",
        "pretrained": False,
    },
    "training": {
        "epochs": 50,
        "batch_size": None,  # 由探测结果自动填充
        "accumulation_steps": 1,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
    },
    "data": {
        "dataset": None,
        "num_workers": 2,
        "pin_memory": True,
        "prefetch_factor": 2,
        "input_size": [3, 224, 224],
        "val_split": 0.2,
    },
    "seed": 42,
    "checkpoint": {
        "save_best_metric": "val_acc",
        "save_last": True,
        "keep_last": 5,
    },
    "watch": {
        "log_gradients": True,
        "log_activations": False,
        "watch_layers": None,
        "log_freq": 50,
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    """从 YAML 文件加载配置，与默认配置合并。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, encoding="utf-8") as f:
        user_config = yaml.safe_load(f)

    return merge_config(copy.deepcopy(DEFAULT_CONFIG), user_config or {})


def merge_config(base: dict, override: dict) -> dict:
    """递归合并两个配置字典。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: dict) -> list[str]:
    """验证配置，返回错误列表。空列表表示配置有效。"""
    errors = []

    if config.get("model") and not isinstance(config["model"].get("name"), str):
        errors.append("model.name 必须为字符串")

    training = config.get("training", {})
    if not isinstance(training.get("epochs"), int) or training["epochs"] < 1:
        errors.append("training.epochs 必须为正整数")
    if training.get("batch_size") is not None and training["batch_size"] < 1:
        errors.append("training.batch_size 必须为正整数或 None")

    valid_opts = {"adam", "sgd", "adamw"}
    if training.get("optimizer", "").lower() not in valid_opts:
        errors.append(f"training.optimizer 必须为 {valid_opts} 之一")

    valid_schedulers = {"cosine", "step", "plateau", "none"}
    if training.get("scheduler", "").lower() not in valid_schedulers:
        errors.append(f"training.scheduler 必须为 {valid_schedulers} 之一")

    # 新增校验
    lr = training.get("lr", None)
    if lr is not None and (not isinstance(lr, (int, float)) or lr <= 0):
        errors.append("training.lr 必须为正数")

    data_cfg = config.get("data", {})
    nw = data_cfg.get("num_workers", 0)
    if not isinstance(nw, int) or nw < 0:
        errors.append("data.num_workers 必须为非负整数")

    input_size = data_cfg.get("input_size", None)
    if input_size is not None:
        if not isinstance(input_size, (list, tuple)) or len(input_size) not in (2, 3):
            errors.append("data.input_size 必须为 [C, H, W] 或 [H, W] 格式的列表")
        elif any(not isinstance(d, int) or d < 1 for d in input_size):
            errors.append("data.input_size 中的各维度必须为正整数")

    seed = config.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append("seed 必须为整数")

    return errors


def config_to_json(config: dict) -> str:
    return json.dumps(config, indent=2, default=str)


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
