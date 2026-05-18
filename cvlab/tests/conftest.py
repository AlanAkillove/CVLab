"""pytest 共享 fixtures。"""

import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_config_path():
    """返回一个临时 YAML 配置文件的路径。"""
    config = {
        "model": {"name": "resnet18", "pretrained": False},
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "optimizer": "adam",
            "lr": 0.001,
        },
        "data": {
            "dataset": "./data",
            "num_workers": 2,
            "input_size": [3, 224, 224],
        },
        "seed": 42,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config, f)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)
