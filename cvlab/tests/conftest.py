"""pytest 共享 fixtures — 为整个测试套件提供通用测试装备。

包含：
- 临时配置文件生成
- 清理临时目录
- 模拟 Database 实例
- i18n 状态自动重置
- pytest 命令行选项注册
- CUDA 相关测试自动跳过
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
import yaml

from cvlab.i18n import current_language, set_language


# ── pytest 钩子 ───────────────────────────────────────────

def pytest_addoption(parser: pytest.Parser) -> None:
    """注册自定义命令行选项。"""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="跳过标记为 slow 的测试",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="运行需要 GPU 的测试",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """根据用户 flag 过滤测试用例。"""
    skip_slow = config.getoption("--skip-slow")
    run_gpu = config.getoption("--run-gpu")

    if skip_slow:
        skip_slow_marker = pytest.mark.skip(reason="通过 --skip-slow 跳过")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow_marker)

    if not run_gpu:
        skip_gpu_marker = pytest.mark.skip(
            reason="需要 --run-gpu 标志；若无 GPU 请添加此标志后重试"
        )
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu_marker)


# ── 核心 fixtures ─────────────────────────────────────────

@pytest.fixture
def sample_config_path() -> Generator[str, None, None]:
    """返回一个临时 YAML 配置文件的路径。

    内容覆盖 model / training / data / seed 等常用字段，
    与 ``cvlab.config.config.DEFAULT_CONFIG`` 兼容。
    """
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


@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """返回一个配置字典（纯内存，无需 I/O）。

    与 ``sample_config_path`` 内容一致，方便在不产生文件的情况下测试。
    """
    return {
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


@pytest.fixture
def minimal_config_dict() -> dict[str, Any]:
    """返回仅含必填字段的最小配置。"""
    return {
        "model": {"name": "resnet18"},
        "training": {"epochs": 2, "optimizer": "adam"},
        "data": {"dataset": "/tmp/data"},
    }


@pytest.fixture
def nested_config_dict() -> dict[str, Any]:
    """返回含深层嵌套的配置，用于测试递归合并。"""
    return {
        "model": {
            "name": "resnet50",
            "pretrained": True,
            "backbone": {
                "type": "bottleneck",
                "layers": [3, 4, 6, 3],
                "stem": {
                    "conv": {"kernel_size": 7, "stride": 2},
                    "pool": {"kernel_size": 3, "stride": 2},
                },
            },
        },
        "training": {
            "optimizer": {
                "name": "adamw",
                "params": {"lr": 0.0001, "weight_decay": 0.01, "betas": [0.9, 0.999]},
            },
            "scheduler": {
                "name": "cosine",
                "params": {"T_max": 50, "eta_min": 1e-6},
            },
        },
    }


@pytest.fixture
def invalid_config_dict() -> dict[str, Any]:
    """返回一个包含多种无效值的配置，用于验证错误累积。"""
    return {
        "model": {"name": 123},  # 应为字符串
        "training": {
            "epochs": -5,          # 负数
            "batch_size": -1,      # 负数
            "optimizer": "rmsprop",  # 非法优化器
            "scheduler": "cyclic",   # 非法 scheduler
        },
        "data": {
            "dataset": "./data",
            "num_workers": -1,     # 不合理但不校验（留给下游）
        },
    }


# ── 临时目录与清理 ───────────────────────────────────────

@pytest.fixture
def isolated_fs(tmp_path: Path) -> Path:
    """提供一个隔离的临时工作目录，测试结束自动清理。

    用法::

        def test_something(isolated_fs):
            # isolated_fs 是一个 Path，当前工作目录已切换至此
            ...

    线程安全，每个测试有独立的 tmp_path。
    """
    old_cwd = Path.cwd()
    os.chdir(str(tmp_path))
    yield tmp_path
    os.chdir(str(old_cwd))


@pytest.fixture
def temp_yaml_file(tmp_path: Path) -> Generator[Path, None, None]:
    """创建一个临时 YAML 文件并返回其 Path。"""
    path = tmp_path / "config.yaml"
    yield path
    if path.exists():
        path.unlink(missing_ok=True)


# ── 模拟数据库 ───────────────────────────────────────────

@pytest.fixture
def mock_db() -> MagicMock:
    """创建一个模拟的 Database 实例。

    默认所有方法返回空值/空列表，调用 ``mock_db.setup_xxx()``
    可以预设返回值::

        def test_xxx(mock_db):
            mock_db.list_experiments.return_value = [
                {"id": "exp_001", "name": "test", ...}
            ]
            ...

    所有方法都可通过 ``mock_db.assert_called_once_with(...)`` 验证调用。
    """
    mock = MagicMock()

    # ── 默认返回值 ────────────────────────────────────────
    mock.list_experiments.return_value = []
    mock.get_experiment.return_value = None
    mock.get_metrics.return_value = []
    mock.create_experiment.return_value = "exp_mock_001"
    mock.delete_experiment.return_value = True
    mock.update_experiment.return_value = True

    # 断言助手
    return mock


@pytest.fixture
def mock_db_with_experiments() -> MagicMock:
    """返回一个预设了若干实验记录的 mock Database。"""
    mock = MagicMock()

    mock.list_experiments.return_value = [
        {
            "id": "exp_001",
            "name": "resnet18-cifar10",
            "status": "completed",
            "created_at": "2025-01-15T10:30:00",
            "config_json": '{"model": {"name": "resnet18"}}',
        },
        {
            "id": "exp_002",
            "name": "resnet50-imagenet",
            "status": "running",
            "created_at": "2025-01-16T14:00:00",
            "config_json": '{"model": {"name": "resnet50"}}',
        },
    ]

    def get_experiment_side_effect(exp_id: str) -> dict | None:
        lookup = {
            "exp_001": mock.list_experiments.return_value[0],
            "exp_002": mock.list_experiments.return_value[1],
        }
        return lookup.get(exp_id)

    mock.get_experiment.side_effect = get_experiment_side_effect
    mock.get_metrics.return_value = [
        {"key": "train/loss", "value": 0.215, "step": 100},
        {"key": "val/acc", "value": 0.892, "step": 100},
    ]

    return mock


# ── 模拟 GPU / 硬件信息 ──────────────────────────────────

@pytest.fixture
def mock_gpu_info() -> dict[str, Any]:
    """返回模拟的 GPU 信息（用于无 GPU 环境下的测试）。"""
    return {
        "available": False,
        "count": 0,
        "devices": [],
        "cuda_version": None,
        "cudnn_version": None,
    }


@pytest.fixture
def mock_gpu_available() -> dict[str, Any]:
    """返回模拟的 GPU 信息（假设 GPU 可用）。"""
    return {
        "available": True,
        "count": 1,
        "devices": [
            {
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total": 24564,
                "memory_free": 22000,
                "compute_capability": (8, 9),
            }
        ],
        "cuda_version": "12.1",
        "cudnn_version": "8.9.0",
    }


# ── i18n 重置 ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_i18n_after_test() -> Generator[None, None, None]:
    """每个测试结束后重置 i18n 语言到默认中文。

    防止测试之间因语言状态互相干扰（i18n 是全局可变状态）。
    """
    yield
    set_language("zh")


# ── 跳过条件 / 检测 ──────────────────────────────────────

def has_cuda() -> bool:
    """检测当前环境是否有可用的 CUDA。"""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


def has_torch() -> bool:
    """检测 PyTorch 是否可用。"""
    try:
        import torch
        return True
    except (ImportError, ModuleNotFoundError):
        return False


requires_cuda = pytest.mark.skipif(
    not has_cuda(),
    reason="此测试需要 CUDA 设备",
)

requires_torch = pytest.mark.skipif(
    not has_torch(),
    reason="此测试需要 PyTorch",
)
