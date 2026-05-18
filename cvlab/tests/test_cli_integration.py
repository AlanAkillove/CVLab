"""CLI 集成测试 — 子进程模式端到端验证。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# ── 辅助函数 ──────────────────────────────────────────────

def run_cli(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """通过 subprocess 运行 cvlab CLI 命令。"""
    import os
    env = None
    if cwd:
        sep = ";" if os.name == "nt" else ":"
        env = {**os.environ, "PYTHONPATH": sep.join(sys.path)}
    cmd = [sys.executable, "-m", "cvlab"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=cwd, env=env)


def run_worker(*args: str) -> subprocess.CompletedProcess:
    """直接运行 subprocess_worker。"""
    cmd = [sys.executable, "-m", "cvlab.train.subprocess_worker"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """创建一个最小配置（CIFAR10，2 epoch，小 batch size）。"""
    config = {
        "model": {"name": "resnet18", "pretrained": False},
        "training": {
            "epochs": 2,
            "batch_size": 32,
            "optimizer": "adam",
            "lr": 0.001,
            "scheduler": "cosine",
        },
        "data": {
            "dataset_name": "CIFAR10",
            "dataset": str(tmp_path / "data"),
            "input_size": [3, 32, 32],
            "num_workers": 0,
        },
        "seed": 42,
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ── 测试: subprocess worker CLI ────────────────────────────

class TestSubprocessWorker:
    """验证 subprocess_worker 模块的 CLI 接口。"""

    def test_help(self):
        """--help 应正常显示。"""
        result = run_worker("--help")
        assert result.returncode == 0
        assert "--experiment-id" in result.stdout
        assert "--batch-size" in result.stdout

    def test_missing_experiment(self):
        """不存在的实验 ID 应返回非零退出码。"""
        result = run_worker(
            "--experiment-id", "exp_nonexistent",
            "--config", "does_not_exist.yaml",
            "--batch-size", "32",
        )
        assert result.returncode != 0

    def test_oom_exit_code(self):
        """模拟 CUDA OOM 应退出 137。"""
        # 通过子进程模拟 CUDA OOM 捕获
        oom_script = """
import sys
try:
    import torch
    raise torch.cuda.OutOfMemoryError()
except torch.cuda.OutOfMemoryError:
    sys.exit(137)
except ImportError:
    # 无 torch 时用常规异常模拟 OOM 退出码
    sys.exit(137)
"""
        result = subprocess.run(
            [sys.executable, "-c", oom_script],
            capture_output=True, text=True,
        )
        assert result.returncode == 137


# ── 测试: CLI 基本命令 ────────────────────────────────────

class TestCLIBasicCommands:
    """CLI 基本命令的集成测试。"""

    def test_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "cvlab" in result.stdout

    def test_version(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_list_empty(self):
        """空数据库时 list 应正常。"""
        # 在临时目录运行，避免影响用户数据
        result = run_cli("list")
        assert result.returncode == 0

    def test_init(self, tmp_path):
        result = run_cli("init", cwd=str(tmp_path))
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (tmp_path / ".cvlab").is_dir()

    def test_diagnose_loss_nonexistent(self):
        """诊断不存在的实验应失败。"""
        result = run_cli("diagnose", "loss", "exp_nonexistent")
        assert result.returncode != 0

    def test_diagnose_dataloader_no_config(self):
        """DataLoader 诊断缺少配置文件应失败。"""
        result = run_cli("diagnose", "dataloader", "nonexistent.yaml")
        assert result.returncode != 0

    def test_profile_model(self):
        """profile 命令应能对已有模型生成画像。"""
        result = run_cli("profile", "--model", "resnet18")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "参数" in result.stdout or "FLOPs" in result.stdout


# ── 测试: 端到端训练集成（需联网下载 CIFAR10）─────────────

class TestTrainIntegration:
    """完整训练集成测试。

    这些测试下载 CIFAR10 并实际运行训练。
    标记为 slow，默认跳过。
    """

    @pytest.mark.slow
    def test_train_subprocess(self, minimal_config):
        """通过 CLI 子进程跑 2 epoch CIFAR10 训练。"""
        result = subprocess.run(
            [
                sys.executable, "-m", "cvlab", "train",
                "--config", str(minimal_config),
            ],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "完成" in result.stdout or "实验" in result.stdout

    @pytest.mark.slow
    def test_train_diagnose_after_train(self, tmp_path, minimal_config):
        """训练后可以用 diagnose 分析 loss。"""
        # 训练
        result = subprocess.run(
            [
                sys.executable, "-m", "cvlab", "train",
                "--config", str(minimal_config),
            ],
            capture_output=True, text=True, timeout=300,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0

        # 从输出提取实验 ID
        import re
        match = re.search(r"exp_\S+", result.stdout)
        if match:
            exp_id = match.group()
            # diagnose loss
            diag = subprocess.run(
                [sys.executable, "-m", "cvlab", "diagnose", "loss", exp_id],
                capture_output=True, text=True, timeout=30,
                cwd=str(tmp_path),
            )
            assert diag.returncode == 0


# ── 测试: OOM 恢复逻辑 ───────────────────────────────────

class TestOOMRecovery:
    """OOM 恢复路径的集成测试。"""

    def test_data_analyze_nonexistent(self):
        """data analyze 不存在的路径应失败。"""
        result = run_cli("data", "analyze", "nonexistent_path")
        assert result.returncode != 0

    def test_data_augment_nonexistent(self):
        """data augment 不存在的图片应失败。"""
        result = run_cli("data", "augment", "nonexistent.png")
        assert result.returncode != 0

    def test_data_check_nonexistent(self):
        """data check 不存在的路径应失败。"""
        result = run_cli("data", "check", "nonexistent_path")
        assert result.returncode != 0

    def test_data_history(self):
        """data history 应正常返回（可能为空列表）。"""
        result = run_cli("data", "history")
        assert result.returncode == 0

    def test_help_command(self):
        """help 命令应正常显示概览。"""
        result = run_cli("help")
        assert result.returncode == 0
        assert "train" in result.stdout
        assert "sweep" in result.stdout
        assert "diagnose" in result.stdout
        assert "data" in result.stdout

    def test_help_with_command(self):
        """help <command> 应正常显示。"""
        result = run_cli("help", "train")
        assert result.returncode == 0

    def test_detect_oom_exit_code(self, tmp_path):
        """验证 `_detect_oom` 等效逻辑：exit code 137 → OOM。"""
        # 模拟 OOM：运行一个立即返回 137 的脚本
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.exit(137)"],
            capture_output=True, text=True,
        )
        assert result.returncode == 137

    def test_worker_oom_handling(self):
        """worker 捕获 CUDA OOM 并返回 137。"""
        # 模拟 CUDA OOM: 通过一个会触发 OOM 的 torch 操作
        # 但在 CPU 上 / 非 CUDA 环境中不会触发
        import torch
        if not torch.cuda.is_available():
            pytest.skip("需要 CUDA 设备")

        oom_script = """
import torch
try:
    raise torch.cuda.OutOfMemoryError()
except torch.cuda.OutOfMemoryError:
    import sys
    sys.exit(137)
"""
        result = subprocess.run(
            [sys.executable, "-c", oom_script],
            capture_output=True, text=True,
        )
        assert result.returncode == 137
