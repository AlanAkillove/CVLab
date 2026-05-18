"""Checkpoint 保存与加载管理。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from cvlab.checkpoint.ema import detect_and_save_ema

if TYPE_CHECKING:
    from cvlab.db.database import Database


class CheckpointManager:
    """Checkpoint 管理。

    Args:
        checkpoint_dir: checkpoint 文件存储目录
        db: 数据库实例（用于记录 checkpoint 元数据）
        experiment_id: 实验 ID
        keep_last: 自动清理时保留的最近 checkpoint 数量
    """

    def __init__(self, checkpoint_dir: str | Path,
                 db: Database | None = None,
                 experiment_id: str | None = None,
                 keep_last: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.experiment_id = experiment_id
        self.keep_last = keep_last

    def save(self, model: nn.Module,
             optimizer: torch.optim.Optimizer | None = None,
             epoch: int = 0,
             metrics: dict[str, float] | None = None,
             is_best: bool = False,
             ema_model: nn.Module | None = None) -> Path:
        """保存 checkpoint。

        自动处理：
        - EMA 权重（如果提供了 ema_model）
        - last 标记更新
        - 旧 checkpoint 清理
        """
        metrics = metrics or {}
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": metrics,
        }

        # 保存原始权重
        filename = f"epoch_{epoch}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(state, filepath)

        # 保存 EMA 权重
        if ema_model is not None:
            ema_path = self.checkpoint_dir / f"epoch_{epoch}_ema.pt"
            ema_state = {
                "epoch": epoch,
                "model_state_dict": ema_model.state_dict(),
                "metrics": metrics,
                "is_ema": True,
            }
            torch.save(ema_state, ema_path)
            if self.db and self.experiment_id:
                self.db.save_checkpoint_record(
                    self.experiment_id, epoch, str(ema_path),
                    metric_name="val_acc",
                    metric_value=metrics.get("val_acc"),
                    is_best=False,
                    is_last=True,
                    is_ema=True,
                    file_size=ema_path.stat().st_size,
                )

        # 更新 best 标记
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            if ema_model is not None:
                ema_best_path = self.checkpoint_dir / "best_ema.pt"
                ema_best_state = {
                    "epoch": epoch,
                    "model_state_dict": ema_model.state_dict(),
                    "metrics": metrics,
                    "is_ema": True,
                }
                torch.save(ema_best_state, ema_best_path)

        # 记录到数据库
        if self.db and self.experiment_id:
            best_metric_name = next(iter(metrics.keys())) if metrics else None
            best_metric_val = next(iter(metrics.values())) if metrics else None
            self.db.save_checkpoint_record(
                self.experiment_id, epoch, str(filepath),
                metric_name=best_metric_name,
                metric_value=best_metric_val,
                is_best=is_best,
                is_last=True,
                is_ema=False,
                file_size=filepath.stat().st_size,
            )
            self._cleanup_old()

        return filepath

    def load(self, epoch: int | None = None,
             best: bool = False,
             ema: bool = False) -> dict | None:
        """加载 checkpoint。

        Args:
            epoch: 指定 epoch 的 checkpoint（与 best 互斥）
            best: 加载最优 checkpoint
            ema: 是否加载 EMA 权重（如果有）
        """
        if best:
            path = self.checkpoint_dir / "best_ema.pt" if ema else self.checkpoint_dir / "best.pt"
        elif epoch is not None:
            suffix = "_ema.pt" if ema else ".pt"
            path = self.checkpoint_dir / f"epoch_{epoch}{suffix}"
        else:
            path = self.checkpoint_dir / "best.pt"

        if not path.exists():
            # 尝试加载 last checkpoint
            path = self._get_last_checkpoint(ema=ema)
            if path is None:
                return None

        return torch.load(path, map_location="cpu", weights_only=True)

    def _get_last_checkpoint(self, ema: bool = False) -> Path | None:
        """获取最近保存的 checkpoint 文件。"""
        suffix = "_ema.pt" if ema else ".pt"
        pattern = re.compile(rf"epoch_(\d+){re.escape(suffix)}")
        candidates: list[tuple[int, Path]] = []
        for f in self.checkpoint_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                candidates.append((int(m.group(1)), f))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _cleanup_old(self) -> None:
        """清理旧 checkpoint，保留最近 keep_last 个。"""
        if self.db and self.experiment_id:
            self.db.cleanup_checkpoints(self.experiment_id, self.keep_last)

    def list_checkpoints(self) -> list[dict]:
        """列出所有 checkpoint。"""
        checkpoints = []
        pattern = re.compile(r"epoch_(\d+)(_ema)?\.pt")
        for f in sorted(self.checkpoint_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                checkpoints.append({
                    "epoch": int(m.group(1)),
                    "path": str(f),
                    "is_ema": m.group(2) == "_ema",
                    "file_size": f.stat().st_size,
                })
        return checkpoints
