"""非侵入式 Hook 注入 - 梯度监控 + 特征图监控。

使用 PyTorch 的 register_forward_hook / register_full_backward_hook，
不需要用户修改模型代码。

性能约束（采样模式）：
- 默认每 50 step 采集一次梯度 L2 范数（log_freq=50）
- 仅监控 watch_layers 指定的层（避免全量注入）
- 非采样 step 不计算范数（零额外开销）
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from cvlab.core.types import GradientReport

if TYPE_CHECKING:
    from cvlab.db.database import Database


class GradientMonitor:
    """梯度健康监控器。

    通过 register_full_backward_hook 注入到指定层，
    以采样模式收集梯度 L2 范数。

    Args:
        model: 目标模型
        layers: 要监控的层名列表 (None = 监控所有含参数的层)
        log_freq: 采样频率（每 N step 采集一次）
        log_activations: 是否同时记录特征图（开销更大）
        experiment_id: 实验 ID（用于记录）
        artifact_dir: 特征图保存路径
        db: 数据库实例
    """

    def __init__(self, model: nn.Module,
                 layers: list[str] | None = None,
                 log_freq: int = 50,
                 log_activations: bool = False,
                 experiment_id: str | None = None,
                 artifact_dir: str | Path | None = None,
                 db: Database | None = None):
        self.model = model
        self.log_freq = log_freq
        self.log_activations = log_activations
        self.experiment_id = experiment_id
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.db = db

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._grad_norms: dict[str, float] = {}
        self._target_layers = layers

        self._register_hooks()

    def _register_hooks(self) -> None:
        """注册 Hook 到目标层。"""
        named_modules = dict(self.model.named_modules())

        if self._target_layers:
            names = [n for n in self._target_layers if n in named_modules]
        else:
            names = [n for n, m in named_modules.items()
                     if any(p.requires_grad for p in m.parameters())]

        for name in names:
            module = named_modules[name]

            # 后向 Hook：采集梯度 L2 范数
            backward_handle = module.register_full_backward_hook(
                self._make_grad_hook(name)
            )
            self._handles.append(backward_handle)

            # 前向 Hook：可选采集特征图
            if self.log_activations:
                forward_handle = module.register_forward_hook(
                    self._make_activation_hook(name)
                )
                self._handles.append(forward_handle)

    def _make_grad_hook(self, name: str):
        """创建梯度 Hook（仅记录 L2 范数，不保存完整梯度张量）。"""
        def hook(module, grad_input, grad_output):
            if grad_output is not None:
                grad = grad_output[0]
                if grad is not None:
                    self._grad_norms[name] = grad.norm(2).item()
        return hook

    def _make_activation_hook(self, name: str):
        """创建特征图 Hook（保存激活值 L2 范数）。"""
        def hook(module, input, output):
            if output is not None:
                if isinstance(output, (tuple, list)):
                    output = output[0]
                act_norm = output.norm(2).item()
                self._grad_norms[f"activation/{name}"] = act_norm
        return hook

    def step(self, global_step: int) -> GradientReport | None:
        """每 step 调用一次。仅在 log_freq 倍数时返回报告。

        返回 None 表示非采样 step（跳过）。
        """
        if global_step % self.log_freq != 0:
            return None

        report = GradientReport(
            layer_grad_norms=dict(self._grad_norms),
            warnings=self._check_warnings(),
            global_step=global_step,
        )

        # 将梯度信息写入数据库
        if self.db and self.experiment_id:
            metrics = {}
            for layer_name, norm in self._grad_norms.items():
                metrics[f"grad_norm/{layer_name}"] = norm
            if metrics:
                self.db.log_metrics(self.experiment_id, metrics, global_step)

        self._grad_norms.clear()
        return report

    def _check_warnings(self) -> list[str]:
        """检测梯度异常。"""
        warnings: list[str] = []
        for name, norm in self._grad_norms.items():
            if norm < 1e-5:
                warnings.append(f"{name}: 梯度消失 (norm={norm:.2e})")
            elif norm > 10.0:
                warnings.append(f"{name}: 梯度爆炸 (norm={norm:.2e})")
        return warnings

    def close(self) -> None:
        """移除所有 Hook。"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
