"""模型性能画像 - 参数量、FLOPs、延迟、内存占用。"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


def _to_scalar(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """将模型输出转为标量 loss，支持单 tensor 或 tuple 输出。"""
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.sum()


@dataclass
class ModelCard:
    name: str
    total_params: int
    trainable_params: int
    flops_macs: int
    flops_giga: float
    params_millions: float
    forward_time_ms: float
    backward_time_ms: float
    memory_peak_mb: float
    memory_input_mb: float
    layer_stats: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """生成文本摘要。"""
        lines = [
            f"模型: {self.name}",
            f"参数: {self.params_millions:.2f}M（训练: {self.trainable_params / 1e6:.2f}M）",
            f"FLOPs: {self.flops_giga:.2f} G",
            "",
            f"前向延迟: {self.forward_time_ms:.2f} ms",
            f"反向延迟: {self.backward_time_ms:.2f} ms",
            f"峰值显存: {self.memory_peak_mb:.1f} MB",
            f"输入占用: {self.memory_input_mb:.1f} MB",
        ]
        return "\n".join(lines)


class ModelProfiler:
    """对 PyTorch 模型进行性能画像。

    用法:
        profiler = ModelProfiler()
        card = profiler.profile(model, input_shape=(1, 3, 224, 224))
        print(card.summary())
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def profile(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        warmup: int = 3,
        repeats: int = 10,
    ) -> ModelCard:
        """对模型进行全面性能画像。

        Args:
            model: PyTorch 模型。
            input_shape: 输入形状，包含 batch 维度，如 (1, 3, 224, 224)。
            dtype: 输入数据类型。
            warmup: 预热次数。
            repeats: 采样次数。

        Returns:
            包含所有性能指标的 ModelCard。
        """
        model = model.to(self.device).eval()
        input_tensor = torch.randn(input_shape, dtype=dtype, device=self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # FLOPs 估算
        flops = self._count_flops(model, input_shape)

        # 前向延迟
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()

        forward_times: list[float] = []
        with torch.no_grad():
            for _ in range(repeats):
                start = time.perf_counter()
                _ = model(input_tensor)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                forward_times.append(time.perf_counter() - start)

        avg_forward = sum(forward_times) / len(forward_times)

        # 反向延迟（仅当模型可训练且需要梯度时）
        backward_times: list[float] = []
        if any(p.requires_grad for p in model.parameters()):
            model.train()
            for _ in range(warmup):
                out = model(input_tensor)
                loss = _to_scalar(out)
                loss.backward()
                model.zero_grad()

            if self.device == "cuda":
                torch.cuda.synchronize()

            for _ in range(repeats):
                start = time.perf_counter()
                out = model(input_tensor)
                loss = _to_scalar(out)
                loss.backward()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                backward_times.append(time.perf_counter() - start)
                model.zero_grad()

            avg_backward = sum(backward_times) / len(backward_times)
        else:
            avg_backward = 0.0

        model.eval()

        # 显存/内存估计
        memory_peak = self._measure_memory(model, input_shape, dtype)

        # 输入张量占用
        input_memory = input_tensor.numel() * input_tensor.element_size() / (1024 * 1024)

        # 逐层统计
        layer_stats = self._layer_statistics(model)

        return ModelCard(
            name=model.__class__.__name__,
            total_params=total_params,
            trainable_params=trainable_params,
            flops_macs=flops,
            flops_giga=flops / 1e9,
            params_millions=total_params / 1e6,
            forward_time_ms=avg_forward * 1000,
            backward_time_ms=avg_backward * 1000,
            memory_peak_mb=memory_peak,
            memory_input_mb=input_memory,
            layer_stats=layer_stats,
        )

    def _count_flops(self, model: nn.Module, input_shape: tuple[int, ...]) -> int:
        """通过 hook 统计 FLOPs（乘加次数）。"""
        flops = 0
        hooks: list[Any] = []

        def conv_hook(module: nn.Module, inp: Any, out: Any) -> None:
            nonlocal flops
            inp = inp[0]
            batch_size = inp.shape[0]
            output_shape = out.shape
            if isinstance(module, nn.Conv2d):
                # MACs = K_h * K_w * C_in * C_out * H_out * W_out
                k_h, k_w = module.kernel_size
                c_in = module.in_channels
                c_out = module.out_channels
                h_out, w_out = output_shape[2], output_shape[3]
                layer_macs = k_h * k_w * c_in * c_out * h_out * w_out
                # 分组卷积处理
                if module.groups > 1:
                    layer_macs = layer_macs // module.groups
                flops += layer_macs * batch_size
            elif isinstance(module, nn.Linear):
                layer_macs = module.in_features * module.out_features
                flops += layer_macs * batch_size

        def register_hooks(module: nn.Module) -> None:
            for child in module.children():
                if isinstance(child, (nn.Conv2d, nn.Linear)):
                    hooks.append(child.register_forward_hook(conv_hook))
                register_hooks(child)

        model = model.to(self.device).eval()
        register_hooks(model)

        with torch.no_grad():
            _ = model(torch.randn(input_shape, device=self.device))

        for h in hooks:
            h.remove()

        return flops

    def _measure_memory(self, model: nn.Module, input_shape: tuple[int, ...],
                        dtype: torch.dtype) -> float:
        """测量模型峰值显存/内存占用（MB）。"""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            input_tensor = torch.randn(input_shape, dtype=dtype, device=self.device)

            with torch.no_grad():
                _ = model(input_tensor)

            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return peak
        else:
            # CPU 模式下返回模型参数占用
            param_memory = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 * 1024)
            return param_memory

    def _layer_statistics(self, model: nn.Module) -> list[dict[str, Any]]:
        """生成逐层统计。"""
        stats: list[dict[str, Any]] = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                stats.append(OrderedDict([
                    ("name", name),
                    ("type", module.__class__.__name__),
                    ("params", params),
                    ("trainable", trainable),
                ]))
        return stats


def flops_to_text(flops: int) -> str:
    """将 FLOPs 转换为人类可读格式。"""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    else:
        return str(flops)
