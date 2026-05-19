"""Batch Size 自动探测。

算法：二分搜索 + 悲观数据注入 + optimizer.step + 精度对齐 + 安全余量。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cvlab.core.types import ProbeCandidate, ProbeResult


class BatchSizeProbe:
    """Batch Size 二分搜索探测。

    在训练开始前运行，自动找到当前模型+数据+硬件配置下
    可用的最大 Batch Size（含 20% 安全余量）。

    Args:
        model: 要训练的模型
        input_shape: 输入张量形状 (C, H, W)
        config: 训练配置（包含 AMP、精度格式等）
        num_gpus: GPU 数量（多卡时每卡独立计算）
    """

    def __init__(self, model: nn.Module, input_shape: tuple[int, ...],
                 config: dict | None = None,
                 num_gpus: int = 1):
        self.model = model
        self.input_shape = input_shape
        self.config = config or {}
        self.num_gpus = num_gpus
        self.device = next(model.parameters()).device

        # 精度格式
        self.use_amp = self.config.get("training", {}).get("amp", False)
        self.use_bf16 = self.config.get("training", {}).get("bf16", False)
        self.dtype = torch.bfloat16 if self.use_bf16 else (torch.float16 if self.use_amp else torch.float32)

    def probe(self) -> ProbeResult:
        """执行二分搜索探测。"""
        candidates: list[ProbeCandidate] = []

        # 估算搜索上界（基于显存总量）
        high = self._estimate_upper_bound()
        low = 1

        # 二分搜索
        best = 1
        while low <= high:
            mid = (low + high) // 2
            success, memory = self._try_batch_size(mid)
            candidates.append(ProbeCandidate(batch_size=mid, memory_gb=memory, success=success))
            if success:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        # 安全余量：多卡不过安全系数（有效 BS = 单卡 BS × GPU 数）
        recommended = best if self.num_gpus > 1 else max(1, int(best * 0.8))

        result = ProbeResult(
            recommended_batch_size=recommended,
            candidates=candidates,
            peak_memory_gb=max((c.memory_gb or 0) for c in candidates if c.success),
            with_amp=self.use_amp or self.use_bf16,
            num_gpus=self.num_gpus,
        )

        self._print_result(result)
        return result

    def _estimate_upper_bound(self) -> int:
        """估算二分搜索上界。"""
        if self.device.type == "cpu":
            # CPU 模式：基于 RAM 估算
            try:
                import psutil
                ram_gb = psutil.virtual_memory().available / (1024**3)
                return min(256, int(ram_gb * 2))
            except Exception:
                return 128

        if not torch.cuda.is_available():
            return 128

        # GPU 模式：基于显存总量 + 模型参数量估算
        total_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        param_mem = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
        # 每张图的特征图显存（粗略估算）
        sample_mem = (
            math.prod(self.input_shape) * 4  # FP32 4 bytes per element
        ) / (1024**3)
        budget = total_mem - param_mem - 0.5  # 留 0.5GB 系统开销
        if sample_mem > 0 and budget > 0:
            return min(512, max(2, int(budget / sample_mem)))
        return 128

    def _try_batch_size(self, batch_size: int) -> tuple[bool, float | None]:
        """尝试给定 batch size，返回 (是否成功, 显存占用 GB)。"""
        if batch_size <= 0:
            return False, None

        try:
            # Step 1: 构造悲观数据（最大分辨率 + 稠密 label）
            x = torch.randn(batch_size, *self.input_shape, device=self.device)
            y = torch.randint(0, 2, (batch_size,), device=self.device)

            # Step 2: 前向（含 AMP）
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.use_amp or self.use_bf16,
            ):
                output = self.model(x)
                # 根据任务类型构造悲观 loss
                if output.shape[-1] == 1:
                    loss = output.mean()  # 回归/检测
                else:
                    loss = nn.functional.cross_entropy(output, y)

            # Step 3: 反向
            loss.backward()

            # Step 4: 优化器步进
            # 创建临时优化器（Adam 会分配动量缓冲区）
            temp_optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
            temp_optimizer.step()
            temp_optimizer.zero_grad()

            # Step 5: 记录显存并清理
            memory_gb = None
            if self.device.type == "cuda" and torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated(self.device) / (1024**3)

            # 清理梯度
            self.model.zero_grad()
            del temp_optimizer
            torch.cuda.empty_cache()

            return True, memory_gb

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # OOM 时清理
            self.model.zero_grad()
            torch.cuda.empty_cache()
            return False, None

    def _print_result(self, result: ProbeResult) -> None:
        """打印探测结果。"""
        gpu_info = ""
        if self.num_gpus > 1:
            gpu_info = f"  GPU 数量：{self.num_gpus}（有效 BS = 单卡 BS × {self.num_gpus}）\n"

        print(f"""
Batch Size 探测结果
─────────────────────────────────
{gpu_info}  输入：{self.input_shape}

  候选值：
""")
        for c in result.candidates:
            mark = "← 推荐" if c.batch_size == result.recommended_batch_size else ""
            status = "[OK]" if c.success else "[FAIL]"
            mem = f"{c.memory_gb:.1f} GB" if c.memory_gb else "N/A"
            print(f"  BS={c.batch_size:<4}  显存占用：{mem:<8}  {status} {mark}")

        print(f"""
  推荐 Batch Size：{result.recommended_batch_size}
  {'（已含 20% 安全余量）' if self.num_gpus == 1 else '（多卡模式，安全余量由每卡独立承担）'}
  {'同时推荐开启 AMP 可进一步提升' if not self.use_amp else ''}
─────────────────────────────────""")
