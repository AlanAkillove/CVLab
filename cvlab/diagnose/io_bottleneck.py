"""I/O 瓶颈诊断 - 检测 DataLoader 是否为训练瓶颈。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IOBottleneckReport:
    is_bottleneck: bool
    avg_data_time: float
    avg_compute_time: float
    gpu_idle_ratio: float
    recommended_workers: int
    recommended_prefetch: int
    recommended_pin_memory: bool
    details: list[str] = field(default_factory=list)


class IOBottleneckDetector:
    """检测 DataLoader 是否为训练瓶颈。

    用法:
        detector = IOBottleneckDetector()
        with detector.profile() as ctx:
            for inputs, labels in loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                ctx.step()  # 标记一次迭代结束
        report = detector.analyze()
    """

    def __init__(self, num_workers: int = 2, prefetch_factor: int = 2, pin_memory: bool = True):
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self._data_times: list[float] = []
        self._compute_times: list[float] = []
        self._iter_times: list[float] = []
        self._last_data_end: float | None = None
        self._iter_start: float | None = None

    def profile(self) -> _ProfilingContext:
        """返回上下文管理器，在训练循环中使用。"""
        return _ProfilingContext(self)

    def analyze(self, warn_threshold: float = 0.3) -> IOBottleneckReport:
        """分析采集的数据，判断 I/O 是否瓶颈。

        Args:
            warn_threshold: GPU 空闲比例超过此值则告警。
        """
        if len(self._data_times) < 5:
            return IOBottleneckReport(
                is_bottleneck=False,
                avg_data_time=0.0,
                avg_compute_time=0.0,
                gpu_idle_ratio=0.0,
                recommended_workers=self.num_workers,
                recommended_prefetch=self.prefetch_factor,
                recommended_pin_memory=self.pin_memory,
                details=["采样不足（至少需要 5 个 step）"],
            )

        # 排除前 2 个 warmup 批次
        data_times = self._data_times[2:]
        compute_times = self._compute_times[2:]

        avg_data = sum(data_times) / len(data_times)
        avg_compute = sum(compute_times) / len(compute_times)
        total_time = avg_data + avg_compute
        gpu_idle_ratio = avg_data / total_time if total_time > 0 else 0

        details: list[str] = []
        is_bottleneck = gpu_idle_ratio > warn_threshold

        if is_bottleneck:
            details.append(
                f"I/O 瓶颈：数据加载占 {gpu_idle_ratio:.1%} 的迭代时间 "
                f"（阈值 {warn_threshold:.0%}）"
            )

        # 推荐配置
        import os
        cpu_count = os.cpu_count() or 4
        recommended_workers = min(cpu_count, 8) if gpu_idle_ratio > 0.3 else self.num_workers
        recommended_prefetch = 4 if gpu_idle_ratio > 0.3 else self.prefetch_factor
        recommended_pin_memory = self.pin_memory

        if gpu_idle_ratio > 0.5:
            details.append(f"建议将 num_workers 从 {self.num_workers} 增加到 {recommended_workers}")
            if recommended_prefetch > self.prefetch_factor:
                details.append(f"建议将 prefetch_factor 从 {self.prefetch_factor} 增加到 {recommended_prefetch}")

        if not self.pin_memory:
            details.append("建议启用 pin_memory=True（可减少 CPU->GPU 传输时间）")

        if gpu_idle_ratio < 0.1 and self.num_workers > 2:
            details.append(f"I/O 开销很小，可考虑减少 num_workers（当前 {self.num_workers}）以节省 CPU")

        return IOBottleneckReport(
            is_bottleneck=is_bottleneck,
            avg_data_time=avg_data * 1000,
            avg_compute_time=avg_compute * 1000,
            gpu_idle_ratio=gpu_idle_ratio,
            recommended_workers=recommended_workers,
            recommended_prefetch=recommended_prefetch,
            recommended_pin_memory=recommended_pin_memory,
            details=details,
        )


class _ProfilingContext:
    """I/O 性能分析的上下文管理器。"""

    def __init__(self, detector: IOBottleneckDetector):
        self._detector = detector
        self._iter_start: float | None = None

    def __enter__(self) -> _ProfilingContext:
        self._detector._iter_start = time.perf_counter()
        self._detector._last_data_end = None
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def step(self) -> None:
        """标记一次训练迭代结束。"""
        now = time.perf_counter()
        d = self._detector

        if d._last_data_end is not None:
            # 计算时间 = 从上一次数据就绪到这次 step 调用
            compute_time = now - d._last_data_end
            d._compute_times.append(compute_time)
        if d._iter_start is not None:
            iter_time = now - d._iter_start
            d._iter_times.append(iter_time)

        # 模拟数据加载时间：距离上次 step 到下次 step 被调用的间隔
        # 实际上更好的方式是 hook DataLoader，这里做近似
        d._iter_start = now
        d._last_data_end = now


class DataLoaderProfiler:
    """通过监测 DataLoader 迭代时间来诊断 I/O 性能。

    直接测量 next(iter(loader)) 的耗时。
    """

    @staticmethod
    def profile_dataloader(loader, num_batches: int = 50) -> dict[str, float]:
        """测量 DataLoader 的迭代性能。

        Args:
            loader: DataLoader 实例
            num_batches: 采样批次数

        Returns:
            包含 mean/std/min/max 加载时间的 dict
        """
        times: list[float] = []
        iterator = iter(loader)
        for i in range(num_batches):
            start = time.perf_counter()
            try:
                _ = next(iterator)
            except StopIteration:
                break
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        if not times:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "batches": 0}

        import numpy as np
        return {
            "mean": float(np.mean(times)) * 1000,
            "std": float(np.std(times)) * 1000,
            "min": float(np.min(times)) * 1000,
            "max": float(np.max(times)) * 1000,
            "batches": len(times),
        }
