"""环境探针 - 系统检测 + 加速选项面板生成。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch

from cvlab.core.types import EnvironmentReport
from cvlab.detect.cpu_info import detect_cpu
from cvlab.detect.gpu_info import (
    check_cuda_mismatch,
    detect_gpus,
    get_recommended_num_workers,
)
from cvlab.detect.os_info import detect_os, is_wsl
from cvlab.detect.storage_info import detect_storage


@dataclass
class AccelerationOption:
    name: str
    enabled: bool
    supported: bool
    benefit: str
    condition: str
    risk: str = ""


@dataclass
class AccelerationPanel:
    options: list[AccelerationOption] = field(default_factory=list)
    recommended_num_workers: int = 2
    recommended_prefetch: int = 2
    recommended_pin_memory: bool = True


class EnvironmentProbe:
    """环境探针：检测系统信息并生成加速选项面板。"""

    def probe(self, data_path: str = ".") -> EnvironmentReport:
        """执行完整环境探测。"""
        os_type, os_version = detect_os()
        cpu_model, cpu_cores, cpu_threads = detect_cpu()
        gpus = detect_gpus()
        num_gpus = len(gpus)
        storage_type, storage_avail = detect_storage(data_path)

        import psutil
        memory = psutil.virtual_memory()

        cuda_mismatch, cuda_info = check_cuda_mismatch()

        report = EnvironmentReport(
            os_type=os_type,
            os_version=os_version,
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            is_wsl=is_wsl(),
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            total_ram_gb=memory.total / (1024**3),
            available_ram_gb=memory.available / (1024**3),
            num_gpus=num_gpus,
            gpus=gpus,
            cuda_mismatch=cuda_mismatch,
            storage_type=storage_type,
            storage_available_gb=storage_avail,
        )
        return report

    def print_report(self, report: EnvironmentReport) -> str:
        """生成格式化的环境报告文本。"""
        lines = [
            "环境探测报告",
            "─────────────────────────────────",
            f"系统：{report.os_type} {report.os_version}",
            f"Python：{report.python_version}",
            f"PyTorch：{report.torch_version}",
            f"CUDA：{report.cuda_version or 'N/A'}",
            f"WSL2：{'是' if report.is_wsl else '否'}",
            "",
            f"CPU：{report.cpu_model} ({report.cpu_cores}核/{report.cpu_threads}线程)",
            f"内存：{report.total_ram_gb:.1f} GB（可用 {report.available_ram_gb:.1f} GB）",
            "",
        ]
        if report.gpus:
            for gpu in report.gpus:
                lines.append(
                    f"GPU {gpu.index}：{gpu.name} "
                    f"({gpu.total_memory_gb:.1f} GB, CC={gpu.compute_capability})"
                )
        else:
            lines.append("GPU：无（CPU 模式）")
        lines.extend([
            "",
            f"存储类型：{report.storage_type}",
            f"可用空间：{report.storage_available_gb:.1f} GB",
            "─────────────────────────────────",
        ])
        return "\n".join(lines)

    def get_acceleration_panel(self, report: EnvironmentReport) -> AccelerationPanel:
        """根据环境报告生成加速选项面板。"""
        panel = AccelerationPanel()
        panel.recommended_num_workers = get_recommended_num_workers()

        has_tensor_core = any(g.supports_tensor_core for g in report.gpus)
        has_bf16 = any(g.supports_bf16 for g in report.gpus)
        has_gpu = report.num_gpus > 0

        panel.options = [
            AccelerationOption(
                name="AMP FP16",
                enabled=has_tensor_core,
                supported=has_tensor_core,
                benefit="显存-40%, 速度~2x",
                condition="需要 Volta+ 架构的 GPU",
                risk="小梯度可能下溢",
            ),
            AccelerationOption(
                name="BF16 精度",
                enabled=False,
                supported=has_bf16,
                benefit="显存-40%, 速度~2x（比 FP16 更稳定）",
                condition="需要 Ampere+ 架构的 GPU",
            ),
            AccelerationOption(
                name="cuDNN Benchmark",
                enabled=False,
                supported=has_gpu,
                benefit="自动选最优算法",
                condition="适合固定输入尺寸",
                risk="首批次较慢，与严格复现不可兼得",
            ),
            AccelerationOption(
                name="torch.compile",
                enabled=True,
                supported=True,
                benefit="训练/推理加速",
                condition="PyTorch 2.0+",
                risk="首次编译有额外耗时",
            ),
            AccelerationOption(
                name="Channels Last",
                enabled=False,
                supported=has_gpu,
                benefit="CNN 卷积层加速",
                condition="部分操作不支持",
            ),
            AccelerationOption(
                name="Gradient Checkpointing",
                enabled=False,
                supported=True,
                benefit="用计算换显存（大模型/小显存场景）",
                condition="所有硬件",
                risk="约 20% 速度开销",
            ),
        ]

        return panel

    def print_panel(self, panel: AccelerationPanel) -> str:
        """生成加速选项面板的格式化文本。"""
        lines = [
            "训练加速配置",
            "─────────────────────────────────",
        ]
        for opt in panel.options:
            icon = "[OK]" if opt.supported else "[--]"
            enabled = "x" if opt.enabled else " "
            lines.append(
                f"  [{enabled}] {opt.name:<25} {icon} {opt.benefit}"
            )
            if not opt.supported:
                lines.append(f"       {opt.condition}")
            if opt.risk:
                lines.append(f"       [WARN] {opt.risk}")
            lines.append("")
        lines.extend([
            "DataLoader 配置",
            f"  num_workers: {panel.recommended_num_workers} ← 基于 CPU 核心数推荐",
            f"  pin_memory:  {'x' if panel.recommended_pin_memory else ' '} "
            "(Windows 下如遇 hang 请关闭)",
            f"  prefetch_factor: {panel.recommended_prefetch}",
            "─────────────────────────────────",
        ])
        return "\n".join(lines)
