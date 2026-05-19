"""GPU 信息检测。

通过 PyTorch + pynvml（可选）检测 GPU 信息。
"""

from __future__ import annotations

import warnings

import torch

from cvlab.core.types import GPUInfo


def detect_gpus() -> list[GPUInfo]:
    """检测所有可用 GPU，返回 GPUInfo 列表。"""
    gpus: list[GPUInfo] = []
    if not torch.cuda.is_available():
        return gpus

    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            cc = (props.major, props.minor)
            free_mem = _get_free_memory(i)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory_gb=props.total_memory / (1024**3),
                free_memory_gb=free_mem,
                compute_capability=cc,
                supports_tensor_core=cc >= (7, 0),
                supports_bf16=cc >= (8, 0),
                driver_version=_get_driver_version(),
            ))
        except Exception:
            gpus.append(GPUInfo(
                index=i,
                name="Unknown GPU",
                total_memory_gb=0.0,
                free_memory_gb=0.0,
            ))
    return gpus


def _get_free_memory(device_idx: int) -> float:
    """获取 GPU 当前可用显存 (GB)。"""
    try:
        return torch.cuda.mem_get_info(device_idx)[0] / (1024**3)
    except Exception:
        return 0.0


def _get_driver_version() -> str:
    """获取 NVIDIA 驱动版本。"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        version = result.stdout.strip()
        return version if version else ""
    except Exception:
        return ""


def check_cuda_mismatch() -> tuple[bool, str]:
    """检查 PyTorch CUDA 编译版本 vs 系统 CUDA Runtime 版本是否匹配。

    Returns:
        (是否不匹配, 描述信息)
        如果 torch 不是在 CUDA 支持下编译的，返回 (False, "")
    """
    if not torch.cuda.is_available():
        return False, ""

    torch_cuda = torch.version.cuda or "?"
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        driver_ver = result.stdout.strip()
        return False, f"PyTorch CUDA: {torch_cuda}, Driver: {driver_ver}"
    except Exception:
        return False, f"PyTorch CUDA: {torch_cuda}"


def get_recommended_num_workers() -> int:
    """推荐 num_workers 值（基于 CPU 物理核心数）。"""
    import os
    import psutil
    cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    return max(2, min(cores, 16))
