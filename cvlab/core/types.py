"""公共类型定义。"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any


@dataclasses.dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: tuple[int, int] | None = None
    supports_tensor_core: bool = False
    supports_bf16: bool = False
    driver_version: str = ""


@dataclasses.dataclass
class EnvironmentReport:
    os_type: str
    os_version: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    is_wsl: bool
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    num_gpus: int = 0
    gpus: list[GPUInfo] = dataclasses.field(default_factory=list)
    cuda_mismatch: bool = False
    storage_type: str = "unknown"  # ssd / hdd / unknown
    storage_available_gb: float = 0.0

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> EnvironmentReport:
        data = json.loads(s)
        if "gpus" in data:
            data["gpus"] = [GPUInfo(**g) for g in data["gpus"]]
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})


@dataclasses.dataclass
class ProbeCandidate:
    batch_size: int
    memory_gb: float | None
    success: bool


@dataclasses.dataclass
class ProbeResult:
    recommended_batch_size: int
    candidates: list[ProbeCandidate]
    peak_memory_gb: float
    with_amp: bool
    num_gpus: int = 1


@dataclasses.dataclass
class GradientReport:
    layer_grad_norms: dict[str, float]
    warnings: list[str]
    global_step: int


@dataclasses.dataclass
class AccelerationConfig:
    amp: bool = False
    bf16: bool = False
    cudnn_benchmark: bool = False
    torch_compile: bool = False
    channels_last: bool = False
    gradient_checkpointing: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ExperimentStatus(str, Enum):
    """实验状态常量。"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class FailureReason(str, Enum):
    """失败原因常量。"""
    OOM = "OOM"
    GRADIENT_EXPLOSION = "GradientExplosion"
    USER_INTERRUPT = "UserInterrupt"
    UNKNOWN = "Unknown"
