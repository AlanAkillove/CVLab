"""CVLab - CV 实验管理平台。

让研究者专注于模型和数据本身，而非工程琐事。
"""

from __future__ import annotations

__version__ = "0.2.5"

# ── 核心 API ──────────────────────────────────────────────
# ── Checkpoint ─────────────────────────────────────────────
from cvlab.checkpoint.manager import CheckpointManager

# ── 配置 ───────────────────────────────────────────────────
from cvlab.config.config import DEFAULT_CONFIG, load_config, validate_config
from cvlab.core.seed import seed_everything
from cvlab.core.tracker import Tracker
from cvlab.core.types import (
    AccelerationConfig,
    EnvironmentReport,
    GPUInfo,
    ProbeCandidate,
    ProbeResult,
)
from cvlab.core.watch import GradientMonitor, GradientReport

# ── 数据集 ─────────────────────────────────────────────────
from cvlab.data.analyze import DatasetAnalyzer, DatasetReport
from cvlab.data.augment import AugmentPreview
from cvlab.data.provenance import DatasetProvenance

# ── 环境检测 ───────────────────────────────────────────────
from cvlab.detect.probe import EnvironmentProbe
from cvlab.diagnose.io_bottleneck import IOBottleneckDetector, IOBottleneckReport

# ── 诊断 ───────────────────────────────────────────────────
from cvlab.diagnose.loss import LossAnomalyReport, LossDetector

# ── 国际化 ─────────────────────────────────────────────────
from cvlab.i18n import _, current_language, get_available_languages, set_language

# ── 探针 ───────────────────────────────────────────────────
from cvlab.probe.batch_size import BatchSizeProbe

# ── 模型 Profile ───────────────────────────────────────────
from cvlab.profile.model_card import ModelCard, ModelProfiler

# ── 报告 ───────────────────────────────────────────────────
from cvlab.report.html_report import HtmlReportGenerator

# ── 权重管理 ───────────────────────────────────────────────
from cvlab.weights.manager import WeightManager

__all__ = [
    # 版本
    "__version__",

    # 核心
    "Tracker",
    "seed_everything",
    "AccelerationConfig",
    "EnvironmentReport",
    "GPUInfo",
    "ProbeCandidate",
    "ProbeResult",
    "GradientMonitor",
    "GradientReport",

    # 配置
    "DEFAULT_CONFIG",
    "load_config",
    "validate_config",

    # 探针
    "BatchSizeProbe",

    # 环境
    "EnvironmentProbe",

    # 诊断
    "LossDetector",
    "LossAnomalyReport",
    "IOBottleneckDetector",
    "IOBottleneckReport",

    # 数据集
    "DatasetAnalyzer",
    "DatasetReport",
    "DatasetProvenance",
    "AugmentPreview",

    # Checkpoint
    "CheckpointManager",

    # 权重
    "WeightManager",

    # Profile
    "ModelProfiler",
    "ModelCard",

    # 报告
    "HtmlReportGenerator",

    # 国际化
    "set_language",
    "current_language",
    "_",
    "get_available_languages",
]
