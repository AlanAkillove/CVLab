"""实验追踪核心类。"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch

from cvlab.config.config import DEFAULT_CONFIG, merge_config, save_config
from cvlab.core.seed import seed_everything
from cvlab.core.types import (
    EnvironmentReport,
    GPUInfo,
)
from cvlab.core.watch import GradientMonitor


class Tracker:
    """实验追踪核心类。

    提供实验创建/加载、Hook 注入、指标记录、Checkpoint 管理等能力。

    用法:
        tracker = Tracker(config={...})          # 创建新实验
        tracker = Tracker(experiment_id="exp_001", resume=True)  # 恢复
        tracker.log({"train/loss": 0.5}, step=1)
        tracker.save_checkpoint(model, optim, epoch=5)
        tracker.finish()
    """

    def __init__(self, experiment_id: str | None = None,
                 config: dict | None = None,
                 cvlab_dir: str | Path = ".cvlab",
                 resume: bool = False):
        """初始化 Tracker。

        Args:
            experiment_id: 实验 ID。为 None 时创建新实验。
            config: 实验配置字典（创建新实验时必填）。
            cvlab_dir: CVLab 数据目录（默认 .cvlab）。
            resume: 是否恢复已有实验（需同时指定 experiment_id）。
        """
        self.cvlab_dir = Path(cvlab_dir)
        self.db = self._get_database()

        if experiment_id and resume:
            self._load_experiment(experiment_id)
        else:
            self._create_experiment(config or {}, experiment_id)

        self._gradient_monitor: GradientMonitor | None = None
        self._ema_model: torch.nn.Module | None = None
        self._watch_layers: list[str] | None = None
        self._log_freq: int = 50

    # ── 数据库连接 ───────────────────────────────────────

    def _get_database(self):
        from cvlab.db.database import Database
        return Database(str(self.cvlab_dir / "cvlab.db"))

    # ── 实验创建/加载 ──────────────────────────────────────

    def _create_experiment(self, config: dict, experiment_id: str | None) -> None:
        import copy
        merged = merge_config(copy.deepcopy(DEFAULT_CONFIG), config)
        seed = merged.get("seed", 42)
        seed_everything(seed)

        env_report = self._probe_environment()
        name = config.get("name", f"experiment_{time.strftime('%m%d_%H%M')}")

        self.experiment_id = self.db.create_experiment(
            name=name,
            config=merged,
            seed=seed,
            env_json=env_report.to_json() if env_report else "{}",
        )

        self.exp_dir = self.cvlab_dir / "experiments" / self.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "artifacts").mkdir(exist_ok=True)

        save_config(merged, self.exp_dir / "config.yaml")
        self._save_script_snapshot()
        self._config = merged

    def _load_experiment(self, experiment_id: str) -> None:
        """加载已有实验。

        Raises:
            ValueError: 实验 ID 不存在。
        """
        exp = self.db.get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"实验 {experiment_id} 不存在")
        self.experiment_id = experiment_id
        self.exp_dir = self.cvlab_dir / "experiments" / experiment_id
        self._config = json.loads(exp["config_json"])
        self.db.update_experiment_status(experiment_id, "running")

    def _probe_environment(self) -> EnvironmentReport | None:
        """快速环境探测（完整探针由 detect 模块负责）。"""
        try:
            import psutil
            report = EnvironmentReport(
                os_type=sys.platform,
                os_version="",
                python_version=sys.version,
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
                is_wsl=False,
                cpu_cores=psutil.cpu_count(logical=False) or 0,
                cpu_threads=psutil.cpu_count(logical=True) or 0,
                total_ram_gb=psutil.virtual_memory().total / (1024**3),
                available_ram_gb=psutil.virtual_memory().available / (1024**3),
                num_gpus=torch.cuda.device_count(),
            )
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    cc = (props.major, props.minor)
                    report.gpus.append(GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory_gb=props.total_memory / (1024**3),
                        free_memory_gb=0.0,
                        compute_capability=cc,
                        supports_tensor_core=cc >= (7, 0),
                        supports_bf16=cc >= (8, 0),
                    ))
            return report
        except Exception as e:
            logger.warning("环境探测失败: %s", e)
            return None

    def _save_script_snapshot(self) -> None:
        """保存调用者训练脚本的快照（hash 去重）。"""
        import hashlib
        try:
            main_script = Path(sys.argv[0]) if sys.argv else None
            if main_script and main_script.exists():
                content = main_script.read_bytes()
                h = hashlib.sha256(content).hexdigest()[:12]
                snapshot_path = self.exp_dir / f"script_{h}.py"
                if not snapshot_path.exists():
                    shutil.copy2(main_script, snapshot_path)
                self.db.update_experiment(
                    self.experiment_id,
                    script_hash=h,
                    command=" ".join(sys.argv),
                )
        except Exception as e:
            logger.warning("脚本快照保存失败: %s", e)

    # ── 属性 ──────────────────────────────────────────────

    @property
    def config(self) -> dict:
        return self._config

    @property
    def exp_path(self) -> Path:
        return self.exp_dir

    # ── Hook 注入 ─────────────────────────────────────────

    def watch(self, model: torch.nn.Module, *,
              log_gradients: bool = True,
              log_activations: bool = False,
              watch_layers: list[str] | None = None,
              log_freq: int = 50) -> GradientMonitor:
        """注入梯度/特征图监控 Hook（非侵入式，不需修改模型代码）。

        返回 GradientMonitor 实例，用于 step() 调用和 close() 清理。
        """
        self._watch_layers = watch_layers
        self._log_freq = log_freq
        self._gradient_monitor = GradientMonitor(
            model=model,
            layers=watch_layers,
            log_freq=log_freq,
            log_activations=log_activations,
            experiment_id=self.experiment_id,
            artifact_dir=self.exp_dir / "artifacts",
            db=self.db,
        )
        return self._gradient_monitor

    # ── 指标记录 ─────────────────────────────────────────

    def log(self, metrics: dict[str, float], step: int) -> None:
        """记录标量指标。

        Args:
            metrics: {指标名: 值} 字典，如 {"train/loss": 0.523, "train/acc": 0.832}。
            step: 当前步数（epoch 或 global step）。
        """
        self.db.log_metrics(self.experiment_id, metrics, step)

    def log_image(self, key: str, image: np.ndarray | torch.Tensor,
                  step: int, caption: str | None = None) -> None:
        """记录图片到 artifacts。

        Args:
            key: 图片标识名。
            image: 图片数据 (H, W, 3)，值范围 [0, 1] 或 [0, 255]。
            step: 当前步数。
            caption: 图片说明（目前保留，暂未使用）。
        """
        import cv2
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        filename = f"step_{step}_{key}.jpg"
        filepath = self.exp_dir / "artifacts" / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.db.save_artifact(self.experiment_id, step, key, "image",
                              file_path=str(filepath))

    def log_confusion_matrix(self, y_true: list, y_pred: list,
                              class_names: list[str], step: int,
                              normalize: bool = True) -> None:
        """记录混淆矩阵。

        Args:
            y_true: 真实标签列表。
            y_pred: 预测标签列表。
            class_names: 类别名称列表。
            step: 当前步数。
            normalize: 是否按行归一化为百分比。
        """
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        data = {
            "matrix": cm.tolist(),
            "class_names": class_names,
            "normalize": normalize,
        }
        self.db.save_artifact(self.experiment_id, step, "confusion_matrix",
                              "confusion_matrix", data_json=json.dumps(data))

    def log_detection(self, key: str, image: np.ndarray,
                       boxes: np.ndarray, scores: np.ndarray,
                       labels: np.ndarray, class_names: list[str],
                       step: int, score_threshold: float = 0.5) -> None:
        """记录检测结果（叠加框的图片）。

        Args:
            key: 结果标识名。
            image: 原始图片 (H, W, 3)。
            boxes: 检测框数组 (N, 4)，格式 [x1, y1, x2, y2]。
            scores: 置信度数组 (N,)。
            labels: 类别索引数组 (N,)。
            class_names: 类别名称列表。
            step: 当前步数。
            score_threshold: 显示阈值，低于此值的框不绘制。
        """
        import cv2
        vis = image.copy()
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            class_name = class_names[int(label)] if class_names else str(label)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{class_name}:{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        filename = f"step_{step}_{key}.jpg"
        filepath = self.exp_dir / "artifacts" / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.db.save_artifact(self.experiment_id, step, key, "detection",
                              file_path=str(filepath))

    def log_segmentation(self, key: str, image: np.ndarray,
                          pred_mask: np.ndarray, step: int,
                          gt_mask: np.ndarray | None = None,
                          alpha: float = 0.5) -> None:
        """记录分割结果（叠加 mask 的图片）。

        Args:
            key: 结果标识名。
            image: 原始图片 (H, W, 3)。
            pred_mask: 预测掩码 (H, W)，值 > 0 为前景。
            step: 当前步数。
            gt_mask: 真实掩码（可选），以红色叠加显示。
            alpha: mask 透明度。
        """
        import cv2
        def _mask_overlay(img, mask, color=(0, 255, 0), alpha=alpha):
            overlay = img.copy()
            mask_bool = mask > 0
            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
            ).astype(np.uint8)
            return overlay
        vis = image.copy()
        vis = _mask_overlay(vis, pred_mask, color=(0, 255, 0))
        if gt_mask is not None:
            # 在右下角叠加 GT
            vis = _mask_overlay(vis, gt_mask, color=(255, 0, 0))
        filename = f"step_{step}_{key}.jpg"
        filepath = self.exp_dir / "artifacts" / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.db.save_artifact(self.experiment_id, step, key, "segmentation",
                              file_path=str(filepath))

    # ── EMA ────────────────────────────────────────────────

    def register_ema(self, ema_model: torch.nn.Module) -> None:
        """注册 EMA 模型。

        save_checkpoint() 会自动保存 EMA 权重。需在训练开始前注册。

        Args:
            ema_model: EMA 模型实例（EMA 权重由用户自行维护）。
        """
        self._ema_model = ema_model

    # ── Checkpoint ─────────────────────────────────────────

    def save_checkpoint(self, model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer | None = None,
                         epoch: int = 0, metrics: dict | None = None,
                         is_best: bool = False) -> Path:
        """保存 Checkpoint（自动处理 EMA 权重）。

        Args:
            model: 模型。
            optimizer: 优化器（可选，恢复训练时需要）。
            epoch: 当前 epoch。
            metrics: 指标字典，如 {"val/acc": 0.85}。
            is_best: 是否标记为最优 checkpoint。

        Returns:
            保存的文件路径。
        """
        from cvlab.checkpoint.manager import CheckpointManager
        mgr = CheckpointManager(self.exp_dir / "checkpoints", self.db,
                                 self.experiment_id)
        return mgr.save(model, optimizer, epoch, metrics or {},
                         is_best, self._ema_model)

    def load_checkpoint(self, epoch: int | None = None,
                         best: bool = False,
                         ema: bool = False) -> dict | None:
        """加载 Checkpoint。

        Args:
            epoch: 指定 epoch（为 None 时加载最新）。
            best: 是否加载最优 checkpoint。
            ema: 是否加载 EMA 权重。

        Returns:
            包含 model_state_dict, optimizer_state_dict 等的字典；无匹配时返回 None。
        """
        from cvlab.checkpoint.manager import CheckpointManager
        mgr = CheckpointManager(self.exp_dir / "checkpoints", self.db,
                                 self.experiment_id)
        return mgr.load(epoch, best, ema)

    # ── 复现命令 ──────────────────────────────────────────

    def get_reproduce_command(self) -> str:
        """生成一键复现命令，包含环境信息、代码版本和 CLI 命令。"""
        exp = self.db.get_experiment(self.experiment_id)
        if not exp:
            return "# 实验不存在"
        lines = [
            f"# 复现实验 {self.experiment_id}",
        ]
        env = json.loads(exp.get("env_json", "{}"))
        if env:
            os_str = env.get("os_type", "?")
            gpu_str = env.get("num_gpus", "?")
            torch_ver = env.get("torch_version", "?")
            lines.append(f"# 环境：{os_str} / {gpu_str} GPU / PyTorch {torch_ver}")
        if exp.get("git_hash"):
            lines.append(f"# 代码版本：git checkout {exp['git_hash']}")
        if exp.get("command"):
            lines.append(exp["command"])
        else:
            lines.append(f"# 训练脚本已保存到: {self.exp_dir}/")
        return "\n".join(lines)

    # ── 完成/关闭 ─────────────────────────────────────────

    def finish(self, status: str = "completed") -> None:
        """标记实验完成，清理 Hook 资源。

        Args:
            status: 最终状态，可选 completed / failed / archived。
        """
        if self._gradient_monitor:
            self._gradient_monitor.close()
        self.db.update_experiment_status(self.experiment_id, status)
        logger.info("Experiment %s finished (status=%s)", self.experiment_id, status)

    def snapshot_dataset(self, dataset_path: str) -> None:
        """记录数据集快照（文件数、总大小），用于数据血缘追踪。

        Args:
            dataset_path: 数据集根路径。
        """
        path = Path(dataset_path)
        if not path.exists():
            return
        total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        file_count = sum(1 for _ in path.rglob("*") if _.is_file())
        self.db.update_experiment(
            self.experiment_id,
            dataset_path=str(path.resolve()),
            dataset_total=total_size,
            dataset_files=file_count,
        )
