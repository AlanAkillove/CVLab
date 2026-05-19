"""Loss 异常检测 - 监控训练过程中的 loss 异常信号。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LossAnomalyReport:
    has_anomaly: bool
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class LossDetector:
    """检测 loss 序列中的异常模式。

    支持：
    - NaN / Inf 检测
    - 梯度爆炸（loss 指数级增长）
    - 梯度消失 / 欠拟合（loss 下降过快或过慢）
    - 平台期（loss 停滞）
    - 学习率异常

    用法:
        detector = LossDetector()
        for loss in losses:
            report = detector.step(loss, step)
            if report.has_anomaly:
                print(report.warnings)
    """

    def __init__(self, window_size: int = 20, patience: int = 10, lr: float | None = None):
        """
        Args:
            window_size: 滑动窗口大小，用于计算均值和方差。
            patience: 检测平台期时容忍的连续无改进步数。
            lr: 当前学习率，用于检测 LR 异常。
        """
        self.window_size = window_size
        self.patience = patience
        self.lr = lr
        self._losses: list[float] = []
        self._steps: list[int] = []
        self._best_loss: float = float("inf")
        self._best_step: int = 0
        self._plateau_count: int = 0
        self._explosion_count: int = 0

    def step(self, loss: float, step: int) -> LossAnomalyReport:
        """输入一个新的 loss 值，返回检测结果。

        Args:
            loss: 当前步的 loss 值。
            step: 当前步数（epoch 或 iteration）。

        Returns:
            包含告警和建议的报告。
        """
        warnings: list[str] = []
        suggestions: list[str] = []

        # NaN / Inf 检测
        if math.isnan(loss) or math.isinf(loss):
            warnings.append(f"Step {step}: loss 为 {'NaN' if math.isnan(loss) else 'Inf'}")
            suggestions.append("检查输入数据中是否包含 NaN，或尝试降低学习率")
            return LossAnomalyReport(has_anomaly=True, warnings=warnings, suggestions=suggestions)

        self._losses.append(loss)
        self._steps.append(step)

        if loss < self._best_loss:
            self._best_loss = loss
            self._best_step = step
            self._plateau_count = 0
        else:
            self._plateau_count += 1

        # 需要至少 window_size 个点才有统计意义
        if len(self._losses) < max(self.window_size, 3):
            return LossAnomalyReport(has_anomaly=False)

        recent = self._losses[-self.window_size:]
        self._steps[-self.window_size:]

        # 爆炸检测：连续 N 步 loss 上升且幅度大
        if len(recent) >= 5:
            deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
            avg_delta = sum(deltas) / len(deltas)
            if avg_delta > 0 and recent[-1] > 3 * self._best_loss:
                self._explosion_count += 1
                if self._explosion_count >= 3:
                    warnings.append(
                        f"Step {step}: loss 持续上升，当前 {loss:.4f}，"
                        f"最佳 {self._best_loss:.4f}（可能梯度爆炸）"
                    )
                    suggestions.append("尝试降低学习率、梯度裁剪（max_norm=1.0）")
            else:
                self._explosion_count = 0
        else:
            self._explosion_count = 0

        # 平台期检测
        if self._plateau_count >= self.patience:
            warnings.append(
                f"Step {step}: loss 已连续 {self._plateau_count} 步未改善"
                f"（最佳 {self._best_loss:.4f}，当前 {loss:.4f}）"
            )
            suggestions.append("考虑降低学习率、调整模型结构或增加数据量")
            self._plateau_count = 0  # 防重复告警

        # 学习率异常检测
        if self.lr is not None and len(self._losses) >= 10:
            initial_loss = self._losses[0]
            if initial_loss > 0:
                relative_change = (initial_loss - loss) / initial_loss
                if self.lr > 0.1 and relative_change < 0.01:
                    warnings.append(
                        f"Step {step}: 学习率 {self.lr} 可能过高（loss 几乎不下降）"
                    )
                    suggestions.append("尝试将学习率降低 10 倍")
                elif self.lr < 1e-6 and relative_change < 0.01 and loss > 0.5:
                    warnings.append(
                        f"Step {step}: 学习率 {self.lr} 可能过低（loss 下降极慢）"
                    )
                    suggestions.append("尝试将学习率提高 10 倍")

        # Loss 突跳检测
        if len(recent) >= 5:
            values = [abs(v) for v in recent]
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            if std_val > 0 and abs(recent[-1] - mean_val) > 5 * std_val:
                warnings.append(
                    f"Step {step}: loss 出现异常突跳（{recent[-1]:.4f}，"
                    f"均值 {mean_val:.4f}，标准差 {std_val:.4f}）"
                )
                suggestions.append("检查该 batch 的数据质量或尝试降低学习率")

        return LossAnomalyReport(
            has_anomaly=len(warnings) > 0,
            warnings=warnings,
            suggestions=suggestions,
        )

    def reset(self) -> None:
        """重置检测器状态。"""
        self._losses.clear()
        self._steps.clear()
        self._best_loss = float("inf")
        self._best_step = 0
        self._plateau_count = 0
        self._explosion_count = 0
