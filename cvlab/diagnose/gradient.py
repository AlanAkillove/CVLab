"""梯度健康诊断 — 加载已记录的梯度范数，分析各层梯度状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cvlab.db.database import Database


@dataclass
class LayerGradStatus:
    name: str
    mean_norm: float
    max_norm: float
    min_norm: float
    recent_norm: float
    status: str  # normal / vanishing / exploding / low / high
    samples: int


@dataclass
class GradientDiagnosisReport:
    experiment_id: str
    layer_statuses: list[LayerGradStatus]
    healthy_count: int
    warning_count: int
    critical_count: int
    total_layers: int
    overall_status: str  # healthy / warning / critical
    suggestions: list[str] = field(default_factory=list)


class GradientDiagnosis:
    """梯度健康诊断。

    从实验指标库中读取已记录的梯度范数，分析各层梯度状态。
    注意：梯度监控必须在训练时通过 Tracker.watch() 注入 Hook 才能采集数据。
    """

    VANISHING_THRESHOLD = 1e-5
    LOW_THRESHOLD = 1e-3
    HIGH_THRESHOLD = 1.0
    EXPLODING_THRESHOLD = 10.0

    def diagnose(self, experiment_id: str, db: Database | None = None) -> GradientDiagnosisReport:
        """对指定实验执行梯度诊断。

        Args:
            experiment_id: 实验 ID。
            db: Database 实例，不传则创建默认实例。

        Returns:
            梯度诊断报告。
        """
        if db is None:
            from cvlab.db.database import Database
            db = Database()

        # 加载所有 grad_norm/ 前缀的指标
        metrics = db.get_metrics(experiment_id)
        grad_metrics = [m for m in metrics if m["key"].startswith("grad_norm/")]

        if not grad_metrics:
            return GradientDiagnosisReport(
                experiment_id=experiment_id,
                layer_statuses=[],
                healthy_count=0,
                warning_count=0,
                critical_count=0,
                total_layers=0,
                overall_status="warning",
                suggestions=["未发现梯度数据。训练时需要通过 Tracker.watch() 注入梯度监控 Hook。"],
            )

        # 按 layer 分组
        layer_data: dict[str, list[float]] = {}
        for m in grad_metrics:
            layer_name = m["key"].replace("grad_norm/", "", 1)
            layer_data.setdefault(layer_name, []).append(m["value"])

        layer_statuses: list[LayerGradStatus] = []
        critical_count = 0
        warning_count = 0
        healthy_count = 0
        suggestions: list[str] = []

        for name, norms in layer_data.items():
            avg_norm = sum(norms) / len(norms)
            max_norm = max(norms)
            min_norm = min(norms)
            recent_norm = norms[-1] if norms else 0.0

            if avg_norm < self.VANISHING_THRESHOLD:
                status = "vanishing"
                critical_count += 1
            elif avg_norm > self.EXPLODING_THRESHOLD:
                status = "exploding"
                critical_count += 1
            elif avg_norm < self.LOW_THRESHOLD:
                status = "low"
                warning_count += 1
            elif avg_norm > self.HIGH_THRESHOLD:
                status = "high"
                warning_count += 1
            else:
                status = "normal"
                healthy_count += 1

            layer_statuses.append(LayerGradStatus(
                name=name,
                mean_norm=avg_norm,
                max_norm=max_norm,
                min_norm=min_norm,
                recent_norm=recent_norm,
                status=status,
                samples=len(norms),
            ))

        # 排序：异常层在前
        _status_order = {"vanishing": 0, "exploding": 1, "low": 2, "high": 3, "normal": 4}
        layer_statuses.sort(key=lambda x: (_status_order.get(x.status, 5), x.name))

        if critical_count > 0:
            overall_status = "critical"
            vanishing = [s for s in layer_statuses if s.status == "vanishing"]
            exploding = [s for s in layer_statuses if s.status == "exploding"]
            if vanishing:
                names = ", ".join(s.name for s in vanishing[:5])
                suggestions.append(
                    f"梯度消失层 ({len(vanishing)}): {names}。"
                    "建议检查激活函数（避免 sigmoid）、添加 BatchNorm、使用残差连接。"
                )
            if exploding:
                names = ", ".join(s.name for s in exploding[:5])
                suggestions.append(
                    f"梯度爆炸层 ({len(exploding)}): {names}。"
                    "建议开启梯度裁剪 (max_norm=1.0)、降低学习率。"
                )
        elif warning_count > 0:
            overall_status = "warning"
            if warning_count > len(layer_statuses) // 2:
                suggestions.append("多数层梯度偏低，考虑增加学习率或检查网络深度是否过大。")
            suggestions.append("建议监控后续训练中梯度变化趋势。")
        else:
            overall_status = "healthy"
            suggestions.append("各层梯度分布正常。")

        return GradientDiagnosisReport(
            experiment_id=experiment_id,
            layer_statuses=layer_statuses,
            healthy_count=healthy_count,
            warning_count=warning_count,
            critical_count=critical_count,
            total_layers=len(layer_statuses),
            overall_status=overall_status,
            suggestions=suggestions,
        )
