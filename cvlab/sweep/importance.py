"""超参重要性分析 — Sweep 完成后分析各超参对目标指标的影响程度。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cvlab.db.database import Database


@dataclass
class ImportanceResult:
    """超参重要性分析结果。"""
    importances: dict[str, float]  # 超参名 → 重要性分数 (0~1)
    total_trials: int
    target_metric: str
    top_params: list[str]          # 按重要性降序
    suggestions: list[str] = field(default_factory=list)


def analyze_importance(sweep_id: str, db: Database | None = None,
                       target_metric: str = "val/acc") -> ImportanceResult:
    """分析 Sweep 中各超参对目标指标的重要性。

    Args:
        sweep_id: Sweep ID。
        db: Database 实例。
        target_metric: 分析用的目标指标名。

    Returns:
        重要性分析结果。
    """
    if db is None:
        from cvlab.db.database import Database
        db = Database()

    sweep = db.get_sweep(sweep_id)
    if not sweep:
        raise ValueError(f"Sweep {sweep_id} 不存在")

    trials = db.get_sweep_trials(sweep_id)
    completed_trials = [t for t in trials if t.get("exp_status") == "completed"]

    if len(completed_trials) < 3:
        raise ValueError(
            f"已完成 trial 不足 ({len(completed_trials)}/3)，"
            "需要至少 3 个完成实验才能分析"
        )

    # 收集特征和指标
    import json
    X_raw: list[dict[str, Any]] = []
    y: list[float] = []

    for t in completed_trials:
        config = json.loads(t["config_json"])
        # 提取展平的超参（非嵌套）
        flat_params = _flatten_config(config)
        X_raw.append(flat_params)

        # 获取目标指标
        exp_id = t["experiment_id"]
        metrics = db.get_metrics(exp_id, keys=[target_metric])
        if not metrics:
            continue
        best_val = max(m["value"] for m in metrics)
        y.append(best_val)

    if len(y) < 3:
        raise ValueError(f"有效指标数据不足 ({len(y)}/3)")

    # 只保留数值型超参
    param_names = sorted(set().union(*(set(p.keys()) for p in X_raw)))
    import numpy as np
    X: list[list[float]] = []
    valid_param_names: list[str] = []
    for pname in param_names:
        values = []
        for p in X_raw:
            v = p.get(pname)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(float(v))
            else:
                values.append(float("nan"))
        # 跳过全是 NaN 或常量列
        unique_vals = set(v for v in values if not np.isnan(v))
        if len(unique_vals) >= 2:
            valid_param_names.append(pname)
            X.append(values)
        # 布尔/枚举参数跳过（需要 one-hot，但目前 skip 简化处理）

    if len(valid_param_names) < 1:
        raise ValueError("没有可分析的数值型超参")

    X_arr = np.array(X, dtype=np.float64).T  # (n_trials, n_params)
    y_arr = np.array(y, dtype=np.float64)

    # 去除含 NaN 的行
    mask = ~np.isnan(X_arr).any(axis=1)
    X_clean = X_arr[mask]
    y_clean = y_arr[mask]

    if len(X_clean) < 3:
        raise ValueError(f"清洗后有效数据不足 ({len(X_clean)}/3)")

    # 随机森林特征重要性
    from sklearn.ensemble import RandomForestRegressor
    n_estimators = min(100, max(10, len(X_clean) * 5))
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=1)
    rf.fit(X_clean, y_clean)

    importances = rf.feature_importances_
    # 归一化到 [0, 1]
    if importances.sum() > 0:
        importances = importances / importances.sum()

    # 构建结果
    param_imp = dict(zip(valid_param_names, importances.tolist(), strict=False))
    sorted_params = sorted(param_imp.items(), key=lambda x: -x[1])

    suggestions = []
    if sorted_params:
        top_name, top_score = sorted_params[0]
        suggestions.append(f"关键超参: {top_name} ({top_score*100:.0f}% 影响)")
        if len(sorted_params) > 1:
            sec_name, sec_score = sorted_params[1]
            if sec_score < top_score * 0.3:
                suggestions.append(f"其他超参影响显著低于 {top_name}，建议专注调优 {top_name}")
            else:
                suggestions.append(f"{sec_name} 也有显著影响 ({sec_score*100:.0f}%)，可联合调优")

    return ImportanceResult(
        importances=param_imp,
        total_trials=len(completed_trials),
        target_metric=target_metric,
        top_params=[name for name, _ in sorted_params],
        suggestions=suggestions,
    )


def _flatten_config(config: dict, prefix: str = "") -> dict[str, Any]:
    """展平嵌套配置字典。"""
    result: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_config(value, full_key))
        else:
            result[full_key] = value
    return result
