"""超参扫描调度器 - 管理 Sweep 的创建、执行和结果收集。"""

from __future__ import annotations

import copy
from typing import Any

from cvlab.db.database import Database
from cvlab.sweep.grid import generate_grid
from cvlab.sweep.random import sample_random


class Sweeper:
    """超参扫描调度器。

    用法:
        sweeper = Sweeper()
        sweep_id = sweeper.create_sweep(
            base_config=base_config,
            strategy="grid",
            params={"training.lr": [0.001, 0.0001]},
        )
        trials = sweeper.get_trials(sweep_id)
    """

    def __init__(self, db: Database | None = None):
        self.db = db or Database()

    def create_sweep(
        self,
        base_config: dict[str, Any],
        strategy: str,
        params: dict[str, Any],
        name: str | None = None,
        seed: int | None = None,
        max_trials: int | None = None,
    ) -> str:
        """创建 Sweep 并生成所有 trial 配置。

        Args:
            base_config: 基础实验配置。
            strategy: "grid" 或 "random"。
            params: 搜索空间定义。
                grid 模式: {"key": [values]} 格式。
                random 模式: {"key": {"type": "choice", "values": [...]}} 格式。
            name: Sweep 名称。
            seed: 随机种子（仅 random 模式）。
            max_trials: 最大 trial 数（仅 random 模式）。

        Returns:
            Sweep ID。
        """
        # 生成一组实验配置
        if strategy == "grid":
            trial_configs = self._prepare_grid(params)
        elif strategy == "random":
            trial_configs = self._prepare_random(params, max_trials or 10, seed)
        else:
            raise ValueError(f"不支持的策略: {strategy}")

        merged_configs = []
        for tc in trial_configs:
            merged = copy.deepcopy(base_config)
            for key, value in tc.items():
                self._set_nested(merged, key, value)
            merged_configs.append(merged)

        sweep_config = copy.deepcopy(base_config)
        sweep_config["_sweep"] = {"strategy": strategy, "params": params}
        if name:
            sweep_config["name"] = name

        # 创建 base experiment 作为 sweep 的锚点
        base_exp_id = self.db.create_experiment(
            name=name or f"sweep_{strategy}",
            config=sweep_config,
        )

        import random as _random
        from datetime import datetime
        ts = datetime.now().strftime('%y%m%d_%H%M%S')
        sweep_id = f"sweep_{ts}_{_random.randint(10000, 99999)}"

        self.db.create_sweep(sweep_id, base_exp_id, sweep_config, strategy)

        for i, config in enumerate(merged_configs):
            trial_exp_id = self.db.create_experiment(
                name=f"{name or 'trial'}_{i}",
                config=config,
            )
            self.db.add_sweep_trial(sweep_id, trial_exp_id, i, config)

        return sweep_id

    def get_trials(self, sweep_id: str) -> list[dict[str, Any]]:
        """获取 Sweep 的所有 trial 及其状态。"""
        return self.db.get_sweep_trials(sweep_id)

    def get_sweep(self, sweep_id: str) -> dict[str, Any] | None:
        """获取 Sweep 信息。"""
        return self.db.get_sweep(sweep_id)

    def get_best_trial(self, sweep_id: str, metric_key: str = "val/acc"
                       ) -> dict[str, Any] | None:
        """获取指定指标最优的 trial。"""
        trials = self.get_trials(sweep_id)
        best = None
        best_val = float("-inf") if "acc" in metric_key or "f1" in metric_key else float("inf")
        for t in trials:
            exp = self.db.get_experiment(t["experiment_id"])
            if not exp:
                continue
            metrics = self.db.get_metrics(t["experiment_id"])
            values = [m["value"] for m in metrics if m["key"] == metric_key]
            if not values:
                continue
            val = values[-1]
            if (("acc" in metric_key or "f1" in metric_key) and val > best_val) or (("loss" in metric_key or "error" in metric_key) and val < best_val):
                best_val = val
                best = {**t, "metric_value": val}
        return best

    def get_top_trials(self, sweep_id: str, metric_key: str = "val/acc",
                        n: int = 5) -> list[dict[str, Any]]:
        """获取指定指标 Top N 的 trial。

        Args:
            sweep_id: Sweep ID。
            metric_key: 目标指标名。
            n: 返回条数。

        Returns:
            按指标排序的 trial 列表，每个包含 trial 信息和 metric_value。
        """
        trials = self.get_trials(sweep_id)
        scored: list[dict[str, Any]] = []
        for t in trials:
            exp = self.db.get_experiment(t["experiment_id"])
            if not exp or exp["status"] != "completed":
                continue
            metrics = self.db.get_metrics(t["experiment_id"])
            values = [m["value"] for m in metrics if m["key"] == metric_key]
            if not values:
                continue
            scored.append({**t, "metric_value": values[-1]})

        # 按指标排序
        maximize = "acc" in metric_key or "f1" in metric_key
        scored.sort(key=lambda x: x["metric_value"], reverse=maximize)
        return scored[:n]

    def _prepare_grid(self, params: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """处理 grid 模式参数。"""
        return generate_grid(params)

    def _prepare_random(self, params: dict[str, dict[str, Any]],
                        n_trials: int, seed: int | None) -> list[dict[str, Any]]:
        """处理 random 模式参数。"""
        return sample_random(params, n_trials, seed)

    @staticmethod
    def _set_nested(config: dict[str, Any], key: str, value: Any) -> None:
        """设置嵌套 key（如 "training.lr"）。"""
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
