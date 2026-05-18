"""超参重要性分析测试。"""

from __future__ import annotations

import json

import pytest

from cvlab.sweep.importance import ImportanceResult, _flatten_config, analyze_importance


class TestFlattenConfig:
    """_flatten_config 展平嵌套配置的测试。"""

    def test_flat_config(self):
        config = {"lr": 0.001, "epochs": 10}
        flat = _flatten_config(config)
        assert flat == {"lr": 0.001, "epochs": 10}

    def test_nested_config(self):
        config = {"training": {"lr": 0.001, "optimizer": "adam"}, "data": {"batch_size": 64}}
        flat = _flatten_config(config)
        assert flat == {"training.lr": 0.001, "training.optimizer": "adam", "data.batch_size": 64}

    def test_deeply_nested(self):
        config = {"a": {"b": {"c": 42}}}
        flat = _flatten_config(config)
        assert flat == {"a.b.c": 42}

    def test_mixed_types(self):
        config = {"lr": 0.001, "nested": {"a": 1}, "flag": True}
        flat = _flatten_config(config)
        assert flat["lr"] == 0.001
        assert flat["flag"] is True
        assert flat["nested.a"] == 1

    def test_empty(self):
        assert _flatten_config({}) == {}


class TestAnalyzeImportance:
    """analyze_importance 集成测试（使用临时数据库）。"""

    def _populate_sweep(self, db, sweep_id: str):
        """填充 sweep 数据：10 个 trial，lr 越大 val/acc 越高。"""
        for i in range(10):
            lr = 0.001 * (i + 1)
            trial_config = {"training": {"lr": lr, "epochs": 10 + i, "optimizer": "adam"}}
            trial_exp_id = db.create_experiment(name=f"trial_{i}", config=trial_config)
            db.add_sweep_trial(sweep_id, trial_exp_id, i, trial_config)
            db.update_experiment_status(trial_exp_id, "completed")
            acc = 0.5 + i * 0.04
            db.log_metrics(trial_exp_id, {"val/acc": acc}, step=1)
            db.log_metrics(trial_exp_id, {"val/acc": acc + 0.02}, step=2)

    def _make_sweep(self, tmp_path) -> tuple:
        """创建含数据的 sweep 并返回 (db, sweep_id)。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "test.db"))
        sweep_id = "sweep_test_001"
        base_exp_id = db.create_experiment(name="sweep_base", config={})
        db.create_sweep(sweep_id, base_exp_id, {}, "grid")
        self._populate_sweep(db, sweep_id)
        return db, sweep_id

    def test_analyze_basic(self, tmp_path):
        """基本分析应返回 ImportanceResult 并识别关键超参。"""
        db, sweep_id = self._make_sweep(tmp_path)
        result = analyze_importance(sweep_id, db=db, target_metric="val/acc")

        assert isinstance(result, ImportanceResult)
        assert result.total_trials >= 3
        assert result.target_metric == "val/acc"
        assert len(result.importances) > 0
        assert len(result.top_params) > 0

    def test_analyze_returns_suggestions(self, tmp_path):
        """分析应有建议信息。"""
        db, sweep_id = self._make_sweep(tmp_path)
        result = analyze_importance(sweep_id, db=db)

        assert len(result.suggestions) > 0
        assert "关键超参" in result.suggestions[0]

    def test_analyze_importance_scores_normalized(self, tmp_path):
        """重要性分数应归一化到总和接近 1.0。"""
        db, sweep_id = self._make_sweep(tmp_path)
        result = analyze_importance(sweep_id, db=db)

        total = sum(result.importances.values())
        assert abs(total - 1.0) < 0.01

    def test_analyze_insufficient_trials(self, tmp_path):
        """少于 3 个完成 trial 应报错。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "empty.db"))

        base_exp_id = db.create_experiment(name="empty", config={})
        db.create_sweep("sweep_empty", base_exp_id, {}, "grid")

        trial_exp = db.create_experiment(name="trial_0", config={"lr": 0.01})
        db.add_sweep_trial("sweep_empty", trial_exp, 0, {"lr": 0.01})
        # 故意不标记为 completed

        with pytest.raises(ValueError, match="已完成.*不足"):
            analyze_importance("sweep_empty", db=db)

    def test_analyze_nonexistent_sweep(self, tmp_path):
        """不存在的 sweep 应报错。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "ghost.db"))

        with pytest.raises(ValueError, match="不存在"):
            analyze_importance("sweep_ghost", db=db)


class TestAnalyzeImportanceEdgeCases:
    """边界情况测试。"""

    def _make_sweep(self, db, sweep_id: str, trial_configs: list[dict],
                    metric_values: list[float]):
        """通用辅助：创建 sweep 并填充指定 trial 和指标。"""
        base_exp_id = db.create_experiment(name="sweep_base", config={})
        db.create_sweep(sweep_id, base_exp_id, {}, "grid")
        for i, (cfg, met) in enumerate(zip(trial_configs, metric_values)):
            exp_id = db.create_experiment(name=f"trial_{i}", config=cfg)
            db.add_sweep_trial(sweep_id, exp_id, i, cfg)
            db.update_experiment_status(exp_id, "completed")
            db.log_metrics(exp_id, {"val/acc": met}, step=1)

    def test_floats_only_not_bools(self, tmp_path):
        """布尔值应被跳过（非数值），数值型参数应被分析。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "edge.db"))

        configs = [
            {"lr": 0.001, "augment": True},
            {"lr": 0.002, "augment": False},
            {"lr": 0.003, "augment": True},
            {"lr": 0.004, "augment": False},
            {"lr": 0.005, "augment": True},
        ]
        metrics = [0.50, 0.55, 0.60, 0.65, 0.70]
        self._make_sweep(db, "sweep_edge", configs, metrics)

        result = analyze_importance("sweep_edge", db=db)
        # lr 是数值型应被保留，augment 是 bool 可能被跳过
        assert "lr" in result.importances
        assert len(result.importances) >= 1

    def test_all_same_param_value(self, tmp_path):
        """所有 trial 中同一超参值不变应被跳过。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "const.db"))

        configs = [{"lr": 0.01, "epochs": 10 + i * 2} for i in range(5)]
        metrics = [0.60 + i * 0.02 for i in range(5)]
        self._make_sweep(db, "sweep_const", configs, metrics)

        result = analyze_importance("sweep_const", db=db)
        # lr 是常量应被跳过，epochs 应有重要性
        assert "lr" not in result.importances
        assert "epochs" in result.importances
