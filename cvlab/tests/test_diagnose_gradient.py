"""梯度诊断模块测试。"""

from __future__ import annotations

import json

import pytest

from cvlab.diagnose.gradient import (
    GradientDiagnosis,
    GradientDiagnosisReport,
    LayerGradStatus,
)


class TestLayerGradStatus:
    """LayerGradStatus dataclass 测试。"""

    def test_create(self):
        status = LayerGradStatus(name="conv1", mean_norm=0.01, max_norm=0.05,
                                  min_norm=0.001, recent_norm=0.02,
                                  status="normal", samples=10)
        assert status.name == "conv1"
        assert status.mean_norm == 0.01
        assert status.status == "normal"


class TestGradientDiagnosis:
    """GradientDiagnosis 单元测试（使用 mock DB）。"""

    def _make_mock_db(self, tmp_path, grad_data: dict[str, list[float]]):
        """创建含梯度指标的 mock 数据库。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "grad.db"))
        exp_id = db.create_experiment(name="grad_test", config={})
        for step in range(len(next(iter(grad_data.values())))):
            for layer_name, values in grad_data.items():
                db.log_metrics(exp_id, {f"grad_norm/{layer_name}": values[step]}, step=step)
        db.update_experiment_status(exp_id, "completed")
        return db, exp_id

    def test_normal_gradients(self, tmp_path):
        """正常梯度应报告 healthy。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [0.01, 0.012, 0.011, 0.009, 0.01],
            "conv2": [0.02, 0.021, 0.019, 0.02, 0.022],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.overall_status == "healthy"
        assert report.total_layers == 2
        assert report.healthy_count == 2

    def test_vanishing_gradient(self, tmp_path):
        """梯度消失应报告 critical。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [1e-6, 1e-7, 5e-7, 1e-6, 5e-7],
            "conv2": [0.01, 0.012, 0.011, 0.009, 0.01],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.overall_status == "critical"
        assert report.critical_count >= 1
        assert any(s.status == "vanishing" for s in report.layer_statuses)

    def test_exploding_gradient(self, tmp_path):
        """梯度爆炸应报告 critical。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [100.0, 200.0, 150.0, 300.0, 250.0],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.overall_status == "critical"
        assert any(s.status == "exploding" for s in report.layer_statuses)

    def test_low_gradient(self, tmp_path):
        """梯度偏低应报告 warning。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [0.0001, 0.00012, 0.00011, 0.00009, 0.0001],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.overall_status == "warning"
        assert any(s.status == "low" for s in report.layer_statuses)

    def test_high_gradient(self, tmp_path):
        """梯度偏高应报告 warning。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [1.5, 2.0, 1.8, 1.6, 1.9],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.overall_status == "warning"
        assert any(s.status == "high" for s in report.layer_statuses)

    def test_mixed_statuses(self, tmp_path):
        """混合状态应正确归类。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [1e-6, 1e-7, 5e-7, 1e-6, 5e-7],  # vanishing
            "conv2": [0.01, 0.012, 0.011, 0.009, 0.01],  # normal
            "conv3": [1.5, 2.0, 1.8, 1.6, 1.9],   # high
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.total_layers == 3
        assert report.critical_count >= 1  # vanishing
        assert report.warning_count >= 1   # high
        assert report.healthy_count >= 1   # normal
        assert report.overall_status == "critical"

    def test_no_grad_metrics(self, tmp_path):
        """无梯度指标应返回 warning + 提示信息。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "empty.db"))
        exp_id = db.create_experiment(name="empty", config={})
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.total_layers == 0
        assert report.overall_status == "warning"
        assert len(report.suggestions) > 0
        assert "未发现梯度数据" in report.suggestions[0]

    def test_nonexistent_experiment(self, tmp_path):
        """不存在的实验应返回无梯度数据。"""
        from cvlab.db.database import Database
        db = Database(str(tmp_path / "ghost.db"))
        report = GradientDiagnosis().diagnose("exp_ghost", db)
        assert report.total_layers == 0

    def test_suggestions_vanishing(self, tmp_path):
        """梯度消失时建议应包含激活函数检查。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [1e-6, 1e-7, 5e-7, 1e-6, 5e-7],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        suggestions = " ".join(report.suggestions)
        assert "激活函数" in suggestions or "BatchNorm" in suggestions

    def test_suggestions_exploding(self, tmp_path):
        """梯度爆炸时建议应包含梯度裁剪。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [100.0, 200.0, 150.0, 300.0, 250.0],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        suggestions = " ".join(report.suggestions)
        assert "梯度裁剪" in suggestions or "max_norm" in suggestions or "学习率" in suggestions

    def test_layer_status_samples_counted(self, tmp_path):
        """每层应有正确的采样数。"""
        db, exp_id = self._make_mock_db(tmp_path, {
            "conv1": [0.01, 0.012, 0.011],
        })
        report = GradientDiagnosis().diagnose(exp_id, db)
        assert report.layer_statuses[0].samples == 3
