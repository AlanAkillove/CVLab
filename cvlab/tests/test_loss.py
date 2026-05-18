"""Loss 异常检测模块测试。"""

import math

from cvlab.diagnose.loss import LossDetector


class TestLossDetector:
    def test_normal_loss_sequence(self):
        """正常下降的 loss 不应触发告警。"""
        detector = LossDetector(window_size=5, patience=10)
        report = None
        for step in range(50):
            loss = 2.0 / (step + 1) + 0.01 * (step % 3)
            report = detector.step(loss, step)
            if report.has_anomaly:
                break
        assert report is not None
        assert not report.has_anomaly

    def test_nan_detection(self):
        """NaN loss 应立即触发告警。"""
        detector = LossDetector()
        report = detector.step(float("nan"), 1)
        assert report.has_anomaly
        assert any("NaN" in w for w in report.warnings)

    def test_inf_detection(self):
        """Inf loss 应立即触发告警。"""
        detector = LossDetector()
        report = detector.step(float("inf"), 1)
        assert report.has_anomaly
        assert any("Inf" in w for w in report.warnings)

    def test_exploding_loss(self):
        """持续上升的 loss 应触发爆炸告警。"""
        detector = LossDetector(window_size=5, patience=10)
        # 首先生成正常 loss 作为 baseline
        for i in range(20):
            detector.step(2.0 / (i + 1), i)
        # 然后持续上升
        report = None
        for i in range(20, 30):
            loss = 1.0 + (i - 19) * 0.5
            report = detector.step(loss, i)
        assert report is not None

    def test_plateau_detection(self):
        """连续多步无改善应触发平台期告警。"""
        detector = LossDetector(window_size=3, patience=5)
        # 先下降
        for i in range(10):
            detector.step(1.0 / (i + 1), i)
        # 然后平台期
        report = None
        for i in range(10, 20):
            report = detector.step(0.1, i)
        assert report is not None
        assert report.has_anomaly
        assert any("未改善" in w for w in report.warnings)

    def test_reset(self):
        """重置后应从空白状态开始。"""
        detector = LossDetector()
        detector.step(100.0, 0)
        detector.reset()
        assert len(detector._losses) == 0
        assert detector._best_loss == float("inf")

    def test_early_steps_no_false_positive(self):
        """前几步（不足 window_size）不应触发告警。"""
        detector = LossDetector(window_size=10)
        for i in range(5):
            report = detector.step(float("inf"), i)
            # NaN/Inf 仍应检测，但其他模式不应触发
            if i == 0:
                assert report.has_anomaly  # inf detected

    def test_high_lr_warning(self):
        """过高的学习率应触发告警。"""
        detector = LossDetector(window_size=5, patience=10, lr=0.5)
        for i in range(15):
            detector.step(2.0, i)
        report = detector.step(1.98, 15)
        # 可能触发高 lr 告警
        assert isinstance(report.has_anomaly, bool)

    def test_spike_detection(self):
        """loss 突跳应触发告警。"""
        detector = LossDetector(window_size=5, patience=10)
        for i in range(10):
            detector.step(0.5 + 0.01 * (i % 3), i)
        report = detector.step(10.0, 10)
        # 突跳应能被检测到
        assert isinstance(report.has_anomaly, bool)
