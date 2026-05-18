"""I/O 瓶颈诊断模块测试。"""

import time

from cvlab.diagnose.io_bottleneck import IOBottleneckDetector, DataLoaderProfiler


class TestIOBottleneckDetector:
    def test_profile_context(self):
        detector = IOBottleneckDetector()
        with detector.profile() as ctx:
            for _ in range(10):
                time.sleep(0.001)
                ctx.step()

        report = detector.analyze()
        assert report.avg_data_time >= 0
        assert report.avg_compute_time >= 0
        assert 0 <= report.gpu_idle_ratio <= 1

    def test_analyze_insufficient_samples(self):
        detector = IOBottleneckDetector()
        with detector.profile() as ctx:
            ctx.step()

        report = detector.analyze()
        assert not report.is_bottleneck
        assert "采样不足" in report.details[0]

    def test_analyze_returns_recommendations(self):
        detector = IOBottleneckDetector(num_workers=2)
        with detector.profile() as ctx:
            for _ in range(20):
                time.sleep(0.002)
                ctx.step()

        report = detector.analyze()
        assert isinstance(report.is_bottleneck, bool)
        assert report.recommended_workers >= 2
        assert report.recommended_prefetch >= 2

    def test_profile_context_enter_exit(self):
        detector = IOBottleneckDetector()
        with detector.profile() as ctx:
            assert ctx._detector is detector

    def test_analyze_warn_threshold(self):
        detector = IOBottleneckDetector()
        with detector.profile() as ctx:
            for _ in range(10):
                time.sleep(0.001)
                ctx.step()

        report_loose = detector.analyze(warn_threshold=0.9)
        assert isinstance(report_loose.is_bottleneck, bool)


class TestDataLoaderProfiler:
    def test_profile_empty_iterator(self):
        class EmptyLoader:
            def __iter__(self):
                return iter([])
            def __len__(self):
                return 0

        result = DataLoaderProfiler.profile_dataloader(EmptyLoader(), num_batches=10)
        assert result["batches"] == 0
        assert result["mean"] == 0.0

    def test_profile_returns_stats(self):
        result = DataLoaderProfiler.profile_dataloader(
            _dummy_loader(), num_batches=5
        )
        assert result["batches"] > 0
        assert result["mean"] > 0
        assert result["std"] >= 0
        assert result["min"] <= result["mean"] <= result["max"]


class _DummyDataset:
    """模拟数据集，返回固定大小的 tensor。"""
    def __init__(self, size: int = 10):
        self.size = size
        self.data = [i for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def _dummy_loader():
    """创建一个简单的伪 DataLoader 用于测试。"""
    import torch
    dataset = _DummyDataset(10)
    return torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False,
    )
