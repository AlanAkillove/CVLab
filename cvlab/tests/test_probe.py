"""BatchSizeProbe 测试（CPU 模式）。"""

import pytest
import torch
import torch.nn as nn

from cvlab.probe.batch_size import BatchSizeProbe


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestBatchSizeProbe:
    def test_probe_returns_result(self):
        model = SimpleCNN()
        probe = BatchSizeProbe(model, (3, 32, 32))
        result = probe.probe()
        assert result.recommended_batch_size >= 1
        assert len(result.candidates) > 0

    def test_probe_candidates_are_recorded(self):
        model = SimpleCNN()
        probe = BatchSizeProbe(model, (3, 32, 32))
        result = probe.probe()
        assert all(isinstance(c.batch_size, int) for c in result.candidates)

    def test_probe_with_config(self):
        model = SimpleCNN()
        config = {"training": {"amp": False, "bf16": False}}
        probe = BatchSizeProbe(model, (3, 32, 32), config=config)
        result = probe.probe()
        assert not result.with_amp
