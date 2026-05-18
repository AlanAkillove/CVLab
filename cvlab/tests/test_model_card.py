"""模型性能画像模块测试。"""

import torch
import torch.nn as nn

from cvlab.profile.model_card import ModelProfiler, flops_to_text


class TinyModel(nn.Module):
    """极简模型用于测试。"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestModelProfiler:
    def test_profile_returns_card(self):
        model = TinyModel()
        profiler = ModelProfiler(device="cpu")
        card = profiler.profile(model, (1, 3, 32, 32), warmup=1, repeats=2)

        assert card.name == "TinyModel"
        assert card.total_params > 0
        assert card.trainable_params > 0
        assert card.params_millions > 0
        assert card.forward_time_ms > 0

    def test_flops_count(self):
        model = TinyModel()
        profiler = ModelProfiler(device="cpu")
        card = profiler.profile(model, (1, 3, 32, 32), warmup=1, repeats=2)

        # Conv2d(3->8, 3x3): 3*8*3*3*32*32 = 221,184
        # Linear(8->10): 8*10 = 80
        # Total: ~221,264
        assert card.flops_macs > 0
        assert card.flops_giga > 0

    def test_layer_statistics(self):
        model = TinyModel()
        profiler = ModelProfiler(device="cpu")
        card = profiler.profile(model, (1, 3, 32, 32), warmup=1, repeats=2)

        assert len(card.layer_stats) > 0
        for stat in card.layer_stats:
            assert "name" in stat
            assert "params" in stat
            assert stat["params"] > 0

    def test_summary_string(self):
        model = TinyModel()
        profiler = ModelProfiler(device="cpu")
        card = profiler.profile(model, (1, 3, 32, 32), warmup=1, repeats=2)

        summary = card.summary()
        assert "TinyModel" in summary
        assert "M" in summary
        assert "ms" in summary

    def test_flops_to_text(self):
        assert flops_to_text(1_000) == "1000"
        assert flops_to_text(1_000_000) == "1.00 M"
        assert flops_to_text(1_000_000_000) == "1.00 G"
        assert flops_to_text(1_000_000_000_000) == "1.00 T"


class TestEdgeCases:
    def test_no_trainable_params(self):
        """没有可训练参数的模型。"""
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        profiler = ModelProfiler(device="cpu")
        card = profiler.profile(model, (1, 3, 32, 32), warmup=1, repeats=2)
        assert card.total_params == 0
        assert card.backward_time_ms == 0.0
