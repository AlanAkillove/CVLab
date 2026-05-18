"""随机种子管理测试。"""

import random

import numpy as np
import torch

from cvlab.core.seed import seed_everything


class TestSeed:
    def test_seed_produces_deterministic_sequence(self):
        seed_everything(42)
        a = random.random()
        b = np.random.randn()
        c = torch.randn(3)

        seed_everything(42)
        assert random.random() == a
        assert np.random.randn() == b
        assert torch.equal(torch.randn(3), c)

    def test_different_seed_different_values(self):
        seed_everything(1)
        vals1 = [random.random() for _ in range(5)]

        seed_everything(2)
        vals2 = [random.random() for _ in range(5)]

        assert vals1 != vals2
