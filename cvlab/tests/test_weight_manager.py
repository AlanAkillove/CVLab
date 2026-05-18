"""权重加载诊断模块测试。"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from cvlab.weights.manager import WeightManager, LoadDiagnostic


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class DifferentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


class MultiLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.mean([2, 3])
        return self.fc(x)


class TestWeightManager:
    def setup_method(self):
        self.manager = WeightManager(cache_dir=tempfile.mkdtemp())

    def test_diagnose_perfect_match(self):
        model = SimpleModel()
        state_dict = model.state_dict()
        diagnostic = self.manager.diagnose(model, state_dict)
        assert diagnostic.success
        assert not diagnostic.has_issues

    def test_diagnose_shape_mismatch(self):
        model = SimpleModel()
        state_dict = model.state_dict()
        # 篡改形状
        state_dict["fc.weight"] = torch.randn(10, 10)
        diagnostic = self.manager.diagnose(model, state_dict)
        assert len(diagnostic.shape_mismatches) > 0

    def test_diagnose_missing_keys(self):
        model = MultiLayerModel()
        state_dict = model.state_dict()
        # 删除一个 key
        del state_dict["fc.weight"]
        diagnostic = self.manager.diagnose(model, state_dict)
        assert "fc.weight" in diagnostic.missing_keys

    def test_diagnose_unexpected_keys(self):
        model = SimpleModel()
        state_dict = model.state_dict()
        state_dict["extra_key"] = torch.randn(5)
        diagnostic = self.manager.diagnose(model, state_dict)
        assert "extra_key" in diagnostic.unexpected_keys

    def test_load_nonexistent_file(self):
        model = SimpleModel()
        diagnostic = self.manager.load(model, "/nonexistent/path.pth")
        assert not diagnostic.success

    def test_load_success(self):
        model = SimpleModel()
        path = Path(tempfile.mkdtemp()) / "weights.pth"
        torch.save(model.state_dict(), path)

        new_model = SimpleModel()
        diagnostic = self.manager.load(new_model, str(path))
        assert diagnostic.success

    def test_cache_info(self):
        info = self.manager.cache_info()
        assert "total_files" in info
        assert "total_size_mb" in info
        assert "cache_dir" in info

    def test_cache_info_with_files(self):
        path = Path(self.manager.cache_dir) / "test.pth"
        torch.save({"test": torch.randn(5)}, path)

        info = self.manager.cache_info()
        assert info["total_files"] >= 1

    def test_compute_hash(self):
        path = Path(tempfile.mkdtemp()) / "weights.pth"
        torch.save({"a": torch.tensor([1.0])}, path)
        h = self.manager.compute_hash(str(path))
        assert h is not None
        assert len(h) == 64  # SHA256

    def test_compute_hash_nonexistent(self):
        h = self.manager.compute_hash("/nonexistent.pth")
        assert h is None

    def test_diff_weights(self):
        path_a = Path(tempfile.mkdtemp()) / "a.pth"
        path_b = Path(tempfile.mkdtemp()) / "b.pth"
        torch.save({"w": torch.ones(5)}, path_a)
        torch.save({"w": torch.zeros(5)}, path_b)

        diffs = self.manager.diff_weights(str(path_a), str(path_b))
        assert len(diffs) > 0
        assert any(d["diff"] == "different" for d in diffs)

    def test_diff_weights_missing_keys(self):
        path_a = Path(tempfile.mkdtemp()) / "a.pth"
        path_b = Path(tempfile.mkdtemp()) / "b.pth"
        torch.save({"w1": torch.ones(5)}, path_a)
        torch.save({"w2": torch.zeros(3)}, path_b)

        diffs = self.manager.diff_weights(str(path_a), str(path_b))
        key_types = {d["diff"] for d in diffs}
        assert "only_in_a" in key_types
        assert "only_in_b" in key_types

    def test_diagnostic_summary_perfect(self):
        diag = LoadDiagnostic(success=True, missing_keys=[], unexpected_keys=[], shape_mismatches=[])
        assert "成功" in diag.summary()

    def test_diagnostic_summary_with_issues(self):
        diag = LoadDiagnostic(
            success=False,
            missing_keys=["fc.weight"],
            unexpected_keys=["extra"],
            shape_mismatches=[{"key": "conv.weight", "expected": [8, 3, 3, 3], "actual": [4, 3, 3, 3]}],
        )
        assert diag.has_issues
        assert "缺失" in diag.summary()
        assert "多余" in diag.summary()
        assert "形状不匹配" in diag.summary()

    def test_ignore_common_buffers(self):
        model = SimpleModel()
        state_dict = model.state_dict()
        # 添加常见的被忽略 buffer
        state_dict["num_batches_tracked"] = torch.tensor(0)
        diagnostic = self.manager.diagnose(model, state_dict)
        assert "num_batches_tracked" not in diagnostic.unexpected_keys
