"""Tracker 核心测试（不需要 GPU）。"""


import pytest
import torch
import torch.nn as nn

from cvlab.core.tracker import Tracker


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.fc = nn.Linear(8 * 30 * 30, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def cvlab_dir(tmp_path):
    d = tmp_path / ".cvlab"
    d.mkdir()
    return str(d)


class TestTracker:
    def test_create_experiment(self, cvlab_dir):
        tracker = Tracker(config={"name": "test_exp"}, cvlab_dir=cvlab_dir)
        assert tracker.experiment_id.startswith("test-exp") or tracker.experiment_id.startswith("exp_")
        assert tracker.exp_dir.exists()
        assert (tracker.exp_dir / "config.yaml").exists()
        assert tracker.config is not None

    def test_log_metrics(self, cvlab_dir):
        tracker = Tracker(config={"name": "test"}, cvlab_dir=cvlab_dir)
        tracker.log({"train/loss": 2.0, "train/acc": 0.5}, step=0)
        tracker.log({"train/loss": 1.5, "train/acc": 0.6}, step=1)
        metrics = tracker.db.get_metrics(tracker.experiment_id)
        assert len(metrics) == 4

    def test_watch_hook(self, cvlab_dir):
        tracker = Tracker(config={"name": "test_watch"}, cvlab_dir=cvlab_dir)
        model = SimpleModel()
        monitor = tracker.watch(model, watch_layers=["conv1"], log_freq=1)
        assert monitor is not None

        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        report = monitor.step(1)
        assert report is not None
        assert "conv1" in report.layer_grad_norms
        assert report.layer_grad_norms["conv1"] > 0

    def test_finish(self, cvlab_dir):
        tracker = Tracker(config={"name": "test_finish"}, cvlab_dir=cvlab_dir)
        eid = tracker.experiment_id
        tracker.finish("completed")
        exp = tracker.db.get_experiment(eid)
        assert exp["status"] == "completed"

    def test_save_and_load_checkpoint(self, cvlab_dir):
        tracker = Tracker(config={"name": "test_ckpt"}, cvlab_dir=cvlab_dir)
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        path = tracker.save_checkpoint(model, optimizer, epoch=5,
                                        metrics={"val_acc": 0.85}, is_best=True)
        assert path.exists()

        loaded = tracker.load_checkpoint(best=True)
        assert loaded is not None
        assert loaded["epoch"] == 5

    def test_reproduce_command(self, cvlab_dir):
        tracker = Tracker(config={"name": "test_repro"}, cvlab_dir=cvlab_dir)
        cmd = tracker.get_reproduce_command()
        assert tracker.experiment_id in cmd

    def test_snapshot_dataset(self, cvlab_dir, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "img1.jpg").write_text("fake")
        (dataset / "img2.jpg").write_text("fake")

        tracker = Tracker(config={"name": "test_data"}, cvlab_dir=cvlab_dir)
        tracker.snapshot_dataset(str(dataset))
        exp = tracker.db.get_experiment(tracker.experiment_id)
        assert exp["dataset_files"] == 2
