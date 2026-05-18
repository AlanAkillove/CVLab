"""CheckpointManager 测试。"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cvlab.checkpoint.manager import CheckpointManager
from cvlab.db.database import Database


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def ckpt_dir(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def db():
    tmp = tempfile.mktemp(suffix=".db")
    database = Database(tmp)
    yield database
    database.close()
    Path(tmp).unlink(missing_ok=True)


class TestCheckpointManager:
    def test_save_and_load_best(self, ckpt_dir, db):
        exp_id = db.create_experiment("test", {})
        mgr = CheckpointManager(ckpt_dir, db, exp_id)
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        path = mgr.save(model, optimizer, epoch=5,
                         metrics={"val_acc": 0.85}, is_best=True)
        assert path.exists()
        assert (ckpt_dir / "best.pt").exists()

        loaded = mgr.load(best=True)
        assert loaded["epoch"] == 5

    def test_save_multiple_and_get_last(self, ckpt_dir, db):
        exp_id = db.create_experiment("test", {})
        mgr = CheckpointManager(ckpt_dir, db, exp_id)
        model = SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        mgr.save(model, opt, epoch=1, metrics={"val_acc": 0.5})
        mgr.save(model, opt, epoch=2, metrics={"val_acc": 0.6})
        mgr.save(model, opt, epoch=3, metrics={"val_acc": 0.7})

        loaded = mgr.load()  # 应加载 last
        assert loaded["epoch"] == 3

    def test_save_with_ema(self, ckpt_dir, db):
        exp_id = db.create_experiment("test", {})
        mgr = CheckpointManager(ckpt_dir, db, exp_id)
        model = SimpleModel()
        ema_model = SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # 将 EMA 模型参数设为不同值
        with torch.no_grad():
            for p in ema_model.parameters():
                p.mul_(0.5)

        mgr.save(model, opt, epoch=5, metrics={"val_acc": 0.85},
                  is_best=True, ema_model=ema_model)

        assert (ckpt_dir / "best_ema.pt").exists()
        loaded_ema = mgr.load(best=True, ema=True)
        assert loaded_ema["is_ema"]

    def test_list_checkpoints(self, ckpt_dir, db):
        exp_id = db.create_experiment("test", {})
        mgr = CheckpointManager(ckpt_dir, db, exp_id)
        model = SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        mgr.save(model, opt, epoch=1)
        mgr.save(model, opt, epoch=2)
        ckpts = mgr.list_checkpoints()
        assert len(ckpts) >= 2

    def test_load_by_epoch(self, ckpt_dir, db):
        exp_id = db.create_experiment("test", {})
        mgr = CheckpointManager(ckpt_dir, db, exp_id)
        model = SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        mgr.save(model, opt, epoch=5)
        loaded = mgr.load(epoch=5)
        assert loaded["epoch"] == 5

    def test_load_nonexistent_returns_none(self, ckpt_dir, db):
        mgr = CheckpointManager(ckpt_dir, db, "nonexistent")
        assert mgr.load(epoch=999) is None
