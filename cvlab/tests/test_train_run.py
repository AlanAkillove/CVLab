"""训练循环模块测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cvlab.train.run import (
    _create_model,
    _create_optimizer,
    _create_scheduler,
    _load_data,
    _train_epoch,
    _validate,
)


class TinyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.fc(x)


# ── _create_model ─────────────────────────────────────────


class TestCreateModel:
    """_create_model 测试。"""

    def test_resnet18_fc(self):
        """resnet18 应正确替换 fc 层输出。"""
        model = _create_model(
            {"model": {"name": "resnet18", "pretrained": False}},
            num_classes=10, device=torch.device("cpu"),
        )
        assert model.fc.out_features == 10

    def test_pretrained_false_creates_model(self):
        """pretrained=False 应正常创建模型。"""
        model = _create_model(
            {"model": {"name": "resnet18", "pretrained": False}},
            num_classes=5, device=torch.device("cpu"),
        )
        assert model.fc.out_features == 5

    def test_unknown_model(self):
        """未知模型名应抛出 ValueError。"""
        with pytest.raises(ValueError, match="不支持的模型"):
            _create_model(
                {"model": {"name": "nonexistent_model"}},
                num_classes=5, device=torch.device("cpu"),
            )


# ── _create_optimizer ─────────────────────────────────────


class TestCreateOptimizer:
    """_create_optimizer 测试。"""

    def test_adam(self):
        """optimizer=adam 应返回 Adam 实例。"""
        model = nn.Linear(10, 5)
        opt = _create_optimizer(model, {"training": {"optimizer": "adam", "lr": 0.001}})
        assert isinstance(opt, torch.optim.Adam)
        assert opt.param_groups[0]["lr"] == 0.001

    def test_adamw(self):
        """optimizer=adamw 应返回 AdamW 实例。"""
        model = nn.Linear(10, 5)
        opt = _create_optimizer(model, {"training": {"optimizer": "adamw", "lr": 0.01}})
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 0.01

    def test_sgd_with_momentum(self):
        """optimizer=sgd 应返回 SGD 实例（momentum=0.9）。"""
        model = nn.Linear(10, 5)
        opt = _create_optimizer(model, {"training": {"optimizer": "sgd", "lr": 0.1}})
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == 0.9
        assert opt.param_groups[0]["lr"] == 0.1

    def test_unknown_optimizer(self):
        """未知优化器应抛出 ValueError。"""
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="不支持的优化器"):
            _create_optimizer(model, {"training": {"optimizer": "unknown"}})

    def test_default_adam_when_missing(self):
        """training dict 缺失 optimizer 时应默认 adam。"""
        model = nn.Linear(10, 5)
        opt = _create_optimizer(model, {"training": {}})
        assert isinstance(opt, torch.optim.Adam)
        assert opt.param_groups[0]["lr"] == 0.001  # 默认 lr


# ── _create_scheduler ─────────────────────────────────────


class TestCreateScheduler:
    """_create_scheduler 测试。"""

    def test_cosine(self):
        """scheduler=cosine 应返回 CosineAnnealingLR。"""
        opt = torch.optim.SGD(nn.Linear(10, 5).parameters(), lr=0.01)
        sched = _create_scheduler(
            opt, {"training": {"scheduler": "cosine", "epochs": 10}}, steps_per_epoch=100,
        )
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step(self):
        """scheduler=step 应返回 StepLR。"""
        opt = torch.optim.SGD(nn.Linear(10, 5).parameters(), lr=0.01)
        sched = _create_scheduler(
            opt, {"training": {"scheduler": "step", "step_size": 5, "gamma": 0.5}}, steps_per_epoch=100,
        )
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)
        assert sched.step_size == 5
        assert sched.gamma == 0.5

    def test_plateau(self):
        """scheduler=plateau 应返回 ReduceLROnPlateau。"""
        opt = torch.optim.SGD(nn.Linear(10, 5).parameters(), lr=0.01)
        sched = _create_scheduler(
            opt, {"training": {"scheduler": "plateau"}}, steps_per_epoch=100,
        )
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_none(self):
        """scheduler=none 应返回 None。"""
        opt = torch.optim.SGD(nn.Linear(10, 5).parameters(), lr=0.01)
        sched = _create_scheduler(
            opt, {"training": {"scheduler": "none"}}, steps_per_epoch=100,
        )
        assert sched is None

    def test_unknown_scheduler(self):
        """未知 scheduler 应抛出 ValueError。"""
        opt = torch.optim.SGD(nn.Linear(10, 5).parameters(), lr=0.01)
        with pytest.raises(ValueError, match="不支持的 scheduler"):
            _create_scheduler(
                opt, {"training": {"scheduler": "unknown"}}, steps_per_epoch=100,
            )

    def test_default_cosine_when_missing(self):
        """training dict 缺失 scheduler 时应默认 cosine。"""
        opt = torch.optim.SGD(nn.Linear(10, 5).parameters(), lr=0.01)
        sched = _create_scheduler(opt, {"training": {}}, steps_per_epoch=100)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


# ── _load_data ────────────────────────────────────────────


class TestLoadData:
    """_load_data 测试。"""

    def test_no_dataset_config(self):
        """无 dataset 配置时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="需要指定"):
            _load_data({"training": {"batch_size": 32}})

    def test_nonexistent_path(self):
        """数据路径不存在时应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            _load_data({
                "training": {"batch_size": 32},
                "data": {"dataset": "/nonexistent/path", "dataset_name": ""},
            })


# ── _train_epoch ──────────────────────────────────────────


class TestTrainEpoch:
    """_train_epoch 测试。"""

    def test_basic_training(self):
        """基本训练应返回非零 loss 和 0-100 准确率。"""
        model = TinyModel(5)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        x = torch.randn(20, 32)
        y = torch.randint(0, 5, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        loss, acc = _train_epoch(model, loader, criterion, optimizer, device)
        assert loss > 0.0
        assert 0.0 <= acc <= 100.0
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_overfitting_small_data(self):
        """小数据上训练多步 loss 应下降。"""
        model = TinyModel(3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        device = torch.device("cpu")

        x = torch.randn(16, 32)
        y = torch.randint(0, 3, (16,))
        loader = DataLoader(TensorDataset(x, y), batch_size=16)

        loss1, _ = _train_epoch(model, loader, criterion, optimizer, device)
        loss2, _ = _train_epoch(model, loader, criterion, optimizer, device)
        assert loss2 < loss1

    def test_model_in_train_mode(self):
        """训练后模型应在 train 模式。"""
        model = TinyModel(3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        x = torch.randn(8, 32)
        y = torch.randint(0, 3, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        _train_epoch(model, loader, criterion, optimizer, device)
        assert model.training

    def test_empty_loader(self):
        """空 DataLoader 应安全返回 (0.0, 0.0)。"""
        model = TinyModel(3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        empty_loader = DataLoader(TensorDataset(torch.randn(0, 32), torch.randint(0, 3, (0,))), batch_size=8)
        loss, acc = _train_epoch(model, empty_loader, criterion, optimizer, device)
        assert loss == 0.0
        assert acc == 0.0


# ── _validate ─────────────────────────────────────────────


class TestValidate:
    """_validate 测试。"""

    def test_basic_validation(self):
        """验证应返回非零 loss 和 0-100 准确率。"""
        model = TinyModel(5)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        x = torch.randn(20, 32)
        y = torch.randint(0, 5, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        loss, acc = _validate(model, loader, criterion, device)
        assert loss > 0.0
        assert 0.0 <= acc <= 100.0
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_no_grad_context(self):
        """验证期间不应跟踪梯度。"""
        model = TinyModel(3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        x = torch.randn(8, 32)
        y = torch.randint(0, 3, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        # 验证前后梯度应不变
        param_before = model.fc.weight.grad
        _validate(model, loader, criterion, device)
        param_after = model.fc.weight.grad
        assert param_before is None or (param_after == param_before).all()

    def test_model_in_eval_mode(self):
        """验证后模型应在 eval 模式。"""
        model = TinyModel(3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        x = torch.randn(8, 32)
        y = torch.randint(0, 3, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        _validate(model, loader, criterion, device)
        assert not model.training
