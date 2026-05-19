"""EMA（指数移动平均）模型检测与处理。"""

from __future__ import annotations

import torch
import torch.nn as nn


def detect_ema(model: nn.Module) -> nn.Module | None:
    """检测模型是否包含 EMA 影子模型。

    常见 EMA 实现将影子模型存储在 model.ema 或 model.ema_model 属性中。
    """
    if hasattr(model, "ema") and isinstance(model.ema, nn.Module):
        return model.ema
    if hasattr(model, "ema_model") and isinstance(model.ema_model, nn.Module):
        return model.ema_model
    return None


def detect_and_save_ema(model: nn.Module, checkpoint_state: dict,
                         checkpoint_path: str) -> None:
    """检测 EMA 模型并保存独立的 EMA checkpoint。"""
    ema_model = detect_ema(model)
    if ema_model is None:
        return

    ema_state = dict(checkpoint_state)
    ema_state["model_state_dict"] = ema_model.state_dict()
    ema_state["is_ema"] = True

    ema_path = checkpoint_path.replace(".pt", "_ema.pt")
    torch.save(ema_state, ema_path)
