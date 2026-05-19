"""cvlab export - 模型导出。

支持导出为 ONNX / TorchScript 格式，含输入输出形状校验。

用法:
    cvlab export --checkpoint <path> --format onnx --input 1x3x224x224
    cvlab export --checkpoint <path> --format torchscript --input 1x3x224x224
    cvlab export --checkpoint <path> --format onnx --opset 17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from cvlab.cli.console import error, header, info, result, success
from cvlab.i18n import _


def cmd_export(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        error(_("Checkpoint 文件不存在: {}").format(checkpoint_path))
        return 1

    # 解析输入形状
    try:
        input_shape = tuple(int(d) for d in args.input.split("x"))
        if len(input_shape) < 1 or len(input_shape) > 4:
            raise ValueError
    except (ValueError, TypeError):
        error(_("输入形状格式错误: {} (需要 Nx[CHW]，如 1x3x224x224)").format(args.input))
        return 1

    header(_("模型导出"))
    result(_("Checkpoint"), str(checkpoint_path))
    result(_("导出格式"), args.format.upper())
    result(_("输入形状"), str(input_shape))

    # 加载 model
    info(_("加载模型..."))
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        error(_("加载 Checkpoint 失败: {}").format(e))
        return 1

    # 尝试从 checkpoint 中恢复模型结构
    model = _reconstruct_model(checkpoint, args)
    if model is None:
        error(_("无法从 checkpoint 重建模型"))
        info(_("提示: 目前仅支持包含 'model_state_dict' 或完整模型序列化的 checkpoint"))
        info(_("自定义模型请先加载后使用 Python API 导出"))
        return 1

    model.eval()

    # 创建 dummy input
    try:
        dummy_input = torch.randn(input_shape)
    except RuntimeError as e:
        error(_("创建 dummy input 失败: {}").format(e))
        return 1

    # 前向验证
    info(_("前向验证..."))
    try:
        with torch.no_grad():
            output = model(dummy_input)
        result(_("输出形状"), str(tuple(output.shape)))
    except Exception as e:
        error(_("前向传播失败: {}").format(e))
        info(_("检查输入形状是否与模型匹配"))
        return 1

    # 导出
    output_path = Path(args.output) if args.output else _default_output_path(checkpoint_path, args.format)

    try:
        if args.format == "onnx":
            _export_onnx(model, dummy_input, output_path, args)
        elif args.format == "torchscript":
            _export_torchscript(model, dummy_input, output_path, args)
        else:
            error(_("不支持的导出格式: {}").format(args.format))
            return 1
    except Exception as e:
        error(_("导出失败: {}").format(e))
        return 1

    # 导出后校验
    file_size = output_path.stat().st_size
    result(_("导出路径"), str(output_path.resolve()))
    result(_("文件大小"), f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / 1024 / 1024:.2f} MB")
    success(_("导出成功"))

    return 0


def _reconstruct_model(checkpoint: dict, args: argparse.Namespace) -> nn.Module | None:
    """从 checkpoint 重建模型。"""
    # 方案1: 完整模型序列化
    if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
        return checkpoint["model"]

    # 方案2: model_state_dict + model_name
    state_dict = None
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # 可能是直接保存的 state_dict
        for key in checkpoint:
            if isinstance(checkpoint[key], torch.Tensor):
                state_dict = checkpoint
                break

    if state_dict is None:
        return None

    # 从 model_name 恢复
    model_name = _detect_model_name(checkpoint)
    if model_name is None:
        return None

    try:
        import torchvision.models as models
        model = getattr(models, model_name)(pretrained=False)
        # 调整分类头
        num_classes = _detect_num_classes(state_dict)
        if num_classes and hasattr(model, "fc") and model.fc.out_features != num_classes:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif num_classes and hasattr(model, "classifier"):
            # 处理 mobilenet 等
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            info(_("缺失 key: {}").format(len(missing)))
        if unexpected:
            info(_("意外 key: {}").format(len(unexpected)))
        return model
    except (ImportError, AttributeError):
        return None


def _detect_model_name(checkpoint: dict) -> str | None:
    """从 checkpoint 元数据或 config 中检测模型名称。"""
    config = checkpoint.get("config", checkpoint.get("config_json", {}))
    if isinstance(config, str):
        import json
        try:
            config = json.loads(config)
        except (json.JSONDecodeError, TypeError):
            config = {}
    model_name = config.get("model", {}).get("name") if isinstance(config, dict) else None
    return model_name


def _detect_num_classes(state_dict: dict) -> int | None:
    """从 state_dict 中检测分类数。"""
    for key in state_dict:
        if (key.endswith(".weight") or key.endswith("bias")) and ("fc" in key or "classifier" in key):
            return state_dict[key].shape[0]
    return None


def _export_onnx(model: nn.Module, dummy_input: torch.Tensor,
                 output_path: Path, args: argparse.Namespace) -> None:
    """导出 ONNX。"""
    info(_("导出 ONNX (opset {})...").format(args.opset))
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        } if args.dynamic else None,
    )


def _export_torchscript(model: nn.Module, dummy_input: torch.Tensor,
                        output_path: Path, args: argparse.Namespace) -> None:
    """导出 TorchScript。"""
    mode = "trace" if args.script_mode == "trace" else "script"
    info(_("导出 TorchScript ({})...").format(mode))

    if mode == "trace":
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path))
    else:
        scripted_model = torch.jit.script(model)
        scripted_model.save(str(output_path))


def _default_output_path(checkpoint_path: Path, fmt: str) -> Path:
    """生成默认输出路径。"""
    stem = checkpoint_path.stem
    if stem.endswith(".pt") or stem.endswith(".pth"):
        stem = Path(stem).stem
    return checkpoint_path.parent / f"{stem}.{fmt}"


def add_subparser(sub) -> None:
    p = sub.add_parser("export", help=_("导出模型 (ONNX / TorchScript)"))
    p.add_argument("--checkpoint", "-c", required=True, help=_("Checkpoint 路径"))
    p.add_argument("--format", "-f", choices=["onnx", "torchscript"], default="onnx",
                   help=_("导出格式 (默认 onnx)"))
    p.add_argument("--input", "-i", default="1x3x224x224",
                   help=_("输入形状，如 1x3x224x224 (默认 1x3x224x224)"))
    p.add_argument("--output", "-o", default=None, help=_("输出路径（自动生成）"))
    p.add_argument("--opset", type=int, default=17, help=_("ONNX opset 版本 (默认 17)"))
    p.add_argument("--dynamic", action="store_true", help=_("ONNX 动态 batch 维度"))
    p.add_argument("--script-mode", choices=["trace", "script"], default="trace",
                   help=_("TorchScript 模式: trace/script (默认 trace)"))
    p.set_defaults(func=cmd_export)
