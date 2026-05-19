"""cvlab profile - 模型性能画像。"""

from __future__ import annotations

import argparse

import torch

from cvlab.i18n import _
from cvlab.cli.console import header, info, result


def cmd_profile(args: argparse.Namespace) -> int:
    header(_("模型性能画像"))

    # 解析输入尺寸
    try:
        input_size = args.input_size or args.input
        input_dims = tuple(int(x) for x in input_size.split("x"))
        input_shape = (1, *input_dims)
    except Exception:
        from cvlab.cli.console import error
        error(_("输入格式错误: {} (需要 CxHxW, 如 3x224x224)").format(input_size))
        return 1

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(_("设备: {}").format(device))

    # 加载模型
    header(_("加载模型"))
    try:
        import torchvision.models as models
        builder = getattr(models, args.model, None)
        if builder is None:
            from cvlab.cli.console import error
            error(_("未知模型: {}").format(args.model))
            return 1
        model = builder(weights=None)
    except Exception as e:
        from cvlab.cli.console import error
        error(_("模型加载失败: {}").format(e))
        return 1

    # 如果指定了 checkpoint，加载权重
    if args.checkpoint:
        from cvlab.weights.manager import WeightManager
        header(_("权重诊断"))
        mgr = WeightManager()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        diagnostic = mgr.diagnose(model, state)
        info(diagnostic.summary())
        model.load_state_dict(state, strict=False)

    # 执行画像
    header(_("性能分析"))
    from cvlab.profile.model_card import ModelProfiler
    profiler = ModelProfiler(device=str(device))
    card = profiler.profile(model, input_shape, warmup=args.num_warmup, repeats=args.num_iter)

    # 使用 Rich 输出
    from cvlab.cli.console import table as print_table
    print_table(_("模型摘要"), [_("指标"), _("值")], [
        [_("参数总量"), f"{card.total_params:,}"],
        [_("可训练参数"), f"{card.trainable_params:,}"],
        [_("参数量 (M)"), f"{card.params_millions:.2f}M"],
        [_("FLOPs (G)"), f"{card.flops_giga:.2f} G"],
        [_("前向延迟"), f"{card.forward_time_ms:.2f} ms"],
        [_("反向延迟"), f"{card.backward_time_ms:.2f} ms"],
        [_("峰值显存"), f"{card.memory_peak_mb:.1f} MB"],
        [_("输入占用"), f"{card.memory_input_mb:.1f} MB"],
    ])

    if card.layer_stats:
        header(_("逐层统计"))
        rows = [[s["name"], s["type"], s["params"], s["trainable"]]
                for s in card.layer_stats[:20]]
        print_table("", [_("层"), _("类型"), _("参数"), _("可训练")], rows)
        if len(card.layer_stats) > 20:
            info(_("... 还有 {} 层").format(len(card.layer_stats) - 20))

    result(_("完成"), _("{}M 参数, {} G FLOPs").format(f"{card.params_millions:.2f}", f"{card.flops_giga:.2f}"))
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("profile", help=_("模型性能画像"))
    p.add_argument("--model", required=True, help=_("模型名称（torchvision 模型名）"))
    p.add_argument("--checkpoint", help=_("Checkpoint 路径（可选）"))
    p.add_argument("--input", default="3x224x224", help=_("输入尺寸 (CxHxW)"))
    p.add_argument("--input-size", dest="input_size", help=_("输入尺寸别名，同 --input"))
    p.add_argument("--device", help=_("运行设备 (cpu/cuda)"))
    p.add_argument("--num-warmup", type=int, default=3, dest="num_warmup", help=_("预热次数（默认 3）"))
    p.add_argument("--num-iter", type=int, default=10, dest="num_iter", help=_("测试迭代数（默认 10）"))
    p.set_defaults(func=cmd_profile)
