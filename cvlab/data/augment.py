"""数据增强预览 - 对单张图片应用增强并可视化效果。"""

from __future__ import annotations

from typing import Any

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class AugmentPreview:
    """增强效果预览工具。

    对单张图片应用一组常见的 CV 增强并显示结果。
    """

    @staticmethod
    def apply_transforms(
        image: torch.Tensor,
        transform_specs: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """对图片应用一组变换并返回结果。

        Args:
            image: 输入图片 (C, H, W)，值范围 [0, 1]。
            transform_specs: 变换定义列表，每项为 {name, params}。
                支持的 name 列表见 TRANSFORM_REGISTRY。

        Returns:
            {变换名称: 结果张量} 的字典。
        """
        results: dict[str, torch.Tensor] = {"original": image.clone()}
        for spec in transform_specs:
            name = spec.get("name", "")
            params = {k: v for k, v in spec.items() if k != "name"}
            fn = TRANSFORM_REGISTRY.get(name)
            if fn is not None:
                try:
                    result = fn(image, **params)
                    results[name] = result
                except Exception as e:
                    results[name + "_error"] = torch.full_like(image, float("nan"))
        return results

    @staticmethod
    def make_grid(results: dict[str, torch.Tensor], ncol: int = 4) -> torch.Tensor:
        """将多张增强结果排列为网格。"""
        if not results:
            return torch.zeros(3, 224, 224)
        ref_shape = next(iter(results.values())).shape
        images = [v for v in results.values() if v.shape == ref_shape]
        if not images:
            return torch.zeros(3, 224, 224)
        return torchvision.utils.make_grid(images, nrow=ncol)


# 内置增强注册表
TRANSFORM_REGISTRY: dict[str, Any] = {
    "brightness": lambda img, factor=0.5: TF.adjust_brightness(img, factor),
    "contrast": lambda img, factor=0.5: TF.adjust_contrast(img, factor),
    "saturation": lambda img, factor=0.5: TF.adjust_saturation(img, factor),
    "hue": lambda img, factor=0.1: TF.adjust_hue(img, factor),
    "rotate": lambda img, angle=30: TF.rotate(img, angle),
    "hflip": lambda img: TF.hflip(img),
    "vflip": lambda img: TF.vflip(img),
    "grayscale": lambda img: TF.rgb_to_grayscale(img, num_output_channels=3),
    "gaussian_blur": lambda img, kernel_size=5: TF.gaussian_blur(img, kernel_size),
    "solarize": lambda img, threshold=0.5: TF.solarize(img, threshold),
    "equalize": lambda img: TF.equalize(img),
    "posterize": lambda img, bits=4: TF.posterize(img, bits),
}
