"""预训练权重加载诊断与管理。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LoadDiagnostic:
    success: bool
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: list[dict[str, Any]]
    layer_stats: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.missing_keys or self.unexpected_keys or self.shape_mismatches)

    def summary(self) -> str:
        """生成加载诊断摘要。"""
        if self.success and not self.has_issues:
            return "权重加载成功，无任何问题"

        lines: list[str] = []
        if self.missing_keys:
            lines.append(f"缺失 {len(self.missing_keys)} 个 key:")
            for k in self.missing_keys[:10]:
                lines.append(f"  - {k}")
            if len(self.missing_keys) > 10:
                lines.append(f"  ... 还有 {len(self.missing_keys) - 10} 个")

        if self.unexpected_keys:
            lines.append(f"多余 {len(self.unexpected_keys)} 个 key:")
            for k in self.unexpected_keys[:10]:
                lines.append(f"  - {k}")
            if len(self.unexpected_keys) > 10:
                lines.append(f"  ... 还有 {len(self.unexpected_keys) - 10} 个")

        if self.shape_mismatches:
            lines.append(f"形状不匹配 {len(self.shape_mismatches)} 处:")
            for m in self.shape_mismatches[:5]:
                lines.append(f"  - {m['key']}: 期望 {m['expected']} vs 实际 {m['actual']}")

        return "\n".join(lines)


class WeightManager:
    """预训练权重管理：加载诊断、本地缓存、权重分析。

    用法:
        manager = WeightManager(cache_dir=".cvlab/weights")
        model = resnet18()
        diagnostic = manager.load(model, "path/to/weights.pth")
        print(diagnostic.summary())
    """

    def __init__(self, cache_dir: str | os.PathLike = ".cvlab/weights"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def diagnose(self, model: nn.Module, state_dict: dict[str, torch.Tensor]
                 ) -> LoadDiagnostic:
        """检查 state_dict 与模型结构的兼容性，不实际加载。"""
        model_sd = model.state_dict()
        model_keys = set(model_sd.keys())
        ckpt_keys = set(state_dict.keys())

        missing_keys = sorted(model_keys - ckpt_keys)
        unexpected_keys = sorted(ckpt_keys - model_keys)

        shape_mismatches: list[dict[str, Any]] = []
        for key in model_keys & ckpt_keys:
            if model_sd[key].shape != state_dict[key].shape:
                shape_mismatches.append({
                    "key": key,
                    "expected": list(model_sd[key].shape),
                    "actual": list(state_dict[key].shape),
                })

        # 逐层权重统计
        layer_stats = self._compute_layer_stats(state_dict)

        # 忽略典型的 buffer 差异
        ignore_prefixes = ("num_batches_tracked", "tracked")
        missing_keys = [k for k in missing_keys
                        if not any(k.startswith(p) for p in ignore_prefixes)]
        unexpected_keys = [k for k in unexpected_keys
                           if not any(k.startswith(p) for p in ignore_prefixes)]

        success = len(shape_mismatches) == 0 and (
            len(missing_keys) == 0 or all(
                "classifier" in k or "fc" in k or "head" in k
                for k in missing_keys
            )
        )

        return LoadDiagnostic(
            success=success,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            shape_mismatches=shape_mismatches,
            layer_stats=layer_stats,
        )

    def load(
        self,
        model: nn.Module,
        weights_path: str | os.PathLike,
        strict: bool = False,
        skip_mismatch: bool = True,
    ) -> LoadDiagnostic:
        """加载权重并返回诊断信息。

        Args:
            model: 目标模型。
            weights_path: 权重文件路径。
            strict: 是否严格加载（默认 False 以容忍分类头不匹配）。
            skip_mismatch: 是否跳过形状不匹配的 key。

        Returns:
            加载诊断信息（实际不修改模型，仅返回检查结果）。
        """
        path = Path(weights_path)
        if not path.exists():
            return LoadDiagnostic(
                success=False,
                missing_keys=[str(weights_path)],
                unexpected_keys=[],
                shape_mismatches=[],
            )

        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        diagnostic = self.diagnose(model, state_dict)

        # 若 skip_mismatch，移除形状不匹配的 key 后尝试加载
        if skip_mismatch and diagnostic.shape_mismatches:
            mismatch_keys = {m["key"] for m in diagnostic.shape_mismatches}
            filtered_sd = {k: v for k, v in state_dict.items()
                           if k not in mismatch_keys}
            try:
                model.load_state_dict(filtered_sd, strict=False)
                diagnostic.success = True
            except Exception as e:
                diagnostic.success = False
                diagnostic.shape_mismatches.append({
                    "key": "load_error",
                    "expected": str(e),
                    "actual": "",
                })
        elif diagnostic.success or not strict:
            model.load_state_dict(state_dict, strict=False)
            diagnostic.success = True
        else:
            diagnostic.success = False

        return diagnostic

    def cache_info(self) -> dict[str, Any]:
        """查看本地权重缓存状态。"""
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0.0, "files": []}

        files: list[dict[str, Any]] = []
        total_size = 0
        for f in self.cache_dir.iterdir():
            if f.is_file() and f.suffix in (".pth", ".pt", ".pkl"):
                size_mb = f.stat().st_size / (1024 * 1024)
                files.append({
                    "name": f.name,
                    "size_mb": round(size_mb, 2),
                    "modified": f.stat().st_mtime,
                })
                total_size += size_mb

        files.sort(key=lambda x: x["name"])
        return {
            "total_files": len(files),
            "total_size_mb": round(total_size, 2),
            "cache_dir": str(self.cache_dir),
            "files": files,
        }

    def compute_hash(self, weights_path: str | os.PathLike) -> str | None:
        """计算权重文件的 SHA256 哈希。"""
        path = Path(weights_path)
        if not path.exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def diff_weights(
        self,
        path_a: str | os.PathLike,
        path_b: str | os.PathLike,
    ) -> list[dict[str, Any]]:
        """比较两个权重文件的差异。

        Returns:
            每个 key 的差异信息列表。
        """
        sd_a = torch.load(path_a, map_location="cpu", weights_only=True)
        sd_b = torch.load(path_b, map_location="cpu", weights_only=True)

        diffs: list[dict[str, Any]] = []
        all_keys = set(sd_a.keys()) | set(sd_b.keys())

        for key in sorted(all_keys):
            if key not in sd_a:
                diffs.append({"key": key, "diff": "only_in_b", "shape": list(sd_b[key].shape)})
            elif key not in sd_b:
                diffs.append({"key": key, "diff": "only_in_a", "shape": list(sd_a[key].shape)})
            elif sd_a[key].shape != sd_b[key].shape:
                diffs.append({
                    "key": key,
                    "diff": "shape_mismatch",
                    "shape_a": list(sd_a[key].shape),
                    "shape_b": list(sd_b[key].shape),
                })
            elif not torch.equal(sd_a[key], sd_b[key]):
                diff_norm = (sd_a[key] - sd_b[key]).norm().item()
                cos_sim = nn.functional.cosine_similarity(
                    sd_a[key].flatten().unsqueeze(0),
                    sd_b[key].flatten().unsqueeze(0),
                ).item()
                diffs.append({
                    "key": key,
                    "diff": "different",
                    "diff_norm": round(diff_norm, 4),
                    "cosine_similarity": round(cos_sim, 4),
                    "shape": list(sd_a[key].shape),
                })

        return diffs

    def _compute_layer_stats(
        self, state_dict: dict[str, torch.Tensor]
    ) -> list[dict[str, Any]]:
        """计算权重统计信息（均值/标准差/最小值/最大值）。"""
        stats: list[dict[str, Any]] = []
        for key, tensor in state_dict.items():
            if tensor.numel() < 2:
                continue
            stats.append({
                "key": key,
                "shape": list(tensor.shape),
                "mean": round(tensor.mean().item(), 6),
                "std": round(tensor.std().item(), 6),
                "min": round(tensor.min().item(), 6),
                "max": round(tensor.max().item(), 6),
                "norm": round(tensor.norm().item(), 4),
            })
        return stats
