"""数据血缘追踪 - 数据集哈希、路径变更检测、版本记录。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetProvenance:
    path: str
    root_hash: str  # 快速 O(1) 根目录状态
    ann_hash: str = ""  # 标注文件 SHA256
    total_files: int = 0
    total_size_bytes: int = 0
    file_count_by_ext: dict[str, int] = field(default_factory=dict)
    recorded_at: str = ""


class ProvenanceTracker:
    """数据血缘追踪器。

    两级设计：
    - Level 1 (O(1)): 仅记录根目录文件列表和总大小，适合每次实验自动调用
    - Level 2 (O(n)): 标注文件完整 SHA256，按需启用
    """

    def __init__(self, provenance_dir: str | os.PathLike = ".cvlab/provenance"):
        self.provenance_dir = Path(provenance_dir)
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, dataset_path: str | os.PathLike,
                 hash_annotations: bool = False) -> DatasetProvenance:
        """对数据集目录进行快照。

        Args:
            dataset_path: 数据集路径。
            hash_annotations: 是否计算标注文件哈希。

        Returns:
            数据集血缘信息。
        """
        root = Path(dataset_path)
        if not root.exists():
            return DatasetProvenance(
                path=str(root), root_hash="", recorded_at=_now_str()
            )

        # Level 1: 根目录状态（O(1)）
        root_state = self._compute_root_hash(root)

        # Level 2: 标注文件哈希
        ann_hash = ""
        if hash_annotations:
            ann_hash = self._compute_annotation_hash(root)

        # 文件统计
        total_files = 0
        total_size = 0
        ext_counter: dict[str, int] = {}
        for f in root.rglob("*"):
            if f.is_file():
                total_files += 1
                total_size += f.stat().st_size
                ext_counter[f.suffix.lower()] = ext_counter.get(f.suffix.lower(), 0) + 1

        prov = DatasetProvenance(
            path=str(root.absolute()),
            root_hash=root_state,
            ann_hash=ann_hash,
            total_files=total_files,
            total_size_bytes=total_size,
            file_count_by_ext=dict(ext_counter),
            recorded_at=_now_str(),
        )

        # 持久化
        self._save_provenance(prov)

        return prov

    def has_changed(self, dataset_path: str | os.PathLike) -> bool:
        """检查数据集自上次快照以来是否发生变化。"""
        root = Path(dataset_path)
        if not root.exists():
            return True

        current_hash = self._compute_root_hash(root)
        history = self._load_provenance(str(root.absolute()))

        if history is None:
            return True

        return current_hash != history.root_hash

    def _compute_root_hash(self, root: Path) -> str:
        """快速 O(1) 根目录哈希：文件列表 + 大小 + 修改时间。"""
        hasher = hashlib.md5()
        try:
            entries = sorted(
                [f for f in root.iterdir() if f.is_file() or f.is_dir()],
                key=lambda x: x.name,
            )
            for entry in entries[:1000]:  # 限制避免超时
                stat = entry.stat()
                line = f"{entry.name}:{stat.st_size}:{stat.st_mtime}"
                hasher.update(line.encode())
        except PermissionError:
            pass
        return hasher.hexdigest()

    def _compute_annotation_hash(self, root: Path) -> str:
        """完整 SHA256 标注文件哈希。"""
        hasher = hashlib.sha256()
        ann_extensions = {".json", ".xml", ".txt", ".csv"}
        files: list[Path] = []
        for ext in ann_extensions:
            files.extend(root.rglob(f"*{ext}"))
        files.sort(key=lambda x: str(x))

        for f in files[:500]:  # 限制数量
            try:
                hasher.update(f.name.encode())
                hasher.update(f.read_bytes()[:1024 * 1024])  # 每文件最多 1MB
            except (PermissionError, OSError):
                pass

        return hasher.hexdigest()

    def _save_provenance(self, prov: DatasetProvenance) -> None:
        """持久化血缘信息。"""
        safe_name = prov.path.replace(":", "_").replace("/", "_").replace("\\", "_")
        path = self.provenance_dir / f"{safe_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "path": prov.path,
                "root_hash": prov.root_hash,
                "ann_hash": prov.ann_hash,
                "total_files": prov.total_files,
                "total_size_bytes": prov.total_size_bytes,
                "file_count_by_ext": prov.file_count_by_ext,
                "recorded_at": prov.recorded_at,
            }, f, indent=2)

    def _load_provenance(self, dataset_path: str) -> DatasetProvenance | None:
        """加载最近一次血缘快照。"""
        safe_name = dataset_path.replace(":", "_").replace("/", "_").replace("\\", "_")
        path = self.provenance_dir / f"{safe_name}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return DatasetProvenance(**data)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """列出所有已记录的数据集快照。"""
        results: list[dict[str, Any]] = []
        for f in sorted(self.provenance_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            results.append({
                "path": data.get("path", ""),
                "total_files": data.get("total_files", 0),
                "total_size_mb": round(data.get("total_size_bytes", 0) / (1024 * 1024), 2),
                "recorded_at": data.get("recorded_at", ""),
            })
        return results


def _now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
