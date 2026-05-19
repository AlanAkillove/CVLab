"""数据分析模块测试。"""

import tempfile
from pathlib import Path

from cvlab.data.analyze import DatasetAnalyzer
from cvlab.data.augment import AugmentPreview
from cvlab.data.provenance import DatasetProvenance, ProvenanceTracker


def _create_fake_image(path: Path, width: int = 32, height: int = 32):
    """创建最小的合法 PNG 图片用于测试。"""
    import struct
    import zlib

    def _make_png(w, h):
        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(h):
            raw += b"\x00" + b"\xff\x00\x00" * w
        idat = _chunk(b"IDAT", zlib.compress(raw))
        iend = _chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    path.write_bytes(_make_png(width, height))


class TestDatasetAnalyzer:
    def test_analyze_nonexistent_path(self):
        analyzer = DatasetAnalyzer("/nonexistent/path")
        report = analyzer.analyze()
        assert report.total_samples == 0
        assert "路径不存在" in report.warnings

    def test_analyze_empty_directory(self, tmp_path):
        analyzer = DatasetAnalyzer(tmp_path)
        report = analyzer.analyze()
        assert report.total_samples == 0

    def test_analyze_with_images(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cls1 = td_path / "cat"
            cls2 = td_path / "dog"
            cls1.mkdir()
            cls2.mkdir()
            _create_fake_image(cls1 / "img1.png")
            _create_fake_image(cls1 / "img2.png")
            _create_fake_image(cls2 / "img3.png")

            analyzer = DatasetAnalyzer(td_path)
            report = analyzer.analyze()
            assert report.total_samples == 3
            assert report.total_classes == 2
            assert report.class_distribution.get("cat") == 2
            assert report.class_distribution.get("dog") == 1
            assert report.total_size_mb > 0

    def test_analyze_flat_directory(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for i in range(5):
                _create_fake_image(td_path / f"img{i}.png")

            analyzer = DatasetAnalyzer(td_path)
            report = analyzer.analyze()
            assert report.total_samples == 5
            assert report.total_classes == 0

    def test_class_balance_score(self):
        balanced = {"a": 10, "b": 10, "c": 10}
        assert DatasetAnalyzer.class_balance_score(balanced) == 1.0

        imbalanced = {"a": 100, "b": 10, "c": 1}
        score = DatasetAnalyzer.class_balance_score(imbalanced)
        assert 0 < score < 1

    def test_class_balance_score_empty(self):
        assert DatasetAnalyzer.class_balance_score({}) == 0.0


class TestProvenanceTracker:
    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "img.jpg").write_text("data")
            (td_path / "ann.json").write_text('{"label": 1}')

            tracker = ProvenanceTracker(provenance_dir=tempfile.mkdtemp())
            prov = tracker.snapshot(td_path)
            assert isinstance(prov, DatasetProvenance)
            assert prov.total_files > 0
            assert prov.total_size_bytes > 0
            assert len(prov.root_hash) == 32  # MD5

    def test_snapshot_with_annotation_hash(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "ann.json").write_text('{"label": 1}')

            tracker = ProvenanceTracker(provenance_dir=tempfile.mkdtemp())
            prov = tracker.snapshot(td_path, hash_annotations=True)
            assert len(prov.ann_hash) == 64  # SHA256

    def test_has_changed(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "img.jpg").write_text("data")

            tracker = ProvenanceTracker(provenance_dir=tempfile.mkdtemp())
            tracker.snapshot(td_path)
            assert not tracker.has_changed(td_path)

            (td_path / "new.jpg").write_text("new")
            assert tracker.has_changed(td_path)

    def test_has_changed_nonexistent(self):
        tracker = ProvenanceTracker(provenance_dir=tempfile.mkdtemp())
        assert tracker.has_changed("/nonexistent")

    def test_nonexistent_path(self):
        tracker = ProvenanceTracker(provenance_dir=tempfile.mkdtemp())
        prov = tracker.snapshot("/nonexistent")
        assert prov.root_hash == ""

    def test_list_snapshots(self):
        tracker = ProvenanceTracker(provenance_dir=tempfile.mkdtemp())
        snaps = tracker.list_snapshots()
        assert isinstance(snaps, list)


class TestAugmentPreview:
    def test_apply_transforms_original_only(self):
        preview = AugmentPreview()
        results = preview.apply_transforms(
            torch_zeros(3, 32, 32),
            [],
        )
        assert "original" in results
        assert len(results) == 1

    def test_apply_hflip(self):
        preview = AugmentPreview()
        results = preview.apply_transforms(
            torch_zeros(3, 32, 32),
            [{"name": "hflip"}],
        )
        assert "hflip" in results


def torch_zeros(c, h, w):
    import torch
    return torch.zeros(c, h, w)
