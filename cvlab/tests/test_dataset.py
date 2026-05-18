"""数据集版本管理测试。"""

from __future__ import annotations

import json

import pytest

from cvlab.db.database import Database


class TestDatasetRegistration:
    """数据集注册测试。"""

    def test_register_dataset(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        ds_id = db.register_dataset("test-ds", str(tmp_path), "test description")
        assert ds_id.startswith("ds_")

        ds = db.get_dataset(ds_id)
        assert ds is not None
        assert ds["name"] == "test-ds"
        assert ds["path"] == str(tmp_path)
        assert ds["description"] == "test description"

    def test_get_datasets_empty(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        datasets = db.get_datasets()
        assert datasets == []

    def test_get_datasets_multiple(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        id1 = db.register_dataset("ds1", "/path/1")
        id2 = db.register_dataset("ds2", "/path/2")
        datasets = db.get_datasets()
        assert len(datasets) == 2
        ids = [d["id"] for d in datasets]
        assert id1 in ids
        assert id2 in ids

    def test_get_dataset_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        assert db.get_dataset("ds_99999") is None


class TestDatasetVersions:
    """数据集版本记录测试。"""

    def test_record_version(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        ds_id = db.register_dataset("test-ds", str(tmp_path))

        db.record_dataset_version(
            dataset_id=ds_id,
            version="v1",
            root_hash="abc123",
            ann_hash="def456",
            total_files=100,
            total_size_bytes=1024 * 1024,
            file_count_by_ext={".jpg": 80, ".txt": 20},
        )

        versions = db.get_dataset_versions(ds_id)
        assert len(versions) == 1
        v = versions[0]
        assert v["version"] == "v1"
        assert v["root_hash"] == "abc123"
        assert v["ann_hash"] == "def456"
        assert v["total_files"] == 100
        assert v["total_size_bytes"] == 1024 * 1024

        # 验证 file_count_by_ext JSON
        ext = v["file_count_by_ext"]
        if isinstance(ext, str):
            ext = json.loads(ext)
        assert ext == {".jpg": 80, ".txt": 20}

    def test_multiple_versions(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        ds_id = db.register_dataset("test-ds", str(tmp_path))

        db.record_dataset_version(ds_id, "v1", "hash1", total_files=50)
        db.record_dataset_version(ds_id, "v2", "hash2", total_files=100)

        versions = db.get_dataset_versions(ds_id)
        assert len(versions) == 2
        # 按时间降序，v2 在前
        assert versions[0]["version"] == "v2"
        assert versions[1]["version"] == "v1"

    def test_version_with_experiment_link(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        exp_id = db.create_experiment(name="test-exp", config={})
        ds_id = db.register_dataset("test-ds", str(tmp_path))

        db.record_dataset_version(
            dataset_id=ds_id,
            version="v1",
            root_hash="hash1",
            experiment_id=exp_id,
        )

        # 通过实验查询关联数据集
        exp_datasets = db.get_experiment_datasets(exp_id)
        assert len(exp_datasets) == 1
        assert exp_datasets[0]["dataset_name"] == "test-ds"
        assert exp_datasets[0]["version"] == "v1"

    def test_get_experiment_datasets_empty(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        exp_id = db.create_experiment(name="test-exp", config={})
        datasets = db.get_experiment_datasets(exp_id)
        assert datasets == []

    def test_latest_version_in_datasets_list(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        ds_id = db.register_dataset("test-ds", str(tmp_path))

        db.record_dataset_version(ds_id, "v1", "hash1", total_files=50)
        db.record_dataset_version(ds_id, "v2", "hash2", total_files=100)

        datasets = db.get_datasets()
        assert len(datasets) == 1
        assert datasets[0]["latest_version"] == "v2"
        assert datasets[0]["latest_files"] == 100


class TestDatasetEdgeCases:
    """数据集边界情况测试。"""

    def test_duplicate_version_ignored(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        ds_id = db.register_dataset("test-ds", str(tmp_path))

        db.record_dataset_version(ds_id, "v1", "hash1")
        db.record_dataset_version(ds_id, "v1", "hash2")  # 重复版本应被 IGNORE

        versions = db.get_dataset_versions(ds_id)
        assert len(versions) == 1
        assert versions[0]["root_hash"] == "hash1"

    def test_version_without_experiment(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        ds_id = db.register_dataset("test-ds", str(tmp_path))

        db.record_dataset_version(ds_id, "v1", "hash1")
        versions = db.get_dataset_versions(ds_id)
        assert len(versions) == 1
        assert versions[0]["experiment_id"] is None

    def test_dataset_without_versions(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        ds_id = db.register_dataset("test-ds", str(tmp_path))
        versions = db.get_dataset_versions(ds_id)
        assert versions == []

    def test_cascade_delete_not_cascade(self, tmp_path):
        """删除 dataset 不应级联删除 experiment（SET NULL）。"""
        db = Database(str(tmp_path / "test.db"))
        exp_id = db.create_experiment(name="test-exp", config={})
        ds_id = db.register_dataset("test-ds", str(tmp_path))

        db.record_dataset_version(
            ds_id, "v1", "hash1", experiment_id=exp_id,
        )

        # 删除数据集
        db._conn.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
        db._conn.commit()

        # 实验应仍然存在
        exp = db.get_experiment(exp_id)
        assert exp is not None
