"""配置模块测试。"""

import tempfile
from pathlib import Path

import pytest
import yaml

from cvlab.config.config import (
    DEFAULT_CONFIG,
    load_config,
    merge_config,
    validate_config,
)


class TestConfig:
    def test_default_config_has_required_keys(self):
        assert "model" in DEFAULT_CONFIG
        assert "training" in DEFAULT_CONFIG
        assert "data" in DEFAULT_CONFIG
        assert "seed" in DEFAULT_CONFIG

    def test_merge_config_simple(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 4}
        merged = merge_config(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3  # 原有值保留
        assert merged["e"] == 4

    def test_load_config(self):
        config = {"training": {"lr": 0.01, "epochs": 100}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(config, f)
            tmp_path = f.name
        try:
            loaded = load_config(tmp_path)
            assert loaded["training"]["lr"] == 0.01
            assert loaded["training"]["epochs"] == 100
            # 默认值应保留
            assert loaded["training"]["optimizer"] == "adam"
            assert loaded["seed"] == 42
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_config_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_validate_valid_config(self):
        errors = validate_config(DEFAULT_CONFIG)
        assert len(errors) == 0

    def test_validate_invalid_optimizer(self):
        config = merge_config(DEFAULT_CONFIG, {"training": {"optimizer": "rmsprop"}})
        errors = validate_config(config)
        assert len(errors) == 1
        assert "optimizer" in errors[0]

    def test_validate_invalid_epochs(self):
        config = merge_config(DEFAULT_CONFIG, {"training": {"epochs": -1}})
        errors = validate_config(config)
        assert len(errors) > 0

    def test_validate_batch_size_none_is_valid(self):
        config = merge_config(DEFAULT_CONFIG, {"training": {"batch_size": None}})
        errors = validate_config(config)
        assert len(errors) == 0
