"""Sweep 模块测试。"""

import tempfile

from cvlab.db.database import Database
from cvlab.sweep.grid import count_grid, generate_grid
from cvlab.sweep.random import sample_random
from cvlab.sweep.sweeper import Sweeper


class TestGrid:
    def test_generate_grid_single_param(self):
        params = {"lr": [0.001, 0.01, 0.1]}
        combos = generate_grid(params)
        assert len(combos) == 3
        assert combos[0]["lr"] == 0.001
        assert combos[2]["lr"] == 0.1

    def test_generate_grid_multi_param(self):
        params = {"lr": [0.001, 0.01], "bs": [32, 64]}
        combos = generate_grid(params)
        assert len(combos) == 4
        assert combos[0] == {"lr": 0.001, "bs": 32}
        assert combos[3] == {"lr": 0.01, "bs": 64}

    def test_generate_grid_empty(self):
        combos = generate_grid({})
        assert combos == [{}]

    def test_count_grid(self):
        params = {"lr": [0.001, 0.01, 0.1], "bs": [32, 64]}
        assert count_grid(params) == 6

    def test_count_grid_single(self):
        params = {"lr": [0.001]}
        assert count_grid(params) == 1


class TestRandom:
    def test_sample_random_choice(self):
        params = {"lr": {"type": "choice", "values": [0.001, 0.01, 0.1]}}
        samples = sample_random(params, n_trials=5, seed=42)
        assert len(samples) == 5
        for s in samples:
            assert s["lr"] in [0.001, 0.01, 0.1]

    def test_sample_random_uniform(self):
        params = {"dropout": {"type": "uniform", "min": 0.0, "max": 1.0}}
        samples = sample_random(params, n_trials=10, seed=42)
        assert len(samples) == 10
        for s in samples:
            assert 0.0 <= s["dropout"] <= 1.0

    def test_sample_random_loguniform(self):
        params = {"lr": {"type": "loguniform", "min": 1e-5, "max": 1e-1}}
        samples = sample_random(params, n_trials=10, seed=42)
        assert len(samples) == 10
        for s in samples:
            assert 1e-5 <= s["lr"] <= 1e-1

    def test_sample_random_int(self):
        params = {"layers": {"type": "int", "min": 1, "max": 5}}
        samples = sample_random(params, n_trials=10, seed=42)
        assert len(samples) == 10
        for s in samples:
            assert 1 <= s["layers"] <= 5

    def test_sample_reproducible_seed(self):
        params = {"lr": {"type": "choice", "values": [0.001, 0.01, 0.1]}}
        a = sample_random(params, n_trials=10, seed=42)
        b = sample_random(params, n_trials=10, seed=42)
        assert a == b


class TestSweeper:
    def _make_db(self):
        return Database(db_path=tempfile.mktemp(suffix=".db"))

    def test_create_sweep_grid(self):
        sweeper = Sweeper(db=self._make_db())
        sweep_id = sweeper.create_sweep(
            base_config={"model": {"name": "test"}, "training": {"epochs": 1}},
            strategy="grid",
            params={"training.lr": [0.001, 0.01]},
            name="test_sweep",
        )
        assert sweep_id.startswith("sweep_")

        trials = sweeper.get_trials(sweep_id)
        assert len(trials) == 2

        sweep = sweeper.get_sweep(sweep_id)
        assert sweep is not None
        assert sweep["strategy"] == "grid"

    def test_create_sweep_random(self):
        sweeper = Sweeper(db=self._make_db())
        sweep_id = sweeper.create_sweep(
            base_config={"model": {"name": "test"}},
            strategy="random",
            params={"training.lr": {"type": "choice", "values": [0.001, 0.01, 0.1]}},
            max_trials=3,
            seed=42,
        )
        trials = sweeper.get_trials(sweep_id)
        assert len(trials) == 3

    def test_get_best_trial(self):
        sweeper = Sweeper(db=self._make_db())
        sweep_id = sweeper.create_sweep(
            base_config={"model": {"name": "test"}, "training": {"epochs": 1}},
            strategy="grid",
            params={"training.lr": [0.001]},
        )
        # 没有指标数据，应返回 None
        best = sweeper.get_best_trial(sweep_id, metric_key="val/acc")
        assert best is None

    def test_set_nested(self):
        config = {"a": {"b": 1}}
        Sweeper._set_nested(config, "a.c", 2)
        assert config["a"]["c"] == 2

        Sweeper._set_nested(config, "x.y.z", 3)
        assert config["x"]["y"]["z"] == 3
