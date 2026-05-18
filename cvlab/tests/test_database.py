"""Database 类测试。"""

import json
import tempfile
from pathlib import Path

import pytest

from cvlab.db.database import Database


@pytest.fixture
def db():
    tmp = tempfile.mktemp(suffix=".db")
    database = Database(tmp)
    yield database
    database.close()
    Path(tmp).unlink(missing_ok=True)


class TestDatabase:
    def test_create_experiment(self, db):
        exp_id = db.create_experiment("test_exp", {"lr": 0.001})
        assert exp_id.startswith("exp_")
        exp = db.get_experiment(exp_id)
        assert exp["name"] == "test_exp"
        assert exp["status"] == "created"
        assert json.loads(exp["config_json"]) == {"lr": 0.001}

    def test_list_experiments(self, db):
        db.create_experiment("exp1", {"lr": 0.001})
        db.create_experiment("exp2", {"lr": 0.01})
        exps = db.list_experiments()
        assert len(exps) == 2

    def test_list_experiments_by_status(self, db):
        e1 = db.create_experiment("e1", {})
        db.create_experiment("e2", {})
        db.update_experiment_status(e1, "running")
        exps = db.list_experiments(status="running")
        assert len(exps) == 1

    def test_update_status_with_reason(self, db):
        exp_id = db.create_experiment("test", {})
        db.update_experiment_status(exp_id, "failed", "OOM")
        exp = db.get_experiment(exp_id)
        assert exp["failure_reason"] == "OOM"

    def test_delete_experiment(self, db):
        exp_id = db.create_experiment("test", {})
        db.delete_experiment(exp_id)
        assert db.get_experiment(exp_id) is None

    def test_log_and_get_metrics(self, db):
        exp_id = db.create_experiment("test", {})
        db.log_metrics(exp_id, {"train/loss": 2.0, "train/acc": 0.5}, step=0)
        db.log_metrics(exp_id, {"train/loss": 1.5, "train/acc": 0.7}, step=1)
        metrics = db.get_metrics(exp_id)
        assert len(metrics) == 4

    def test_log_metrics_upsert(self, db):
        exp_id = db.create_experiment("test", {})
        db.log_metrics(exp_id, {"train/loss": 1.0}, step=0)
        db.log_metrics(exp_id, {"train/loss": 2.0}, step=0)  # upsert
        metrics = db.get_metrics(exp_id)
        train_losses = [m for m in metrics if m["key"] == "train/loss"]
        assert train_losses[-1]["value"] == 2.0

    def test_metrics_dataframe(self, db):
        import pandas as pd
        exp_id = db.create_experiment("test", {})
        db.log_metrics(exp_id, {"loss": 2.0, "acc": 0.5}, step=0)
        db.log_metrics(exp_id, {"loss": 1.5, "acc": 0.7}, step=1)
        df = db.get_metrics_dataframe(exp_id)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"acc", "loss"}
        assert len(df) == 2

    def test_tags(self, db):
        exp_id = db.create_experiment("test", {})
        db.add_tag(exp_id, "baseline")
        db.add_tag(exp_id, "lr-sweep")
        db.add_tag(exp_id, "baseline")  # duplicate, should be ignored
        tags = db.get_tags(exp_id)
        assert len(tags) == 2
        db.remove_tag(exp_id, "baseline")
        assert len(db.get_tags(exp_id)) == 1

    def test_checkpoint_lifecycle(self, db):
        exp_id = db.create_experiment("test", {})
        db.save_checkpoint_record(exp_id, epoch=10, path="epoch_10.pt",
                                   metric_name="val_acc", metric_value=0.85,
                                   is_best=True, is_last=True, file_size=1000)
        db.save_checkpoint_record(exp_id, epoch=20, path="epoch_20.pt",
                                   metric_name="val_acc", metric_value=0.88,
                                   is_best=True, is_last=True, file_size=1000)
        best = db.get_best_checkpoint(exp_id)
        assert best["epoch"] == 20
        last = db.get_last_checkpoint(exp_id)
        assert last["epoch"] == 20

    def test_artifacts(self, db):
        exp_id = db.create_experiment("test", {})
        db.save_artifact(exp_id, step=10, key="pred_samples",
                          type_="image", file_path="epoch_10.png")
        artifacts = db.get_artifacts(exp_id)
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "image"

    def test_search_by_metric(self, db):
        e1 = db.create_experiment("e1", {})
        e2 = db.create_experiment("e2", {})
        db.update_experiment_status(e1, "completed")
        db.update_experiment_status(e2, "completed")
        db.log_metrics(e1, {"val_acc": 0.85}, step=0)
        db.log_metrics(e2, {"val_acc": 0.95}, step=0)
        results = db.search_experiments(metric_key="val_acc", metric_min=0.9)
        assert len(results) == 1
        assert results[0]["id"] == e2

    def test_sweep_lifecycle(self, db):
        exp_id = db.create_experiment("sweep_root", {})
        trial_1 = db.create_experiment("trial1", {"lr": 0.001})
        trial_2 = db.create_experiment("trial2", {"lr": 0.01})
        db.create_sweep("sweep_001", exp_id, {"lr": [0.001, 0.01]}, "grid")
        db.add_sweep_trial("sweep_001", trial_1, 0, {"lr": 0.001})
        db.add_sweep_trial("sweep_001", trial_2, 1, {"lr": 0.01})
        trials = db.get_sweep_trials("sweep_001")
        assert len(trials) == 2

    def test_update_experiment(self, db):
        exp_id = db.create_experiment("test", {})
        db.update_experiment(exp_id, notes="hello world")
        exp = db.get_experiment(exp_id)
        assert exp["notes"] == "hello world"
