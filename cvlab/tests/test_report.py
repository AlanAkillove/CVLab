"""HTML 报告模块测试。"""

import tempfile
from pathlib import Path

from cvlab.core.utils import flatten_dict
from cvlab.db.database import Database
from cvlab.report.html_report import HtmlReportGenerator


class TestHtmlReportGenerator:
    def test_generate_raises_on_nonexistent(self):
        gen = HtmlReportGenerator()
        try:
            gen.generate("nonexistent_exp")
            assert False, "应该抛出异常"
        except ValueError:
            pass

    def test_generate_returns_html(self):
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        exp_id = db.create_experiment(
            name="test_exp",
            config={"model": {"name": "test"}, "training": {"lr": 0.001}},
        )
        db.log_metrics(exp_id, {"train/loss": 0.5, "val/acc": 0.8}, step=1)

        gen = HtmlReportGenerator(db=db)
        html = gen.generate(exp_id)

        assert "<!DOCTYPE html>" in html
        assert "test_exp" in html
        assert "0.5000" in html or "0.5" in html
        assert "0.001" in html

    def test_save_creates_file(self):
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        exp_id = db.create_experiment(
            name="save_test",
            config={"model": {"name": "test"}},
        )

        gen = HtmlReportGenerator(db=db)
        output = Path(tempfile.mktemp(suffix=".html"))
        result = gen.save(exp_id, str(output))
        assert result.exists()
        html = result.read_text(encoding="utf-8")
        assert "save_test" in html

    def test_flatten(self):
        flat = flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        assert flat == {"a.b": 1, "a.c": 2, "d": 3}

    def test_flatten_list(self):
        flat = flatten_dict({"a": [1, 2, 3]})
        assert "a" in flat
