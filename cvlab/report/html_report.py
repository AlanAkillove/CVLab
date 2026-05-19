"""HTML 报告生成 - 将实验数据渲染为自包含的 HTML 文件。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from cvlab.core.utils import flatten_dict
from cvlab.db.database import Database


class HtmlReportGenerator:
    """实验报告生成器。

    用法:
        gen = HtmlReportGenerator()
        html = gen.generate("exp_001")
        gen.save("exp_001", "report_exp_001.html")
    """

    def __init__(self, db: Database | None = None, template_dir: str | os.PathLike | None = None):
        self.db = db or Database()
        template_dir = template_dir or Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )

    def generate(self, experiment_id: str, title: str | None = None) -> str:
        """生成实验的 HTML 报告。

        Args:
            experiment_id: 实验 ID。
            title: 报告标题，默认使用实验名称。

        Returns:
            HTML 字符串。
        """
        exp = self.db.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"实验 {experiment_id} 不存在")

        template = self.env.get_template("report.html.j2")

        # 解析指标
        db_metrics = self.db.get_metrics(experiment_id)
        final_metrics: dict[str, float] = {}
        if db_metrics:
            for m in db_metrics:
                key = m["key"]
                if key not in final_metrics or m["step"] > 0:
                    final_metrics[key] = m["value"]

        # 解析配置
        flat_config: dict[str, Any] = {}
        if exp.get("config_json"):
            try:
                config = json.loads(exp["config_json"])
                flat_config = flatten_dict(config)
            except json.JSONDecodeError:
                flat_config = {"raw": exp["config_json"]}

        # 解析环境
        env_info: dict[str, Any] = {}
        if exp.get("env_json"):
            try:
                env_info = json.loads(exp["env_json"])
                env_info = flatten_dict(env_info)
            except json.JSONDecodeError:
                env_info = {"raw": exp["env_json"]}

        # Checkpoints
        checkpoints = self.db.get_checkpoints(experiment_id) or []

        warnings: list[str] = []
        if exp.get("failure_reason"):
            warnings.append(f"实验失败: {exp['failure_reason']}")

        return template.render(
            title=title or exp.get("name", "实验报告"),
            experiment_id=experiment_id,
            status=exp.get("status", "unknown"),
            created_at=(exp.get("created_at", "")[:19] if exp.get("created_at") else ""),
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            warnings=warnings,
            metrics=final_metrics,
            config=flat_config,
            environment=env_info,
            checkpoints=checkpoints,
            command=exp.get("command", ""),
        )

    def save(self, experiment_id: str, output_path: str | os.PathLike,
             title: str | None = None) -> Path:
        """生成并保存 HTML 报告到文件。"""
        html = self.generate(experiment_id, title=title)
        path = Path(output_path)
        path.write_text(html, encoding="utf-8")
        return path
