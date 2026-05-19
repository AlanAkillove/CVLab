"""SQLite 数据库操作层。"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from cvlab.db.schema import SCHEMA_SQL


class Database:
    """SQLite 数据库封装，线程安全。"""

    def __init__(self, db_path: str = ".cvlab/cvlab.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        """每个线程持有一个独立连接。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── 实验 CRUD ──────────────────────────────────────────

    def create_experiment(self, name: str, config: dict, seed: int | None = None,
                          env_json: str = "{}") -> str:
        import hashlib
        import random
        from datetime import datetime

        # 生成短 ID: {slug}_{MMDD}_{4char_hash}
        # 例如: resnet18-cifar10_0520_a3f2
        # 保留排序性（日期前缀）、语义（模型/数据集名）、唯一性（短 hash）
        from cvlab.core.utils import slugify

        # 从 config 中提取模型名和数据集名作为 ID 前缀
        model_name = config.get("model", {}).get("name", "") if isinstance(config, dict) else ""
        dataset_name = ""
        if isinstance(config, dict):
            dataset_name = config.get("data", {}).get("dataset_name", "") or ""

        # 构建语义前缀
        if name:
            semantic = name
        elif model_name:
            semantic = model_name
            if dataset_name:
                semantic = f"{model_name}_{dataset_name}"
        else:
            semantic = "exp"

        slug = slugify(semantic, max_len=24)
        date_part = datetime.now().strftime('%m%d')
        short_hash = hashlib.md5(
            f"{datetime.now().isoformat()}_{random.random()}".encode()
        ).hexdigest()[:4]

        exp_id = f"{slug}_{date_part}_{short_hash}"
        now = self._now()
        self._conn.execute(
            """INSERT INTO experiments (id, name, status, created_at, updated_at,
               config_json, env_json, seed)
               VALUES (?, ?, 'created', ?, ?, ?, ?, ?)""",
            (exp_id, name, now, now, json.dumps(config), env_json, seed),
        )
        self._conn.commit()
        return exp_id

    def get_experiment(self, experiment_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_experiments(self, status: str | None = None,
                         tag: str | None = None,
                         limit: int = 100, offset: int = 0) -> list[dict]:
        where = []
        params: list[Any] = []
        if status:
            where.append("e.status = ?")
            params.append(status)
        if tag:
            where.append("e.id IN (SELECT experiment_id FROM tags WHERE tag = ?)")
            params.append(tag)
        where_clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self._conn.execute(
            f"""SELECT e.* FROM experiments e {where_clause}
                ORDER BY e.created_at DESC LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_experiment_status(self, experiment_id: str, status: str,
                                  failure_reason: str | None = None) -> None:
        now = self._now()
        if failure_reason:
            self._conn.execute(
                """UPDATE experiments SET status=?, updated_at=?, failure_reason=?
                   WHERE id=?""",
                (status, now, failure_reason, experiment_id),
            )
        else:
            self._conn.execute(
                "UPDATE experiments SET status=?, updated_at=? WHERE id=?",
                (status, now, experiment_id),
            )
        self._conn.commit()

    def update_experiment(self, experiment_id: str, **kwargs) -> None:
        kwargs["updated_at"] = self._now()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        values = [*list(kwargs.values()), experiment_id]
        self._conn.execute(
            f"UPDATE experiments SET {sets} WHERE id=?", values
        )
        self._conn.commit()

    def delete_experiment(self, experiment_id: str) -> None:
        self._conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
        self._conn.commit()

    # ── 指标 ───────────────────────────────────────────────

    def log_metrics(self, experiment_id: str, metrics: dict[str, float],
                     step: int) -> None:
        rows = [(experiment_id, step, k, v) for k, v in metrics.items()]
        self._conn.executemany(
            """INSERT OR REPLACE INTO metrics (experiment_id, step, key, value)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def get_metrics(self, experiment_id: str,
                     keys: list[str] | None = None) -> list[dict]:
        if keys:
            placeholders = ",".join("?" for _ in keys)
            rows = self._conn.execute(
                f"""SELECT step, key, value FROM metrics
                    WHERE experiment_id=? AND key IN ({placeholders})
                    ORDER BY step ASC""",
                (experiment_id, *keys),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT step, key, value FROM metrics
                   WHERE experiment_id=? ORDER BY step ASC""",
                (experiment_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_metrics_dataframe(self, experiment_id: str,
                                keys: list[str] | None = None) -> pd.DataFrame:
        """返回 pandas DataFrame，每列为指标，索引为 step。"""
        import pandas as pd
        rows = self.get_metrics(experiment_id, keys)
        data: dict[int, dict[str, float]] = {}
        for r in rows:
            data.setdefault(r["step"], {})[r["key"]] = r["value"]
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.name = "step"
        return df

    # ── 标签 ───────────────────────────────────────────────

    def add_tag(self, experiment_id: str, tag: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO tags (experiment_id, tag) VALUES (?, ?)",
            (experiment_id, tag),
        )
        self._conn.commit()

    def remove_tag(self, experiment_id: str, tag: str) -> None:
        self._conn.execute(
            "DELETE FROM tags WHERE experiment_id=? AND tag=?",
            (experiment_id, tag),
        )
        self._conn.commit()

    def get_tags(self, experiment_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT tag FROM tags WHERE experiment_id=?", (experiment_id,)
        ).fetchall()
        return [r["tag"] for r in rows]

    # ── Checkpoint ──────────────────────────────────────────

    def save_checkpoint_record(self, experiment_id: str, epoch: int,
                                 path: str, metric_name: str | None = None,
                                 metric_value: float | None = None,
                                 is_best: bool = False,
                                 is_last: bool = False,
                                 is_ema: bool = False,
                                 file_size: int | None = None) -> None:
        self._conn.execute(
            """INSERT INTO checkpoints
               (experiment_id, epoch, path, metric_name, metric_value,
                is_best, is_last, is_ema, created_at, file_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, epoch, path, metric_name, metric_value,
             int(is_best), int(is_last), int(is_ema), self._now(), file_size),
        )
        # 更新旧 best 标记
        if is_best:
            self._conn.execute(
                """UPDATE checkpoints SET is_best=0
                   WHERE experiment_id=? AND id != (
                       SELECT id FROM checkpoints
                       WHERE experiment_id=? AND is_best=1
                       ORDER BY id DESC LIMIT 1
                   )""",
                (experiment_id, experiment_id),
            )
        if is_last:
            self._conn.execute(
                """UPDATE checkpoints SET is_last=0
                   WHERE experiment_id=? AND is_ema=? AND id != (
                       SELECT id FROM checkpoints
                       WHERE experiment_id=? AND is_last=1 AND is_ema=?
                       ORDER BY id DESC LIMIT 1
                   )""",
                (experiment_id, int(is_ema), experiment_id, int(is_ema)),
            )
        self._conn.commit()

    def get_checkpoints(self, experiment_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE experiment_id=? ORDER BY epoch ASC""",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_checkpoint(self, experiment_id: str,
                              ema: bool = False) -> dict | None:
        row = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE experiment_id=? AND is_best=1 AND is_ema=?
               ORDER BY id DESC LIMIT 1""",
            (experiment_id, int(ema)),
        ).fetchone()
        return dict(row) if row else None

    def get_last_checkpoint(self, experiment_id: str,
                              ema: bool = False) -> dict | None:
        row = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE experiment_id=? AND is_last=1 AND is_ema=?
               ORDER BY id DESC LIMIT 1""",
            (experiment_id, int(ema)),
        ).fetchone()
        return dict(row) if row else None

    def cleanup_checkpoints(self, experiment_id: str,
                              keep_last: int = 5) -> None:
        """清理旧 checkpoint，只保留最近 N 个。"""
        rows = self._conn.execute(
            """SELECT id, path FROM checkpoints
               WHERE experiment_id=? AND is_best=0 AND is_last=0
               ORDER BY created_at DESC""",
            (experiment_id,),
        ).fetchall()
        # 保留 keep_last 个非 best/last checkpoint，删除更旧的
        if len(rows) > keep_last:
            delete_ids = [r["id"] for r in rows[keep_last:]]
            placeholders = ",".join("?" for _ in delete_ids)
            self._conn.execute(
                f"DELETE FROM checkpoints WHERE id IN ({placeholders})",
                delete_ids,
            )
            self._conn.commit()

    # ── Artifacts ──────────────────────────────────────────

    def save_artifact(self, experiment_id: str, step: int, key: str,
                       type_: str, file_path: str | None = None,
                       data_json: str | None = None) -> None:
        self._conn.execute(
            """INSERT INTO artifacts
               (experiment_id, step, key, type, file_path, data_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (experiment_id, step, key, type_, file_path, data_json),
        )
        self._conn.commit()

    def get_artifacts(self, experiment_id: str,
                       keys: list[str] | None = None) -> list[dict]:
        if keys:
            placeholders = ",".join("?" for _ in keys)
            rows = self._conn.execute(
                f"""SELECT * FROM artifacts
                    WHERE experiment_id=? AND key IN ({placeholders})
                    ORDER BY step ASC""",
                (experiment_id, *keys),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE experiment_id=? ORDER BY step ASC",
                (experiment_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Sweep ───────────────────────────────────────────────

    def create_sweep(self, sweep_id: str, experiment_id: str,
                      config: dict, strategy: str) -> None:
        self._conn.execute(
            """INSERT INTO sweeps (id, experiment_id, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (sweep_id, experiment_id, json.dumps(config), strategy, self._now()),
        )
        self._conn.commit()

    def add_sweep_trial(self, sweep_id: str, experiment_id: str,
                         trial_index: int, config: dict) -> None:
        self._conn.execute(
            """INSERT INTO sweep_trials
               (sweep_id, experiment_id, trial_index, config_json)
               VALUES (?, ?, ?, ?)""",
            (sweep_id, experiment_id, trial_index, json.dumps(config)),
        )
        self._conn.commit()

    def list_sweeps(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sweeps ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_sweep(self, sweep_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sweeps WHERE id=?", (sweep_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_sweep_trials(self, sweep_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT st.*, e.status as exp_status, e.config_json as exp_config
               FROM sweep_trials st
               LEFT JOIN experiments e ON st.experiment_id = e.id
               WHERE st.sweep_id=? ORDER BY st.trial_index""",
            (sweep_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 数据集 ──────────────────────────────────────────────

    def register_dataset(self, name: str, path: str,
                          description: str = "") -> str:
        """注册一个数据集。

        Args:
            name: 数据集名称。
            path: 数据集路径。
            description: 可选描述。

        Returns:
            数据集 ID。
        """
        import random
        ds_id = f"ds_{random.randint(10000, 99999)}"
        now = self._now()
        self._conn.execute(
            """INSERT INTO datasets (id, name, path, description, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ds_id, name, path, description, now, now),
        )
        self._conn.commit()
        return ds_id

    def get_datasets(self) -> list[dict]:
        """列出所有已注册的数据集（含最新版本号）。"""
        rows = self._conn.execute(
            """SELECT d.*,
                      (SELECT version FROM dataset_versions
                       WHERE dataset_id = d.id ORDER BY recorded_at DESC LIMIT 1
                      ) as latest_version,
                      (SELECT total_files FROM dataset_versions
                       WHERE dataset_id = d.id ORDER BY recorded_at DESC LIMIT 1
                      ) as latest_files
               FROM datasets d ORDER BY d.updated_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_dataset(self, dataset_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        return dict(row) if row else None

    def record_dataset_version(
        self, dataset_id: str, version: str, root_hash: str,
        ann_hash: str = "", total_files: int = 0,
        total_size_bytes: int = 0,
        file_count_by_ext: dict | None = None,
        experiment_id: str | None = None,
    ) -> None:
        """记录数据集的一个新版本。

        Args:
            dataset_id: 数据集 ID。
            version: 版本号（如 v1, v2 或 hash 前缀）。
            root_hash: 根目录哈希。
            ann_hash: 标注文件 SHA256。
            total_files: 文件总数。
            total_size_bytes: 总大小（字节）。
            file_count_by_ext: 按扩展名统计的文件数。
            experiment_id: 关联的实验 ID（可选）。
        """
        self._conn.execute(
            """INSERT OR IGNORE INTO dataset_versions
               (dataset_id, version, root_hash, ann_hash, total_files,
                total_size_bytes, file_count_by_ext, recorded_at, experiment_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (dataset_id, version, root_hash, ann_hash, total_files,
             total_size_bytes,
             json.dumps(file_count_by_ext or {}),
             self._now(), experiment_id),
        )
        # 更新数据集的时间戳
        self._conn.execute(
            "UPDATE datasets SET updated_at=? WHERE id=?",
            (self._now(), dataset_id),
        )
        self._conn.commit()

    def get_dataset_versions(self, dataset_id: str) -> list[dict]:
        """获取数据集的所有版本记录。"""
        rows = self._conn.execute(
            """SELECT dv.*, e.name as experiment_name
               FROM dataset_versions dv
               LEFT JOIN experiments e ON dv.experiment_id = e.id
               WHERE dv.dataset_id = ? ORDER BY dv.recorded_at DESC""",
            (dataset_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_experiment_datasets(self, experiment_id: str) -> list[dict]:
        """获取实验关联的所有数据集版本。"""
        rows = self._conn.execute(
            """SELECT dv.*, d.name as dataset_name, d.path as dataset_path
               FROM dataset_versions dv
               JOIN datasets d ON dv.dataset_id = d.id
               WHERE dv.experiment_id = ? ORDER BY dv.recorded_at DESC""",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 搜索 ───────────────────────────────────────────────

    def search_experiments(self, tag: str | None = None,
                            status: str | None = None,
                            text: str | None = None,
                            metric_key: str | None = None,
                            metric_min: float | None = None,
                            metric_max: float | None = None,
                            limit: int = 100) -> list[dict]:
        where = []
        params: list[Any] = []

        if status:
            where.append("e.status = ?")
            params.append(status)
        if tag:
            where.append("e.id IN (SELECT experiment_id FROM tags WHERE tag = ?)")
            params.append(tag)
        if text:
            where.append("(e.name LIKE ? OR e.notes LIKE ?)")
            params.extend([f"%{text}%", f"%{text}%"])
        if metric_key is not None:
            sub = "SELECT experiment_id FROM metrics WHERE key = ?"
            sub_params: list[Any] = [metric_key]
            if metric_min is not None:
                sub += " AND value >= ?"
                sub_params.append(metric_min)
            if metric_max is not None:
                sub += " AND value <= ?"
                sub_params.append(metric_max)
            where.append(f"e.id IN ({sub})")
            params.extend(sub_params)

        where_clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self._conn.execute(
            f"""SELECT DISTINCT e.* FROM experiments e {where_clause}
                ORDER BY e.created_at DESC LIMIT ?""",
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
