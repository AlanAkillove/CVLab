"""Schema 迁移工具（预留）。"""

from cvlab.db.schema import SCHEMA_SQL


def run_migrations(db_path: str) -> None:
    """执行所有待应用的迁移。当前版本直接初始化 Schema。"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
