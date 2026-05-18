"""SQLite 数据库 Schema。"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'created',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    env_json        TEXT NOT NULL DEFAULT '{}',
    seed            INTEGER,
    git_hash        TEXT,
    script_hash     TEXT,
    command         TEXT,
    dataset_path    TEXT,
    dataset_total   INTEGER,
    dataset_files   INTEGER,
    dataset_ann_hash TEXT,
    failure_reason  TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    step            INTEGER NOT NULL,
    key             TEXT NOT NULL,
    value           REAL NOT NULL,
    UNIQUE(experiment_id, step, key)
);

CREATE TABLE IF NOT EXISTS tags (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    tag             TEXT NOT NULL,
    UNIQUE(experiment_id, tag)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    epoch           INTEGER NOT NULL,
    path            TEXT NOT NULL,
    metric_name     TEXT,
    metric_value    REAL,
    is_best         INTEGER DEFAULT 0,
    is_last         INTEGER DEFAULT 0,
    is_ema          INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    file_size       INTEGER
);

CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    step            INTEGER NOT NULL,
    key             TEXT NOT NULL,
    type            TEXT NOT NULL,
    file_path       TEXT,
    data_json       TEXT
);

CREATE TABLE IF NOT EXISTS sweeps (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    config_json     TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS sweep_trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id        TEXT NOT NULL REFERENCES sweeps(id) ON DELETE CASCADE,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    trial_index     INTEGER NOT NULL,
    config_json     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id, step);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(experiment_id, key);
CREATE INDEX IF NOT EXISTS idx_tags_experiment ON tags(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_sweep_trials_sweep ON sweep_trials(sweep_id);

CREATE TABLE IF NOT EXISTS datasets (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    path            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version         TEXT NOT NULL,
    root_hash       TEXT NOT NULL,
    ann_hash        TEXT DEFAULT '',
    total_files     INTEGER DEFAULT 0,
    total_size_bytes INTEGER DEFAULT 0,
    file_count_by_ext TEXT DEFAULT '{}',
    recorded_at     TEXT NOT NULL,
    experiment_id   TEXT REFERENCES experiments(id) ON DELETE SET NULL,
    UNIQUE(dataset_id, version)
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_id ON dataset_versions(dataset_id);
"""
