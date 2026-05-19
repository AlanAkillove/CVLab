# Changelog

## 0.2.0 (2026-05-19) — Internationalization & Enterprise Upgrade

### i18n System (New)
- Full internationalization (i18n) framework with `cvlab/i18n/` module
- Chinese (zh) and English (en) translation support via JSON files
- Language auto-detection from system locale, env var `CVLAB_LANG`, or CLI `--lang` flag
- `_()` translation function with format string support
- Language switcher in Streamlit UI (top-right corner, instant switch)
- All CLI modules (train, sweep, profile, weights, diagnose, data) translated
- All 6 UI pages (experiments, detail, compare, sweep, diagnostics, datasets) i18n-ready
- `cvlab.i18n` exported as public API

### Bug Fixes
- **Critical**: Fixed config shallow merge in `Tracker._create_experiment()` — now uses `merge_config()` for deep recursive merge
- **Critical**: Fixed `snapshot_dataset()` hardcoded `dataset_total=0` — now correctly passes `total_size`
- Fixed unused `import stat` in `detect/storage_info.py`
- Fixed unused `import EnvironmentProbe` in `ui/components/accelerations.py`
- Fixed `print()` → `logging` in `core/tracker.py:finish()`

### Chinese Encoding
- All `open()` calls now explicitly use `encoding="utf-8"` (config, detect modules, i18n)
- All `subprocess.run()` calls with `text=True` now use `encoding="utf-8", errors="replace"` (detect modules)
- UTF-8 encoding enforced across YAML, JSON, HTML, CSS file operations

### Testing Infrastructure
- **36 new i18n tests**: translation, language switching, format strings, threading safety
- **27 new CLI smoke tests**: version, help, init, list, show, lang flag, error handling
- **52 enhanced config tests** (up from 8): merge, load, validate, serialization, immutability
- Enhanced `conftest.py` with rich fixtures, mock DB, GPU mocking, test isolation
- pytest markers: `requires_cuda`, `requires_torch`, skip-slow/run-gpu options

### DevOps & Developer Experience
- Dockerfile (CPU) + Dockerfile.gpu (CUDA) — multi-stage, minimal images
- GitHub Actions CI workflow (lint → test → docker build)
- `.editorconfig` for consistent coding style
- `.pre-commit-config.yaml` with ruff, mypy, trailing whitespace checks
- `Makefile` with common commands (setup, test, lint, docker, pre-commit)
- Enhanced `pyproject.toml` with ruff, mypy, coverage configs
- `.dockerignore` for optimized builds

### UI Enhancements
- Language switcher in Streamlit UI (top-right corner, dropdown)
- Dark mode support (CSS media query + manual toggle)
- Swiss Design improvements: consistent typography, scrollbar styling
- All UI text wrapped with `_()` for automatic translation
- Sidebar footer with version info

### Code Quality
- Extracted shared `cvlab/core/utils.py` with `flatten_dict()` (replaces 3 duplicated implementations)
- Fixed `checkpoint/manager.py` error on empty metrics dict
- Consistent `encoding="utf-8"` across all file I/O operations
- Improved `except Exception` granularity in detect modules

## 0.1.0 (2026-05-18)

- Initial release
- Experiment tracking with Tracker API
- Gradient monitoring via PyTorch hooks
- Hyperparameter sweep (grid + random search)
- Batch size auto-detection
- Environment diagnostics
- Checkpoint management with EMA support
- Streamlit web UI for experiment management
- Dataset version tracking and provenance
- CLI for train, profile, diagnose, sweep, data, weights
- HTML report generation
