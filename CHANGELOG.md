# Changelog

## 0.2.4 (2026-05-19) — Readable Experiment IDs

### UX Improvement
- **Redesigned experiment ID format**: `{semantic-slug}_{MMDD}_{4char-hash}`
  - Old: `exp_250519_123456_789_1234` (27 chars, no semantics)
  - New: `resnet18-cifar10_0519_a3f2` or `my-experiment_0519_x7k9` (readable at a glance)
  - Falls back: `name → model_dataset → exp`
- `cvlab list` now shows model name column
- `cvlab show` header now displays experiment name prominently

### Bug Fixes
- Fixed test_tracker.py assertion for new ID format

## 0.2.3 (2026-05-19) — Feature Growth

### New Features
- **`cvlab tag`** — experiment tag management CLI
  ```bash
  cvlab tag add exp_001 baseline       # add tag
  cvlab tag remove exp_001 baseline    # remove tag
  cvlab tag list                       # list all tags
  cvlab tag search baseline            # search experiments by tag
  ```
- **`cvlab note`** — experiment note editing CLI
  ```bash
  cvlab note exp_001                    # view note
  cvlab note exp_001 "lr too high"     # set note
  cvlab note exp_001 --clear           # clear note
  ```
- **`cvlab export`** — model export to ONNX / TorchScript
  ```bash
  cvlab export --checkpoint best.pt --format onnx --input 1x3x224x224
  cvlab export --checkpoint best.pt --format torchscript --script-mode trace
  ```
  Includes input shape validation, forward pass verification, auto model reconstruction.
- **Prediction sample timeline** — when `log_images: true` in config, automatically saves prediction visualizations every N epochs during training
- **Webhook notification system** — training complete/failure/OOM notifications
  ```bash
  cvlab train --config config.yaml --webhook https://hooks.slack.com/xxx
  ```
  Auto-detects Slack / Feishu / DingTalk / generic webhook formats.

## 0.2.2 (2026-05-19) — Third-Party Library Usability

### New Feature
- **`cvlab ui` command** — launch Streamlit Web UI with auto-resolved path
  ```bash
  cvlab ui                    # default port 8501
  cvlab ui --port 8502 --lang en
  cvlab ui --host 0.0.0.0     # LAN access
  ```
  Users no longer need to locate `app.py` manually. Three fallback resolution strategies:
  development path → `importlib.resources` → `__file__` inspection

### Documentation
- Added third-party library usage guide to README/README_EN
- `cvlab ui` documented in command reference tables
- Web UI sections updated to show `cvlab ui` command instead of `streamlit run`
- USAGE.md: added `cvlab ui` command reference + FAQ entry
- All CLI invocation examples now show the `cvlab` command (not `python -m`)

## 0.2.1 (2026-05-19) — Expert Review Fixes

### New Feature
- **`cvlab compare` CLI command** — Rich highlighted terminal table for multi-experiment comparison
  ```bash
  cvlab compare exp_001 exp_002 --metric val/acc
  ```
  Shows config diffs (yellow highlights) and metric summaries in side-by-side format

### Bug Fixes
- **OOM recovery improvements**:
  - Batch size reduction changed from halving to 20% (x0.8), less aggressive recovery
  - Broadened OOM detection: now catches both `torch.cuda.OutOfMemoryError` and `RuntimeError` with "CUDA out of memory" message
  - Non-OOM errors (exit code 1) correctly bypass retry logic
- Fixed `%` formatting inconsistency with `_()` in `cli/main.py` — now uses `.format()` uniformly
- Fixed UI navigation titles not wrapped in `_()` in `app.py`
- Fixed `Dockerfile.gpu` redundant `python3-pip` install and missing non-root user
- Fixed `test_report.py` 2 failing tests (outdated `_flatten` references)
- Updated version to 0.2.1 across `pyproject.toml` and `__init__.py`

### Documentation
- **README.md/README_EN.md**: Streamlined features to top 5 (was 9 collapsed sections), added screenshot placeholder, added compare to command table, updated ToC
- **USAGE.md**: 
  - Added CV-specific Python API examples (confusion matrix, detection, segmentation)
  - Updated all OOM descriptions to "减小 20%" (was "减半")
  - Clarified `cvlab data augment` scope (CLI static vs UI interactive)
  - Added model limitation docs with Python API guidance
  - Updated FAQ to reference `cvlab compare` CLI command
- Updated CHANGELOG with v0.2.1 entry

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
