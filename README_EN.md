<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue.svg?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange.svg?style=flat-square" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/i18n-zh%20%7C%20en-blueviolet.svg?style=flat-square" alt="i18n">
  <br>
  <a href="README.md"><img src="https://img.shields.io/badge/中文-%F0%9F%87%A8%F0%9F%87%B3-white.svg?style=flat-square" alt="ZH"></a>
</p>

<h1 align="center">CVLab</h1>
<p align="center"><b>CV Experiment Management Platform</b> — Let researchers focus on models and data, not engineering chores</p>

<p align="center">
  Lightweight · Non-intrusive · CV-specialized · Zero service dependencies
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Command Reference](#command-reference)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

CVLab is a lightweight, CV-specialized training experiment management tool. It doesn't require model inheritance changes, doesn't bind to a framework, and doesn't deploy services. One `pip install`, three lines of code, and you get experiment tracking, gradient monitoring, data analysis, and hyperparameter sweep capabilities.

**Language Support**: [中文](README.md) | English (default)

---

## Key Features

<details open>
<summary><b>📊 Experiment Tracking</b></summary>

- Automatic experiment ID generation, deterministic seeding, environment snapshots, and script copies
- Scalar metric logging with SQLite persistence (WAL mode, zero dependencies)
- Image, confusion matrix, detection box, and segmentation mask visualization
- Checkpoint management (auto-rotation, EMA weights, best/last tagging)
- One-click reproduction command generation

</details>

<details>
<summary><b>🔌 Non-intrusive Hook Injection</b></summary>

- Gradient monitoring via `register_full_backward_hook` (sampling mode)
- Vanishing/exploding gradient alerts (thresholds: vanish < 1e-5, explosion > 10)
- Layer-specific monitoring and activation value sampling
- Zero model code modifications required

</details>

<details>
<summary><b>🖥️ Environment Detection & Acceleration</b></summary>

- Auto-detect OS / Python / PyTorch / CUDA versions
- GPU model, memory, Compute Capability, TensorCore support
- WSL2 detection, CUDA version mismatch warnings
- 6 training acceleration options (AMP FP16, BF16, cuDNN Benchmark, torch.compile, Channels Last, Gradient Checkpointing)

</details>

<details>
<summary><b>🎯 Batch Size Auto-Detection</b></summary>

- Binary search algorithm with 20% safety margin
- Pessimistic data injection (max resolution + dense labels)
- AMP/BF16 aligned probing
- Multi-GPU aware

</details>

<details>
<summary><b>🔍 Training Diagnostics</b></summary>

- **I/O Bottleneck Detection**: Analyze DataLoader load time vs compute time, recommend optimal num_workers
- **Loss Anomaly Detection**: NaN/Inf, explosion, plateau, spike, LR anomalies
- **Model Performance Profile**: FLOPs, parameter count, forward/backward latency, peak memory
- **Weight Loading Diagnostics**: missing/unexpected keys, shape checks, dual weight file diff

</details>

<details>
<summary><b>⚡ Hyperparameter Sweep</b></summary>

- Grid search (Cartesian product enumeration)
- Random search (choice/uniform/loguniform/int distributions)
- Trial management + best trial search
- Random forest hyperparameter importance analysis

</details>

<details>
<summary><b>📁 Dataset Analysis</b></summary>

- Class distribution, image format statistics, size distribution, class balance scoring
- Data provenance tracking (two-level: O(1) root directory snapshot + annotation SHA256)
- Change detection, snapshot listing

</details>

<details>
<summary><b>🌐 Visual UI</b></summary>

- Streamlit multi-page (experiment list, detail, compare, sweep, environment diagnostics)
- Plotly interactive metric curve overlay
- Language switching (中文 / English)
- Dark mode support

</details>

<details>
<summary><b>📄 Report Generation</b></summary>

- Self-contained HTML reports (Jinja2 templates)
- Hyperparameters, environment, metrics, checkpoints, reproduction commands

</details>

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlanAkillove/CVLab.git
cd CVLab

# Recommended: use uv (or pip)
pip install uv
uv venv
uv sync --dev

# Or use pip directly
pip install -e .

# Install PyTorch (CPU version if no GPU)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Three Lines of Code

```python
import torch
import torch.nn as nn
from cvlab import Tracker

# 1. Create experiment (auto-records environment, saves script snapshot)
tracker = Tracker(config={
    "name": "my_experiment",
    "model": {"name": "SimpleCNN"},
    "training": {"epochs": 5, "batch_size": 64, "lr": 0.001},
})

# 2. Inject Hook (non-intrusive gradient monitoring)
monitor = tracker.watch(model, log_gradients=True, watch_layers=["conv1"])

# 3. Normal training loop
for epoch in range(5):
    for inputs, labels in loader:
        loss = criterion(model(inputs), labels)
        loss.backward()
        monitor.step(global_step)
        optimizer.step()

    # 4. Log metrics
    tracker.log({"train/loss": loss.item(), "train/acc": acc}, step=epoch)
    tracker.save_checkpoint(model, optimizer, epoch=epoch)

# 5. Finish experiment
tracker.finish()
print(tracker.get_reproduce_command())
```

Full example: [examples/quickstart.py](examples/quickstart.py)

---

## Usage Examples

### CLI Mode

```bash
# Language switching
cvlab --lang en help        # English (default)
cvlab --lang zh help        # 中文

# Initialize
cvlab init

# Train (with YAML config)
cvlab train --config examples/cifar10.yaml

# List experiments
cvlab list

# Show experiment details
cvlab show exp_250519_123456_7890

# Diagnostics
cvlab diagnose loss exp_250519_123456_7890

# Hyperparameter sweep
cvlab sweep create --config sweep.yaml

# Model profiling
cvlab profile --model resnet18

# Dataset analysis
cvlab data analyze ./dataset
```

### Config-Driven Training

Create `config.yaml`:

```yaml
model:
  name: resnet18
  pretrained: false
training:
  epochs: 50
  batch_size: null          # auto-detect max batch size
  optimizer: adam
  lr: 0.001
  scheduler: cosine
data:
  dataset: ./data
  num_workers: 2
seed: 42
```

```bash
cvlab train --config config.yaml
```

### Web UI

```bash
streamlit run cvlab/ui/app.py
```

Open `http://localhost:8501` in your browser. Language switching and dark mode toggle are available in the top-right corner.

---

## Command Reference

| Command | Description |
|---------|-------------|
| `cvlab init` | Initialize CVLab in current directory |
| `cvlab train --config <yaml>` | Start training |
| `cvlab list [--status] [--tag]` | List experiments |
| `cvlab show <experiment_id>` | Show experiment details |
| `cvlab diagnose loss <exp_id>` | Loss anomaly diagnosis |
| `cvlab diagnose gradient <exp_id>` | Gradient health diagnosis |
| `cvlab diagnose dataloader <config>` | DataLoader performance diagnosis |
| `cvlab sweep create --config <yaml>` | Create hyperparameter sweep |
| `cvlab sweep analyze <sweep_id>` | Analyze hyperparameter importance |
| `cvlab profile --model <name>` | Model performance profiling |
| `cvlab weights info <model>` | Weight information |
| `cvlab weights download <model>` | Download pretrained weights |
| `cvlab data analyze <path>` | Dataset analysis |
| `cvlab data check <path>` | Data provenance check |
| `cvlab data history` | Dataset snapshot history |

---

## Project Structure

```
cvlab/
├── core/           # Core: Tracker, Watch Hook, Seed management
├── db/             # SQLite persistence (WAL mode, zero deps)
├── config/         # YAML config loading & validation
├── detect/         # Environment detection: OS/CPU/GPU/Storage
├── probe/          # Batch size auto-detection
├── diagnose/       # Training diagnostics: I/O, Loss, Gradients
├── profile/        # Model performance profiling
├── weights/        # Pretrained weight management
├── sweep/          # Hyperparameter sweep (Grid + Random)
├── data/           # Data analysis & provenance tracking
├── report/         # HTML report generation
├── checkpoint/     # Checkpoint management
├── train/          # Training subprocess management
├── i18n/           # Internationalization (中文 / English)
├── cli/            # argparse CLI entry point
├── ui/             # Streamlit web UI
└── tests/          # Test suite (150+ tests)
```

---

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Framework Agnostic** | Native `nn.Module`, no CVLab base class required |
| **Non-Intrusive** | Hook sampling mode, no training loop modification |
| **Zero Service Dependencies** | SQLite local storage, `pip install` + `import cvlab` |
| **CV Specialized** | Built-in detection box / segmentation mask / confusion matrix visualization |
| **i18n First** | Bilingual from day one |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OS: Windows / Linux / macOS

---

## Contributing

Thank you for considering contributing to CVLab!

1. **Report Issues**: Open a GitHub Issue with full error information, Python version, and reproduction steps
2. **Submit Code**: Fork → Create feature branch → Commit changes (with tests) → Open Pull Request
3. **Development Guide**: See [CONTRIBUTING.md](CONTRIBUTING.md)

### Local Development

```bash
git clone https://github.com/AlanAkillove/CVLab.git
cd CVLab
uv venv && uv sync --dev
uv run pytest              # Run tests
uv run ruff check cvlab/   # Lint check
```

---

## License

MIT © CVLab Contributors

---

<p align="center">
  <a href="README.md"><b>中文 README</b></a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/AlanAkillove/CVLab/issues">Issues</a>
</p>
