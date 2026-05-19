<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue.svg?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange.svg?style=flat-square" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/i18n-zh%20%7C%20en-blueviolet.svg?style=flat-square" alt="i18n">
  <br>
  <a href="README_EN.md"><img src="https://img.shields.io/badge/English-%F0%9F%87%AC%F0%9F%87%A7-white.svg?style=flat-square" alt="EN"></a>
</p>

<h1 align="center">CVLab</h1>
<p align="center"><b>CV 实验管理平台</b> — 让研究者专注于模型和数据本身，而非工程琐事</p>

<p align="center">
  轻量级 · 非侵入式 · CV 专精 · 零服务依赖
</p>

---

## 目录

- [简介](#简介)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [命令参考](#命令参考)
- [了解更多](#了解更多)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 简介

CVLab 是一个轻量级、CV 专精的训练实验管理工具。装一个 `pip install`，加三行代码，就能获得实验追踪、梯度监控、诊断分析和超参扫描能力。

**语言支持**: 中文（默认）| [English](README_EN.md)

---

## 核心特性

- **📊 实验追踪** — 自动记录指标、图片、混淆矩阵、检测框和分割 Mask。SQLite 持久化，零服务依赖。一键复现命令。
- **🔌 非侵入式梯度监控** — 基于 PyTorch `register_full_backward_hook` 的采样模式，无需修改模型代码。梯度消失/爆炸自动告警。
- **🖥️ 环境探针 + 加速诊断** — 自动检测 GPU / CUDA / 存储类型，推荐 6 项训练加速选项（AMP、BF16、torch.compile 等）。
- **🎯 Batch Size 自动探测** — 二分搜索 + 20% 安全余量，含 OOM 自动恢复机制（每次减小 20%，最多重试 2 次）。
- **🔍 训练诊断套件** — I/O 瓶颈检测、Loss 异常分析、模型性能画像（FLOPs/参数量/延迟）、超参重要性分析。

> 完整功能列表见 [USAGE.md](USAGE.md)（数据集分析、超参扫描、HTML 报告、可视化 UI 等）。

---

## 截图

> *界面截图准备中 —— 欢迎贡献！*

---

## 快速开始

### 安装

```bash
# 方式一：git clone 安装（当前推荐）
git clone https://github.com/AlanAkillove/CVLab.git
cd CVLab
pip install -e .

# 方式二：使用 uv（更快）
pip install uv
uv venv && uv sync --dev

# 首次使用前安装 PyTorch（CPU 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 查看是否安装成功
cvlab --version
```

> **注意**: CVLab 目前通过 GitHub 分发，暂未上架 PyPI。`pip install cvlab` 会安装其他同名包，请使用上方 git clone 方式。

### 三行代码接入

```python
import torch
import torch.nn as nn
from cvlab import Tracker

# 1. 创建实验（自动记录环境、保存脚本快照）
tracker = Tracker(config={
    "name": "my_experiment",
    "model": {"name": "SimpleCNN"},
    "training": {"epochs": 5, "batch_size": 64, "lr": 0.001},
})

# 2. 注入 Hook（非侵入式梯度监控）
monitor = tracker.watch(model, log_gradients=True, watch_layers=["conv1"])

# 3. 正常训练
for epoch in range(5):
    for inputs, labels in loader:
        loss = criterion(model(inputs), labels)
        loss.backward()
        monitor.step(global_step)
        optimizer.step()

    # 4. 记录指标
    tracker.log({"train/loss": loss.item(), "train/acc": acc}, step=epoch)
    tracker.save_checkpoint(model, optimizer, epoch=epoch)

# 5. 完成实验
tracker.finish()
print(tracker.get_reproduce_command())
```

完整示例见 [examples/quickstart.py](examples/quickstart.py)。

---

## 使用示例

### 快速体验（30 秒，无 GPU 也可运行）

```bash
cvlab init
cvlab train --config examples/tiny_cnn.yaml
```

### CLI 模式

```bash
# 中英文切换
cvlab --lang en help        # English
cvlab --lang zh help        # 中文（默认）

# 初始化
cvlab init

# 训练（使用 YAML 配置）
cvlab train --config examples/cifar10.yaml

# 列出实验
cvlab list

# 查看实验详情
cvlab show exp_250519_123456_7890

# 诊断分析
cvlab diagnose loss exp_250519_123456_7890

# 超参扫描
cvlab sweep create --config sweep.yaml

# 模型画像
cvlab profile --model resnet18

# 数据集分析
cvlab data analyze ./dataset
```

### 配置驱动训练

创建 `config.yaml`:

```yaml
model:
  name: resnet18
  pretrained: false
training:
  epochs: 50
  batch_size: null          # 自动探测最大 batch size
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
# 自动解析 app.py 路径，无需手动查找
cvlab ui

# 指定端口和语言
cvlab ui --port 8502 --lang en
```

在浏览器中打开 `http://localhost:8501`，右上角可切换语言和深色模式。

> 作为第三方库安装后，无需知道 `app.py` 的安装路径，`cvlab ui` 会自动处理。

---

## 命令参考

| 命令 | 说明 |
|------|------|
| `cvlab init` | 在当前目录初始化 CVLab |
| `cvlab train --config <yaml>` | 启动训练 |
| `cvlab list [--status] [--tag]` | 列出实验 |
| `cvlab show <experiment_id>` | 查看实验详情 |
| `cvlab diagnose loss <exp_id>` | Loss 异常诊断 |
| `cvlab diagnose gradient <exp_id>` | 梯度健康诊断 |
| `cvlab diagnose dataloader <config>` | DataLoader 性能诊断 |
| `cvlab compare <exp_1> <exp_2> [exp_3...]` | 多实验 CLI 对比（Rich 高亮表格） |
| `cvlab sweep create --config <yaml>` | 创建超参扫描 |
| `cvlab sweep analyze <sweep_id>` | 分析超参重要性 |
| `cvlab profile --model <name>` | 模型性能画像 |
| `cvlab weights info <model>` | 权重信息 |
| `cvlab weights download <model>` | 下载预训练权重 |
| `cvlab data analyze <path>` | 数据集分析 |
| `cvlab data check <path>` | 数据血缘检查 |
| `cvlab data history` | 数据集快照历史 |
| `cvlab ui [--port] [--lang]` | 启动 Web 界面 |

---

## 项目结构

```
cvlab/
├── core/           # 核心：Tracker、Watch Hook、Seed 管理
├── db/             # SQLite 持久化（WAL 模式，零依赖）
├── config/         # YAML 配置加载与验证
├── detect/         # 环境检测：OS/CPU/GPU/存储
├── probe/          # Batch Size 自动探测
├── diagnose/       # 训练诊断：I/O、Loss、梯度
├── profile/        # 模型性能画像
├── weights/        # 预训练权重管理
├── sweep/          # 超参扫描（Grid + Random）
├── data/           # 数据分析、血缘追踪
├── report/         # HTML 报告生成
├── checkpoint/     # Checkpoint 管理
├── train/          # 训练子进程管理
├── i18n/           # 国际化（中文 / English）
├── cli/            # argparse CLI 入口
├── ui/             # Streamlit Web 界面
└── tests/          # 测试套件（150+ 测试）
```

---

## 设计原则

| 原则 | 说明 |
|------|------|
| **不绑定框架** | 原生 `nn.Module`，不需要继承任何 CVLab 基类 |
| **不干扰训练** | Hook 采样模式，不修改训练循环结构 |
| **零服务依赖** | SQLite 本地存储，`pip install` + `import cvlab` 直接使用 |
| **CV 专精** | 检测框/分割掩码/混淆矩阵内置可视化 |
| **国际化优先** | 从设计之初就支持中英双语 |

---

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- 操作系统：Windows / Linux / macOS

---

## 贡献指南

感谢你考虑为 CVLab 贡献！

1. **报告问题**: 在 GitHub Issues 提交，包含完整错误信息、Python 版本和复现步骤
2. **提交代码**: Fork 仓库 → 创建特性分支 → 提交变更（包含测试）→ 发起 Pull Request
3. **开发指南**: 见 [CONTRIBUTING.md](CONTRIBUTING.md)

### 本地开发

```bash
git clone https://github.com/AlanAkillove/CVLab.git
cd CVLab
uv venv && uv sync --dev
uv run pytest            # 运行测试
uv run ruff check cvlab/  # 代码检查
```

---

## 许可证

MIT © CVLab Contributors

---

<p align="center">
  <a href="README_EN.md"><b>English README</b></a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/AlanAkillove/CVLab/issues">Issues</a>
</p>
