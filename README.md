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
- [项目结构](#项目结构)
- [设计原则](#设计原则)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 简介

CVLab 是一个轻量级、CV 专精的训练实验管理工具。它不要求你改模型继承、不绑定框架、不部署服务。装一个 `pip install`，加三行代码，就能获得实验追踪、梯度监控、数据分析和超参扫描能力。

**语言支持**: 中文（默认）| [English](README_EN.md)

---

## 核心特性

<details open>
<summary><b>📊 实验追踪</b></summary>

- 自动创建实验 ID、固定随机种子、保存环境快照和脚本副本
- 标量指标记录与 SQLite 持久化（WAL 模式，零依赖）
- 图片、混淆矩阵、检测框、分割掩码可视化
- Checkpoint 管理（自动轮转、EMA 权重、best/last 标记）
- 一键复现命令生成

</details>

<details>
<summary><b>🔌 非侵入式 Hook 注入</b></summary>

- 基于 `register_full_backward_hook` 采样模式梯度监控
- 梯度消失/爆炸告警（阈值: vanish < 1e-5, explosion > 10）
- 支持指定层监控，支持激活值采样
- 零修改模型代码

</details>

<details>
<summary><b>🖥️ 环境检测与加速配置</b></summary>

- 自动检测 OS / Python / PyTorch / CUDA 版本
- GPU 型号、显存、Compute Capability、TensorCore 支持
- WSL2 检测、CUDA 版本不匹配告警
- 6 项训练加速选项（AMP FP16、BF16、cuDNN Benchmark、torch.compile、Channels Last、Gradient Checkpointing）

</details>

<details>
<summary><b>🎯 Batch Size 自动探测</b></summary>

- 二分搜索算法，安全余量 20%
- 悲观数据注入（最大分辨率 + 密集标签）
- AMP/BF16 对齐探测
- 多 GPU 感知

</details>

<details>
<summary><b>🔍 训练诊断</b></summary>

- **I/O 瓶颈检测**: 分析 DataLoader 加载时间 vs 计算时间，推荐最优 num_workers
- **Loss 异常检测**: NaN/Inf、爆炸、平台期、突跳、LR 异常
- **模型性能画像**: FLOPs、参数量、前向/反向延迟、峰值显存
- **权重加载诊断**: missing/unexpected keys、形状检查、双权重文件差异对比

</details>

<details>
<summary><b>⚡ 超参扫描</b></summary>

- Grid 搜索（笛卡尔积枚举）
- Random 搜索（choice/uniform/loguniform/int 四种分布）
- Trial 管理 + 最佳 Trial 查找
- 随机森林超参重要性分析

</details>

<details>
<summary><b>📁 数据集分析</b></summary>

- 类别分布、图片格式统计、尺寸分布、类别平衡度评分
- 数据血缘追踪（两级：O(1) 根目录快照 + 标注文件 SHA256）
- 变化检测、快照列表

</details>

<details>
<summary><b>🌐 可视化 UI</b></summary>

- Streamlit 多页面（实验列表、详情、对比、Sweep、环境诊断）
- Plotly 交互式指标曲线叠加
- 语言切换（中文 / English）
- 深色模式支持

</details>

<details>
<summary><b>📄 报告生成</b></summary>

- 自包含 HTML 报告（Jinja2 模板）
- 超参、环境、指标、Checkpoints、复现命令

</details>

---

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/AlanAkillove/CVLab.git
cd CVLab

# 推荐使用 uv（也可以使用 pip）
pip install uv
uv venv
uv sync --dev

# 或直接使用 pip
pip install -e .

# 安装 PyTorch（如无 GPU）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

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
streamlit run cvlab/ui/app.py
```

在浏览器中打开 `http://localhost:8501`，右上角可切换语言和深色模式。

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
| `cvlab sweep create --config <yaml>` | 创建超参扫描 |
| `cvlab sweep analyze <sweep_id>` | 分析超参重要性 |
| `cvlab profile --model <name>` | 模型性能画像 |
| `cvlab weights info <model>` | 权重信息 |
| `cvlab weights download <model>` | 下载预训练权重 |
| `cvlab data analyze <path>` | 数据集分析 |
| `cvlab data check <path>` | 数据血缘检查 |
| `cvlab data history` | 数据集快照历史 |

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
