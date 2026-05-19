# CVLab 使用指南

CVLab 是一个 CV（计算机视觉）实验管理平台，帮助研究者专注于模型和数据本身，而非工程琐事。

> **v0.2.x 预览版** — 功能在快速迭代中，API 可能有小幅变化。PyPI 发布前稳定锁定。

## 目录

- [安装](#安装)
  - [依赖说明](#依赖说明)
- [快速开始](#快速开始)
  - [1. 初始化](#1-初始化)
  - [2. 准备训练配置](#2-准备训练配置)
  - [3. 启动训练](#3-启动训练)
  - [4. 查看实验](#4-查看实验)
  - [5. 诊断分析](#5-诊断分析)
  - [6. 超参扫描](#6-超参扫描)
  - [7. 多实验对比](#7-多实验对比)
- [命令参考](#命令参考)
  - [`cvlab train`](#cvlab-train)
  - [`cvlab compare`](#cvlab-compare)
  - [`cvlab list`](#cvlab-list)
  - [`cvlab show`](#cvlab-show)
  - [`cvlab diagnose`](#cvlab-diagnose)
  - [`cvlab data`](#cvlab-data)
  - [`cvlab profile`](#cvlab-profile)
  - [`cvlab weights`](#cvlab-weights)
  - [`cvlab sweep`](#cvlab-sweep)
  - [`cvlab estimate`](#cvlab-estimate)
  - [`cvlab init`](#cvlab-init)
  - [`cvlab ui`](#cvlab-ui)
  - [`cvlab help`](#cvlab-help)
- [Python API — CV 专项功能](#python-api--cv-专项功能)
  - [混淆矩阵](#混淆矩阵)
  - [检测框可视化](#检测框可视化)
  - [分割掩码叠加](#分割掩码叠加)
  - [完整示例](#完整示例)
- [关于模型支持](#关于模型支持)
- [训练最佳实践](#训练最佳实践)
  - [选择 Batch Size](#选择-batch-size)
  - [监控梯度](#监控梯度)
  - [恢复中断训练](#恢复中断训练)
- [Python API 速览](#python-api-速览)
- [数据存储](#数据存储)
- [常见问题](#常见问题)
  - [Q: 提示 ModuleNotFoundError](#q-提示-modulenotfounderror)
  - [Q: CUDA Out of Memory](#q-cuda-out-of-memory)
  - [Q: 训练时 DataLoader 成为瓶颈](#q-训练时-dataloader-成为瓶颈)
  - [Q: 如何复现实验？](#q-如何复现实验)
  - [Q: 如何对比多个实验？](#q-如何对比多个实验)
  - [Q: 如何启动 Web UI？](#q-如何启动-web-ui)

## 安装

```bash
# 克隆项目后，进入目录安装
cd CVLab
pip install -e .
```

### 依赖说明

CVLab 需要 PyTorch 2.0+（推荐 CUDA 版本）。首次使用前请确保：

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 输出 True 表示 CUDA 可用
```

完整依赖清单见 `pyproject.toml`。

## 快速开始

### 1. 初始化

```bash
# 在工作目录初始化 CVLab（创建 .cvlab/ 存储目录）
cvlab init
```

### 2. 准备训练配置

创建 YAML 配置文件，例如 `cifar10.yaml`：

```yaml
model:
  name: resnet18
  pretrained: false

training:
  epochs: 50
  batch_size: 128           # 设为 null 可自动探测最大 batch size
  optimizer: adam
  lr: 0.001
  scheduler: cosine

data:
  dataset_name: CIFAR10
  input_size: [3, 32, 32]
  num_workers: 2
  val_split: 0.1
  augment: true

seed: 42
```

完整配置模板见 `examples/cifar10_full.yaml`。

### 3. 启动训练

```bash
cvlab train --config cifar10.yaml
```

训练过程会：
- 自动探测最佳 Batch Size（若配置为 `null`）
- 记录 loss、acc 等指标到 SQLite
- 监控各层梯度范数
- 在每个 epoch 结束时保存 Checkpoint
- 遇到 OOM 时自动减小 Batch Size（每次 20%）重试

### 4. 查看实验

```bash
# 列出所有实验
cvlab list

# 按状态筛选
cvlab list --status completed

# 查看实验详情
cvlab show exp_001
```

### 5. 诊断分析

```bash
# Loss 异常分析
cvlab diagnose loss exp_001

# 梯度健康诊断
cvlab diagnose gradient exp_001

# 学习率与 Loss 联动分析
cvlab diagnose lr-loss exp_001

# DataLoader 性能诊断
cvlab diagnose dataloader config.yaml

# 全面诊断（组合以上所有）
cvlab diagnose experiment exp_001
```

### 6. 超参扫描

```bash
# 创建 Sweep 配置文件 sweep.yaml
# 然后运行
cvlab sweep create --config sweep.yaml

# 分析 Sweep 结果（超参重要性）
cvlab sweep analyze sweep_001 --metric val/acc
```

### 7. 多实验对比

```bash
# CLI 对比（终端输出 Rich 高亮表格）
cvlab compare exp_001 exp_002
cvlab compare exp_001 exp_002 --metric val/acc

# 或使用 Streamlit UI 交互式对比
streamlit run cvlab/ui/app.py
```

---

## 命令参考

### `cvlab train`

训练模型。

```
用法: cvlab train --config <config> [options]

必选:
  --config, -c <path>    配置文件路径 (YAML)

可选:
  --resume <exp_id>      从指定实验恢复训练
  --seed <int>           覆盖随机种子
  --batch-size <int>     覆盖 Batch Size
  --epochs <int>         覆盖训练轮数
  --lr <float>           覆盖学习率
  --name <str>           实验名称
```

训练以子进程方式运行，主进程负责监控和 OOM 恢复。

**OOM 恢复机制**：
1. 训练子进程检测到 CUDA OOM（通过 `torch.cuda.OutOfMemoryError` 或 `RuntimeError` 中 "CUDA out of memory" 关键词）
2. 退出码 137 通知主进程
3. 主进程自动将 Batch Size 减小 20%（乘以 0.8），最多重试 2 次
4. 若最小 Batch Size (1) 仍 OOM，标记实验为 failed

> **注意**：非 OOM 错误（exit code 1）不会触发重试，直接标记失败。

### `cvlab compare`

多实验指标对比（v0.2.1 新增）。

```
用法: cvlab compare <exp_id> <exp_id> [<exp_id> ...] [options]

必选:
  exp_id    实验 ID，至少 2 个

可选:
  --metric, -m <str>    指定对比的指标（默认显示所有共同指标）
```

输出 Rich 高亮对比表格，不同配置值以黄色标注，便于快速定位差异。

### `cvlab list`

列出实验。

```
用法: cvlab list [options]

可选:
  --status <str>    按状态筛选 (created/running/completed/failed/archived)
  --tag <str>       按标签筛选
  --limit <int>     最大条数 (默认 100)
  --json            输出 JSON 格式（可管道消费）
  --csv             输出 CSV 格式（可导入 Excel）
```

```bash
# JSON 输出
cvlab list --json | jq '.[].id'

# CSV 输出
cvlab list --csv > experiments.csv
```

### `cvlab show`

查看实验详情。

```
用法: cvlab show <experiment_id>
```

显示信息包括：名称、状态、创建时间、Seed、超参配置、指标汇总、Checkpoints、环境信息。

### `cvlab diagnose`

训练诊断工具集。

```
用法: cvlab diagnose <subcommand> [options]

子命令:
  loss <exp_id>              分析训练/验证 loss 曲线异常
  gradient <exp_id>          诊断各层梯度健康状态
  lr-loss <exp_id>           学习率与 Loss 联动分析
  dataloader <config.yaml>   诊断 DataLoader I/O 瓶颈
  experiment <exp_id>        全面诊断（组合以上所有）

dataloader 选项:
  --num-batches <int>    采样批次数 (默认 50)
```

**Loss 诊断**检测的异常类型：
- NaN/Inf loss
- Loss 突然跳升
- Loss 出现平台期
- Loss 发散（持续上升）

**梯度诊断**使用记录的梯度范数（需通过 Tracker.watch() 在训练时注入 Hook）：
- 梯度消失：平均范数 < 1e-5
- 梯度爆炸：平均范数 > 10.0
- 梯度偏低：平均范数 < 1e-3
- 梯度偏高：平均范数 > 1.0

### `cvlab data`

数据集管理工具。

```
用法: cvlab data <subcommand> [options]

子命令:
  analyze <path>          分析数据集统计信息
  augment <image>         预览数据增强效果（CLI 静态输出）
  check <path>            检查数据集血缘状态
  history                 列出数据集快照历史

augment 选项:
  --columns, -c <int>     网格列数 (默认 4)

check 选项:
  --hash-annotations      计算标注文件 SHA256 哈希
```

> **关于 augment**：CLI 版 `cvlab data augment` 提供静态网格预览（单图多种增强对比）。交互式增强预览（参数滑块实时调整）请使用 Streamlit UI。

**data analyze** 输出：
- 样本总数、类别数
- 图片格式分布
- 类别分布（前 20 类）
- 类别平衡度分数
- 图片尺寸范围（最小/最大/平均）
- 标注文件状态

**data check** 使用两级血缘追踪：
- Level 1 (O(1))：根目录文件列表 + 总大小哈希，每次自动调用
- Level 2 (O(n))：标注文件完整 SHA256，按需启用

### `cvlab profile`

模型画像工具，分析模型参数量、FLOPs、推理速度。

```
用法: cvlab profile --model <name> [options]

必选:
  --model <name>    模型名称 (如 resnet18, resnet50)

可选:
  --device <str>      运行设备 (cpu/cuda, 默认自动选择)
  --input-size        输入尺寸 (默认 3x224x224)
  --checkpoint        加载 Checkpoint 路径
  --num-warmup        预热步数 (默认 3)
  --num-iter          测试迭代数 (默认 10)
```

输出：
- 模型参数量（总数 / 可训练）
- 理论 FLOPs
- 推理延迟（前向 / 反向）
- 峰值显存占用

> **注意**：`cvlab profile` 目前支持 `torchvision.models` 中的分类模型。自定义模型（YOLO、ViT、DETR 等）请使用 Python API。

### `cvlab weights`

预训练权重管理。

```
用法: cvlab weights <subcommand> [options]

子命令:
  info <model_name>     显示模型权重信息
  download <model_name> 下载预训练权重到本地缓存
  list                  列出已缓存的权重文件
```

**weights info** 会对指定模型（如 `resnet50`）显示：
- 模型参数量
- 预期权重大小
- 本地缓存状态

**weights download** 从 torchvision hub 下载预训练权重并缓存到本地。

### `cvlab sweep`

超参扫描。

```
用法: cvlab sweep <subcommand> [options]

子命令:
  create                  创建并启动 Sweep
    --config, -c <path>     Sweep 配置文件路径 (必选)
    --name <str>            Sweep 名称
    --seed <int>            随机种子
    --max-trials <int>      最大 trial 数（random 模式）

  analyze <sweep_id>      分析超参重要性
    --metric <str>          目标指标名 (默认 val/acc)
```

**Sweep 配置示例** (`sweep.yaml`)：

```yaml
_sweep:
  strategy: grid           # grid | random
  params:
    training.lr:
      values: [0.1, 0.01, 0.001]
    training.optimizer:
      values: [sgd, adam]
    data.augment:
      values: [true, false]

# 基础配置（所有 trial 共享）
model:
  name: resnet18
  pretrained: false
training:
  epochs: 30
  batch_size: 64
data:
  dataset_name: CIFAR10
  input_size: [3, 32, 32]
```

**analyze 子命令**使用随机森林回归评估各超参对目标指标的影响程度，输出：
- 各超参的重要性分数（0~1，归一化）
- 关键超参建议
- 联合调优建议

**top 子命令**（v0.2.5 新增）显示 Sweep 中指标最优的 N 个 trial：
```
cvlab sweep top <sweep_id> [--metric val/acc] [-n 5]
```
输出 Rich 排名表格，第一名绿色高亮。用于快速找到最佳参数组合。

### `cvlab estimate`

预计算训练耗时（v0.2.5 新增）。

```
用法: cvlab estimate --config <yaml> [options]

必选:
  --config, -c <path>    配置文件路径 (YAML)

可选:
  --batches <int>        采样 batch 数 (默认 5，含预热)
  --batch-size <int>     覆盖 Batch Size
  --epochs <int>         覆盖训练轮数
  --cpu                  强制使用 CPU
```

在正式训练前跑少量 batch，估算每 epoch 时间和总耗时，
帮助判断当前配置是否值得跑完。输出：
- 平均每 batch 时间（数据加载 vs 计算 分解）
- 吞吐量（samples/sec）
- 每 epoch 估算时间
- 总耗时（按配置的 epochs）
- 峰值显存
- DataLoader 瓶颈检测（数据加载占比 > 50% 时告警）

```bash
cvlab estimate --config examples/tiny_cnn.yaml
cvlab estimate --config config.yaml --batches 3 --epochs 100
```

### `cvlab init`

在当前目录初始化 CVLab 存储。

```
用法: cvlab init
```

创建 `.cvlab/` 目录（含默认 `config.yaml` 模板）用于存放 SQLite 数据库、Checkpoints、Artifacts。

### `cvlab ui`

启动 Streamlit Web 界面（v0.2.1 新增）。

```
用法: cvlab ui [options]

可选:
  --port, -p <int>    端口号（默认 8501）
  --host <str>        监听地址（默认 127.0.0.1，使用 0.0.0.0 允许外部访问）
  --lang <str>        界面语言（zh/en，默认自动检测）
```

自动解析 `app.py` 路径，用户无需手动寻找。

```bash
# 默认启动
cvlab ui

# 指定端口和语言
cvlab ui --port 8502 --lang en

# 局域网访问
cvlab ui --host 0.0.0.0
```

> 需要先安装 streamlit：`pip install streamlit`

### `cvlab help`

显示命令帮助概览。

```
用法: cvlab help [command]
```

---

## Python API — CV 专项功能

除了基本的指标记录和梯度监控，CVLab 的 Tracker API 提供了 CV 领域的专项可视化能力：

### 混淆矩阵

```python
tracker.log_confusion_matrix(
    y_true=[0, 1, 2, 1, 0],
    y_pred=[0, 1, 1, 1, 0],
    class_names=["cat", "dog", "bird"],
    step=epoch,
    normalize=True,   # 按行归一化为百分比
)
```

自动保存到 Artifacts，在 Streamlit UI 和 HTML 报告中可查看。

### 检测框可视化

```python
import numpy as np

tracker.log_detection(
    key="val_detections",
    image=image_np,                          # (H, W, 3)
    boxes=np.array([[10, 20, 100, 200], ...]),  # (N, 4), [x1, y1, x2, y2]
    scores=np.array([0.95, 0.87, ...]),      # (N,)
    labels=np.array([0, 1, ...]),            # (N,)
    class_names=["person", "car", ...],
    step=epoch,
    score_threshold=0.5,                     # 低于此阈值的框不绘制
)
```

### 分割掩码叠加

```python
tracker.log_segmentation(
    key="val_seg",
    image=image_np,        # (H, W, 3)
    pred_mask=pred_mask,   # (H, W), 值 > 0 为前景
    step=epoch,
    gt_mask=gt_mask,       # (H, W), 可选，以红色叠加显示
    alpha=0.5,             # mask 透明度
)
```

### 完整示例

更多 CV 专项示例见 `examples/quickstart.py`。

---

## 关于模型支持

`cvlab train` 和 `cvlab profile` 的 CLI 命令目前支持 `torchvision.models` 中的分类模型（resnet、densenet、mobilenet、efficientnet 等）。

**如果你使用自定义模型**（YOLO、ViT、DETR、UNet 等），请使用 Python API：

```python
from cvlab import Tracker

tracker = Tracker(config={...})
# Tracker 接受任意 nn.Module，无需继承 CVLab 基类
monitor = tracker.watch(model, log_gradients=True)
# ... 正常训练循环
```

---

## 训练最佳实践

### 选择 Batch Size

建议将配置中的 `batch_size` 设为 `null`，CVLab 会自动执行 BatchSizeProbe 探测最大可用值（含 20% 安全余量）。探测过程：
1. 从 1 开始指数增长
2. 结合 optimizer.step() 模拟真实训练
3. 回退到最后一个不 OOM 的值
4. 乘以 0.8 安全系数

### 监控梯度

训练时 Tracker 自动 Hook 模型的各层梯度。训练后可通过诊断命令查看：

```bash
cvlab diagnose gradient exp_001
```

若发现梯度消失，建议：
- 检查激活函数（避免 sigmoid）
- 添加 BatchNorm
- 使用残差连接

若发现梯度爆炸，建议：
- 开启梯度裁剪（`gradient_clip: 1.0`）
- 降低学习率

### 恢复中断训练

```bash
# 从最佳 Checkpoint 恢复
cvlab train --config config.yaml --resume exp_001
```

自动恢复：
- 模型权重（最佳 epoch）
- 优化器状态
- 学习率调度器状态
- 历史指标

---

## Python API 速览

```python
from cvlab import Tracker, seed_everything
from cvlab.config.config import load_config

# 设置随机种子
seed_everything(42)

# 创建 Tracker（自动创建实验记录）
tracker = Tracker(config={"model": {"name": "resnet18"}, ...})
exp_id = tracker.experiment_id

# 注入梯度监控 Hook
monitor = tracker.watch(model, log_gradients=True, log_freq=50)

# 记录指标
tracker.log({"train/loss": 0.523, "train/acc": 0.832}, step=epoch)

# 记录图片
tracker.log_image("pred_samples", image_tensor, step=epoch)

# 记录混淆矩阵
tracker.log_confusion_matrix(y_true, y_pred, class_names, step=epoch)

# 记录检测框
tracker.log_detection("detect", img, boxes, scores, labels, class_names, step=epoch)

# 记录分割掩码
tracker.log_segmentation("seg", img, pred_mask, step=epoch, gt_mask=gt_mask)

# 保存 Checkpoint
tracker.save_checkpoint(model, optimizer, epoch=epoch,
                        metrics={"val/acc": 0.85}, is_best=True)

# 标记完成
tracker.finish("completed")
```

更多示例见 `examples/quickstart.py` 和 `examples/full_pipeline.py`。

---

## 数据存储

CVLab 所有数据存储在 `.cvlab/` 项目目录：

```
.cvlab/
├── cvlab.db              # SQLite 数据库（实验元数据、指标）
├── experiments/          # 实验数据
│   ├── exp_001/
│   │   ├── checkpoints/  # 模型权重
│   │   │   ├── epoch_30.pt
│   │   │   ├── epoch_50.pt
│   │   │   ├── best.pt
│   │   │   └── best_ema.pt
│   │   ├── artifacts/    # 图片、混淆矩阵等
│   │   └── script_snapshot.py  # 训练脚本快照
│   └── exp_042/
└── provenance/           # 数据集血缘快照
```

迁移项目时，打包整个 `.cvlab/` 目录即可。

---

## 常见问题

### Q: 提示 ModuleNotFoundError

确保已安装所有依赖：
```bash
pip install -e .
```

### Q: CUDA Out of Memory

CVLab 会自动重试（每次减小 20% Batch Size）。若仍失败，尝试：
- 减小模型规模
- 减小 `input_size`
- 启用 `amp: true`（混合精度）
- 手动设置更小的 `batch_size`

### Q: 训练时 DataLoader 成为瓶颈

使用 `cvlab diagnose dataloader config.yaml` 诊断。若加载时间 > 50ms/batch：
- 增加 `num_workers`
- 启用 `pin_memory: true`
- 确保数据在 SSD 上
- 检查是否有大量的实时增强

### Q: 如何复现实验？

实验记录中包含完整的配置和随机种子。使用 `cvlab show exp_001` 查看，或通过 `tracker.get_reproduce_command()` 获取一键复现命令。

### Q: 如何对比多个实验？

```bash
# 方式一：CLI 命令行对比
cvlab compare exp_001 exp_002 [--metric val/acc]

# 方式二：Streamlit UI 交互式对比
cvlab ui  # → Compare 页面
```

CLI 对比输出 Rich 高亮表格，适合终端内快速查看差异。UI 对比提供交互式曲线叠加，适合深度分析。

### Q: 如何启动 Web UI？

```bash
# 只需一条命令，自动解析 app.py 路径
cvlab ui

# 指定端口
cvlab ui --port 8502 --lang en
```

> 需要先安装 streamlit：`pip install streamlit`
