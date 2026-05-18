# CVLab 使用指南

CVLab 是一个 CV（计算机视觉）实验管理平台，帮助研究者专注于模型和数据本身，而非工程琐事。

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
- 遇到 OOM 时自动减半 Batch Size 重试

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
1. 训练子进程因 CUDA OOM 崩溃时（exit code 137）
2. 主进程自动将 Batch Size 减半
3. 最多重试 2 次
4. 若最小 Batch Size (1) 仍 OOM，标记实验为 failed

### `cvlab list`

列出实验。

```
用法: cvlab list [options]

可选:
  --status <str>    按状态筛选 (created/running/completed/failed/archived)
  --tag <str>       按标签筛选
  --limit <int>     最大条数 (默认 20)
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
  augment <image>         预览数据增强效果
  check <path>            检查数据集血缘状态
  history                 列出数据集快照历史

augment 选项:
  --columns, -c <int>     网格列数 (默认 4)

check 选项:
  --hash-annotations      计算标注文件 SHA256 哈希
```

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

### `cvlab init`

在当前目录初始化 CVLab 存储。

```
用法: cvlab init
```

创建 `.cvlab/` 目录用于存放 SQLite 数据库、Checkpoints、Artifacts。

### `cvlab help`

显示命令帮助概览。

```
用法: cvlab help [command]
```

---

## 配置文件参考

### 完整配置选项

```yaml
# 模型
model:
  name: resnet18                # 模型名称 (支持 torchvision.models 中的分类模型)
  pretrained: false             # 是否加载 ImageNet 预训练权重
  pretrained_weights: null      # 自定义权重路径（优先级高于 pretrained）
  output_classes: 10            # 分类数（默认根据数据集自动确定）

# 训练
training:
  epochs: 50                    # 训练轮数
  batch_size: 128               # Batch Size（设为 null 自动探测）
  optimizer: adam               # sgd | adam | adamw
  lr: 0.001                     # 学习率
  weight_decay: 0.0001          # 权重衰减
  momentum: 0.9                 # SGD 动量（仅 optimizer=sgd 时）
  scheduler: cosine             # cosine | step | none
  lr_step_size: 30              # StepLR 步长（仅 scheduler=step 时）
  lr_gamma: 0.1                 # StepLR 衰减率
  warmup_epochs: 0              # 学习率预热轮数
  label_smoothing: 0.0          # 标签平滑
  gradient_clip: null           # 梯度裁剪最大范数
  early_stop_patience: null     # 早停耐心值
  amp: false                    # 是否启用自动混合精度

# 数据
data:
  dataset_name: CIFAR10         # 数据集名称
  dataset: ./data               # 数据集根路径（自动下载到该路径）
  input_size: [3, 32, 32]       # 输入尺寸 [C, H, W]
  num_workers: 2                # DataLoader 工作进程数
  val_split: 0.1                # 验证集比例（自动划分）
  augment: true                 # 是否使用数据增强
  pin_memory: true              # DataLoader pin_memory
  persistent_workers: false     # DataLoader persistent_workers

# 日志
logging:
  log_interval: 10              # 指标记录间隔（步数）
  grad_log_freq: 50             # 梯度范数记录间隔
  save_checkpoint: true         # 是否保存 Checkpoint
  checkpoint_top_k: 3           # 最多保留的最佳 Checkpoint 数
  log_images: false             # 是否记录预测样本图片

# 实验
seed: 42                        # 随机种子
name: null                      # 实验名称（自动生成）
tags: []                        # 标签列表
notes: ""                       # 备注
```

### 配置模板

`examples/` 目录包含多个配置模板：

| 文件 | 说明 |
|------|------|
| `cifar10.yaml` | 最小 CIFAR10 配置 |
| `cifar10_full.yaml` | 完整 CIFAR10 配置（带注释） |
| `tiny_cnn.yaml` | 极简 CNN 配置（快速调试用） |
| `imagenet_style.yaml` | ImageNet 风格大模型配置 |

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

## Python API

除了 CLI，CVLab 提供 Python API 供脚本调用：

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

CVLab 会自动重试（减半 Batch Size）。若仍失败，尝试：
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

建议使用 Streamlit UI 进行对比：
```bash
streamlit run cvlab/ui/app.py
```

该 UI 提供实验列表、详情查看、多实验对比等功能。
