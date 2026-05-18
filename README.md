# CVLab

> **Preview v0.1.0** — 预览版本，API 可能发生变化。

CV 实验管理平台 — 让研究者专注于模型和数据本身，而非工程琐事。

```bash
pip install -e .
```

## 一句话

CVLab 是一个轻量级、CV 专精的训练实验管理工具。它不要求你改模型继承、不绑定框架、不部署服务。装一个 `pip install`，加三行代码，就能获得实验追踪、梯度监控、数据分析和超参扫描能力。

## 快速开始

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
        monitor.step(global_step)  # 每 N 步采样梯度范数
        optimizer.step()

    # 4. 记录指标
    tracker.log({"train/loss": loss.item(), "train/acc": acc}, step=epoch)
    tracker.save_checkpoint(model, optimizer, epoch=epoch)

# 5. 完成实验
tracker.finish()
print(tracker.get_reproduce_command())
```

完整示例见 [examples/quickstart.py](examples/quickstart.py)。

## 功能

### 实验追踪
- 自动创建实验 ID、固定随机种子、保存环境快照和脚本副本
- 标量指标记录与 SQLite 持久化
- 图片、混淆矩阵、检测框、分割掩码可视化
- Checkpoint 管理（自动轮转、EMA 权重、best/last 标记）
- 一键复现命令生成

### 非侵入式 Hook 注入
- `register_full_backward_hook` 采样模式梯度监控
- 梯度消失/爆炸告警（阈值: vanish < 1e-5, explosion > 10）
- 支持指定层监控，支持激活值采样

### 环境检测与加速配置
- 自动检测 OS/Python/PyTorch/CUDA 版本
- GPU 型号、显存、Compute Capability、TensorCore 支持
- WSL2 检测、CUDA 版本不匹配告警
- 6 项训练加速选项（AMP FP16、BF16、cuDNN Benchmark、torch.compile、Channels Last、Gradient Checkpointing）

### Batch Size 自动探测
- 二分搜索算法，安全余量 20%
- 悲观数据注入（最大分辨率 + 密集标签）
- AMP/BF16 对齐探测
- 多 GPU 感知

### 训练诊断
- **I/O 瓶颈检测**：分析 DataLoader 加载时间 vs 计算时间比例，推荐最优 num_workers
- **Loss 异常检测**：NaN/Inf、爆炸、平台期、突跳、LR 异常
- **模型性能画像**：FLOPs (MACs)、参数量、前向/反向延迟、峰值显存
- **权重加载诊断**：missing/unexpected keys、形状检查、双权重文件差异对比

### 超参扫描
- Grid 搜索（笛卡尔积枚举）
- Random 搜索（choice/uniform/loguniform/int 四种分布）
- Trial 管理、最佳 Trial 查找

### 数据集分析
- 类别分布、图片格式统计、尺寸分布
- 类别平衡度评分
- 数据血缘追踪（两级：O(1) 根目录快照 + 标注文件 SHA256）
- 变化检测、快照列表

### 可视化 UI
- Streamlit 多页面（实验列表、详情、对比、Sweep、环境诊断）
- Plotly 交互式指标曲线叠加
- 超参配置自动对比
- Sweep Trial 指标柱状对比

### 报告生成
- 自包含 HTML 报告（Jinja2 模板）
- 超参、环境、指标、Checkpoints、复现命令

### 诊断面板（Streamlit UI）
- 环境探针结果、加速选项一键评估
- DataLoader 优化建议

## API 速览

```python
from cvlab import Tracker

# 创建/恢复实验
tracker = Tracker(config={...})                    # 创建新实验
tracker = Tracker(experiment_id="exp_001")          # 恢复已有实验

# 实验追踪
tracker.log({"train/loss": 0.5}, step=1)            # 记录指标
tracker.log_image("samples", image, step=1)         # 记录图片
tracker.log_confusion_matrix(y_true, y_pred, ...)   # 记录混淆矩阵

# 可视化
tracker.log_detection("detection", img, boxes, ...) # 检测框标注
tracker.log_segmentation("seg", img, pred_mask, ...)# 分割掩码叠加

# Checkpoint
tracker.save_checkpoint(model, optim, epoch=5, ...) # 自动处理 EMA
tracker.load_checkpoint(best=True)                  # 加载最优权重

# Hook 监控
monitor = tracker.watch(model, log_gradients=True)
monitor.step(global_step)                           # 采样梯度分析

# 数据集快照
tracker.snapshot_dataset("./data")                  # O(1) 数据血缘

# 完成
tracker.finish("completed")
print(tracker.get_reproduce_command())
```

```bash
# CLI 命令概览
cvlab help                                # 显示命令帮助概览
cvlab train --config config.yaml          # 启动训练
cvlab list                                # 列出实验
cvlab show exp_001                        # 显示实验详情
cvlab diagnose loss exp_001               # 诊断 loss 异常
cvlab diagnose gradient exp_001           # 诊断梯度健康
cvlab sweep create --config sweep.yaml    # 超参扫描
cvlab sweep analyze sweep_001             # 超参重要性分析
cvlab profile --model resnet18            # 模型性能画像
cvlab weights info resnet18               # 权重信息
cvlab weights download resnet50           # 下载预训练权重
cvlab data analyze ./dataset              # 数据集分析
cvlab data check ./dataset                # 数据血缘检查

# Streamlit UI（单独启动）
streamlit run cvlab/ui/app.py             # 启动 Web 界面
```

详细示例见 [examples/](examples/) 目录。

## 项目结构

```
cvlab/
├── core/           # 核心：Tracker、Watch Hook、Seed 管理
├── db/             # SQLite 持久化（零依赖，WAL 模式）
├── detect/         # 环境检测：OS/CPU/GPU/存储
├── probe/          # Batch Size 自动探测
├── diagnose/       # 训练诊断：I/O、Loss、预测可视化
├── profile/        # 模型性能画像
├── weights/        # 预训练权重管理
├── sweep/          # 超参扫描（Grid + Random）
├── data/           # 数据分析、血缘追踪、增强预览
├── report/         # HTML 报告生成
├── config/         # YAML 配置管理
├── cli/            # argparse CLI 入口
├── ui/             # Streamlit Web 界面
└── tests/          # 152+ 测试，覆盖核心模块
```

## 设计原则

- **不绑定框架**：原生 `nn.Module`，不需要继承任何 CVLab 基类
- **不干扰训练**：Hook 采样模式，不修改训练循环结构
- **零服务依赖**：SQLite 本地存储，`pip install` + `import cvlab` 直接使用
- **CV 专精**：检测框/分割掩码/混淆矩阵内置可视化

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- 操作系统：Windows / Linux / macOS

## 许可证

MIT
