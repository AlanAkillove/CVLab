# CVLab 开发路线图

> 从 v0.2.7 到 v0.3.0：从"能用的工具"到"离不开的工作台"。
>
> 在 PyPI 发布之前，v0.2.x 系列保持预览版状态。每个版本完成的功能将在 CHANGELOG 中记录。

---

## 当前状态 (v0.2.6)

已实现的核心模块：

| 模块 | 状态 |
|------|------|
| 实验追踪（超参+指标） | ✅ 完成 |
| Checkpoint 管理 + EMA | ✅ 完成 |
| 梯度健康监控 + 告警 | ✅ 完成 |
| Loss 异常检测 | ✅ 完成 |
| 超参扫描（Grid+Random） | ✅ 完成 |
| 超参重要性分析 | ✅ 完成 |
| OOM 容错恢复 | ✅ 完成 |
| Batch Size 自动探测 | ✅ 完成 |
| 模型 Profiling | ✅ 完成 |
| 预训练权重管理 | ✅ 完成 |
| 数据集版本血缘 | ✅ 完成 |
| 多实验对比（CLI + UI） | ✅ 完成 |
| HTML 报告生成 | ✅ 完成 |
| Webhook 通知 | ✅ 完成 |
| ONNX/TorchScript 导出 | ✅ 完成 |
| 数据集分析（CIFAR/VOC/YOLO） | ✅ 完成 |
| 数据增强预览（交互式 UI） | ✅ 完成 |
| 预测时间轴（UI） | ✅ 完成 |
| LR-Loss 联动叠加图（UI） | ✅ 完成 |
| 数据集质量分析与建议 | ✅ 完成 |
| 预训练耗时估算 | ✅ 完成 |
| 可管道输出的实验列表 (JSON/CSV) | ✅ 完成 |

---

## v0.2.7 — 训练诊断深化

过拟合早期预警比单纯的 val loss 检测更智能，在泛化差距扩大、loss 方差增大、置信度分化时提前介入。

### 过拟合风险评估

```bash
cvlab diagnose overfitting <exp_id>
```

输出：
- 泛化差距（Train/Val Acc 差距）及扩大速率
- Val Loss 标准差变化趋势
- 预测置信度分化程度（Train vs Val）
- 综合风险等级及建议措施（增大 Dropout / 增强增强策略 / 加入 Mixup）

### LR Finder

```bash
cvlab diagnose lr-finder --config config.yaml
```

Leslie Smith 方法，自动搜索最优起始学习率：
- 从 1e-7 到 1e+1 指数增长
- Loss 下降最陡处的 1/10 为推荐起始 LR
- Loss 开始发散前为推荐最大 LR
- 保存 LR-Loss 曲线图，支持一键写入 config.yaml

### Dead Neurons 检测

```bash
cvlab diagnose dead-neurons <exp_id>
```

ReLU 神经元死亡统计，按层输出死亡率：
- 正常 (< 5%) / 偏高 (5-20%) / 严重 (> 20%)
- 严重层建议：检查初始化方式 / 换用 LeakyReLU / 降低 weight_decay

**工作量估算**：3-4 天（核心是诊断逻辑，现有 diagnose 框架可扩展）

---

## v0.2.8 — 数据工程工具

### 数据集格式转换

```bash
cvlab data convert --input ./dataset --from coco --to yolo --output ./dataset_yolo
cvlab data convert --from voc --to coco
```

支持格式：COCO JSON / VOC XML / YOLO txt / 平铺目录
转换内容：
- 标注坐标格式转换（归一化 ↔ 绝对坐标）
- category_id 映射表自动生成
- BBox 边界裁剪（超出图片时自动修正）
- 自动生成 data.yaml 或同名 JSON

### 数据集切分与采样

```bash
cvlab data split ./dataset --train 0.8 --val 0.1 --test 0.1 --stratify
cvlab data sample ./dataset --n 1000 --stratify --output ./dataset_debug
```

- 分层抽样（按类别比例保持 train/val/test 平衡）
- 小子集采样（调试用，1000 张快速验证代码正确性）

### 标注一致性检查

```bash
cvlab data check-annotations ./dataset
```

- BBox 标注风格分析（是否有不同标注策略）
- 坐标精度分布分析（整数 vs 浮点，推断标注工具）
- 空标注 / 异常 BBox / 超界 BBox 检测
- 多标注员策略差异报告

**工作量估算**：4-5 天（格式转换逻辑 + 切分 + 标注质检）

---

## v0.2.9 — 可解释性工具

### Grad-CAM 时间轴

训练中自动对固定验证样本生成 Grad-CAM 热力图：

```bash
cvlab analyze gradcam <exp_id> --samples 8
```

- UI 中展示 Epoch 维度的时间轴滑块
- 每一行的样本固定，列是不同 Epoch 的热力图
- 错误样本的 Grad-CAM 自动标注（true label + pred label）
- 观察模型关注区域从"整体形状"到"局部判别特征"的演变

### 特征空间可视化

```bash
cvlab analyze feature-space --checkpoint best.pt --dataset ./val
```

- t-SNE / UMAP 降维到 2D
- UI 中交互式散点图（按类别着色）
- 量化指标：类内距离 / 类间距离 / 分离度分数
- 可导出为 HTML 交互图
- 支持按 Epoch 对比特征空间演变

### 难样本挖掘

```bash
cvlab analyze hard-examples <exp_id> --n 20
```

- 找出持续预测错误的样本
- 分析错误模式（高混淆类别 / 系统性错误 / 边界样本）
- 连同 Grad-CAM 热力图一起展示
- 建议：加入训练集 / 专项增强

**工作量估算**：5-6 天（Grad-CAM 实现最复杂，需要钩子注册）

---

## v0.2.10 — 实验设计辅助

### 消融实验管理

```yaml
# ablation.yaml
base_config: configs/best.yaml
ablations:
  - name: "w/o Augmentation"
    override:
      data.augment: false
  - name: "SGD vs Adam"
    override:
      training.optimizer: sgd
```

```bash
cvlab ablation run --config ablation.yaml
cvlab ablation report --id ablation_001
```

自动生成对比表格：组件 / val_acc / Δ / 结论
支持一键导出 LaTeX 表格（论文友好）

### 实验模板库

```bash
cvlab template list
cvlab template use classification/resnet_cifar10 --output my_config.yaml
```

常见 CV 任务的最佳实践配置，附带"为什么这样配"的注释。

内置模板（初始）：
- `classification/resnet_cifar10` / `vit_imagenet`
- `detection/yolov8_coco` / `faster_rcnn`
- `segmentation/deeplabv3`

### 实验目标追踪

```bash
cvlab goal set --metric val_acc --target 90.0 --deadline "2026-05-25"
cvlab goal status
```

- 设定目标和截止日期
- 自动追踪进度（当前最佳 / 差距 / 剩余天数）
- 基于历史实验预估还需多少次实验
- 提示最可能突破的方向

**工作量估算**：4-5 天（消融实验框架 + 模板仓库 + 目标追踪状态机）

---

## v0.2.11 — 部署与落地工具

### 模型压缩向导

```bash
cvlab compress --checkpoint best.pt --target mobile
```

交互式向导，根据目标场景推荐压缩方案：
- 方案A：INT8 量化（无损推荐）
- 方案B：剪枝 + 量化（需 fine-tune）
- 方案C：换用轻量模型族

每项给出预期：参数量 / FPS / 模型大小 / 精度影响

### 推理速度基准

```bash
cvlab benchmark --checkpoint best.pt --input 1x3x224x224
```

对比 FP32 / FP16 / INT8 / torch.compile / TensorRT 的延迟、FPS、显存、精度损失。

### 训练预算估算（增强版）

现有 `cvlab estimate` 的基础上增强：
- 预测完成时间（含时区转换）
- Windows 电源计划提醒
- "先跑 5 epoch 确认趋势"建议
- 训练结束后自动执行钩子 (`on_finish`)

**工作量估算**：4-5 天（压缩向导交互最多，推理基准次之）

---

## v0.3.0 — 知识积累与协作（发布版）

### 跨实验知识库

```bash
cvlab insights --dataset cifar10 --model resnet
```

基于所有历史实验，自动沉淀：
- 某模型+数据集组合的最佳 lr 范围 / batch size 倾向 / 增强效果统计
- 高失败率的配置模式
- "从未尝试过但值得探索"的建议

### 实验日志与时间线

```bash
cvlab journal
```

按照时间线展示实验，记录"为什么"：
- 每条实验自动关联前一天
- 手动补充笔记（失败原因 / 灵感 / next step）
- 目标驱动的实验序列标记

### 实验分享与合并

```bash
cvlab share exp_003 --output exp_003_share.zip
cvlab merge ./teammate_cvlab.db --output merged.db
```

- 生成自包含的分享包（不含权重，含可查看的 HTML 报告）
- 合并多人实验数据库（跨机器跨团队的实验对比）

### PyPI 发布

- `pip install cvlab` 即可安装
- 提供 Docker 镜像（集成环境一键部署）
- README 加入快速部署教程

**工作量估算**：5-6 天（跨实验数据分析最复杂，日志系统次之）

---

## 总工期估算

| 版本 | 主题 | 工期 |
|------|------|------|
| v0.2.7 | 训练诊断深化 | 3-4 天 |
| v0.2.8 | 数据工程工具 | 4-5 天 |
| v0.2.9 | 可解释性工具 | 5-6 天 |
| v0.2.10 | 实验设计辅助 | 4-5 天 |
| v0.2.11 | 部署与落地 | 4-5 天 |
| v0.3.0 | 知识积累 + 发布 | 5-6 天 |
| **总计** | | **25-31 天** |

## 优先级说明

如果时间有限，以下是最值得先做的 3 个（投入产出比最高）：

1. **LR Finder** (v0.2.7) — 一劳永逸解决"该用多大 lr"的日常纠结
2. **数据集格式转换** (v0.2.8) — 每天在不同框架间搬数据的刚需
3. **消融实验管理** (v0.2.10) — 论文写作的核心工具

这三个做完之后，日常研究的核心痛点基本解决，可以优先发 PyPI。
