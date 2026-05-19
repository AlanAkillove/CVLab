# CVLab 用户体验审计报告

**审计日期**: 2026-05-19  
**审计范围**: F:/Projects/CVLab (v0.2.3)  
**审计目标**: 从用户使用体验角度评估 CLI、配置、错误处理、安装入门、信息设计、CI/DevOps 六个维度

---

## 严重程度定义

| 级别 | 说明 |
|------|------|
| 🔴 **P0 - Critical** | 用户无法完成任务，或会因迷惑而放弃使用 |
| 🟠 **P1 - High** | 显著降低效率，频繁引发困惑或误操作 |
| 🟡 **P2 - Medium** | 体验不佳但不阻塞流程，积累后影响满意度 |
| 🔵 **P3 - Low** | 信息设计/观感优化，有则更好 |

---

## 1. CLI 使用体验

### 1.1 `cvlab init` — 完成后无下一步引导

**问题**: `cvlab init` 只创建了一个 `.cvlab/` 目录，输出仅一行：
```
  [OK]  Initialization complete: F:/Projects/xxx/.cvlab
```
用户得到这个信息后完全不知道下一步做什么。没有提示创建配置文件、没有指向示例、没有引导运行第一条命令。

**影响**: 新用户在此停滞，需要折返回 README 寻找下一步。

**严重程度**: 🟠 **P1 - High**

**建议**: 在 init 后追加多行输出，形成"欢迎面板"：
```
✅ CVLab 已初始化！

接下来你可以：
  1. 创建训练配置:  cp examples/cifar10.yaml config.yaml
  2. 启动训练:       cvlab train --config config.yaml
  3. 查看实验:       cvlab list

快速体验: cvlab help
文档:     https://github.com/AlanAkillove/CVLab#readme
```

---

### 1.2 默认 `cvlab` 无参数输出帮助信息 — 缺乏欢迎感

**问题**: 运行 `cvlab`（不带任何参数）直接打印 argparse 默认帮助文本，风格机械：
```
usage: cvlab [-h] [--version] [--lang {zh,en}] {train,compare,sweep,...} ...
CVLab - CV Experiment Management Platform

positional arguments:
  ...
```
没有欢迎语，没有快速上手提示，没有版本号。新用户的"第一印象"是冰冷的命令行用法表。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 在无参数时打印一个简短欢迎信息 + 关键命令概览，再引导用户使用 `cvlab help`。例如：
```
CVLab v0.2.3 — CV Experiment Management Platform

快速开始:
  cvlab init             初始化
  cvlab train --config  开始训练
  cvlab list             查看实验
  cvlab ui               启动 Web 界面

查看全部命令: cvlab help
```

---

### 1.3 `cvlab compare` 错误信息不友好

**问题**: 当只传一个实验 ID 时报错：
```
[FAIL] 至少选择 2 个实验
```
这个信息没有指出正确的用法格式，用户可能不知道要写 `cvlab compare exp_001 exp_002`。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 修复建议：
```
[FAIL] 至少选择 2 个实验
用法: cvlab compare <exp_id> <exp_id> [exp_id ...]
示例: cvlab compare exp_001 exp_002
```

---

### 1.4 `cvlab profile --input` vs `--input-size` 别名混淆

**问题**: `profile` 子命令同时定义了 `--input` 和 `--input-size`（dest 相同 `input_size`），但 help 文本中同时显示这两个参数，用户会困惑该用哪个、有什么不同。实际上一个是另一个的别名，但不仔细看代码无法知道。

**严重程度**: 🔵 **P3 - Low**

**建议**: 移除 `--input-size` 别名，统一使用 `--input`。或者在 help 中标注 `--input-size` 是 `--input` 的别名。

---

### 1.5 `cvlab data check` — "血缘"概念对用户不直观

**问题**: "数据血缘检查"（Provenance Check）是一个数据库领域的专业术语，对 CV 研究者来说不够直观。Help 文本说"检查数据血缘状态"，用户可能不理解这是做什么的。

**严重程度**: 🔵 **P3 - Low**

**建议**: 在 help 和命令输出中添加类比说明或改述，例如：
```
cvlab data check <path>    检查数据集是否变化（当你修改了数据集后，此命令会提示）
```

---

### 1.6 `cvlab weights info/download` — 不列出支持的模型

**问题**: `weights info <name>` 和 `weights download <name>` 的 help 只写"模型名称"，用户不知道哪些模型可用，需要试错。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 
1. 在 help 文本中注明"支持 torchvision 分类模型，如 resnet18, resnet50, mobilenet_v3 等"
2. 提供一个 `cvlab weights list-remote` 命令，列出所有可下载的模型
3. 或者当模型名无效时，建议用户使用 `cvlab profile --help` 查看更多模型示例

---

### 1.7 训练子进程启动期间无进度反馈

**问题**: 当 `cvlab train` 启动子进程时，输出一行：
```
[INFO] 启动训练子进程 (attempt 1/3)
```
然后可能 5-30 秒没有任何输出（取决于 PyTorch 初始化、数据集下载等）。用户不知道程序是否卡住了。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 在子进程启动后添加一个简单的 Spinner 或定期输出 "加载中..." 提示。或者将子进程的 stdout/stderr 实时回传到主进程终端。

---

### 1.8 `cvlab diagnose dataloader` 导入私有函数

**问题**: `_diag_dataloader` 中 `from cvlab.train.run import _load_data` 导入了一个私有函数（下划线前缀）。这违反了封装原则，如果 `_load_data` 签名或行为发生变化，诊断命令会悄然失效。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 
1. 将 `_load_data` 改为公共函数 `load_data`（或提取到 `cvlab.data` 模块）
2. 或者在诊断命令中提供更友好的出错信息，明确告知失败原因

---

### 1.9 `cvlab help` 信息密度过高

**问题**: `cvlab help` 打印一个包含所有命令、描述、用法的表格，信息量很大。14 个命令的用法挤在一个表格里，每行的 usage 可能跨多行，阅读负担重。

**严重程度**: 🔵 **P3 - Low**

**建议**: 
1. 将命令按功能分组显示（训练类、诊断类、数据类、管理类）
2. 先显示分组概览，再提示用户 `cvlab <command> --help` 查看详情

---

## 2. 配置体验

### 2.1 `cvlab init` 不生成配置模板

**问题**: 初始化后目录是空的，用户需要手动创建或从 examples/ 复制配置文件。若用户不知道 examples 目录存在（README 未强调），就只能从头写 YAML。

**严重程度**: 🟠 **P1 - High**

**建议**: `cvlab init` 在 `.cvlab/` 下生成一个默认模板 `config.yaml`，内容为最小可用配置，所有字段带注释：
```yaml
# CVLab 训练配置 — 由 cvlab init 自动生成
# 完整示例见 examples/cifar10_full.yaml
model:
  name: resnet18          # torchvision 分类模型名
training:
  epochs: 10
  batch_size: null        # null=自动探测
  optimizer: adam
  lr: 0.001
data:
  dataset_name: CIFAR10   # 或使用 dataset: ./my_data (ImageFolder)
  input_size: [3, 32, 32]
```

---

### 2.2 配置验证覆盖不足

**问题**: `validate_config` 仅校验：
- `model.name` 为字符串
- `epochs` 为正整数
- `batch_size` 为正或 None
- `optimizer` 在 {adam, sgd, adamw} 内
- `scheduler` 在 {cosine, step, plateau, none} 内

**不校验的项目**:
- `data.dataset` 路径是否存在
- `data.num_workers` 是否 >= 0（可导致 crash）
- `data.input_size` 格式（如 [3, 224] 长度不对会悄悄失败）
- `lr` 是否为正数（负学习率不会报错）
- `seed` 类型是否为 int

**严重程度**: 🟡 **P2 - Medium**

**建议**: 扩充验证规则，尤其对可能导致无声失败或 crash 的字段做提前检验：
```python
if not isinstance(training.get("lr"), (int, float)) or training["lr"] <= 0:
    errors.append("training.lr 必须为正数")
if data_cfg.get("num_workers", 0) < 0:
    errors.append("data.num_workers 不能为负数")
```

---

### 2.3 无独立 Sweep YAML 模板文件

**问题**: USAGE.md 中内嵌了 sweep 配置示例，但 `examples/` 目录下没有独立的 `sweep.yaml` 文件供用户直接复制编辑。用户如果需要用 sweep，必须从文档中手动抄写 YAML 内容。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 在 `examples/` 下添加 `sweep.yaml` 文件，内容对标 USAGE.md 中的示例，方便用户 `cp examples/sweep.yaml .` 后直接修改。

---

### 2.4 `_sweep` 前缀泄露实现细节

**问题**: Sweep 配置文件使用 `_sweep` 键（带下划线前缀）来定义扫描参数。这个下划线通常表示"内部/私有"，用户看到会困惑——这是应该手动写的配置字段，还是由工具自动生成的？

**严重程度**: 🔵 **P3 - Low**

**建议**: 改为更直观的键名如 `sweep:` 或 `hyperopt:`，去除下划线前缀。

---

### 2.5 `batch_size: null` 对非 YAML 熟练用户不友好

**问题**: 配置中的 `batch_size: null` 表示"自动探测"，这是 YAML 中对 `None` 的标准写法。但对不熟悉 YAML 的用户来说，`null` 可能看着像拼写错误。如果用户想"不设置"而直接省略这个字段，自动探测不会触发（因为 DEFAULT_CONFIG 中 `batch_size` 是存在的）。

**严重程度**: 🔵 **P3 - Low**

**建议**: 在注释中写清楚两种写法：
```yaml
batch_size: null   # 设为 null 自动探测最大 batch size
# 或指定固定值: batch_size: 128
# 完全省略此字段则使用默认值 64
```

---

## 3. 错误处理

### 3.1 异常堆栈泄露到终端

**问题**: `cvlab train --resume` 模式下，当训练失败时会打印完整堆栈：
```python
except Exception as e:
    error(_("训练失败: {}").format(e))
    import traceback
    console.print(traceback.format_exc())  # ← 堆栈直接输出
    return 1
```
对非开发者用户来说，堆栈追踪是噪音和恐吓信息。

**严重程度**: 🟠 **P1 - High**

**建议**: 
1. 默认只显示友好错误信息
2. 添加 `--verbose` / `--debug` 全局标志，仅在开启时显示堆栈
3. 或将堆栈写入日志文件，提示用户 `详情见 .cvlab/last_error.log`

---

### 3.2 部分命令异常捕获过于宽泛

**问题**: 多处使用 `except Exception as e`（export, diagnose dataloader, profile 等），但捕获后的处理仅仅是 `error(_("... {}").format(e))`。用户得到的错误信息是 Python 异常字符串（如 `list index out of range`），没有业务层面的解释和修复建议。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 区分可预期的业务异常（如文件不存在、格式错误）和不可预期的系统异常。对前者提供友好提示 + 修复建议；对后者统一处理为"发生未知错误，详情见日志"。

---

### 3.3 数据库初始化失败无友好提示

**问题**: 多个 CLI 命令中直接使用 `db = Database()`，如果 SQLite 数据库无法创建（权限问题、磁盘满、路径不可写），会抛出原始 `sqlite3.OperationalError` 异常，没有任何包装。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 在 Database 初始化处捕获常见异常并转译：
```python
try:
    db = Database()
except sqlite3.OperationalError as e:
    error(_("无法创建数据库: {}").format(e))
    info(_("请检查 .cvlab/ 目录的写入权限"))
    info(_("或者运行 cvlab init 重新初始化"))
    return 1
```

---

### 3.4 `cvlab export` 模型重建静默失败

**问题**: `_reconstruct_model` 函数尝试从 checkpoint 重建模型时，搜索路径是硬编码的（`model_state_dict` → `state_dict` → 遍历所有 Tensor key）。如果都不匹配，`_detect_model_name` 返回 None，最终输出：
```
[FAIL] 无法从 checkpoint 重建模型
[INFO] 提示: 目前仅支持包含 'model_state_dict' 或完整模型序列化的 checkpoint
```
这个信息没有告诉用户如何解决（比如用 Python API 手动加载），也没有提及具体缺了什么。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 在错误信息中包含更多诊断信息：
- 输出 checkpoint 中实际包含的 key 列表（前 10 个）供用户参考
- 提供 Python API 替代方案的具体代码示例

---

### 3.5 没有全局 `--verbose` / `--debug` 标志

**问题**: 全局参数只有 `--version` 和 `--lang`，没有控制日志详细程度的标志。遇到问题时用户无法获取更多调试信息，开发者也无法快速定位问题。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 添加全局 `--verbose` 标志，在详细模式下：
- 显示 `train_classification` 中的更多内部日志
- 保留子进程 stdout/stderr 输出
- 显示堆栈追踪而非隐藏
- 显示配置加载的合并过程

---

## 4. 安装与入门

### 4.1 PyPI 安装不可用但文档暗示可用

**问题**: README 写道"装一个 `pip install`，加三行代码，就能获得..."，但项目并未发布到 PyPI（pyproject.toml 中 version=0.2.3，但实际只能 `git clone + pip install -e .`）。用户尝试 `pip install cvlab` 会安装到一个同名的其他包或报错。

**严重程度**: 🔴 **P0 - Critical**

**建议**: 
1. **短期**: 修改 README 安装说明，明确写 `git clone ... && cd CVLab && pip install -e .`，删除暗示 PyPI 可用的表述
2. **长期**: 将包发布到 PyPI，实现真正的 `pip install cvlab`

---

### 4.2 README 中 CIFAR10 首次运行体验不佳

**问题**: 快速开始使用 CIFAR10 作为示例。但：
1. `dataset_name: CIFAR10` 会触发 `download=True`，首次下载约 170MB，在普通网络下需要 1-5 分钟
2. 下载期间 CLI 无进度条（`torchvision.datasets` 不提供回调接口）
3. 用户可能以为程序卡死

**严重程度**: 🟡 **P2 - Medium**

**建议**: 
1. 在下载前打印明确提示："正在下载 CIFAR10 数据集（约 170MB），请稍候..."
2. 考虑提供一个不使用数据集下载的最简示例（如随机生成数据的 Toy 模式）
3. 或者在 `tiny_cnn.yaml` 中增加下载提示

---

### 4.3 无 `cvlab --version` 版本号显示在帮助中

**问题**: 查看版本信息需要显式运行 `cvlab --version`，但 help 文本中不显示当前版本号。用户无法一眼知道安装的是哪个版本。

**严重程度**: 🔵 **P3 - Low**

**建议**: 在 `cvlab` 无参数输出或 `cvlab help` 的开头显示版本号：
```
CVLab v0.2.3 — CV Experiment Management Platform
```

---

### 4.4 缺少"最简体验路径"文档

**问题**: 虽然 `examples/tiny_cnn.yaml` 提供了快速验证配置（5 epoch, CIFAR10, w/ GPU 可选），但 README 的快速开始部分没有指向它。用户按 README 的引导会用到默认的 50 epoch 配置，等待时间较长。

**严重程度**: 🔵 **P3 - Low**

**建议**: 在 README 快速开始中新增一个"30 秒体验"区块：
```bash
# 快速体验（无需 GPU，1 分钟内出结果）
cvlab init
cvlab train --config examples/tiny_cnn.yaml
```

---

## 5. 信息过载

### 5.1 训练过程中每 epoch 输出冗余

**问题**: `train_classification` 在每个 epoch 结束时输出详细指标行：
```
Epoch   1/50 | train loss: 1.5234 | train acc: 45.23% | val acc: 52.10% | lr: 1.00e-03 | 12.3s
Epoch   2/50 | train loss: 1.2345 | train acc: 55.67% | val acc: 58.90% | lr: 1.00e-03 | 11.8s
...
Epoch  50/50 | ...
```
上方已有的 Rich 进度条已经显示了 epoch 进度。对 90+ epoch 的训练，这会产生 90 行几乎相同格式的输出。终端会迅速被填满，用户查找早期信息时需要回滚很多屏。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 
1. 默认只每 N 个 epoch 输出（如每 10 个 epoch），或在进度条中直接显示关键指标
2. 提供一个 `--quiet` 模式，只在进度条中显示信息
3. 或者提供 `--log-interval` 参数控制打印频率

---

### 5.2 `cvlab profile` 默认输出完整逐层统计

**问题**: 对于 resnet50 这类有 50+ 层的模型，`profile` 输出：
```
逐层统计
┌─────┬──────────────────────────┬───────────┬────────┬──────────┐
│ #   │ 层                        │ 类型      │ 参数   │ 可训练   │
├─────┼──────────────────────────┼───────────┼────────┼──────────┤
│   1 │ conv1                     │ Conv2d    │  9,408 │ True     │
│   2 │ bn1                       │ BatchNorm │    256 │ True     │
...
│  20 │ layer4.2.conv3            │ Conv2d    │  ...   │ ...      │
└─────┴──────────────────────────┴───────────┴────────┴──────────┘
... 还有 30+ 层
```
对大多数用户，前 20 行已足够，剩余的几十行是噪音。但"还有 30+ 层"这句话仍暗示用户错过了信息。

**严重程度**: 🔵 **P3 - Low**

**建议**: 
1. 默认只显示摘要指标（参数总量、FLOPs、延迟、显存）
2. 逐层统计改为 `--layers` 标志控制，或默认只显示前 5 层 + 尾部 5 层
3. 提供 `--summary-only` 选项

---

### 5.3 `cvlab show` 全量 JSON 配置转储

**问题**: 当实验配置较大时（如 sweep 配置含多个 trial params），`console.print(json.dumps(config, indent=2, default=str))` 会输出数十行原始 JSON，在终端中占据大量空间。用户通常只关心关键超参（lr, batch_size, optimizer）而非完整结构。

**严重程度**: 🔵 **P3 - Low**

**建议**: 
1. 默认只显示常用字段（模型名、lr、batch_size、epochs、优化器、数据集）
2. 完整 JSON 配置通过 `--full` / `--verbose` 标志查看
3. 或使用折叠面板输出

---

### 5.4 训练完成后报告生成信息多余

**问题**: 训练完成后，用户看到：
```
[header] 报告生成
[OK] 报告: /path/to/exp_xxx.html
[WARN] 报告生成失败: ...
```
训练已经结束，用户此时关心的是最终指标（已经显示过了），"报告生成"这个步骤的输出打断了"训练完成"的收束感。

**严重程度**: 🔵 **P3 - Low**

**建议**: 将报告生成作为后台任务静默执行，或者在训练完成摘要之后再显示。为用户提供一个 `--no-report` 选项跳过此步骤。

---

## 6. CI/DevOps 用户体验

### 6.1 Makefile 中 `python3` 在 Windows 上不可用

**问题**: `PYTHON ?= python3` 在 Linux/macOS 上可以，但在 Windows（Git Bash / MSYS2）上 `python3` 不存在，应该是 `python`。此外，Windows 上通常没有 `uname` 命令（Git Bash 有但路径不同）。

**严重程度**: 🟠 **P1 - High**

**建议**: 
```makefile
# 自动检测 Windows
ifeq ($(OS),Windows_NT)
    PYTHON ?= python
    UV ?= uv
else
    PYTHON ?= python3
    UV ?= uv
endif
```

---

### 6.2 CI 中 lint 和 test 串行执行

**问题**: GitHub Actions 中，test job 设置了 `needs: lint`，意味着 lint 必须通过测试才能运行。这两个任务没有依赖关系，串行执行浪费了 CI 时间（lint ~1min + test ~3min = ~4min 串行 vs ~3min 并行）。

**严重程度**: 🔵 **P3 - Low**

**建议**: 移除 `needs: lint` 让 lint 和 test 并行执行。仅在快速失败的场景（如 deploy 之前）保留串行检查。

---

### 6.3 无 `make test-file` 开发用快捷目标

**问题**: 开发过程中经常需要单独运行某个测试文件，但 make help 中只有 `test` / `test-full` / `test-cov` / `test-quick`。开发者要么完整运行所有测试，要么手动敲 `uv run pytest cvlab/tests/test_config.py -v`。

**严重程度**: 🔵 **P3 - Low**

**建议**: 添加：
```makefile
test-file: ## Run a specific test file (usage: make test-file FILE=test_config)
	$(UV) run $(PYTEST) cvlab/tests/$(FILE) -v --tb=short
```

---

### 6.4 测试覆盖度阈值未设置

**问题**: `test-cov` 目标生成 coverage 报告但不设阈值，即使覆盖率下降 CI 也不会失败。目前的覆盖率基线是多少无从得知。

**严重程度**: 🔵 **P3 - Low**

**建议**: 在 `pyproject.toml` 中设置：
```toml
[tool.coverage.report]
fail_under = 70  # 至少 70%
```
并将 coverage 检查加入 CI pipeline 或 pre-merge check。

---

### 6.5 缺少端到端集成测试

**问题**: 测试套件（25+ 文件）以单元测试为主，没有覆盖完整的用户流程。例如：
- `cvlab init` → 检查 `.cvlab/` 存在且数据库可连接
- `cvlab train --config examples/tiny_cnn.yaml` → 训练跑完 1 个 epoch → 检查 checkpoint 是否保存
- `cvlab list` → 检查新实验是否出现在列表中

这些端到端测试虽然慢，但对回归保护至关重要。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 新增一个 `test_e2e.py`（加 `@pytest.mark.slow` 标记），用最小配置跑一个完整流程（1 epoch），验证：
1. 命令执行成功（exit code 0）
2. 预期文件生成（checkpoint, db 记录）
3. Web UI 能否启动（只验证 import，不实际打开浏览器）

---

## 7. 其他发现

### 7.1 English translations encoding issue

在 Windows 上读取 `en.json` 时，python 默认编码 `gbk` 导致解码失败。虽然文件本身以 UTF-8 存储，但某些场景下（如 `open()` 不带 `encoding` 参数）会导致乱码。`cvlab/i18n/__init__.py` 中确实使用了 `encoding="utf-8"` 打开了文件。但 `__init__.py` 尝试了 `locale.getdefaultlocale()` 获取系统语言——在 Windows 中文系统上返回 `zh_CN`，导致无法自动切换到英文。

**严重程度**: 🟡 **P2 - Medium**

**建议**: 确认所有 `en.json` 中的非 ASCII 字符（如果有）正确编码。可以考虑使用纯 ASCII 的英文翻译避免编码问题。

### 7.2 `__init__.py` 大量延迟导入可能隐藏 ImportError

**问题**: `cvlab/__init__.py` 从多个子模块导入大量类，但部分子模块需要 torch/torchvision 等重量级依赖。如果某个依赖缺失，会在 `from cvlab import Tracker` 时抛出 ImportError，而非在调用具体功能时。用户可能只想要 `Tracker` 却被要求安装 `streamlit` 等其他依赖。

**严重程度**: 🔵 **P3 - Low**

**建议**: 将 `__init__.py` 中的导入改为惰性导入（Lazy Import），或只在用户实际使用时才尝试加载。保留核心类（Tracker, seed_everything）为立即导入。

### 7.3 `.cvlab/` 数据库文件不宜 git 追踪但文档未说明

**问题**: `.gitignore` 已忽略 `.cvlab/`，但文档中未提醒用户。如果用户将 `.cvlab/` 加入到 git 追踪中（手动 `git add -f`），二进制数据库文件会导致仓库膨胀。

**严重程度**: 🔵 **P3 - Low**

**建议**: 在 USAGE.md 的数据存储章节添加提示："`.cvlab/` 目录已默认在 `.gitignore` 中，请勿将其加入版本控制。"

---

## 总结

| 维度 | P0 Critical | P1 High | P2 Medium | P3 Low | 总分 |
|------|:-----------:|:-------:|:---------:|:------:|:----:|
| CLI 使用体验 | 0 | 1 | 4 | 3 | 8 |
| 配置体验 | 0 | 1 | 3 | 2 | 6 |
| 错误处理 | 0 | 1 | 3 | 0 | 4 |
| 安装与入门 | 1 | 1 | 1 | 2 | 5 |
| 信息过载 | 0 | 0 | 1 | 3 | 4 |
| CI/DevOps | 0 | 1 | 1 | 3 | 5 |
| **合计** | **1** | **5** | **13** | **13** | **32** |

### 优先修复清单

1. **🔴 README 不暗示 PyPI 可用** — 修改安装说明，避免用户尝试 `pip install cvlab`
2. **🟠 `cvlab init` 后无引导** — 添加欢迎面板和下一步指引
3. **🟠 `cvlab init` 不生成配置模板** — 初始化时创建 config.yaml 模板
4. **🟠 异常堆栈泄露** — 引入 `--verbose` 标志，默认隐藏堆栈
5. **🟠 Makefile 在 Windows 上不可用** — 自动检测 OS 设置 `PYTHON`
6. **🟡 训练输出冗余** — 每 epoch 输出改为可配置频率
7. **🟡 配置验证覆盖不足** — 扩充验证规则
8. **🟡 缺少端到端测试** — 添加最小 E2E 流程
9. **🟡 子进程启动无反馈** — 添加 Spinner 或定期输出
10. **🟡 无 sweep YAML 模板** — 添加 `examples/sweep.yaml`
