# Landau 阻尼 x-v 神经代理项目报告

**报告对象：** `landau-damping-surrogate-standardized`  
**报告日期：** 2026-08-27  
**项目状态：** 标准化整理后的当前维护版  
**直接上游：** [`kaneman777/landau-damping-surrogate`](https://github.com/kaneman777/landau-damping-surrogate)  
**固定上游 commit：** `9490dbff6b322bf7f43bbcff49af48bb9ff65fd7`

## 1. 执行摘要

本项目是在 GitHub 上游 `kaneman777/landau-damping-surrogate` 基础上进行修改和扩展形成的 Landau 阻尼 x-v 相空间神经代理系统。上游主要围绕一维静电 PIC、参数扫描以及场能时间曲线的 MLP/FNO 实验；当前改进版将研究对象推进到 1D1V 相空间分布，建立了 case-level HDF5 数据合同、二维条件 FNO、一步和多步自回归 Stepper、正性约束、Poisson/电场/能量闭合诊断以及完整的科学资产管理结构。

本次标准化整理采用以下原则：

1. 项目根目录只服务于当前改进版；
2. GitHub 上游项目作为固定、只读参考材料单独保存；
3. 整理前 notebook 和适配示例归入项目历史材料；
4. 代码、配置、数据、模型、结果、日志和元数据不混放；
5. 正式数据、模型和参考结果全部使用路径、大小和 SHA256 标识；
6. 历史精确缓存与 M0/M2 保守缓存按数值语义分离；
7. 文档明确区分“已验证能力”“冻结历史指标”和“尚未闭环环节”。

当前工作副本约 `565 MB`，包含 38 个包内 Python 模块、13 个脚本、7 个测试文件、两份正式缓存、两个正式 checkpoint、三个 NPZ smoke 结果，以及 50 个文件的固定上游快照。核心包源码约 `13,055` 行。

## 2. 项目谱系与代码边界

### 2.1 演化关系

```text
Antoine Tavant / 1d-pic-electrostatic
  └─ 一维静电 PIC 教学实现
      ↓ fork / surrogate experiments
kaneman777 / landau-damping-surrogate
  ├─ PIC 参数扫描
  ├─ (Te, Lx, t) -> 场能点的 MLP
  └─ (Te, Lx) -> log10(E) 曲线的 FNO-style 模型
      ↓ 本项目持续修改和扩展
当前标准化改进版
  ├─ 1D1V x-v 相空间 HDF5 数据
  ├─ (k, alpha, t) -> delta_f 的 Snapshot FNO
  ├─ delta_f_t -> delta_f_{t+0.5} 的 Stepper FNO
  ├─ 多步递归、mean anchor 与正性训练
  ├─ 密度模、Poisson、电场、场能和守恒诊断
  └─ 标准化配置、测试、结果与资产管理
```

### 2.2 当前维护代码

当前版本唯一的安装来源是 [`src/landau_surrogate/`](../../src/landau_surrogate)。日常开发只修改根目录下的：

```text
src/
scripts/
tests/
configs/
docs/
```

### 2.3 上游参考材料

上游材料位于 [`reference/upstream/`](../../reference/upstream)：

- `landau-damping-surrogate-original/`：commit `9490dbf` 的完整 tracked-file 快照；
- `landau-damping-surrogate-9490dbf.tar.gz`：相同 commit 的固定归档；
- `UPSTREAM_SOURCE.json`：仓库、commit、日期、许可证和归档哈希；
- `UPSTREAM_TREE.sha256`：展开后 50 个文件的逐文件哈希。

这些文件不参与 setuptools 包发现、当前测试或 CLI。更新上游时应新增快照，不得覆盖现有快照。

### 2.4 整理前历史材料

整理前保留的 notebook 和已适配 PIC 示例位于：

```text
reference/project_history/legacy-adaptation/
```

它们不是 GitHub 上游的逐字节原件，也不是当前工作入口，只用于追溯研究过程。

### 2.5 许可证判断

GitHub 上游 README 写有 MIT，但其实际 tracked `LICENSE.md` 是 GNU GPL v3。当前项目以许可证文件为准，继续采用 GNU GPL v3，并在 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) 中保留直接上游和 Antoine Tavant 的归属链。

## 3. 科学问题与物理合同

项目面向归一化 1D1V 静电 Vlasov--Poisson 系统：

```text
∂f/∂t + v ∂f/∂x - E ∂f/∂v = 0
n_e(x,t) = ∫ f(x,v,t) dv
ρ(x,t) = 1 - n_e(x,t)
∂E/∂x = ρ
mean_x(E) = 0
```

数据并不直接拟合完整分布，而是使用训练集背景：

```text
delta_f = f - f_bg
sigma_train = sqrt(mean_train(delta_f^2))
normalized_delta_f = delta_f / sigma_train
```

`f_bg` 和 `sigma_train` 只由训练 case 估计并冻结，验证和测试不能重新估计。使用单一全局 RMS 是为了保留不同 `alpha` 对应的物理幅值差异。

case 按完整 `(k, alpha)` 轨迹划分：

| Split | case 数 | 时刻数/case | 样本数 |
|---|---:|---:|---:|
| Train | 70 | 31 | 2170 |
| Validation | 14 | 31 | 434 |
| Test | 26 | 31 | 806 |
| 合计 | 110 | 31 | 3410 |

同一 case 的所有时刻始终处于同一 split，避免轨迹级信息泄漏。

## 4. 当前标准目录结构

```text
landau-damping-surrogate-standardized/
├── README.md                       中文总入口
├── pyproject.toml                  包、依赖、CLI 和测试配置
├── requirements*.txt               基础依赖与正式环境参考
├── Makefile                        常用安装/测试/校验命令
├── src/landau_surrogate/           当前维护的 Python 包
├── scripts/                        CLI 包装和资产校验
├── tests/                          单元与合同测试
├── examples/                       小型合成缓存示例
├── configs/
│   ├── environment/                Conda 环境参考
│   ├── cli_presets/                推理与训练参数参考
│   └── paths.example.toml          标准路径表
├── docs/
│   ├── science/                    数学、模型和物理诊断
│   ├── data/                       数据合同与资产目录
│   ├── operations/                 快速开始和训练边界
│   ├── results/                    冻结指标与验证状态
│   ├── reports/                    当前详细项目报告
│   └── history/                    来源谱系和历史报告
├── data/
│   ├── raw/                        原始数据占位和规则
│   └── processed/
│       ├── historical_exact/       当前模型严格兼容缓存
│       └── conservative_m02/       保守缓存
├── models/
│   ├── snapshot/                   Snapshot checkpoint
│   └── rollout/                    Stage 9B checkpoint
├── results/
│   ├── smoke/                      冻结参考输出
│   └── runs/                       新运行输出
├── logs/                           新运行日志
├── artifacts/
│   ├── manifests/                  SHA256 与机器可读清单
│   ├── inspections/                缓存/checkpoint 检查快照
│   ├── model_cards/                模型卡和指标
│   └── release/                    原发布清单
└── reference/
    ├── upstream/                   GitHub 上游只读快照
    └── project_history/            整理前历史材料
```

## 5. 源码模块说明

### 5.1 `data/`：HDF5 数据接口

[`src/landau_surrogate/data/`](../../src/landau_surrogate/data) 提供：

- `snapshot_cache.py`：单时刻样本读取；
- `rollout_cache.py`：同一 case 内相邻时刻配对和完整序列读取；
- `rollout_windows.py`：多步训练窗口；
- `conservative.py`：控制体积平均、梯形权重和 M0/M2 修正。

HDF5 reader 使用进程局部、延迟打开的文件句柄，避免 DataLoader fork 后共享失效句柄。Pair 和 Window reader 会检查 case-major、时间连续和网格形状合同。

### 5.2 `models/`：二维条件 FNO

[`snapshot_fno.py`](../../src/landau_surrogate/models/snapshot_fno.py) 实现：

```text
(normalized k, alpha, t)
+ physical k, alpha, t
+ x/v coordinate channels
-> Conditional 2D FNO
-> normalized delta_f(x,v,t)
```

Snapshot 支持单头和双头。正式模型使用双头，将输出精确分解为：

```text
mean_delta(v,t) = mean_x(delta_f)
nonzero(x,v,t) = delta_f - mean_delta
mean_x(nonzero) = 0
```

[`stepper_fno.py`](../../src/landau_surrogate/models/stepper_fno.py) 除条件和坐标外，还输入：

```text
current full field
current mean field
current nonzero field
```

正式 Stepper 使用 residual 参数化，预测 mean/nonzero 增量，并恢复下一时刻完整场。

FNO 对 x-v 做二维 FFT。x 为周期方向；v 为非周期方向，因此只在 v 方向做零填充。FFT 固定使用 float32，避免 CUDA 对非二次幂速度网格的半精度限制。

### 5.3 `losses/`：训练和科学约束

[`src/landau_surrogate/losses/`](../../src/landau_surrogate/losses) 包含：

- 全局 MSE 与相对误差；
- x 方向 mode-1 误差；
- 相速度附近 `|v-v_phase|<=0.5` 的共振区误差；
- 空间平均分量误差；
- 增量误差；
- 低/中/高频谱误差和周期梯度误差；
- 完整分布软正性损失。

正性约束作用于：

```text
f_pred = f_bg + sigma_train * normalized_delta_f_pred
```

由于扰动本身允许为负，因此不能对 `delta_f` 直接做非负截断。

### 5.4 `training/`：四阶段训练

[`src/landau_surrogate/training/`](../../src/landau_surrogate/training) 包含：

1. `snapshot.py`：直接 Snapshot 训练；
2. `one_step.py`：teacher-forced 一步 Stepper；
3. `multistep.py`：2→4→8 horizon curriculum；
4. `positivity.py`：完整分布正性 fine-tuning。

训练按 validation 指标保存最佳 epoch，再在 test 上报告结果。训练参数、环境、CSV 指标、图像和 artifact SHA256 会写入输出目录。

### 5.5 `inference/`：生产推理

[`snapshot.py`](../../src/landau_surrogate/inference/snapshot.py) 从 checkpoint 自包含的网格、背景和归一化尺度恢复：

```text
normalized_delta_f
delta_f
f_phase
mean_delta
nonzero
```

[`rollout.py`](../../src/landau_surrogate/inference/rollout.py) 可从 NPZ 初态或 HDF5 case/time 启动递归 Rollout。标准化版增加了缓存 SHA256 校验：如果 HDF5 与 checkpoint 的 `cache_contract` 不一致，默认拒绝运行。只有明确的兼容性实验才能使用 `--allow-cache-mismatch`。

### 5.6 `diagnostics/` 与 `physics/`

诊断链为：

```text
normalized delta_f
  -> physical delta_f
  -> f_phase
  -> n_e = ∫f dv
  -> rho = 1 - n_e
  -> periodic Poisson
  -> E(x,t)
  -> field / kinetic / total energy
```

同时计算 density mode-1 复振幅、相位、有效衰减率、频率、速度矩、高 x 模能量、扰动放大和长 Rollout 有限性。

### 5.7 `pic/`

[`src/landau_surrogate/pic/`](../../src/landau_surrogate/pic) 是上游 PIC 教学代码的包路径适配版本。它用于来源追溯和参考模拟，不是当前 Snapshot/Rollout 推理的必经模块，也尚未达到生产 PIC 求解器的测试标准。

## 6. 数据资产管理

### 6.1 历史精确抽样缓存

```text
规范路径：$LANDAU_DATA_ROOT/processed/historical_exact/
      landau_deltaf_cache_x128_v193_v1.h5
大小：186,152,364 bytes
SHA256：84fd51e09e3898555a690aef3df57625f48bc374dab3a27fa5856cf750801a2c
```

该缓存是现有 Snapshot 和 Rollout checkpoint 的严格兼容缓存。其压缩方式是从 mother grid 做精确 stride 取点，不保证速度矩严格守恒。

### 6.2 M0/M2 保守缓存

```text
规范路径：$LANDAU_DATA_ROOT/processed/conservative_m02/
      landau_deltaf_cache_x128_v193_conservative_m02_v2.h5
大小：271,364,016 bytes
SHA256：f00a6618a29c96a9f4e1c5cfea5d3372656fd925355678eecf907b3332892236
```

该缓存使用控制体积平均和光滑 M2 投影，使粗网格梯形积分在浮点精度内保持 M0 与 M2。它适用于物理闭合审计和未来重训练，但不是当前 checkpoint 的 drop-in 数据替代。

### 6.3 未包含的 mother 数据

```text
landau_xv_master_110_v1.h5
shape：110 × 31 × 512 × 1537
SHA256：f92b58323b627ed526c29028abc0da0869173f6c380e8df4b76f424f89d1c722
```

mother HDF5 已迁移到 `$LANDAU_DATA_ROOT/raw/pic/`，应保持只读并受控；源码树不保存数据实体。

## 7. 模型资产管理

### 7.1 Snapshot checkpoint

```text
路径：models/snapshot/snapshot_best.pt
大小：34,299,788 bytes
SHA256：a45058a899d7338ef0acbf398dac02e61ffab4c164c123d84382b2265889984b
stage：stage8d1d_final_baseline
```

任务为 `(k, alpha, t) -> normalized delta_f(x,v,t)`。正式结构是 width 32、4 层、x modes 16、v modes 32 的 dual-head 条件 FNO。

### 7.2 Stage 9B Rollout checkpoint

```text
路径：models/rollout/rollout_positivity_best.pt
大小：68,599,602 bytes
SHA256：b94434678c1a1cdd8d22e5e94b4bfa2ba2ddfc2ba8b7fc4e0acded76c9077dd5
stage：stage9b_final_positivity_rollout
```

该文件同时包含 Stepper 和 Snapshot mean-anchor 权重，因此体积约为单一 Snapshot checkpoint 的两倍。正式参数包括 residual Stepper、`dt=0.5`、`mean_anchor_beta=0.5` 和 `absolute_hinge` 正性候选。

PyTorch checkpoint 使用 pickle 容器，只能加载可信来源。

## 8. 配置、输出、日志和元数据

### 8.1 配置

[`configs/`](../../configs) 只放环境和运行参数：

- `environment/conda.yml`：参考环境；
- `paths.example.toml`：数据、模型、结果和日志标准路径；
- `cli_presets/inference_*.json`：已验证推理参数；
- `cli_presets/training_reference.json`：历史正式参数参考。

当前程序以 CLI 参数为最终事实来源，JSON preset 不会被隐式读取或覆盖 CLI。

### 8.2 新运行输出

统一写入：

```text
results/runs/<run_id>/
```

建议每个 run 保存：

```text
command.txt
config.json
environment.json
metrics.csv / metrics.json
prediction.npz 或 best.pt
plots/
artifact_sha256.txt
```

### 8.3 日志

日志写入：

```text
logs/<run_id>/
```

日志只记录运行过程，不与科学结果、模型或数据混放。

### 8.4 资产清单

[`artifacts/manifests/assets.json`](../../artifacts/manifests/assets.json) 当前登记：

- 2 个正式 HDF5；
- 2 个正式 checkpoint；
- 6 个 smoke NPZ/JSON 结果；
- 1 个固定 GitHub 上游归档。

共 11 项。使用：

```bash
python scripts/verify_assets.py --require-all
```

`SOURCE_TREE.sha256` 记录当前源码、配置、文档和历史适配材料；`UPSTREAM_TREE.sha256` 单独记录上游原始快照，二者不混合。

## 9. 推理和数据流

### 9.1 Snapshot

```text
用户条件 (k, alpha, t)
  -> 范围检查与归一化
  -> 条件特征和坐标通道
  -> Snapshot FNO
  -> normalized delta_f
  -> × sigma_train
  -> delta_f + f_bg
  -> NPZ + JSON metadata
```

### 9.2 Rollout

```text
可信 HDF5 / NPZ 初态
  -> shape、finite、时间范围检查
  -> HDF5 SHA256 与 checkpoint contract 校验
  -> Stepper 预测下一时刻 mean/nonzero
  -> Snapshot 预测下一时刻 mean
  -> beta=0.5 mean anchor
  -> 预测结果递归反馈
  -> 完整时间序列 NPZ + JSON metadata
```

## 10. 冻结科学结果

根据保留模型卡：

| 指标 | Snapshot | Stage 9B Rollout |
|---|---:|---:|
| 测试 case-macro trajectory/snapshot relative L2 | 0.20597 | 0.22414 |
| horizon-30 case-macro relative L2 | 不适用 | 0.27950 |
| density mode-1 complex relative L2 | — | 0.08348 |
| density mode-1 phase MAE | — | 0.42363 rad |
| 最大预测/真值范数比 | — | 1.09615 |

正性指标：

```text
预测负值比例均值：0.01539
真值负值比例均值：0.01323
负值比例 excess 均值：0.00752
hard-clamp 相对变化均值：3.16e-05
```

频谱指标显示中、高 x 模相对误差约为 1，x 梯度误差也较高。因此模型主要适合低频主模和整体轨迹，不应描述为对细尺度相空间结构的高保真替代。

## 11. 本次实际验证

在 Conda 环境 `landau-pic-surrogate` 中完成：

| 验证项 | 结果 |
|---|---|
| Python 源码编译 | 通过 |
| 单元/合同测试 | `11/11` 通过 |
| pip 依赖一致性 | 通过 |
| CLI `--help` 冒烟 | 通过 |
| 数据/模型/结果/上游归档哈希 | 11 项全部通过 |
| 上游展开快照逐文件哈希 | 50/50 通过 |
| 上游归档 tracked 文件数 | 50 |
| Snapshot 真实推理 | 通过，finite |
| Snapshot 与保留 smoke 最大差异 | `0.0` |
| 两步 Rollout | 通过，shape `[3,128,193]`，finite |
| Rollout 与保留 30 步结果前缀最大差异 | `0.0` |
| mean/nonzero 分解误差 | 约 `1.16e-10` 或更小 |
| 错误缓存保护 | 保守缓存被正确拒绝 |

当前 Conda 环境的 editable 安装指向本标准化项目的 `src/landau_surrogate`，不会从 `reference/` 导入代码。

## 12. 代码审核结论与风险

### 12.1 高优先级：训练链 final 封装未闭环

公开训练脚本产生研究候选 checkpoint：

```text
stage8d1d_fno_ablation
stage8d2a_stepper
stage8d2b_multistep_candidate
stage9b_positivity_candidate
```

而后续阶段或生产推理要求：

```text
stage8d1d_final_baseline
stage8d2a_final_stepper
stage8d2c_final_rollout
stage9b_final_positivity_rollout
```

历史候选比较、mean-anchor 选择以及 final schema 封装工具未完整进入发布代码。因此现有正式模型可以推理，但不能声称仅使用公开脚本即可无人工步骤地复现完整训练链。

### 12.2 已处理：缓存语义混用

整理前文档容易把保守缓存理解为现有 checkpoint 的训练缓存。现在已完成：

- 路径分层；
- 文档明确用途；
- 资产清单记录兼容关系；
- Rollout 默认 SHA256 校验。

训练 warm-start 阶段未来仍应统一加入相同的 cache contract 校验。

### 12.3 中优先级：训练模块过长且重复

四个训练主函数包含大量重复的设备、AMP、JSON/CSV、哈希、环境和指标代码。建议后续抽取：

```text
training/common.py
artifacts/checkpoint_schema.py
evaluation/common.py
```

### 12.4 中优先级：checkpoint 安全

`torch.load(..., weights_only=False)` 会反序列化 pickle。当前 checkpoint 结构需要非权重元数据，但必须明确限定可信文件；对外发布可考虑安全容器或将张量与 JSON 元数据分离。

### 12.5 中优先级：测试覆盖仍有限

当前测试覆盖数据合同、模型 shape、Poisson、正性、保守压缩和资产 schema，但尚未自动覆盖：

- 完整训练阶段交接；
- final checkpoint 封装；
- 30 步真实模型 CI；
- 所有物理指标回归；
- PIC 生产正确性；
- 外部 holdout。

### 12.6 低优先级：PIC 参考代码

PIC 子包保留较强的上游教学代码风格，测试和接口完整度明显低于神经代理主链。后续若不再用于数据生成，可考虑从安装包移入可选 `reference` 或独立 extras；若继续使用，应单独重构和验证。

## 13. 后续维护规则

### 13.1 修改代码

只修改根目录当前代码，不修改 `reference/upstream/`：

```text
src/
scripts/
tests/
configs/
docs/
```

### 13.2 添加数据或模型

1. 放入对应 `data/processed/<semantic_name>/` 或 `models/<task>/`；
2. 计算大小和 SHA256；
3. 更新 `artifacts/manifests/assets.json`；
4. 写明来源、归一化和兼容 checkpoint；
5. 运行 `verify_assets.py --require-all`。

### 13.3 添加运行结果

新结果放入 `results/runs/<run_id>/`，不得覆盖 `results/smoke/`。日志放入同名 `logs/<run_id>/`。

### 13.4 更新上游参考

1. 不覆盖 commit `9490dbf` 快照；
2. 新建按 commit 命名的归档/目录；
3. 新增来源 JSON 和逐文件哈希；
4. 更新差异报告；
5. 检查许可证变化。

## 14. 推荐下一步

按优先级建议：

1. 实现候选选择与 `candidate -> final` checkpoint 标准封装工具；
2. 为所有训练 warm-start 增加 cache SHA256 和 normalization contract 校验；
3. 抽取训练公共基础设施，缩短超长 `main()`；
4. 增加真实 artifact 的最小集成测试；
5. 建立新的外部 holdout；
6. 决定 PIC 子包是继续维护、变为可选组件，还是完全留在参考区；
7. 为当前标准化目录建立独立 Git 仓库，并使用 Git LFS 或外部 artifact storage 管理 HDF5/checkpoint。

## 15. 报告对应入口

- 项目入口：[`README.md`](../../README.md)
- 文档索引：[`docs/README.md`](../README.md)
- 上游来源：[`UPSTREAM_LINEAGE.md`](../history/UPSTREAM_LINEAGE.md)
- 上游差异：[`reference/DIFFERENCES.md`](../../reference/DIFFERENCES.md)
- 快速开始：[`QUICKSTART.md`](../operations/QUICKSTART.md)
- 训练边界：[`TRAINING_PIPELINE.md`](../operations/TRAINING_PIPELINE.md)
- 数据资产：[`ASSET_CATALOG.md`](../data/ASSET_CATALOG.md)
- 验证状态：[`VALIDATION_STATUS.md`](../results/VALIDATION_STATUS.md)
- 机器清单：[`assets.json`](../../artifacts/manifests/assets.json)

本报告描述的是标准化后的当前维护版本；`docs/history/` 中带有 `ORIGINAL` 或旧 Stage 名称的文件仅作历史证据，不应覆盖本报告的当前口径。
