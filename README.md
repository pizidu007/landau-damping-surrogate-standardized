# Landau 阻尼 x-v 神经代理项目（标准化整理版）

本项目是在 [kaneman777/landau-damping-surrogate](https://github.com/kaneman777/landau-damping-surrogate) 基础上持续修改和改进形成的 x-v 相空间神经代理。当前目录是标准化后的唯一维护版本；GitHub 上游原项目及整理前历史材料已放入独立的 `reference/`，不参与默认安装和运行。

上游项目本身 fork 自 Antoine Tavant 的 1D electrostatic PIC。完整谱系、固定 commit 和差异见[上游来源说明](docs/history/UPSTREAM_LINEAGE.md)与[差异报告](reference/DIFFERENCES.md)。

## 新成员从这里开始

**先看成果：[关键结果与可视化](docs/results/KEY_RESULTS_zh-CN.md) · [checkpoint 下载与使用](docs/results/CHECKPOINTS_zh-CN.md)。** 已公开 2026-09-09 结果快照和 16 个可校验权重，最新 Round 11 全部九组结果及失败边界一并保留。

第一次接触项目，请从[新成员学习路线](docs/onboarding/README.md)开始，不需要先读完历史报告或运行大规模训练：

1. [背景与核心概念](docs/onboarding/01_BACKGROUND_zh-CN.md)：从分布函数、Landau 阻尼、矩与热流，讲到为什么需要神经闭合。
2. [项目逻辑与代码地图](docs/onboarding/02_PROJECT_LOGIC_zh-CN.md)：区分各条模型路线，解释数据、训练、自由推进和研究阶段的关系。
3. [任务定义与验收标准](docs/onboarding/03_TASK_DEFINITIONS_zh-CN.md)：输入输出、数组形状、A/B/C 对照、数据使用边界、指标与交付物。
4. [术语与符号速查](docs/onboarding/04_GLOSSARY_zh-CN.md)：遇到不熟悉的缩写或符号时查阅。

准备动手时，再读[协作与共享服务器入门](docs/operations/COLLABORATION_zh-CN.md)。Git 仓库包含源码、配置、测试、文档、来源记录和精选图表/指标；选定模型通过 [GitHub Release](https://github.com/pizidu007/landau-damping-surrogate-standardized/releases/tag/research-assets-2026-09-09) 下载。完整数据、实验数组、日志和中间权重保留在共享服务器。下文历史资产链接描述的是完整服务器目录，新 clone 只需按用途连接或下载对应资产。

当前研究已扩展至 Gkeyll 连续体参考数据和具有历史输入的流体热流闭合；最新实验口径见 [Round 11 协议](docs/experiments/LANDAU_CLOSURE_ROUND11_PROTOCOL_zh-CN.md)。下文的 Snapshot/Rollout 指标属于较早分支，不代表 Round 11 的结果。

## 1. 当前状态

- Python 包、训练脚本、推理脚本和测试已完整保留。
- GitHub 上游项目已固定为 commit `9490dbf`，完整快照和归档只读保存在 `reference/upstream/`。
- 两份正式 HDF5 缓存、Snapshot checkpoint、Stage 9B Rollout checkpoint 已按用途归档。
- 历史 smoke 输出、模型卡、检查结果和原始发布清单已保留。
- 当前正式 checkpoint 是在“历史精确抽样缓存”上训练的；M0/M2 保守缓存用于物理闭合审计和未来重新训练，二者不可作为同一种缓存静默混用。
- Snapshot 和 Rollout 推理可直接运行；公开训练代码保留了候选阶段，但候选到最终 checkpoint 的历史选择/封装步骤并未完整公开，详见[训练流程说明](docs/operations/TRAINING_PIPELINE.md)。

## 2. 项目解决的问题

项目研究归一化 1D1V 静电 Vlasov--Poisson Landau 阻尼：

```text
∂f/∂t + v ∂f/∂x - E ∂f/∂v = 0
n_e(x,t) = ∫ f(x,v,t) dv
ρ(x,t) = 1 - n_e(x,t)
∂E/∂x = ρ,  mean_x(E)=0
```

学习目标有两类：

```text
Snapshot: (k, alpha, t) -> normalized delta_f(x,v,t)
Stepper:  (normalized delta_f_t, k, alpha, t)
          -> normalized delta_f_{t+0.5}
```

`delta_f = f - f_bg`，归一化尺度只由训练 case 估计。最终 Rollout 递归使用自身预测，并以 `beta=0.5` 混合 Snapshot 模型给出的空间平均分量。正性损失约束完整分布 `f_bg + delta_f_pred`，不是约束扰动本身为正。

更完整的科学说明见：

- [详细中文项目报告](docs/reports/PROJECT_REPORT_zh-CN.md)
- [方法与模型](docs/science/METHOD.md)
- [物理诊断](docs/science/PHYSICS_DIAGNOSTICS.md)
- [数据格式](docs/data/DATA_FORMAT.md)
- [资产目录](docs/data/ASSET_CATALOG.md)

## 3. 标准目录结构

```text
landau-damping-surrogate-standardized/
├── src/landau_surrogate/       Python 包源码
├── scripts/                    稳定 CLI 入口与资产校验工具
├── tests/                      无正式数据依赖的单元/合同测试
├── examples/                   小型合成数据示例
├── configs/                    环境、路径和 CLI 参数预设
├── docs/
│   ├── science/                数学模型、网络方法、物理诊断
│   ├── data/                   数据合同和资产说明
│   ├── operations/             安装、推理、训练、复现说明
│   ├── results/                冻结结果和当前验证状态
│   ├── reports/                当前详细项目报告
│   └── history/                原清理报告、阶段映射和旧说明
├── data/processed/              指向外部正式数据根的兼容符号链接
├── models/
│   ├── snapshot/               直接 Snapshot 模型
│   ├── rollout/                最终正性 Rollout 模型
│   └── closure/                PIC 热流闭合模型及部署参数
├── results/
│   ├── published/              公开的精选图表与指标快照
│   ├── smoke/                  已验证的小型推理结果
│   └── runs/                   新实验输出（运行时创建）
├── logs/                       新运行日志（运行时创建）
├── artifacts/
│   ├── manifests/              哈希、大小和用途清单
│   ├── inspections/            缓存/checkpoint 检查快照
│   ├── model_cards/            模型与保守缓存元数据
│   └── release/                原发布清单
└── reference/
    ├── upstream/               GitHub 原项目固定快照、归档和哈希
    └── project_history/        整理前的历史适配材料
```

## 4. 环境与安装

已有 Conda 环境：

```bash
conda activate landau-pic-surrogate
python -m pip install -e .
```

也可以参考 [configs/environment/conda.yml](configs/environment/conda.yml) 创建新环境。GPU 环境应先安装与本机 CUDA 匹配的 PyTorch。

正式数据统一位于：

```bash
export LANDAU_DATA_ROOT=/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized
```

源码树不再保存大型数据实体；旧的 `data/processed/...` 路径仅保留兼容符号链接。

## 5. 首次检查

下面的 `--require-all` 适用于已备齐历史资产的共享服务器工作目录。
新 clone 请按 [checkpoint 指南](docs/results/CHECKPOINTS_zh-CN.md)下载所需权重，
再运行对应示例；不需要补齐全部历史资产才能上手。

```bash
python scripts/verify_assets.py --require-all
python -m pytest -q

python scripts/inspect_cache.py \
  data/processed/historical_exact/landau_deltaf_cache_x128_v193_v1.h5

python scripts/inspect_checkpoint.py \
  models/rollout/rollout_positivity_best.pt
```

注意：PyTorch checkpoint 使用 pickle 容器，只应加载可信来源的文件。

## 6. 推理示例

Snapshot：

```bash
python scripts/infer_snapshot.py \
  --checkpoint models/snapshot/snapshot_best.pt \
  --k 0.55 --alpha 0.040 --time 7.5 \
  --output results/runs/snapshot_k055_a0040_t075.npz \
  --device cuda:0
```

30 步自由 Rollout：

```bash
python scripts/infer_rollout.py \
  --checkpoint models/rollout/rollout_positivity_best.pt \
  --cache data/processed/historical_exact/landau_deltaf_cache_x128_v193_v1.h5 \
  --cache-split test --cache-split-position 0 --cache-time-index 0 \
  --steps 30 \
  --output results/runs/test_case0_rollout.npz \
  --device cuda:0
```

完整操作说明见 [快速开始](docs/operations/QUICKSTART.md)。

## 7. 数据与模型口径

| 资产 | 用途 | SHA256 前缀 |
|---|---|---|
| `historical_exact/...v1.h5` | 复现现有 Snapshot/Rollout checkpoint | `84fd51e0` |
| `conservative_m02/...v2.h5` | M0/M2 闭合审计、未来重训练 | `f00a6618` |
| `models/snapshot/snapshot_best.pt` | 单时刻重建 | `a45058a8` |
| `models/rollout/rollout_positivity_best.pt` | 最终正性自由 Rollout | `b9443467` |
| `models/closure/closure_fno_pic_v1.pt` | PIC 热流闭合；部署时与 HP 混合 | `6d2628f0` |

详细哈希、文件大小和兼容关系记录在 [artifacts/manifests/assets.json](artifacts/manifests/assets.json)。

## 8. 冻结结果摘要

正式元数据记录：

- Snapshot 测试 case-macro relative L2：约 `0.2060`；
- 最终正性 Rollout 测试轨迹 case-macro relative L2：约 `0.2241`；
- horizon-30 case-macro relative L2：约 `0.2795`；
- 密度主模相位 MAE：约 `0.424 rad`；
- 预测完整分布负值比例均值约 `1.54%`，真值本身约 `1.32%`；
- 中高频谱误差仍较大，模型更适合低频主模和整体轨迹，不是细尺度高保真替代品。

见 [当前验证状态](docs/results/VALIDATION_STATUS.md)和[冻结结果](docs/results/RESULTS.md)。

## 9. 输出和日志约定

新运行统一写入：

```text
results/runs/<run_id>/     NPZ、JSON、CSV、图像和 checkpoint
logs/<run_id>/             控制台日志和环境记录
```

推荐的 `run_id`：`YYYYMMDD_HHMMSS_<task>_<tag>`。不要把新输出写回 `src/`、`data/processed/`、`models/` 或 `artifacts/model_cards/`。

## 10. 已知限制

- 只验证 `t=0..15`；更长 Rollout 仅检查有限性，没有真值准确度保证。
- 上一条只针对原 x-v 神经代理；新增热流闭合分支使用独立 sealed test 验证到 `t=60`。
- 正性是软约束，不保证严格质量或总能量守恒。
- 现有测试集在多阶段研究中曾被观察；正式发表应增加外部 holdout。
- `pic/` 是上游教学代码的适配参考实现，不是当前代理推理主链。
- 历史 checkpoint 的候选选择与最终封装步骤尚未完全产品化。

## 11. 许可证

项目采用 GNU GPL v3。直接上游是 `kaneman777/landau-damping-surrogate`，其 PIC 来源为 Antoine Tavant 的 1D electrostatic PIC，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 12. 非线性 CUDA 数据扩展

新的生成器在 CUDA 上保持粒子和时间历史常驻，直接沉积 M0--M3、热流与热流散度，并可稀疏保存完整相空间。先运行：

```bash
python scripts/generate_nonlinear_pic.py \
  --profile smoke --run-id audit_smoke_v1 --device cuda:0
```

配置见 `configs/data/nonlinear_pic_v1.json`，完整说明见 [CUDA 数据生成](docs/operations/CUDA_DATA_GENERATION.md)和[强非线性数据方案](docs/science/NONLINEAR_DATA_PLAN.md)。正式生成不允许回退到 CPU。

## 13. PIC 热流闭合六阶段研究

基于60条非线性 CUDA-PIC 轨迹，已经完成标签审计、传统闭合基线、FNO
消融训练、流体闭环、rollout 微调和四组结果可视化。纯 FNO 在 sealed
test 上将 `dq/dx` 相对误差从校准 HP 的 `0.985` 降到 `0.309`；长期部署
采用 validation 冻结的 `25% FNO + 75% HP` 混合闭合，5/5 测试轨迹均
稳定积分到 `t=60`，且没有密度或压力钳制。

- [中文研究报告](docs/reports/PIC_HEAT_FLUX_CLOSURE_zh-CN.md)
- [模型卡](models/closure/MODEL_CARD.md)
- [冻结部署参数](models/closure/deployment.json)
- [机器可读指标](results/closure_fno_v1/final_report/study_summary.json)
- [公开的八张结果图](results/published/2026-09-09/pic_closure_v1/figures/)
