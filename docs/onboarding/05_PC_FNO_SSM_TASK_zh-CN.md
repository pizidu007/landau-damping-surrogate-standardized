# 个人 PC 任务包：小型 FNO + SSM 记忆 + 非线性残差闭合

本任务面向暂时无法访问共享服务器、使用 RTX 3060 笔记本的新成员。
建议按 6 GB 显存做保守配置；实际可用显存以本机为准。
不需要 Gkeyll、服务器账户或完整相空间数据。

**研究问题：在相同流体任务上，较小的空间网络配合因果记忆，能否改善长期热流闭合？**
完整数据、对照入口和可直接训练的简单 SSM 起始实现已经提供。
后续工作是修改模型、训练和消融；该原型还没有证明能通过长期物理精度门。

## 1. 为什么这些小数组足够

原始 195 条合格案例的 HDF5 合计约 134.4 GiB，其中主要体积来自完整 `f(x,v,t)`。
这里保留低阶流体状态及闭合监督，不发布速度网格上的分布：

| 数组 | 形状 / 类型 | 含义 |
|---|---|---|
| `state` | `[4001,4,128]`，float32 | 按 `[n,u,p,E]` 排列；已按 Round11 的 mode 16 规则处理 |
| `gradient` | `[4001,128]`，float32 | 真值 `dq/dx`，同一 mode 16 带宽，单位与流体方程一致 |
| `time` | `[4001]`，float64 | `0,0.02,...,80`，完整连续时间序列 |
| `dt` | 标量，float64 | `0.02` |
| `x_over_l` | `[128]`，float64 | 周期网格的归一化单元中心 `(j+0.5)/128` |
| `reference_state` | `[801,4,128]`，float32 | 仅 validation/旧 test：未滤波参考状态，供最终误差计算 |
| `reference_time` | `[801]`，float64 | 仅 validation/旧 test：`0,0.1,...,80` |

每条轨迹的 `K`、`alpha`、物理类别、原有 split、SHA256 和来源在 `manifest.json`。
所有案例的空间点数和时间长度相同，因此可以批处理，但不能把不同案例拼成一条时间序列。

`p` 是压力，不是温度；温度为 `p/n`。原数据从中心矩重建压力后才滤波。
`gradient` 是热流梯度，不是热流本身，也不是状态下一帧。
真实 `gradient` 只作为监督/诊断，不能作为部署输入。
该数据合同支持**梯度形式的残差闭合**；不支持重建完整相空间分布或检验速度空间细丝结构。

train/validation 的 `state` 和 `gradient` 与原 Round11 缓存**逐元素、逐 dtype 完全一致**。
只是把同样的数组用标准 NPZ 的 LZMA 方法无损压缩；没有 float16 量化、缩短时间、
降低空间分辨率或抽掉中间帧。`numpy.load(..., allow_pickle=False)` 可以直接读取。
mode 16 是原实验既有的带宽限制，不是为减小这次下载临时加的限制。

未滤波参考仍保留在验证集：不能改用更平滑的滤波参考来让新模型显得更准确。
现有均匀缓存来自原始输出触发时间上的插值，导出时没有再次重采样它。

## 2. 一个压缩包，完整数据直接使用

最终交付是单个 `landau-fno-ssm-kit-2026-09-09.zip`，约 1.1 GB。
解压后，`data/` 已包含全部 139 条训练、20 条验证和 36 条旧诊断测试轨迹，
`checkpoints/` 有 A/B/C seed 0，`src/` 与 `scripts/` 是可直接安装、运行的代码。
不需要访问服务器、另行下载或拼接分包。默认训练使用完整 train，比较使用完整 validation。

先看包内首页的三个步骤：安装、训练、比较。这里是需要修改代码时才查阅的细节。
所有路径都相对于解压后的根目录；新输出在 `runs/`，不会覆盖数据或已有权重。

数据不需要全部放进显存。每个案例解压到内存的状态和标签约 9.8 MiB，
训练按案例逐条读入 CPU，按连续时间块送 GPU。NPZ 不能直接 memory-map；
本例每条轨迹只解压一次，避免为每个随机窗口重复解压。

36 条 test 是此前使用过的旧诊断集，不是新的盲测；默认训练和比较都不读取它。
新的独立 holdout 不在包中。清单还保留一个 24/6 的 starter 子集索引用于调试，
不影响默认全量使用，也不要求拆包。它按物理类别内的参数排序均匀取样，不依赖模型误差。

## 3. 可选的接口检查

正常实验只需运行首页命令。需要排查安装/数组问题时，可执行：

```bash
python scripts/inspect_pc_closure_data.py --data-root data --verify
python scripts/inspect_pc_closure_data.py --data-root data --smoke-backward --device cpu
```

检查应看到 139 train / 20 validation；反向传播检查使用随机 FNO，不生成训练结论。
没有 CUDA 时仍可用 CPU 检查。起始实现是纯 PyTorch，不依赖 Mamba 的扩展安装。

## 4. 读取序列和残差标签

在自己的训练脚本中使用：

```python
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from landau_surrogate.data.portable_closure import ClosureSequenceDataset, hp_gradient

root = Path('data')
stats = json.loads((root / 'statistics_full.json').read_text())
dataset = ClosureSequenceDataset(root, subset='full', length=64, burn_in=64, stride=64)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
batch = next(iter(loader))
U = batch['state']                  # [B,128,4,128]；前 64 帧是记忆预热
target = batch['gradient']          # [B,128,128]
valid = batch['valid']              # 负时间补齐处 False，不得推进 SSM
loss_mask = batch['loss_mask']      # 预热前缀不计损失
linear = hp_gradient(U, batch['K'][:, None], stats['hp_scale'])
residual_target = target - linear   # 供非线性残差头训练；仍是 dq/dx 的单位

mean = torch.tensor(stats['input_mean'])[None, None, :, None]
std = torch.tensor(stats['input_std'])[None, None, :, None]
normalized_state = (U - mean) / std
```

HP 形式的线性参考在此明确规定为 `g_linear = c * |d/dx|(p-n)`，保留 mode 0..16。
这是线性温度扰动近似；不是把 `p-n` 当作非线性温度 `p/n-1` 的恒等式。
`c` 在相应训练子集上拟合为非负最小二乘系数，不沿用 PIC 数据上校准的数值。
默认的 `statistics_full.json` 仅使用 139 条 train；调试索引对应的 `statistics_starter.json` 仅使用 24 条 train。
若自己换成别的解析闭合，需同时重新定义残差标签和对应的消融基线。

数据中的负时间槽用初态补齐，`valid=False`；不能将它们视为真正观测过的历史。
`time` 返回这些槽的实际相对物理时间，方便显式处理初态边界。

`ClosureSequenceDataset` 用于先跑通窗口训练。后期建议按案例逐条读取完整序列，
从 t=0 递推、按 64/128 步切块做截断反向传播，块间保留数值 hidden state 并 detach。
独立随机窗口前补 64 帧只提供 1.28 的预热，**不能保证重现 t=0 开始的长期 SSM 状态**。
要比较长记忆，需增加预热、保留连续块状态，并始终使用完整因果回放做 validation。

## 5. 要实现的模型与更新逻辑

已提供纯 PyTorch 小型对角 SSM，见 `src/landau_surrogate/models/pc_fno_ssm.py`。
这是一份可修改的起点，不是对最佳架构的结论。

```text
当前状态 U_t=[n,u,p,E] 与 K
            │
       小型空间 FNO → 空间特征 z_t
            │                │
            │          因果 SSM 的记忆 h_t
            └────────┬───────┘
              非线性残差头 → δg_t
                              │
g_linear(U_t,K) ────────────── + → mode≤16、零空间均值 → g_t=dq/dx
                                                        │
                                                  流体方程推进
```

FNO 负责同一时刻的空间耦合；SSM 负责过去状态的信息；残差头根据当前空间特征和记忆
修正线性闭合。不要先做全空间平均再送进 SSM，以免丢失相位与空间结构。
可以对每个空间位置的 FNO 特征维护共享参数的 SSM，或者明确保留复数傅里叶模态记忆。

一个可检查的递推约定是：

```text
h_0 = 0
δg_t = decoder(z_t, h_t)
g_t = P_{1..16}[g_linear(U_t,K) + δg_t]
h_{t+dt} = A_bar(dt) h_t + B_bar(dt) z_t
```

`P_{1..16}` 去掉零模态并限制到 16 阶。非线性可放在空间编码和残差头中；
记忆的线性状态转移可以先用稳定对角参数化，例如 `A=-softplus(a)` 后按 `dt` 离散化。
纯实对角衰减模块是简单起点，不等同于完整 S4/Mamba；若需要表达振荡记忆，可另做二维振荡块的消融。

**RK4 子步与 hidden state：**旧 Round11 求解器在一次时间步内会多次调用闭合网络。
不能在每次 `forward()` 时偷偷修改持久 hidden state，否则一次物理步会更新四次甚至更多次记忆。
先明确一个一致的约定：本步所有 RK 试算使用同一份 `h_t`，接受状态后只提交一次
`h_{t+dt}=update(h_t,z(U_t),dt)`；试算或拒绝的步骤不提交记忆。
训练中的序列输出也必须采用“先读 h_t 预测当前，再推进记忆”的同一顺序。
更精细的联合流体/记忆积分可以作为后续改进，需要额外一致性测试。

`evaluate_pc_closure.py` 已支持 A/B/C、解析闭合和新的 SSM checkpoint。
显式记忆推进在 `src/landau_surrogate/fluid/pc_ssm.py`，已有未来无泄漏、块间一致性和
每个接受步只提交一次记忆的测试。改成其他 SSM 后仍需保持这些规则。

默认起始配置：空间宽度 24、FNO 2 层、17 个 rFFT 槽（含 0..16）、
每特征 8 个 SSM 状态，batch 1、反向长度 64。一次只训练一个候选，不跑三种子并行。
先用 float32；FFT 和复数运算不要未经检查就全部转 half。
显存不足时依次降 batch、反向长度、宽度，而不是静默改评估分辨率。
这些是保守起始参数，尚未在这台具体笔记本上测量峰值显存或训练时长。

## 6. 先建立能跑的旧模型对照

实验包中的 `checkpoints/round11_A_seed0.pt`、`round11_B_seed0.pt`、
`round11_C_seed0.pt` 均为先前公开的原始权重，未裁剪、未重新训练。
取同一个 seed 0 是预先指定的对照选择，不根据子集成绩挑“最好”或“最弱”模型。

| 对照 | 作用 |
|---|---|
| `zero` | 无热流梯度的流体基线 |
| `hp` | 在所选 continuum train 上拟合的线性闭合 |
| Round11 A seed 0 | 只看当前状态 + K 的历史已有 FNO |
| Round11 B seed 0 | 非均匀 8 槽、跨度 2.0 的历史 FNO |
| Round11 C seed 0 | 在 B 输入外加入初始 alpha |

离线评价在真实历史上预测 `dq/dx`，用于检查标签和模型接口：

```bash
python scripts/evaluate_pc_closure.py --data-root data --baseline round11 --checkpoint checkpoints/round11_B_seed0.pt --mode offline --subset full --limit 1 --device cpu --output-dir results/runs/pc_B_offline
```

先跑一条 2 个时间单位的 CPU 自由推进：

```bash
python scripts/evaluate_pc_closure.py --data-root data --baseline hp --horizon 2 --device cpu --output-dir results/runs/pc_hp_smoke
python scripts/evaluate_pc_closure.py --data-root data --baseline round11 --checkpoint checkpoints/round11_B_seed0.pt --horizon 2 --device cpu --output-dir results/runs/pc_B_smoke
```

通常直接运行首页的 `compare_pc_closures.py`，即可对全部 20 条 validation 比较 HP、A/B/C 和新模型。
单模型入口中，`--device cuda:0 --horizon 80 --limit 0` 也是完整评价；
A/C 换 checkpoint 路径即可，`zero` 不需要 checkpoint。
每次使用新的输出目录，避免覆盖先前对照。

脚本逐案例运行，保留原 checkpoint 中的归一化、K/alpha 处理、反射规则、history span 和调用频率。
它不读取 checkpoint 中的服务器数据路径，也不要求恢复服务器目录结构。
物理步长固定 0.02，输出步长 0.1；所有中间 RK 状态与接受状态都检查有限性、n>0、p>0。
首次失效后停止该案例，后续数组为 NaN，误差不填零，不夹紧负密度/压力。
CPU、不同 GPU 和逐案例 batch 与原批处理之间可能存在浮点数差异，
相同数据与规则也不保证长时间不稳定轨迹的数值输出逐 bit 相同。

输出 `provenance.json`、`summary.json` 和逐案例预测 NPZ。
`horizon<80` 明确记录为短程接口检查，不算完整精度门通过。
离线标签误差、短程状态误差、完整 t=80 误差是三个不同结果，不可混称。

## 7. 公平比较：两类表分开写

**历史参考表：**把新模型与提供的 A/B/C 权重放到同样的案例、步长、带宽、
初态、未滤波参考和 t=80 区间评价，报告每个模型的训练数据量与参数量。
旧 A/B/C 使用 139 条训练案例及既有初始化/训练过程。
只用 starter 的 24 条轨迹训练的新模型可以与它们比较表现，
但不能把差异直接归因于“SSM 架构优劣”。

**受控消融表：**自己在相同训练划分、参数/训练预算规则和种子上重训：

1. 小型 FNO，直接预测闭合，无记忆。
2. 小型 FNO + SSM，直接预测闭合。
3. 小型 FNO，预测线性闭合之外的非线性残差，无记忆。
4. 小型 FNO + SSM + 非线性残差闭合。

这张表才能区分记忆与残差各自的作用。起始训练脚本已通过 `--variant` 支持这四组，
默认全部使用完整 139/20 划分；若先用子集调试，请明确记录训练数据量。
拟合尺度、HP 系数、归一化只用所选 train；调参/选 checkpoint 用 validation。
若使用 alpha，明确把任务列为 alpha 条件输入版本；不能称为“仅从历史辨识初始参数”。

## 8. 验收顺序与交付物

第一阶段：读懂数据合同，跑通数据校验、FNO 反向检查和三个旧 checkpoint 的接口。
第二阶段：实现 SSM 递推与残差目标；验证未来帧不会影响过去输出、换案例清空记忆、
负时间补齐不推进记忆、每个接受步只提交一次状态、闭合空间均值约为零。
第三阶段：完成训练/验证曲线和上述消融，再进入完整自由推进。

全程验证从给定初态与 `h_0=0` 开始，只允许把自己的预测状态送入记忆。
不使用真实 q、未来矩场或真实历史为模型补充信息，不从训练时的样本索引恢复隐藏的真值轨迹。

至少报告：

- 完成 t=80 的案例数与首次失效时间，所有失败案例保留。
- t=30..80 场能 log10-RMSE、扰动状态相对 L2、电场一阶模态相位误差。
- 质量、动量、总能量漂移和密度/压力最小值。
- 相同案例的轨迹图；比较两个模型时同时报告完成数和共同完成案例上的配对差值。
- 参数量、峰值显存、训练/推理用时、设备和 batch；没有达到相同精度前不宣称同精度加速。

完整 Round11 参考门限是全部 20 条 validation 完成，晚期中位场能误差 ≤0.3、
扰动误差 ≤0.25、相位误差 ≤0.35，并满足守恒及分物理类别的门限。
完整规则见仓库 `configs/training/continuum_history_closure_round11_quality.json`。
提供的九个 Round11 历史模型都未通过完整门限；其中三个已放进实验包。
现有结果是研究起点，不是已经解决问题的部署模型。

最终提交：可复现代码、个人配置、数据清单 SHA256、使用的案例/种子、checkpoint、
离线与自由推进结果、消融表及失败分析。不要提交整个数据包到 Git 历史。

## 9. 背景阅读

- [项目扫盲、逻辑与任务定义](https://github.com/pizidu007/landau-damping-surrogate-standardized/tree/main/docs/onboarding)
- [旧结果与看图说明](https://github.com/pizidu007/landau-damping-surrogate-standardized/blob/main/docs/results/KEY_RESULTS_zh-CN.md)
- [FNO 原论文](https://arxiv.org/abs/2010.08895)：空间算子网络的出处。
- [S4 原论文](https://arxiv.org/abs/2111.00396)及[作者代码](https://github.com/state-spaces/s4)：状态空间序列建模的背景。

本项目的“FNO + 简单 SSM + 残差闭合”是结合现有任务提出的实验路线，
不代表上述论文已经证明它能解决这里的非线性 Landau 长期闭合问题。
