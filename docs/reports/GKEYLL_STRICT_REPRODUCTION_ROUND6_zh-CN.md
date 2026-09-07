# Gkeyll–FNO 严格单案例复现 Round 6

## 结论

六个阶段已经全部完成。本轮最重要的结果是：按照论文明确描述移除
GroupNorm、使用 ReLU，并对完整 8000 帧同轨迹数据训练后，严格单案例
`k=0.35, A=0.10` 的长时间闭环得到显著改善；但仍未完全达到预设的论文级
验收阈值，也不能跨参数泛化。

最佳模型为：

- 网络：4层 FNO、width 64、16个内部 Fourier 模；
- 结构：无 GroupNorm、ReLU；
- 训练：完整 `t=0–40` 的8000帧，seed 2；
- 部署：只保留 mode 1–8；
- 流体：primitive 中心矩、Ampère、RK4、`dt=0.002`；
- 第一时间步：使用真实初始热流梯度。

最佳检查点：`results/gkeyll_round6/stage3/A7_full_seed2/best.pt`。

## 1. 与上一轮相比的提升

| `t=40` 指标 | 旧 GroupNorm/GELU mode-8 | Round 6 | 改善 |
|---|---:|---:|---:|
| 场能 log10 RMSE | 0.3987 | **0.1672** | 58.1% |
| 密度相对 L2 | 0.00703 | **0.00440** | 37.4% |
| 速度相对 L2 | 0.2086 | **0.08256** | 60.4% |
| 压力相对 L2 | 0.02568 | **0.00901** | 64.9% |
| 闭环路径热流相对 L2 | 0.7704 | **0.4008** | 48.0% |
| 1%状态误差首次时间 | 14.26 | **23.28** | 推迟9.02 |
| 10%状态误差首次时间 | 18.11 | **31.47** | 推迟13.36 |

新模型无密度/压力钳制，稳定完成20000个 RK4 步。

![长时间场能](../../results/gkeyll_round6/final_report/best_long_time_field_energy.png)

## 2. 阶段一：基线与数据合同

固定使用自行生成且明确标注的中心矩轨迹，避免 Zenodo 公开 MAT 文件中
`p_new/q_new` 原始矩语义和两个文件哈希相同的问题。真实热流 oracle 已证明
同一流体求解器在 `t=40` 的状态误差约 `5.3e-5`、场能误差约 `0.0016`，
因此本轮不再改变 PDE。

## 3. 阶段二：结构消融

8种结构、2个 seed 共完成16次训练和16次 `t=10` 闭环。主要结果：

- 去掉 GroupNorm 但保留 GELU 的 A1 会失稳；
- 去掉 GroupNorm 并改用 ReLU 后，A2/A5/A7 的两 seed 短闭环都稳定；
- A7（ReLU、16模）两 seed 场能误差均值为 `0.000482`；
- A2（ReLU、24模）为 `0.000515`；
- 旧结构 A0 均值为 `0.0633`，且 seed 波动显著。

这说明论文明确写出的 ReLU 不是无关紧要的实现细节；离线误差最小也不等价于
闭环最稳定。

![结构筛选](../../results/gkeyll_round6/final_report/stage2_offline_vs_closed_loop.png)

## 4. 阶段三：完整8000帧训练

A2 和 A7 各训练3个 seed、固定150轮。全轨迹离线相对误差为：

- A2：`0.00134–0.00172`；
- A7：`0.00162–0.00208`。

尽管所有模型离线误差都很低，6个原生模态 `t=40` 闭环的场能误差范围仍为
`0.231–0.954`，部分 seed 发生钳制。最终由 A7 seed 2 胜出，说明闭环模型选择
必须包含自由运行指标。

## 5. 阶段四：长闭环、谱截断和真实状态重启

### 5.1 谱截断

| 最大部署模 | 场能 log10 RMSE |
|---:|---:|
| 8 | **0.1672** |
| 12 | 0.2260 |
| 16 | 0.2308 |
| 24 | 0.2807 |

mode 8 最优，高模仍然主要增加闭环敏感性。

### 5.2 真实状态重启

从 `t=0,5,...,35` 的真实状态分别重启1、2、5时间单位：

- horizon 1：平均场能误差 `2.70e-5`，最大 `5.55e-5`；
- horizon 2：平均 `8.11e-5`，最大 `1.45e-4`；
- horizon 5：平均 `9.72e-4`，最大 `0.00346`。

这证明非线性状态本身可以被局部准确推进，当前剩余问题是连续自由运行中的
长期相位积累，而不是某个 `t≥20` 区间完全学不会。

![重启热图](../../results/gkeyll_round6/final_report/truth_restart_error_heatmap.png)

## 6. 阶段五：rollout 课程训练

完成 B0–B5：监督基线、谱惩罚、H10、H50、H200 和 H200+谱惩罚。

| 模型 | `t=40` 场能 log10 RMSE |
|---|---:|
| 监督基线 | **0.1672** |
| H10 | 0.2940 |
| H50 | 0.2970 |
| H200 | 0.2878 |
| H200+谱惩罚 | 0.2970 |

短 curriculum 对局部损失有效，却改变了长周期相位，所有候选均被拒绝，没有覆盖
监督基线。10–200步只对应 `0.02–0.4` 时间单位，远短于 bounce period；下一轮
若继续 rollout 训练，应使用相位对齐、多重 shooting 或跨一个完整振荡周期的
伴随/截断训练，而不是继续增加短窗 epoch。

![消融](../../results/gkeyll_round6/final_report/mode_and_rollout_ablation.png)

## 7. 阶段六：跨参数检查

将单案例最佳模型直接用于：

| 案例 | 场能 log10 RMSE | 结果 |
|---|---:|---|
| `(0.35,0.075)` | 1.159 | 失稳并钳制 |
| `(0.40,0.10)` | 1.537 | 失稳并钳制 |

因此最佳模型只能命名为“论文同案例复现模型”，不能替代 Round 5 多案例模型。
要得到通用闭合，仍需使用带 `k,A` 条件、多案例数据和 case-wise split 的模型。

## 8. 验收结果

| 验收项 | 目标 | 结果 | 状态 |
|---|---:|---:|---|
| 场能 log10 RMSE | ≤0.15 | 0.1672 | 未通过（差11.5%） |
| 密度相对 L2 | ≤0.005 | 0.00440 | 通过 |
| 速度相对 L2 | ≤0.08 | 0.08256 | 略未通过 |
| 压力相对 L2 | ≤0.015 | 0.00901 | 通过 |
| 闭环热流相对 L2 | ≤0.20 | 0.4008 | 未通过 |
| 1%状态误差时间 | ≥20 | 23.28 | 通过 |
| 稳定推进到 `t=40` | 完成 | 完成、无钳制 | 通过 |

本轮应判定为“显著改善、部分通过，但尚未达到论文级完整复现”。

## 9. 可视化与数据产出

- `results/gkeyll_round6/final_report/round6_summary.json`
- `results/gkeyll_round6/final_report/best_feedback_audit.json`
- `results/gkeyll_round6/final_report/best_long_time_field_energy.png`
- `results/gkeyll_round6/final_report/best_moments_truth_prediction_error.png`
- `results/gkeyll_round6/final_report/stage2_offline_vs_closed_loop.png`
- `results/gkeyll_round6/final_report/mode_and_rollout_ablation.png`
- `results/gkeyll_round6/final_report/truth_restart_error_heatmap.png`
- Gkeyll 真值相空间图：
  `/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized/gkeyll/paper_match_single_v1/figures/phase_space_full_and_resonant.png`

![三矩演化](../../results/gkeyll_round6/final_report/best_moments_truth_prediction_error.png)

相空间图只代表 Gkeyll kinetic truth；三矩流体模型没有演化分布函数，不能把它
表述为 FNO 预测的相空间。
