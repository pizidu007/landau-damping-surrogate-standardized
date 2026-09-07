# Gkeyll 单案例复现：Round 8 稳定性约束与论文对齐对比

## 1. 本轮目标

本轮固定上一轮参数扫描得到的网络规模 `width=128, modes=32`，不再改变模型容量，只考察训练约束能否降低随机种子敏感性并改善长时间闭环推进。同时，在完全相同的流体方程、Ampere 场推进、时间步长和谱截断下，引入标准 Hammett–Perkins（HP）闭合，按论文逻辑比较：

1. Gkeyll 动理学真值；
2. Fluid + HP closure；
3. Fluid + pure-supervised FNO closure。

所有 FNO 图均不包含 rollout 微调。论文对齐图采用上一轮已经验证通过的纯监督最优检查点 `Round7 / width128_modes32 / seed1`，避免用本轮效果更差的新约束模型替换现有最优结果。

## 2. 固定设置与训练约束

- 模型：FNO1d，`width=128, modes=32`
- 随机种子：0、1、2
- 部署谱截断：mode 8
- 闭环推进：primitive moment formulation + Ampere field solver
- 时间步长：`dt=0.002`
- 初始闭合量：Gkeyll 真值
- 闭合调用：RK stage evaluation
- CPU：每个进程最多 2 核；训练和闭环计算使用 GPU 0

本轮比较五种损失配置：

| 配置 | 监督损失 | mode-8 部署损失 | 频谱一致性 | 闭合功/RMS 约束 |
|---|---:|---:|---:|---:|
| C0 baseline | 1.00 | 0 | 0 | 0 |
| C1 balanced | 1.00 | 1.00 | 0.10 | 0 |
| C2 deployment-focused | 0.25 | 1.00 | 0.25 | 0 |
| C3 deployment-only | 0 | 1.00 | 0.25 | 0 |
| C4 work/RMS | 1.00 | 1.00 | 0.10 | 0.05 / 0.05 |

## 3. 稳定性约束结果

### 3.1 短时间闭环

以三个种子的场能 `log10 RMSE` 统计：

| 配置 | t≤10 中位数 | 种子极差 | 判断 |
|---|---:|---:|---|
| C0 baseline | 2.335e-4 | 2.741e-4 | 基线 |
| C1 balanced | **1.946e-4** | **3.870e-5** | 短时最好，种子方差显著下降 |
| C2 deployment-focused | 2.556e-4 | 3.591e-4 | 未超过 C1 |
| C3 deployment-only | 2.885e-4 | 3.571e-4 | 牺牲全频监督后没有收益 |
| C4 work/RMS | 2.527e-4 | 1.826e-4 | 有一定稳健化，但不及 C1 |

C1 说明部署滤波损失和频谱一致性对短时间推进确实有效：中位误差下降约 16.7%，三种子的极差下降约 85.9%。

### 3.2 长时间闭环

将短时最优 C1 推进到 `t=40` 后：

| 配置 | 三种子场能 log10 RMSE | 中位数 | 极差 | 完整推进 |
|---|---|---:|---:|---|
| C0 baseline | 0.3102, **0.0379**, 0.1788 | 0.1788 | 0.2723 | 3/3 |
| C1 balanced | 0.2559, 0.1423, 0.7301 | 0.2559 | 0.5878 | 2/3；seed2 在 t=27.818 失稳 |

因此，本轮约束只改善了局部的一步映射和短时闭环，没有解决非线性阶段的离分布累积误差。C1 的长时中位误差比基线反而高约 43%，且出现密度/压力钳位和提前终止。C2–C4 的短时结果均未超过 C1，所以没有继续消耗算力做 t=40 推进。

这说明当前长时失败的主要矛盾不是网络宽度或模数，也不是简单的输出频谱约束；它来自闭环状态逐渐离开监督数据流形后，模型从未在训练中见过自己的误差状态。

## 4. 论文对齐：真值、HP 与纯监督 FNO

标准 HP 系数取 `sqrt(8/pi)=1.595769`，与 FNO 使用相同 mode-8 谱截断和同一流体推进器。代表 FNO 取 Round7 的纯监督最优 `seed1`。

### 4.1 长时间场能

![长时间场能：真值、HP 与 FNO](/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized/results/gkeyll_round8/final_report/paper_field_energy_truth_hp_fno.png)

HP 能表示线性 Landau 阻尼，但在进入非线性回升以后继续过度阻尼并产生相位偏差；纯监督 FNO 则复现了阻尼、最低点和后续回升，直到 `t=40` 仍与真值基本重合。这一相对表现与论文中 Fluid+HP 和 Fluid+ML 的对照一致。

### 4.2 低阶矩时空演化

![低阶矩：真值、HP 与 FNO](/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized/results/gkeyll_round8/final_report/paper_moments_truth_hp_fno.png)

FNO 对密度、速度和压力的振幅与相位保持良好；HP 的振幅随时间明显衰减，并在中后期形成系统性误差。

### 4.3 误差图（论文 Fig. 6 排列方式）

![FNO 与 HP 的低阶矩绝对误差](/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized/results/gkeyll_round8/final_report/paper_moment_errors_fno_hp.png)

### 4.4 热通量梯度闭合量

![热通量梯度闭合：真值、HP 与 FNO](/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized/results/gkeyll_round8/final_report/paper_closure_truth_hp_fno.png)

闭合量采用与部署一致的 mode-8 滤波后统计：

| 闭合 | 相对 L2 | RMSE | 相关系数 |
|---|---:|---:|---:|
| HP | 1.1331 | 3.6157e-2 | 0.4447 |
| pure-supervised FNO | **0.00458** | **1.4607e-4** | **0.99999** |

闭环 `t≤40` 指标：

| 闭合 | 场能 log10 RMSE | n 相对 L2 | u 相对 L2 | p 相对 L2 | 钳位次数 |
|---|---:|---:|---:|---:|---:|
| HP | 1.1773 | 0.01520 | 0.63164 | 0.04289 | 0 |
| pure-supervised FNO | **0.03791** | **0.000186** | **0.006253** | **0.000755** | 0 |

## 5. 结论与下一步

1. 固定 `128/32` 是合理的；当前最好结果已经能在这个单案例上清楚复现论文的核心物理结论：HP 只能维持线性阻尼趋势，FNO 学到非线性热通量闭合后能够恢复场能回升。
2. 本轮简单稳定性损失没有产生新的长时最优模型，所以正式展示继续使用 Round7 纯监督 seed1。
3. 下一轮若目标是让三个随机种子都稳定，而不是只保留最佳种子，应直接训练闭环误差状态：在 `t=20–35` 强非线性区间加入 2–8 步可微短 rollout，或用流体推进产生的偏移状态做离线扰动增强；checkpoint 选择也应加入短闭环指标。
4. 这不会改变“纯监督结果”的展示口径：rollout-aware 模型应作为稳定性消融单独报告，不能与纯监督复现图混在一起。

## 6. 产物

- 图表与汇总：`results/gkeyll_round8/final_report/`
- 稳定性训练：`results/gkeyll_round8/training/`
- t≤10 闭环：`results/gkeyll_round8/t10/`
- C1 的 t≤40 闭环：`results/gkeyll_round8/t40/`
- HP 的 t≤40 闭环：`results/gkeyll_round8/hp_standard_t40/`
- 配置：`configs/training/gkeyll_round8_stability_constraints.json`
- 机器可读摘要：`results/gkeyll_round8/final_report/round8_paper_comparison_summary.json`

