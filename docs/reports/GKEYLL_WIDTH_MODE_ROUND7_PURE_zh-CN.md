# Gkeyll 严格单案例复现 Round 7：FNO 宽度–模数扫描与纯监督结果

## 1. 本轮目标与约束

本轮只改变 FNO 宽度与内部 Fourier 模数，检查模型容量对监督闭合精度和闭环推进精度的影响。所有训练使用同一个论文参数单案例 Gkeyll 轨迹：

- `k=0.35, A=0.10, t=0–40`；
- 输入为密度、速度、压力，标签为热通量梯度 `∂xq`；
- 4 层 FNO、ReLU、无 GroupNorm；
- `paper_full` 全轨迹监督协议，100 epochs，batch size 256；
- 流体闭环固定使用 Ampère 场推进、primitive 方程、RK4 stage 级闭合、`dt=0.002`；
- 部署时统一截断到 maximum mode 8；
- 不使用任何 rollout 微调。

扫描网格为 5 个宽度 `32, 48, 64, 96, 128` 与 5 个内部模数 `8, 12, 16, 24, 32`，共 25 个模型。先用 seed 0 做全网格，再对短时闭环前五名补 seed 1、2，最后对三种子中位数前两名做 `t=40` 评估。

## 2. 宽度–模数扫描

seed 0 的最低监督相对 L2 为：

- width 128 / modes 32：`9.97e-4`；
- width 128 / modes 24：`1.05e-3`；
- width 96 / modes 24：`1.19e-3`。

短时闭环的单 seed 排序与监督误差排序不同。seed 0 的前五名为：

| 结构 | t≤10 场能 log-RMSE | clamp |
|---|---:|---:|
| width 128 / modes 8 | 0.000211 | 0 |
| width 128 / modes 32 | 0.000234 | 0 |
| width 128 / modes 12 | 0.000260 | 0 |
| width 96 / modes 24 | 0.000302 | 0 |
| width 128 / modes 16 | 0.000313 | 0 |

这说明离线标签拟合不是闭环模型选择的充分条件。内部 modes 8 的监督误差约 `5e-3`，但在 seed 0 的短时闭环上反而最好；高频标签拟合能力更强，也可能把对 PDE 推进不利的误差带入闭环。

## 3. 三随机种子复核

按三种子 `t≤10` 场能误差的中位数排序：

| 结构 | seed 0 | seed 1 | seed 2 | 中位数 |
|---|---:|---:|---:|---:|
| width 128 / modes 12 | 0.000260 | 0.000209 | 0.000217 | **0.000217** |
| width 128 / modes 32 | 0.000234 | 0.000158 | 0.000432 | **0.000234** |
| width 128 / modes 8 | 0.000211 | 0.000282 | 0.000379 | 0.000282 |
| width 128 / modes 16 | 0.000313 | 0.000204 | 0.000285 | 0.000285 |
| width 96 / modes 24 | 0.000302 | 0.000471 | 0.000460 | 0.000460 |

因此进入长时评估的是 128/12 与 128/32。

## 4. t=40 纯监督闭环结果

| 结构/seed | 完成时间 | clamp | 场能 log-RMSE | 密度 rel-L2 | 速度 rel-L2 | 压力 rel-L2 |
|---|---:|---:|---:|---:|---:|---:|
| 128/32 seed 1 | 40.000 | 0 | **0.03791** | **0.000186** | **0.00625** | **0.000755** |
| 128/32 seed 2 | 40.000 | 0 | 0.17883 | 0.00954 | 0.10795 | 0.01410 |
| 128/32 seed 0 | 40.000 | 0 | 0.31020 | 0.01206 | 0.30428 | 0.03946 |
| 128/12 seed 2 | 40.000 | 0 | 0.18290 | 0.00406 | 0.08273 | 0.01056 |
| 128/12 seed 1 | 40.000 | 1630 | 0.37119 | 0.05138 | 1.14019 | 0.16337 |
| 128/12 seed 0 | 32.808 | 7310 | 0.91631 | 6.70e14 | inf | inf |

最终推荐结构是 width 128 / modes 32，最佳 checkpoint 是 seed 1。它的 teacher-forced `∂xq` 指标为：

- relative L2：`9.72e-4`；
- RMSE：`3.10e-5`；
- correlation：`0.99999953`。

128/32 的三种子 `t=40` 场能误差中位数为 `0.17883`，范围为 `0.03791–0.31020`。因此本轮找到了明显更好的“可能最优”模型，但随机种子敏感性仍然较强，不能把最佳 seed 当成稳定可重复的论文级结果。

## 5. 相比 Round 6 的提升

Round 6 的纯监督模型（width 64 / modes 16 / seed 2）在相同 `t=40` 协议下场能 log-RMSE 为 `0.16723`。Round 7 最佳 seed 1 为 `0.03791`：

- 场能误差降低约 **77.3%**，约为原来的 **1/4.41**；
- 密度误差从 `0.00440` 降到 `0.000186`，约改善 **23.6 倍**；
- 速度误差从 `0.08256` 降到 `0.00625`，约改善 **13.2 倍**；
- 压力误差从 `0.00901` 降到 `0.000755`，约改善 **11.9 倍**。

提升主要来自更大的宽度与更多内部 Fourier 模式提高了监督闭合精度，同时部署端继续保留 mode 8 截断，抑制了高频闭环误差。另一方面，128/32 的三种子中位数 `0.17883` 略差于 Round 6 的单一已选 seed，因此真正尚未解决的是训练初始化导致的长期闭环方差。

## 6. 纯监督可视化

以下图只包含 Gkeyll 真值与纯监督 FNO；没有 rollout 微调模型或微调曲线：

- `results/gkeyll_round7/final_report/pure_supervised_width_mode_scan.png`
- `results/gkeyll_round7/final_report/pure_supervised_long_time_field_energy.png`
- `results/gkeyll_round7/final_report/pure_supervised_moments_truth_prediction_error.png`
- `results/gkeyll_round7/final_report/pure_supervised_closure_truth_prediction_error.png`

长时场能图同时画出最佳 seed、三种子中位数和三种子范围，用于区分“最佳可达到效果”和“典型可重复效果”。时空图统一使用横轴 `x/L`、纵轴 time，避免坐标语义混淆。

## 7. 文件与复现入口

- 扫描配置：`configs/training/gkeyll_round7_width_modes.json`
- 最佳纯监督 checkpoint：`results/gkeyll_round7/refine_training/w128_m32_seed1/best.pt`
- 最佳长时 rollout：`results/gkeyll_round7/t40/w128_m32_seed1/rollout.npz`
- 纯监督汇总：`results/gkeyll_round7/final_report/round7_pure_summary.json`
- 出图程序：`src/landau_surrogate/tools/plot_gkeyll_round7_pure.py`

## 8. 下一步

下一轮不建议继续无限增大 width/modes。更值得做的是固定 128/32，减少随机种子方差：增加 seed 数，检查训练过程中闭环稳定指标，保存多个低监督误差 checkpoint 并用短时闭环二次选择；随后再测试轻量的闭环一致性正则。当前结果已经证明纯监督 FNO 可以显著逼近论文单案例效果，但“稳定地得到好 seed”仍是达到论文级复现的主要差距。
