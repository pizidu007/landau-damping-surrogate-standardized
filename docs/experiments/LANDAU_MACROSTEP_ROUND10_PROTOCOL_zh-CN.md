# Landau Round 10：保守多场历史宏步演化协议

## 1. 冻结问题

本轮不再优化瞬时热流闭合，而是回答：

> 在完整未见参数轨迹上，低阶保守场的历史是否能支持跨越 50 个原小步
> 的稳定、准确自由演化？

主状态固定为：

```text
U = [M0, M1, M2, E]
```

模型学习未来状态残差：

```text
U[t-L:t], K, alpha, DeltaT -> U[t+DeltaT] - U[t]
```

不输入绝对时间，不在历史种子之后使用真值回灌。

## 2. 数据和划分

- 数据：`continuum_v1` 中通过冻结 QC 的 195 条 CUDA Gkeyll 轨迹；
- canonical split：139 train / 20 validation / 36 diagnostic test；
- 模型选择只使用 train 和 validation；
- 当前 36 条 test 已在旧闭合研究中打开，仅作为 diagnostic test；
- 正式最终结论需要模型、阈值冻结后新增约 12 条 sealed holdout。

所有归一化统计只能由 train 轨迹计算。

## 3. 第一阶段固定对照

| 编号 | 模型 | 历史长度 | 宏步 |
|---|---|---:|---:|
| M0 | residual FNO1d | 1 | 5 个输出间隔，约 `DeltaT=0.1` |
| M1 | residual FNO1d | 4 | 同上 |
| M2 | periodic residual U-Net1d | 4 | 同上 |
| C0 | 已冻结的 closure FNO + `dt=0.002` RK4 | 当前论文基线 | 小步 |

M0 和 M1 的 width、modes、layers 相同，以隔离历史信息的作用。M2 用于避免
把结论绑定到 FNO 架构。

## 4. 训练与物理处理

- history 和 target 均按同一宏步间隔采样；
- rollout curriculum：1 -> 4 -> 16 个自由宏步；
- 损失：标准化全状态、空间频谱、电场一阶复模、总能量和正性；
- 每步硬投影：质量均值、动量均值、周期 Poisson 电场和部署频谱；
- 不进行密度或压力钳位；一旦低于冻结下限，记为物理失败；
- checkpoint 只按 validation 多步指标选择；test 不参与选择。

## 5. 评价

每个案例必须保存完整 `M0/M1/M2/E` 轨迹，至少报告：

- 标准化全状态和各通道相对 L2；
- 相对平衡态的扰动场 L2；
- 电场一阶复模振幅误差和相位 MAE；
- 归一化场能 log10-RMSE；
- 质量、动量和总能量漂移；
- 最小密度、最小压力、失败时间；
- 端到端 wall time 和网络调用次数。

## 6. 阶段门

只有同时满足以下条件才进入 `DeltaT≈0.5`：

1. M1 在 validation 的完整自由 rollout 中全部达到 `t=80`，零正性违规；
2. M1 相对 M0 的关键长期指标有稳定优势，且三随机种子结论一致；
3. diagnostic test 场能 log10-RMSE 中位数相对冻结 closure 基线 `0.794`
   至少改善 25%；
4. 完整场和相位指标同步改善，不接受只改善稳定性；
5. 同硬件、同输出频率下相对 closure+RK4 至少达到 10 倍端到端加速。

若 M1 不优于 M0，则历史窗口假设不成立；若二者均失败，再进行带 `M3/q`
或可演化隐状态的条件实验，而不是继续扩大无历史闭合网络。

## 7. 预注册执行顺序

1. 合同测试和 CPU smoke；
2. M0/M1 seed0 小规模 GPU pilot，检查损失和闭环；
3. 冻结必要的数值修复；
4. M0/M1/M2 三随机种子正式训练；
5. validation 全程自由 rollout；
6. 完全冻结模型和阈值后，打开 diagnostic test；
7. 若通过阶段门，生成新的 sealed holdout 并进行一次最终评估。
