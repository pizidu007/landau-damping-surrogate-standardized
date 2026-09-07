# 当前验证状态

## 已验证

- Python 3.11 环境下全部源码可编译；
- 11 个数据、模型、Poisson、正性、保守压缩和资产合同测试通过；
- 公开 CLI 的 `--help` 可运行；
- 四个正式数据/模型资产 SHA256 与清单一致；
- 真实 Snapshot 推理输出有限，并满足 mean/nonzero 精确分解；
- 真实 Rollout 推理输出有限；
- 当前重新推理结果与保留 smoke 结果一致。
- Rollout 会拒绝与 checkpoint 哈希不匹配的缓存，除非显式使用实验性覆盖参数。
- GitHub 直接上游固定为 commit `9490dbf`；展开快照和归档均包含 50 个 tracked 文件，归档 SHA256 已纳入资产校验。

## 正式元数据结果

- Snapshot 测试 case-macro relative L2：`0.20597`；
- Stage 9B 测试轨迹 case-macro relative L2：`0.22414`；
- Stage 9B horizon-30 case-macro relative L2：`0.27950`；
- 密度 mode-1 phase MAE：`0.42363 rad`；
- 预测完整分布负值比例均值：`0.01539`；
- 真值完整分布负值比例均值：`0.01323`。

## 尚未充分验证

- 从零开始的完整候选训练和 final checkpoint 封装；
- 新机器上的位级训练复现；
- `t>15` 的真值准确度；
- 严格质量与总能量守恒；
- PIC 子包作为生产模拟器的正确性；
- 外部、从未参与研发选择的 holdout 数据。
