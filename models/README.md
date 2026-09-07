# 模型目录

- `snapshot/snapshot_best.pt`：`(k, alpha, t) -> delta_f(x,v,t)` 的直接模型。
- `rollout/rollout_positivity_best.pt`：带 Snapshot mean anchor 和软正性约束的最终递归模型。

Checkpoint 内含模型配置、网格、归一化尺度、背景分布和来源哈希。文件是 PyTorch pickle 容器，只加载可信来源。模型卡位于 `artifacts/model_cards/`。
