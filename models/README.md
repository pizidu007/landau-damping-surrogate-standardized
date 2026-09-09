# 模型目录

选定权重已发布到 GitHub Release，参见[下载与使用指南](../docs/results/CHECKPOINTS_zh-CN.md)。
在项目根目录运行 `python scripts/download_checkpoints.py --list` 可查看清单；
下载器核对 SHA256 并恢复相对路径，权重实体仍不进入 Git 历史。

- `snapshot/snapshot_best.pt`：`(k, alpha, t) -> delta_f(x,v,t)` 的直接模型。
- `rollout/rollout_positivity_best.pt`：带 Snapshot mean anchor 和软正性约束的最终递归模型。

Checkpoint 内含模型配置、网格、归一化尺度、背景分布和来源哈希。文件是 PyTorch pickle 容器，只加载可信来源。模型卡位于 `artifacts/model_cards/`。
