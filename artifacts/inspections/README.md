# 检查快照

这里的 JSON 文件从原工作目录原样保留，用于证明当时缓存和 checkpoint 的结构。Checkpoint 自身保存了历史训练路径，因此检查结果中可能出现旧机器绝对路径；这些路径仅用于溯源，不是当前标准路径。

当前有效路径以 `artifacts/manifests/assets.json` 和 `configs/paths.example.toml` 为准。
