# 资产元数据

- `manifests/`：文件路径、大小、SHA256、角色和兼容关系；`SOURCE_TREE.sha256` 记录整理后源码、配置和文档树。
- `inspections/`：缓存与 checkpoint 的只读结构检查结果。
- `model_cards/`：冻结模型指标和限制。
- `release/`：原源码发布清单，用于溯源。

使用 `python scripts/verify_assets.py --require-all` 验证正式数据、模型、保留的 smoke 结果和固定的 GitHub 上游归档。
