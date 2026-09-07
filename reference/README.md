# 参考材料区

本目录与当前改进版代码严格分离，不属于默认安装、测试、训练或推理路径。

## 目录

- `upstream/landau-damping-surrogate-original/`：GitHub 上游项目在固定 commit 的完整 tracked-file 快照，便于直接阅读。
- `upstream/landau-damping-surrogate-9490dbf.tar.gz`：同一 commit 的确定性归档，防止嵌套 `.gitignore` 或后续文件操作破坏原始材料。
- `upstream/UPSTREAM_SOURCE.json`：来源、commit、许可证和归档哈希。
- `upstream/UPSTREAM_TREE.sha256`：展开快照的逐文件哈希。
- `project_history/legacy-adaptation/`：整理前项目保留的 notebook 和已适配 PIC 示例；它们不是原始上游的逐字节副本。

## 使用原则

日常开发只修改项目根目录的 `src/`、`scripts/`、`tests/`、`configs/` 和 `docs/`。不要从 `reference/` 导入模块，不要把这里的旧入口当作当前工作流。

需要对照上游时，以 `UPSTREAM_SOURCE.json` 记录的 commit 为准。更新上游材料时应创建新的 commit 快照和差异记录，不要覆盖本次快照。
