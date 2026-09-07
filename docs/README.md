# 文档索引

## 新成员先读

从[新成员学习路线](onboarding/README.md)开始。该入口提供阅读顺序、贯穿示例、CPU 入门练习和第一项任务的交付标准。

- [背景与核心概念](onboarding/01_BACKGROUND_zh-CN.md)：分布函数、Landau 阻尼、无量纲参数、矩方程、热流闭合与 FNO。
- [项目逻辑与代码地图](onboarding/02_PROJECT_LOGIC_zh-CN.md)：历史路线与当前任务、数据流、训练/部署流程、研究阶段和代码入口。
- [任务定义与验收标准](onboarding/03_TASK_DEFINITIONS_zh-CN.md)：输入输出合同、A/B/C 对照、因果历史、评价指标、质量门和实验交付模板。
- [术语与符号速查](onboarding/04_GLOSSARY_zh-CN.md)：物理、数值、机器学习与研究评价术语。

## 科学说明

- `science/METHOD.md`：表示、FNO、递归训练、mean anchor 与正性目标。
- `science/PHYSICS_DIAGNOSTICS.md`：密度模、Poisson、电场、场能和共振窗口。
- `science/NONLINEAR_DATA_PLAN.md`：强非线性参数域、M0--M3 标签和接受门槛。

## 数据与资产

- `data/DATA_FORMAT.md`：HDF5 数据合同。
- `data/ASSET_CATALOG.md`：当前标准路径、哈希和兼容关系。
- `data/DATA_AND_CHECKPOINTS_ORIGINAL.md`：整理前说明，仅作历史参考。

## 操作

- [协作与共享服务器入门](operations/COLLABORATION_zh-CN.md)：新成员安装、连接共享数据/模型、运行测试及协作约定。
- `operations/QUICKSTART.md`：安装、检查和推理。
- `operations/TRAINING_PIPELINE.md`：训练阶段及当前缺失的 final 封装环节。
- `operations/CUDA_DATA_GENERATION.md`：CUDA-only PIC 生成、分片和恢复。
- `operations/REPRODUCIBILITY_ORIGINAL.md`：整理前复现说明，仅作历史参考。

## 结果与历史

- `reports/PROJECT_REPORT_zh-CN.md`：贴合当前目录结构的详细中文项目报告。
- `results/VALIDATION_STATUS.md`：本次整理后的实际验证状态。
- `results/RESULTS.md`：冻结正式指标。
- `history/`：旧阶段映射、源码清理报告、发布说明和交付检查表。
- `history/UPSTREAM_LINEAGE.md`：GitHub 上游、原始 PIC 与当前改进版的谱系。

## 上游参考材料

- `../reference/README.md`：参考区使用规则。
- `../reference/DIFFERENCES.md`：当前版本与上游任务、数据和模型的差异。
