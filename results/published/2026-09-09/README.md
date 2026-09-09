# 2026-09-09 公开结果快照

请从[关键结果与看图指南](../../../docs/results/KEY_RESULTS_zh-CN.md)开始，
再看[checkpoint 下载与使用](../../../docs/results/CHECKPOINTS_zh-CN.md)。

- `round11/`：九组完整验证汇总、180 条案例 CSV、配对比较、正式配置、oracle 与质量门、全部案例 PDF。
- `round10/`：三种子验证汇总、原先选定的三种模型的训练/validation/旧 test 记录、结果图谱。
- `pic_closure_v1/`：部署配置、研究汇总、八张结果图。
- `legacy/`：旧 Snapshot/Rollout 模型卡；最终 positivity Rollout 的卡为 `positivity_model_card.json`。
- `provenance.json`：复制文件的原始路径、大小与 SHA256；CSV 的来源与缺失值说明。

原图与原 JSON 不修改数值，CSV 从 Round 11 汇总导出。这里没有训练数据或权重实体。
Round 11 的所有 checkpoint 都未通过质量门；图表发布不代表模型已经达标。
本目录是固定快照，后续结果请另建日期目录。
