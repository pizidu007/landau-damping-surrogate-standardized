# 数据目录

正式数据已经迁移到：

```text
/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized
```

本目录只保留兼容符号链接和说明，不再保存大型数据实体。可用环境变量
`LANDAU_DATA_ROOT` 覆盖默认数据根目录。

## `processed/historical_exact`

历史精确步长抽样缓存。现有 Snapshot 和 Stage 9B Rollout checkpoint 的 `cache_contract.cache_sha256` 指向该文件。复现既有模型结果时必须使用它。

## `processed/conservative_m02`

控制体积平均加 M0/M2 修正的保守缓存。用于 Poisson/能量闭合审计和未来重训练。它与历史缓存形状相同但数值语义不同，不能仅凭形状认为兼容。

原始 512×1537 mother HDF5 位于外部数据根的 `raw/pic/`。禁止用派生缓存覆盖原始数据。
