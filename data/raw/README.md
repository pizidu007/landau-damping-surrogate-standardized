# 原始数据占位说明

当前标准化项目不包含全分辨率 mother HDF5。原始数据应位于独立、只读、受控的数据存储中，并通过 SHA256 标识；不要复制进源码 Git，也不要用 `processed/` 中的派生缓存覆盖它。

已知 mother 数据标识见 `docs/data/ASSET_CATALOG.md`。
