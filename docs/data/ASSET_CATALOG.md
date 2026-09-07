# 数据与模型资产目录

## 正式资产

| 类型 | 标准路径 | 说明 |
|---|---|---|
| PIC mother | `$LANDAU_DATA_ROOT/raw/pic/landau_xv_master_110_v1.h5` | 110-case 全分辨率母数据 |
| 历史缓存 | `$LANDAU_DATA_ROOT/processed/historical_exact/landau_deltaf_cache_x128_v193_v1.h5` | 现有 checkpoint 的严格兼容缓存 |
| 保守缓存 | `$LANDAU_DATA_ROOT/processed/conservative_m02/landau_deltaf_cache_x128_v193_conservative_m02_v2.h5` | M0/M2 闭合与未来重训练 |
| Snapshot | `models/snapshot/snapshot_best.pt` | 单时刻直接重建 |
| Rollout | `models/rollout/rollout_positivity_best.pt` | 最终正性递归模型 |

机器可读哈希见 `artifacts/manifests/assets.json`。

## 兼容性要求

两个缓存均为 110 case、31 时刻、128×193 网格，使用同一训练归一化尺度，但压缩语义不同。现有 checkpoint 内嵌缓存哈希为历史缓存的 `84fd51...`。保守缓存不是它的无差别替代品。

case 划分为：70 train、14 validation、26 test。所有时刻始终跟随所属 case，不跨 split。

默认数据根为：

```text
/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized
```

可通过 `LANDAU_DATA_ROOT` 覆盖。源码树中的旧缓存路径为兼容符号链接。

## Mother 数据

```text
landau_xv_master_110_v1.h5
shape: 110 × 31 × 512 × 1537
SHA256: f92b58323b627ed526c29028abc0da0869173f6c380e8df4b76f424f89d1c722
```

该文件已经迁移到外部受控数据根，不应提交到源码仓库。
