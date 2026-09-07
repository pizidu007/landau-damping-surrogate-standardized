
# 原始数据与 checkpoint 说明（历史存档）

> 本文是整理前的原始说明，路径和“文件是否存在”的描述可能已经过时。当前标准路径和兼容关系以 `docs/data/ASSET_CATALOG.md` 与 `artifacts/manifests/assets.json` 为准。

Do not commit these large files to the source repository. Distribute them via a
private shared drive or a release asset and verify SHA256 after download.

## Full-resolution mother HDF5

```text
landau_xv_master_110_v1.h5
SHA256 f92b58323b627ed526c29028abc0da0869173f6c380e8df4b76f424f89d1c722
shape: 110 cases x 31 times x 512 x 1537
```

## Selected conservative training cache

```text
landau_deltaf_cache_x128_v193_conservative_m02_v2.h5
SHA256 f00a6618a29c96a9f4e1c5cfea5d3372656fd925355678eecf907b3332892236
size: 271,364,016 bytes
```

## Direct snapshot checkpoint

```text
snapshot best.pt
SHA256 a45058a899d7338ef0acbf398dac02e61ffab4c164c123d84382b2265889984b
```

## Final positivity rollout checkpoint

The supplied source-review bundle did not include the final 34 MB `best.pt`.
Before public handoff, copy the formal Stage 9B final checkpoint from the
research storage and record its SHA256 in this document or in a release asset
manifest.

## Suggested local layout

```text
data/
  landau_deltaf_cache_x128_v193_conservative_m02_v2.h5
checkpoints/
  snapshot_best.pt
  rollout_positivity_best.pt
```

Both `data/` and `checkpoints/` are ignored by the repository configuration.
