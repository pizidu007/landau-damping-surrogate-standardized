# 快速开始

以下命令均从项目根目录执行。

## 1. 激活环境

```bash
conda activate landau-pic-surrogate
python -m pip install -e .
```

## 2. 验证安装和资产

```bash
python -m pytest -q
python scripts/verify_assets.py --require-all
```

## 3. 检查数据和模型

```bash
python scripts/inspect_cache.py \
  data/processed/historical_exact/landau_deltaf_cache_x128_v193_v1.h5

python scripts/inspect_checkpoint.py \
  models/rollout/rollout_positivity_best.pt
```

## 4. Snapshot 推理

```bash
mkdir -p results/runs/manual_snapshot
python scripts/infer_snapshot.py \
  --checkpoint models/snapshot/snapshot_best.pt \
  --k 0.55 --alpha 0.040 --time 7.5 \
  --output results/runs/manual_snapshot/prediction.npz \
  --device cuda:0
```

## 5. Rollout 推理

```bash
mkdir -p results/runs/manual_rollout
python scripts/infer_rollout.py \
  --checkpoint models/rollout/rollout_positivity_best.pt \
  --cache data/processed/historical_exact/landau_deltaf_cache_x128_v193_v1.h5 \
  --cache-split test --cache-split-position 0 --cache-time-index 0 \
  --steps 30 \
  --output results/runs/manual_rollout/prediction.npz \
  --device cuda:0
```

输出 NPZ 同目录会生成 JSON 元数据。除非明确研究外推，否则不要使用 `--allow-extrapolation`。
