
# 原始复现说明（历史存档）

> 本文保留整理前的命令和研究语境。当前推理路径以 `docs/operations/QUICKSTART.md` 为准；尤其不要把保守缓存当作现有 checkpoint 的严格兼容训练缓存。

## 1. Validate artifacts

Use a local SHA256 tool, then inspect their embedded contracts:

```bash
python scripts/inspect_cache.py data/landau_deltaf_cache_x128_v193_conservative_m02_v2.h5
python scripts/inspect_checkpoint.py checkpoints/rollout_positivity_best.pt
```

## 2. Snapshot training

The historical snapshot trainer also expects the frozen parameter-IDW baseline
case-metrics CSV because the original experiment reports win rates against that
baseline.

```bash
python scripts/train_snapshot.py \
  --cache data/landau_deltaf_cache_x128_v193_conservative_m02_v2.h5 \
  --stage8d1a-case-metrics /path/to/baseline_case_metrics.csv \
  --output-dir outputs/snapshot \
  --candidate-name modes_v32 \
  --variant dual_head \
  --condition-mode phase_aware \
  --mode formal \
  --epochs 50 \
  --modes-v 32 \
  --device cuda:0
```

## 3. One-step stepper

```bash
python scripts/train_one_step.py \
  --cache data/landau_deltaf_cache_x128_v193_conservative_m02_v2.h5 \
  --output-dir outputs/one_step \
  --mode formal \
  --epochs 50 \
  --batch-size 32 \
  --device cuda:0
```

Run `--help` because the original trainer preserves all physics-loss and AMP
arguments.

## 4. Multi-step rollout

Warm-start from the selected one-step checkpoint and train with the historical
2 -> 4 -> 8 curriculum:

```bash
python scripts/train_multistep.py --help
```

## 5. Positivity fine-tuning

Warm-start from the selected rollout parent:

```bash
python scripts/train_positivity.py --help
```

The final formal run used the `absolute_hinge` positivity objective. The full
set of exact historical command lines was not included in the source-review
bundle; inspect the accepted JSON/config metadata or preserve the original run
log when preparing a publication-grade replication package.

## 6. Validation policy

- split by complete `(k, alpha)` cases;
- estimate background and global RMS only on training cases;
- choose checkpoints and candidates using validation metrics;
- report test metrics after selection;
- keep a new external holdout for future publication-grade claims because the
  existing test set was observed during multi-stage development.
