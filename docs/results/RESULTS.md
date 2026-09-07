
# Frozen formal results

These values summarize the supplied formal metadata. They are included for
traceability; rerunning the code requires the external data and checkpoints.

## Snapshot baseline

- task: `(k, alpha, t) -> normalized delta_f(x,v,t)`
- test case-macro relative L2: approximately `0.2060`
- horizon recursion: not applicable
- selected checkpoint SHA256:
  `a45058a899d7338ef0acbf398dac02e61ffab4c164c123d84382b2265889984b`

## One-step to multi-step rollout

- one-step Stage 8D-2A trajectory macro L2: approximately `0.2602`
- multi-step Stage 8D-2B trajectory macro L2: approximately `0.2337`
- one-step horizon-30 L2: approximately `0.3438`
- multi-step horizon-30 L2: approximately `0.2898`
- multi-step improved 25 of 26 formal test cases

## Selected final rollout parent

The selected Stage 8D-2C candidate used a 50% snapshot mean anchor.

- test trajectory case-macro relative L2: `0.2263`
- test horizon-30 case-macro relative L2: `0.2810`
- test density-mode-1 phase MAE: `0.4428 rad`

## Positivity-constrained final model

The selected Stage 9B `absolute_hinge` candidate reduced excess negative-grid
fraction from roughly `15.6%` to below `1%` in the primary group comparison,
while keeping trajectory error near the parent. It also slightly improved the
density main-mode and field diagnostics, but mass/kinetic/total-energy drift
was not solved.

## Conservative cache

Selected cache:

```text
landau_deltaf_cache_x128_v193_conservative_m02_v2.h5
SHA256 f00a6618a29c96a9f4e1c5cfea5d3372656fd925355678eecf907b3332892236
size 271,364,016 bytes
```

It uses control-volume averaging and an M0/M2 correction. The cache passed the
field/closure acceptance checks for all 110 cases.
