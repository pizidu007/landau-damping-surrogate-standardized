# PIC heat-flux FNO v2 (history-5 research checkpoint)

`closure_fno_pic_v2_history5.pt` is the best **offline closure** checkpoint
from the second experimental round.  It consumes five consecutive
`(density, velocity, pressure)` states plus the normalized fundamental wave
number, predicts the mode-24-filtered central heat flux, and obtains its
gradient spectrally.

- validation relative L2: `0.172714`
- reused test relative L2: `0.168382`
- reused test correlation: `0.984183`
- SHA-256: `f4386b238df3c57f4ddf83a9454930930cbb01ed284bfbd457347a99fa958238`

This checkpoint is **not the deployment default**.  Pure-FNO validation
rollouts become unstable between approximately `t=24` and `t=34`.  A 75% HP
blend is stable through `t=60`, but its validation field-energy log10 RMSE
(`1.550`) is worse than the v1 deployment checkpoint (`1.393`).  Therefore
`deployment.json` intentionally remains on the v1 FNO/HP hybrid.

The evaluation set was already opened in the first round, so the v2 test
number is a reused regression measurement rather than a newly sealed model
selection result.  Use validation metrics for all v2 design decisions.
