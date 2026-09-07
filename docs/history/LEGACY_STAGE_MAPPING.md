
# Legacy stage mapping

The cleaned public surface uses functional names. The checkpoint schema still
contains historical stage identifiers for compatibility.

| Public module/command | Original source stage |
|---|---|
| `data.snapshot_cache` | Stage 8D-1A cache dataset |
| `models.snapshot_fno` / `train_snapshot.py` | Stage 8D-1D snapshot FNO |
| `data.rollout_cache`, `models.stepper_fno`, `train_one_step.py` | Stage 8D-2A |
| `data.rollout_windows`, `train_multistep.py` | Stage 8D-2B |
| `diagnostics.rollout` and mean-anchor logic | Stage 8D-2C |
| `losses.spectral` | Stage 9A |
| `losses.positivity`, `train_positivity.py`, `infer_rollout.py` | Stage 9B |
| `physics.closure` | Stage 10A |
| `data.conservative`, cache build/audit tools | Stage 10A-2 |

Stage names are implementation history, not the recommended user-facing
workflow.
