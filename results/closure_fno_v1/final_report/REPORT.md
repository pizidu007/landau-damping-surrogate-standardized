# PIC-derived FNO heat-flux closure v1

## Outcome

- Pure FNO test closure relative L2: `0.308576` (HP `0.984870`, improvement `68.7%`).
- Deployed 75% HP hybrid closure relative L2: `0.770147`.
- Hybrid fluid test field-energy log10 RMSE: `1.308028` (HP `1.627332`, improvement `19.6%`).
- All five sealed test parameter pairs reached `t=60` without density or pressure clamps.
- Observed shared-GPU wall-time speedup over the formal PIC generator: `1.21x`; the 10x target is not yet met.

## Interpretation

Predicting the smoother heat flux and differentiating spectrally was substantially more accurate than predicting its gradient directly. Pure FNO closure was not stable for every long rollout. Validation-only selection chose a 75% calibrated HP blend, which trades some pointwise closure accuracy for robust long-time integration. The result is a successful PIC-based closure proof of concept, but not yet a full reproduction of the paper's nonlinear bounce-frequency accuracy or computational speedup.
