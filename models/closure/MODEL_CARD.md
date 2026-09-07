# PIC heat-flux closure FNO v1

## Intended use

This checkpoint predicts the one-dimensional electron heat flux `q(x)` from
PIC-derived density, velocity, pressure, and physical fundamental wavenumber.
The fluid source term is obtained with an exact periodic spectral derivative.

The deployed reduced model uses

```text
dqdx = 0.25 * dqdx_FNO + 0.75 * dqdx_HP
maximum retained fluid mode = 24
RK4 substep dt = 0.02
```

The HP blend and spectral cutoff were selected only on the validation split.
The five test parameter pairs were opened after all model and deployment
choices were frozen.

## Data and architecture

- Dataset: `nonlinear_cuda_pic_m03_v1`, 60 CUDA-PIC cases.
- Split: 30 train, 15 validation, 15 test cases, grouped by `(k, alpha)`.
- Training labels: three-seed mean, because single-seed third moments are noisy.
- Model: four-layer periodic 1D FNO, width 48, 24 Fourier coefficients.
- Selected parameterization: predict `q`, then differentiate spectrally.
- Training: supervised closure training followed by two-step differentiable
  multi-moment fluid rollout fine tuning.

## Sealed-test results

- Pure FNO `dqdx` relative L2: `0.308576`; correlation: `0.952298`.
- Calibrated HP `dqdx` relative L2: `0.984870`.
- Deployed hybrid `dqdx` relative L2: `0.770147`.
- All five hybrid rollouts reached `t=60` with zero density/pressure clamps.
- Hybrid field-energy log10 RMSE: `1.308028`; HP: `1.627332`.
- Density/velocity/pressure relative-L2 improvements over HP:
  `25.8% / 41.7% / 27.5%`.

## Limitations

The pure FNO closure is substantially more accurate pointwise but is not
stable for every long free rollout. The 75% HP blend is therefore a deliberate
accuracy–stability tradeoff. The hybrid does not yet reproduce every nonlinear
growth rate or bounce frequency. Shared-GPU wall time gives only `1.21x`
speedup over the present highly optimized CUDA-PIC generator, so kernel launch,
compilation, batching, and/or a cheaper local closure are needed before this is
a computationally compelling replacement.
