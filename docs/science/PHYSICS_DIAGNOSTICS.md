
# Physics diagnostics

## Density perturbation

```text
delta_n(x,t) = integral_v delta_f(x,v,t) dv
```

The first density Fourier mode is emphasized because the initial condition
excites the fundamental spatial wave. For a domain `L=2*pi/k`, the physical
wave number `k` corresponds to the discrete mode index `m=1`.

## Mode amplitude and phase

```text
A_1(t) = |delta_n_hat_1(t)|
phase_error = arg(delta_n_hat_1_pred * conj(delta_n_hat_1_truth))
```

Phase is numerically ill-conditioned when the truth amplitude is close to zero.
Low-amplitude times should be masked rather than interpreted as catastrophic
model failure.

## Periodic Poisson contract

```text
rho = 1 - n_e
ik_m E_hat_m = rho_hat_m
E_hat_0 = 0
```

The mother-density audit obtained worst relative errors no larger than
`4.61e-6` for reconstructed electric field and `3.13e-6` for field energy,
with the internal Fourier Poisson residual near `1e-18`. These numbers validate
the diagnostic implementation, not the neural-model accuracy.

## Field energy

```text
W_E(t) = 1/2 * integral_x E(x,t)^2 dx
```

Deep minima in logarithmic field-energy plots often occur near electric-field
zero crossings. They should be interpreted together with the mode amplitude,
envelope and phase.

## Resonance window

The training/evaluation mask is case dependent:

```text
v_phase = omega_fit / k
|v - v_phase| <= 0.5
```

The `+-0.5` width is an engineering evaluation window, not a sharp theoretical
resonance boundary.
