
# Method

## Representation

The normalized perturbation is decomposed along the periodic x direction:

```text
mean_delta(v,t) = mean_x(normalized_delta_f)
nonzero(x,v,t) = normalized_delta_f - mean_delta
normalized_delta_f = mean_delta + nonzero
```

`mean_delta` is the spatial Fourier mode `m=0`; `nonzero` contains all
`m != 0` modes. It is not the same quantity as the density `m=1` diagnostic,
which is computed only after integrating over velocity.

## Conditional snapshot FNO

Input: normalized and physical `(k, alpha, t)` plus coordinate channels.
Output: either one full field or two explicit heads for `mean_delta` and
`nonzero`.

## Conditional stepper FNO

Input: current normalized x-v state, `(k, alpha, t)`, coordinates and explicit
mean/nonzero channels. The model can be parameterized as:

- `absolute`: predict the next state components directly;
- `residual`: predict component increments and add them to the current state.

## Teacher forcing and free rollout

One-step training uses a true current PIC frame. Free inference starts with one
true initial state, then recursively feeds each prediction back into the model.
Multi-step unrolled training reduces the resulting exposure bias.

## Mean anchor

The selected Stage 8D-2C rollout blended only the predicted spatial mean with
the direct snapshot mean:

```text
m_final = (1-beta) * m_stepper + beta * m_snapshot
```

with `beta=0.5`. The nonzero component remains the stepper prediction.

## Positivity objective

The perturbation may be negative. The physical constraint is applied to the
reconstructed complete distribution:

```text
f_pred = f_bg + delta_f_pred >= 0
```

The selected soft hinge objective greatly reduces excess negative-grid
fraction but does not enforce exact mass or total-energy conservation.
