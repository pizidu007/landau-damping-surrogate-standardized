
# Compact x-v cache format

The compact HDF5 cache is a **derived, read-only** training product. It must not
replace or modify the full-resolution mother HDF5.

## Root attributes

Important attributes include:

- `status = COMPLETE`
- `dataset_version`
- `source_dataset_sha256`
- `normalization_sha256`
- `x_stride`, `v_stride`

## Groups

### `/grids`

- `normalized_x`: periodic x coordinate, shape `[Nx]`
- `velocity`: velocity coordinate, shape `[Nv]`
- `phase_time`: output times, shape `[Nt]`

The formal model grid is `Nx=128`, `Nv=193`, `Nt=31`.

### `/cases`

- `case_id`: UTF-8 identifier such as `k0p45_a0p005`
- `k`, `alpha`
- `phase_velocity`
- `split_code`: train=0, validation=1, test=2
- `group_code`: validation/interstitial/boundary/legacy grouping

### `/samples`

Samples are ordered case-major and time-contiguous:

```text
sample_index = case_index * Nt + time_index
```

Datasets include:

- `field`: normalized perturbation, shape `[Ncase*Nt, Nx, Nv]`
- `condition`: normalized `(k, alpha, t)`, shape `[Nsample, 3]`
- `physical_condition`: physical `(k, alpha, t)`, shape `[Nsample, 3]`
- `case_index`, `time_index`, `split_code`, `group_code`

## Normalization

```text
delta_f = f - f_bg
sigma_train = sqrt(mean_train(delta_f^2))
normalized_delta_f = delta_f / sigma_train
```

`f_bg` and `sigma_train` are estimated from training cases only and frozen for
validation/test cases. A single global scale is used so the physical amplitude
differences between different `alpha` values are retained.
