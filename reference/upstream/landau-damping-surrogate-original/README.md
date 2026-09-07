# 1D-PIC Plasma Simulation + Surrogate (WIP)

Fork of [Antoine Tavant's 1D-PIC (electrostatic)](https://github.com/antoinelpp/1d-pic-electrostatic), extended with experiments on **neural surrogates** for Landau damping runs.  
Everything here is **work in progress** — models, metrics, and scripts may change.

## What’s in this repo

| Area | Contents |
|------|----------|
| **PIC** | 1D electrostatic PIC; field energy \( \sum E^2 \) (and time series) exported for ML. |
| **Data generation** | `main.py` runs a **random** \((T_e, L_x)\) sweep and saves `sweep_2d_results/data/*.npy` (not committed — see [.gitignore](.gitignore)). |
| **Surrogate** | `surrogate/`: MLP baseline (`train.py`) and **FNO-style** curve model (`train_fno.py`, `fno_model.py`). |

## Current status (2026-03)

- **PIC**: Runs and is validated against a standard Landau damping test; Landau-type setups are used to build training curves.
- **Data**: Each `.npy` stores `t`, `energy`, `te`, `lx`, `kld`, `gamma` (and friends) for surrogate loading.
- **Surrogate**
  - **MLP** (`surrogate/train.py`): pointwise prediction with optional physics-informed penalty (decay tendency). Inputs are typically \([T_e, L_x, t]\) after preprocessing.
  - **FNO path** (`surrogate/train_fno.py`): predicts a **full \(\log_{10}(E)\) curve** (length 1000 after trim) from \((T_e, L_x)\) only. Preprocessing: transient cut, log transform, Savitzky–Golay smoothing (see `data_loader.py`).
  - Train/val split and best-checkpoint saving are implemented in `train_fno.py`; validation plots are written to `fno_val_samples.png` (ignored by git unless you force-add).

### Limitations (honest)

- Surrogates are **not production-grade**: generalization depends strongly on coverage in \((T_e, L_x)\), and the FNO head design / normalization are still being iterated on.
- The **MLP** path can still show non-physical bumps for some parameters, is sensitive to high-frequency numerical noise in PIC outputs, and does not yet have systematic error benchmarks.
- **Sweep data and plots are not in Git** (large); reproduce by running `main.py` locally.

### Possible next steps

1. Stronger **output normalization** and/or **band-limited decoder** for smoother curves.
2. Optional extra inputs (e.g. `kld`, \(\gamma\)) if read from `.npy`.
3. Systematic benchmarks (per-parameter errors, failure maps).

### MLP vs FNO comparison (current snapshot)

The repo now includes a direct comparator (`surrogate/compare_models.py`) that evaluates both models on the same split/seed.

Example command:

```powershell
python surrogate/compare_models.py --split test --seed 42 --gamma_theory_floor 1e-8
```

Latest test-split snapshot (seed=42, n_test=28):

| Model | Curve MSE (log10 E) | Curve MAE (log10 E) |
|------|----------------------|---------------------|
| MLP | 0.195 | 0.329 |
| FNO-style | 0.834 | 0.693 |

Interpretation: in the current setup, the pointwise MLP baseline is more accurate than the current FNO-style curve model.  
This does **not** rule out neural operators in general; it indicates the present FNO-style head/configuration is not yet a good fit for this dataset.

Suggested plots for slides/README (good + failure case):

- `surrogate/outputs/run_20260323_160045_lambda5e-4/mlp_quick_te500_lx0.0100.png` (reasonable agreement)
- `surrogate/outputs/run_20260323_160045_lambda5e-4/mlp_quick_te2000_lx0.0100.png` (clear mismatch / stress case)

---

## Getting started

### Prerequisites

- Python 3.10+ recommended (3.13 used in development)
- Dependencies: see [`requirements.txt`](requirements.txt) (includes **PyTorch**, **NumPy**, **SciPy** for smoothing, **Matplotlib**, **Numba**, etc.)

### venv (Windows / PowerShell)

```powershell
cd path\to\1D_PIC
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### Run order (typical)

1. **Generate PIC data** (long-running; adjust `num_samples` in `main.py` as needed):

   ```powershell
   python main.py
   ```

   Outputs under `sweep_2d_results/data/` (and plots under `sweep_2d_results/plots/`).

2. **Train FNO surrogate** (from repo root, with venv activated):

   ```powershell
   python surrogate/train_fno.py
   ```

3. **Train MLP baseline** (optional):

   ```powershell
   python surrogate/train.py
   ```

4. **Quick inference test** (if paths match your data layout):

   ```powershell
   python surrogate/quick_test.py
   ```

---

## Git: what is *not* pushed

Large or regenerable artifacts are listed in [`.gitignore`](.gitignore), including:

- `sweep_2d_results/`, `*.npy`, typical `data/` folders  
- `*.png` / `*.jpg` / `*.pdf` (plot outputs)  
- `*.pth` and `surrogate/models/` (saved weights & norms)  
- `.venv/`

**Commit the code; regenerate data and weights locally** (or attach artifacts elsewhere if you need to share them).

---

## Authors & acknowledgments

- **Surrogate extensions**: (add your name / GitHub)  
- **Original PIC**: [Antoine Tavant](https://github.com/antoinelpp/1d-pic-electrostatic)


## License

MIT License (see repository license file if present).
