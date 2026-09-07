import numpy as np
import torch
import glob
import os
from scipy.signal import savgol_filter


def load_and_preprocess_fno(data_dir="sweep_2d_results/data"):
    """
    Load PIC sweep .npy files for FNO training.

    Returns
    -------
    X : torch.Tensor [N, 2]
        Raw (Te, Lx); input normalization is done in train_fno using train split only.
    Y : torch.Tensor [N, T]
        log10(field energy) curves, fixed length T=1000.
    gamma_theory : torch.Tensor [N]
        Linear-theory γ from each file if present; else nan.
    t_curve : torch.Tensor [N, T]
        Time axis aligned with each row of Y (for γ_eff fits).
    """
    all_params = []
    all_curves = []
    all_gamma = []
    all_t_curve = []

    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        files = glob.glob(os.path.join("../", data_dir, "*.npy"))

    if not files:
        print("Error: data files not found.")
        return None, None, None, None

    print(f"Loading {len(files)} simulations for FNO...")

    target_length = 1000

    for f in files:
        data = np.load(f, allow_pickle=True).item()
        te, lx, t, energy = data["te"], data["lx"], data["t"], data["energy"]
        g = data.get("gamma", np.nan)
        if not np.isfinite(g):
            g = np.nan

        start_idx = int(len(t) * 0.05)
        e_s = energy[start_idx:]
        t_s = np.asarray(t[start_idx:], dtype=np.float64)

        log_e = np.log10(np.maximum(e_s, 1e-25))

        if len(log_e) > 51:
            log_e = savgol_filter(log_e, window_length=51, polyorder=3)

        if len(log_e) >= target_length:
            log_e = log_e[:target_length]
            t_row = t_s[:target_length]
            all_params.append([te, lx])
            all_curves.append(log_e)
            all_gamma.append(float(g))
            all_t_curve.append(t_row)

    if not all_curves:
        print("Error: no curves met length >= 1000 after preprocessing.")
        return None, None, None, None

    X = torch.tensor(all_params, dtype=torch.float32)
    Y = torch.tensor(np.stack(all_curves), dtype=torch.float32)
    gamma_theory = torch.tensor(all_gamma, dtype=torch.float32)
    t_curve = torch.tensor(np.stack(all_t_curve), dtype=torch.float64)

    print("FNO preprocessing complete (raw Te/Lx; no normalization in loader).")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    return X, Y, gamma_theory, t_curve


def load_and_preprocess(data_dir="sweep_2d_results/data", max_points_per_file=2048):
    """
    Pointwise dataset for MLP: each row is [Te, Lx, t] -> normalized log10(E).
    Same per-file smoothing as FNO; saves surrogate/models/norm_params.pth with input
    min/max and output mean/std.

    Parameters
    ----------
    max_points_per_file : int or None
        Each run is subsampled to at most this many time samples (evenly spaced along the curve).
        Default 2048 avoids ~1e5 timesteps per run -> tens of millions of rows (unusable with
        physics-informed autograd). Pass None to keep every timestep (slow).
    """
    all_inputs = []
    all_targets = []

    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        files = glob.glob(os.path.join("../", data_dir, "*.npy"))

    if not files:
        print("Error: data files not found.")
        return None, None

    for f in files:
        data = np.load(f, allow_pickle=True).item()
        te, lx, t, energy = data["te"], data["lx"], data["t"], data["energy"]
        start_idx = int(len(t) * 0.05)
        e_s = energy[start_idx:]
        t_s = np.asarray(t[start_idx:], dtype=np.float64)

        log_e = np.log10(np.maximum(e_s, 1e-25))
        if len(log_e) > 51:
            log_e = savgol_filter(log_e, window_length=51, polyorder=3)

        n = len(log_e)
        if max_points_per_file is not None and n > max_points_per_file:
            idx = np.linspace(0, n - 1, num=max_points_per_file, dtype=np.int64)
            t_s = t_s[idx]
            log_e = log_e[idx]

        for ti, yi in zip(t_s, log_e):
            all_inputs.append([te, lx, float(ti)])
            all_targets.append(float(yi))

    if not all_inputs:
        print("Error: no data points loaded.")
        return None, None

    X = torch.tensor(all_inputs, dtype=torch.float32)
    Y = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(1)

    in_min = X.min(dim=0).values
    in_max = X.max(dim=0).values
    X_norm = (X - in_min) / (in_max - in_min).clamp_min(1e-8)
    out_mean = Y.mean(dim=0)
    out_std = Y.std(dim=0).clamp_min(1e-8)
    Y_norm = (Y - out_mean) / out_std

    os.makedirs("surrogate/models", exist_ok=True)
    torch.save(
        {"min": in_min, "max": in_max, "out_mean": out_mean, "out_std": out_std},
        "surrogate/models/norm_params.pth",
    )

    mpp = "all" if max_points_per_file is None else str(max_points_per_file)
    print(f"MLP preprocessing: {X_norm.shape[0]} points, input dim 3 (max_points_per_file={mpp}).")
    return X_norm, Y_norm


if __name__ == "__main__":
    load_and_preprocess_fno()
