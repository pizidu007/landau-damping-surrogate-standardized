import argparse
import numpy as np
import torch

from data_loader import load_and_preprocess_fno
from fno_model import FNO1d
from model import LandauSurrogate
from metrics import gamma_eff_from_log10_curve


def _split_sizes(n: int, val_fraction: float, test_fraction: float):
    n_test = min(max(0, int(round(n * test_fraction))), max(0, n - 2))
    n_val = max(1, min(int(round(n * val_fraction)), max(1, n - n_test - 1)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_test = 0
        n_val = max(1, n - 1)
        n_train = n - n_val
    return n_train, n_val, n_test


def _normalize_minmax(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min).clamp_min(1e-8)


def _curve_metrics(pred, true):
    diff = pred - true
    mse = torch.mean(diff**2).item()
    mae = torch.mean(torch.abs(diff)).item()
    return mse, mae


def _gamma_eff_batch(y_log10, t_curve):
    y_np = y_log10.detach().cpu().numpy()
    t_np = t_curve.detach().cpu().numpy()
    out = []
    for i in range(y_np.shape[0]):
        out.append(gamma_eff_from_log10_curve(y_np[i], t_np[i]))
    return torch.tensor(out, dtype=torch.float64)


def _mean_rel_err(pred, ref, min_ref_abs=1e-8):
    pred = torch.as_tensor(pred, dtype=torch.float64).flatten()
    ref = torch.as_tensor(ref, dtype=torch.float64).flatten()
    mask = torch.isfinite(pred) & torch.isfinite(ref) & (torch.abs(ref) > min_ref_abs)
    if mask.sum() == 0:
        return float("nan"), float("nan"), 0
    rel = torch.abs(pred[mask] - ref[mask]) / torch.abs(ref[mask])
    return rel.mean().item(), rel.median().item(), int(mask.sum().item())


def _predict_fno(model, x_raw, fno_norm):
    in_min = fno_norm.get("in_min", fno_norm.get("min"))
    in_max = fno_norm.get("in_max", fno_norm.get("max"))
    if in_min is None or in_max is None:
        raise KeyError("FNO norm file needs in_min/in_max (or legacy min/max).")
    x = _normalize_minmax(x_raw, in_min, in_max)
    with torch.no_grad():
        y_n = model(x)
    out_mean = fno_norm.get("out_mean", torch.tensor(0.0))
    out_std = fno_norm.get("out_std", torch.tensor(1.0))
    y = y_n * out_std.clamp_min(1e-8) + out_mean
    return y


def _predict_mlp(model, x_raw, t_curve, mlp_norm):
    # x_raw: [N,2] = [Te, Lx], t_curve: [N,T]
    n, t_steps = t_curve.shape
    te = x_raw[:, 0:1].repeat(1, t_steps)
    lx = x_raw[:, 1:2].repeat(1, t_steps)

    inp = torch.stack([te, lx, t_curve.float()], dim=-1)  # [N,T,3]
    inp_flat = inp.reshape(-1, 3)
    inp_norm = _normalize_minmax(inp_flat, mlp_norm["min"], mlp_norm["max"])

    with torch.no_grad():
        y_n = model(inp_norm).reshape(n, t_steps)
    y = y_n * mlp_norm["out_std"].clamp_min(1e-8) + mlp_norm["out_mean"]
    return y


def run_compare(
    val_fraction=0.15,
    test_fraction=0.15,
    seed=42,
    target_split="test",
    gamma_theory_floor=1e-8,
):
    x_raw, y_true, gamma_theory, t_curve = load_and_preprocess_fno()
    if x_raw is None:
        raise RuntimeError("No FNO data found.")

    n = x_raw.shape[0]
    n_train, n_val, n_test = _split_sizes(n, val_fraction, test_fraction)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    if target_split == "val":
        idx = idx_val
    elif target_split == "train":
        idx = idx_train
    else:
        idx = idx_test if idx_test.numel() > 0 else idx_val

    x_sel = x_raw[idx]
    y_sel = y_true[idx]
    t_sel = t_curve[idx]
    g_sel = gamma_theory[idx]

    # Load models + norms
    fno_norm = torch.load("surrogate/models/fno_norm_params.pth", map_location="cpu")
    mlp_norm = torch.load("surrogate/models/norm_params.pth", map_location="cpu")

    fno = FNO1d(modes=8, width=64, out_steps=y_true.shape[1])
    fno.load_state_dict(torch.load("surrogate/models/fno_landau_model.pth", map_location="cpu"))
    fno.eval()

    mlp = LandauSurrogate()
    mlp.load_state_dict(torch.load("surrogate/models/landau_model.pth", map_location="cpu"))
    mlp.eval()

    y_fno = _predict_fno(fno, x_sel, fno_norm)
    y_mlp = _predict_mlp(mlp, x_sel, t_sel, mlp_norm)

    mse_fno, mae_fno = _curve_metrics(y_fno, y_sel)
    mse_mlp, mae_mlp = _curve_metrics(y_mlp, y_sel)

    g_true_eff = _gamma_eff_batch(y_sel, t_sel)
    g_fno_eff = _gamma_eff_batch(y_fno, t_sel)
    g_mlp_eff = _gamma_eff_batch(y_mlp, t_sel)

    rel_fno_theory, med_fno_theory, n1 = _mean_rel_err(g_fno_eff, g_sel, min_ref_abs=gamma_theory_floor)
    rel_mlp_theory, med_mlp_theory, n2 = _mean_rel_err(g_mlp_eff, g_sel, min_ref_abs=gamma_theory_floor)
    rel_fno_true, med_fno_true, n3 = _mean_rel_err(g_fno_eff, g_true_eff, min_ref_abs=1e-12)
    rel_mlp_true, med_mlp_true, n4 = _mean_rel_err(g_mlp_eff, g_true_eff, min_ref_abs=1e-12)

    print("=" * 70)
    print(
        f"Split={target_split}, n_total={n}, n_train={n_train}, n_val={n_val}, "
        f"n_test={n_test}, n_eval={idx.numel()}, seed={seed}"
    )
    print("=" * 70)
    print("Metric (log10 curve domain unless noted):")
    print(f"  FNO  curve MSE: {mse_fno:.6f}, curve MAE: {mae_fno:.6f}")
    print(f"  MLP  curve MSE: {mse_mlp:.6f}, curve MAE: {mae_mlp:.6f}")
    print("")
    print("Gamma metrics:")
    print(
        f"  FNO  mean/median |Δγ_eff|/|γ_theory|: {rel_fno_theory:.6f}/{med_fno_theory:.6f} "
        f"(n={n1}, |γ_theory|>{gamma_theory_floor:g}), "
        f"mean/median |Δγ_eff|/|γ_eff,true|: {rel_fno_true:.6f}/{med_fno_true:.6f} (n={n3})"
    )
    print(
        f"  MLP  mean/median |Δγ_eff|/|γ_theory|: {rel_mlp_theory:.6f}/{med_mlp_theory:.6f} "
        f"(n={n2}, |γ_theory|>{gamma_theory_floor:g}), "
        f"mean/median |Δγ_eff|/|γ_eff,true|: {rel_mlp_true:.6f}/{med_mlp_true:.6f} (n={n4})"
    )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--gamma_theory_floor", type=float, default=1e-8)
    args = parser.parse_args()
    run_compare(
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        target_split=args.split,
        gamma_theory_floor=args.gamma_theory_floor,
    )
