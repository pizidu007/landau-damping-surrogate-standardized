import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data_loader import load_and_preprocess_fno
from fno_model import FNO1d
from metrics import gamma_eff_from_log10_curve, batch_gamma_metrics


def _split_sizes(n: int, val_fraction: float, test_fraction: float):
    """Train / val / test counts; test may be 0 for very small n."""
    n_test = min(max(0, int(round(n * test_fraction))), max(0, n - 2))
    n_val = max(1, min(int(round(n * val_fraction)), max(1, n - n_test - 1)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_test = 0
        n_val = max(1, n - 1)
        n_train = n - n_val
    return n_train, n_val, n_test


def _normalize_inputs(X: torch.Tensor, in_min: torch.Tensor, in_max: torch.Tensor) -> torch.Tensor:
    denom = (in_max - in_min).clamp_min(1e-8)
    return (X - in_min) / denom


def _normalize_outputs(Y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (Y - mean) / std.clamp_min(1e-8)


def _denormalize_outputs(Yn: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return Yn * std.clamp_min(1e-8) + mean


def _gamma_eff_batch(y_log10: torch.Tensor, t_curve: torch.Tensor):
    """y_log10 [B,T], t_curve [B,T] -> list of gamma_eff (float)."""
    y_np = y_log10.detach().cpu().numpy()
    t_np = t_curve.detach().cpu().numpy()
    out = []
    for i in range(y_np.shape[0]):
        out.append(gamma_eff_from_log10_curve(y_np[i], t_np[i]))
    return torch.tensor(out, dtype=torch.float64)


def evaluate_split(
    model,
    x_norm,
    y_raw,
    gamma_ref,
    t_curve,
    mean,
    std,
    criterion,
    device,
):
    """Curve MSE in log10 space; mean relative error on γ_eff vs stored theory γ."""
    model.eval()
    with torch.no_grad():
        pred_n = model(x_norm.to(device))
        pred_raw = _denormalize_outputs(pred_n.cpu(), mean, std)
        mse = criterion(pred_raw, y_raw).item()
        gamma_pred = _gamma_eff_batch(pred_raw, t_curve)
    rel_gamma, n_gamma = batch_gamma_metrics(gamma_pred, gamma_ref)
    return mse, rel_gamma, n_gamma


def train_fno(
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    epochs: int = 200,
    plot_val_samples: int = 5,
):
    # 1. Load raw arrays
    pack = load_and_preprocess_fno()
    if pack[0] is None:
        raise RuntimeError("load_and_preprocess_fno() returned no data.")
    X, Y_raw, gamma_theory, t_curve = pack

    n = X.shape[0]
    if n < 2:
        raise RuntimeError(f"Need at least 2 simulations, got {n}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train, n_val, n_test = _split_sizes(n, val_fraction, test_fraction)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)

    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    X_train, Y_train = X[idx_train], Y_raw[idx_train]

    in_min = X_train.min(dim=0).values
    in_max = X_train.max(dim=0).values
    out_mean = Y_train.mean(dim=0)
    out_std = Y_train.std(dim=0)

    X_norm = _normalize_inputs(X, in_min, in_max)
    Y_norm = _normalize_outputs(Y_raw, out_mean, out_std)

    os.makedirs("surrogate/models", exist_ok=True)
    norm_path = "surrogate/models/fno_norm_params.pth"
    torch.save(
        {
            "in_min": in_min,
            "in_max": in_max,
            "out_mean": out_mean,
            "out_std": out_std,
            "seed": seed,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
        },
        norm_path,
    )
    print(f"Saved normalization (train-only stats) to {norm_path}")

    train_loader = DataLoader(
        TensorDataset(X_norm[idx_train], Y_norm[idx_train]),
        batch_size=max(1, min(32, n_train)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_norm[idx_val], Y_norm[idx_val]),
        batch_size=max(1, min(32, n_val)),
        shuffle=False,
    )

    print(
        f"Split: train={n_train}, val={n_val}, test={n_test} "
        f"(seed={seed}, val_frac={val_fraction}, test_frac={test_fraction})"
    )

    model = FNO1d(modes=8, width=64, out_steps=Y_raw.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None

    model_path = "surrogate/models/fno_landau_model.pth"

    print("Starting FNO training (loss on normalized targets; metrics in log10(E))...")

    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_mse_log, val_rel_g, n_g_val = evaluate_split(
                model,
                X_norm[idx_val],
                Y_raw[idx_val],
                gamma_theory[idx_val],
                t_curve[idx_val],
                out_mean,
                out_std,
                criterion,
                device,
            )

        if val_mse_log < best_val_mse:
            best_val_mse = val_mse_log
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            g_str = f"{val_rel_g:.6f}" if np.isfinite(val_rel_g) else "nan"
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"train norm MSE: {avg_train:.6f} | "
                f"val log10 MSE: {val_mse_log:.6f} | "
                f"val |Δγ|/|γ| (mean): {g_str} (n={n_g_val}) "
                f"(best val log10 MSE: {best_val_mse:.6f})"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Best val log10 MSE: {best_val_mse:.6f}. Saved to {model_path}")

    # Final metrics on val and test (curve MSE in log10; γ relative error)
    model.eval()
    for name, idx in [("val", idx_val), ("test", idx_test)]:
        if idx.numel() == 0:
            continue
        with torch.no_grad():
            mse_log, rel_g, n_g = evaluate_split(
                model,
                X_norm[idx],
                Y_raw[idx],
                gamma_theory[idx],
                t_curve[idx],
                out_mean,
                out_std,
                criterion,
                device,
            )
        g_str = f"{rel_g:.6f}" if np.isfinite(rel_g) else "nan"
        print(
            f"[{name}] log10 curve MSE: {mse_log:.6f} | "
            f"mean |Δγ|/|γ|: {g_str} (valid γ count: {n_g})"
        )

    # Plot validation curves (denormalized log10 E)
    n_plot = min(plot_val_samples, len(idx_val))
    if n_plot == 0:
        return

    cols = min(3, n_plot)
    rows = (n_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    model.eval()
    with torch.no_grad():
        for i in range(n_plot):
            j = int(idx_val[i])
            x = X_norm[j : j + 1].to(device)
            y_true = Y_raw[j].cpu().numpy()
            pred = _denormalize_outputs(model(x).cpu(), out_mean, out_std).numpy().squeeze()
            ax = axes[i]
            ax.plot(y_true, label="Target (log10 E)", alpha=0.7)
            ax.plot(pred, label="FNO pred", linewidth=2)
            ax.set_title(f"Val idx {i + 1}/{n_plot}")
            ax.legend(fontsize=8)
            ax.set_xlabel("index")
            ax.set_ylabel("log10(E)")

    for j in range(n_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("FNO: validation set (held-out simulations)", fontsize=12)
    fig.tight_layout()
    out_png = "fno_val_samples.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {n_plot} validation curves to {out_png}")


if __name__ == "__main__":
    train_fno()
