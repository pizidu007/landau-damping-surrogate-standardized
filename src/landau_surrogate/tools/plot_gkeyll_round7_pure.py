"""Plot Round-7 results using only Gkeyll truth and pure supervised FNO models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from landau_surrogate.data.huang2025 import load_huang_mat
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


WIDTHS = (32, 48, 64, 96, 128)
MODES = (8, 12, 16, 24, 32)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def checkpoint_path(results_root: Path, width: int, modes: int, seed: int) -> Path:
    stage = "grid" if seed == 0 else "refine_training"
    return results_root / stage / f"w{width}_m{modes}_seed{seed}" / "best.pt"


def plot_scan(results_root: Path, output_dir: Path) -> None:
    offline = np.full((len(WIDTHS), len(MODES)), np.nan)
    closed_loop = np.full_like(offline, np.nan)
    for row, width in enumerate(WIDTHS):
        for column, modes in enumerate(MODES):
            identifier = f"w{width}_m{modes}_seed0"
            offline[row, column] = read_json(
                results_root / "grid" / identifier / "summary.json"
            )["test"]["relative_l2"]
            closed_loop[row, column] = read_json(
                results_root / "t10" / identifier / "summary.json"
            )["field_energy_log10_rmse"]

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for axis, values, title in (
        (axes[0], offline, r"Supervised $\partial_xq$ relative $L_2$"),
        (axes[1], closed_loop, r"Pure-FNO closed loop: $t\leq10$ field log-RMSE"),
    ):
        image = axis.imshow(np.log10(values), origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(MODES)), MODES)
        axis.set_yticks(range(len(WIDTHS)), WIDTHS)
        axis.set_xlabel("internal Fourier modes")
        axis.set_ylabel("FNO width")
        axis.set_title(title)
        for row in range(len(WIDTHS)):
            for column in range(len(MODES)):
                axis.text(
                    column, row, f"{values[row, column]:.1e}", ha="center", va="center",
                    color="white" if np.log10(values[row, column]) < np.nanmedian(np.log10(values)) else "black",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axis, label="log10(metric)", shrink=0.86)
    figure.suptitle("Round 7 width–mode scan (seed 0, pure supervised training)")
    figure.tight_layout()
    figure.savefig(output_dir / "pure_supervised_width_mode_scan.png", dpi=200)
    plt.close(figure)


def plot_long_time(
    results_root: Path, width: int, modes: int, output_dir: Path,
) -> tuple[int, list[dict]]:
    rollouts = []
    summaries = []
    for seed in (0, 1, 2):
        identifier = f"w{width}_m{modes}_seed{seed}"
        rollouts.append(load_rollout(results_root / "t40" / identifier / "rollout.npz"))
        summaries.append(read_json(results_root / "t40" / identifier / "summary.json"))
    best_seed = int(np.argmin([value["field_energy_log10_rmse"] for value in summaries]))
    time = rollouts[0]["time"]
    truth = rollouts[0]["truth_field_energy"]
    normalizer = float(truth[0])
    predictions = np.stack([value["field_energy"] / normalizer for value in rollouts])

    figure, axis = plt.subplots(figsize=(9.4, 4.9))
    axis.fill_between(
        time, predictions.min(axis=0), predictions.max(axis=0),
        color="#56B4E9", alpha=0.22, label="pure supervised FNO: 3-seed range",
    )
    axis.semilogy(
        time, np.median(predictions, axis=0), color="#0072B2", lw=1.4, ls="--",
        label="pure supervised FNO: 3-seed median",
    )
    axis.semilogy(
        time, predictions[best_seed], color="#D55E00", lw=1.9,
        label=f"best pure supervised FNO: seed {best_seed}",
    )
    axis.semilogy(time, truth / normalizer, color="black", lw=2.0, label="Gkeyll truth")
    axis.set_xlabel("time")
    axis.set_ylabel("normalized electric-field energy")
    axis.set_title(f"Long-time pure-FNO closure: width={width}, modes={modes}")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "pure_supervised_long_time_field_energy.png", dpi=210)
    plt.close(figure)
    return best_seed, summaries


def plot_moments(
    results_root: Path, width: int, modes: int, seed: int, output_dir: Path,
) -> None:
    rollout = load_rollout(
        results_root / "t40" / f"w{width}_m{modes}_seed{seed}" / "rollout.npz"
    )
    time, truth, prediction = rollout["time"], rollout["truth"], rollout["prediction"]
    figure, axes = plt.subplots(3, 3, figsize=(12.8, 10.0), sharex=True, sharey=True)
    names = ("density n", "velocity u", "pressure p")
    for row, name in enumerate(names):
        combined = np.concatenate((truth[:, row].ravel(), prediction[:, row].ravel()))
        if row == 0:
            limits = tuple(np.quantile(combined, (0.005, 0.995)))
            cmap = "viridis"
        else:
            scale = max(float(np.quantile(np.abs(combined), 0.995)), 1.0e-8)
            limits = (-scale, scale)
            cmap = "RdBu_r"
        error = np.abs(prediction[:, row] - truth[:, row])
        for column, value, title in (
            (0, truth[:, row], "Gkeyll truth"),
            (1, prediction[:, row], "pure supervised FNO-fluid"),
        ):
            image = axes[row, column].imshow(
                value, origin="lower", extent=(0, 1, time[0], time[-1]),
                aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1],
            )
            axes[row, column].set_title(f"{name}: {title}")
            figure.colorbar(image, ax=axes[row, column], shrink=0.78)
        error_image = axes[row, 2].imshow(
            error, origin="lower", extent=(0, 1, time[0], time[-1]), aspect="auto",
            cmap="magma", vmin=0.0, vmax=max(float(np.quantile(error, 0.995)), 1.0e-12),
        )
        axes[row, 2].set_title(f"{name}: absolute error")
        figure.colorbar(error_image, ax=axes[row, 2], shrink=0.78)
        axes[row, 0].set_ylabel("time")
    for axis in axes[-1]:
        axis.set_xlabel("x/L")
    figure.suptitle(f"Best pure supervised closed loop (seed {seed})")
    figure.tight_layout()
    figure.savefig(output_dir / "pure_supervised_moments_truth_prediction_error.png", dpi=190)
    plt.close(figure)


def predict_closure(
    trajectory_path: Path, checkpoint_file: Path, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    normalization = checkpoint["normalization"]
    mean = torch.tensor(normalization["input_mean"], device=device).reshape(1, 3, 1)
    std = torch.tensor(normalization["input_std"], device=device).reshape(1, 3, 1)
    target_mean = float(normalization["target_mean"])
    target_std = float(normalization["target_std"])
    trajectory = load_huang_mat(trajectory_path)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(trajectory.state), 256):
            state = torch.from_numpy(trajectory.state[start:start + 256]).to(device)
            dummy_k = torch.zeros(len(state), device=device)
            value = model((state - mean) / std, dummy_k) * target_std + target_mean
            predictions.append(value.cpu().numpy())
    return trajectory.time, trajectory.heat_flux_gradient, np.concatenate(predictions)


def plot_closure(
    trajectory_path: Path, checkpoint_file: Path, device: torch.device, output_dir: Path,
) -> dict[str, float]:
    time, truth, prediction = predict_closure(trajectory_path, checkpoint_file, device)
    error = prediction - truth
    scale = max(float(np.quantile(np.abs(np.concatenate((truth.ravel(), prediction.ravel()))), 0.995)), 1.0e-12)
    error_scale = max(float(np.quantile(np.abs(error), 0.995)), 1.0e-12)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharex=True, sharey=True)
    for axis, value, title, limit in (
        (axes[0], truth, r"Gkeyll truth: $\partial_xq$", scale),
        (axes[1], prediction, r"pure supervised FNO: $\partial_xq$", scale),
        (axes[2], error, "prediction − truth", error_scale),
    ):
        image = axis.imshow(
            value, origin="lower", extent=(0, 1, time[0], time[-1]), aspect="auto",
            cmap="RdBu_r", vmin=-limit, vmax=limit,
        )
        axis.set_title(title)
        axis.set_xlabel("x/L")
        figure.colorbar(image, ax=axis, shrink=0.82)
    axes[0].set_ylabel("time")
    figure.suptitle("Teacher-forced heat-flux-gradient closure")
    figure.tight_layout()
    figure.savefig(output_dir / "pure_supervised_closure_truth_prediction_error.png", dpi=200)
    plt.close(figure)
    return {
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(truth)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "correlation": float(np.corrcoef(truth.ravel(), prediction.ravel())[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--modes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_scan(args.results_root, args.output_dir)
    best_seed, summaries = plot_long_time(
        args.results_root, args.width, args.modes, args.output_dir
    )
    plot_moments(args.results_root, args.width, args.modes, best_seed, args.output_dir)
    closure_metrics = plot_closure(
        args.trajectory, checkpoint_path(args.results_root, args.width, args.modes, best_seed),
        torch.device(args.device), args.output_dir,
    )
    summary = {
        "selected": {"width": args.width, "modes": args.modes, "best_t40_seed": best_seed},
        "t40_by_seed": summaries,
        "teacher_forced_closure": closure_metrics,
        "included_models": "Gkeyll truth and pure supervised FNO only",
        "rollout_fine_tuning_included": False,
    }
    (args.output_dir / "round7_pure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
