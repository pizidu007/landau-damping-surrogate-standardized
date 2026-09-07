"""Create paper-aligned Gkeyll truth, HP-closure, and FNO-closure comparisons."""
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
from landau_surrogate.fluid.multimoment_1d import spectral_filter
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def select_median_seed(results_root: Path) -> tuple[int, list[dict]]:
    summaries = [
        read_json(results_root / "t40" / f"C1_balanced_seed{seed}" / "summary.json")
        for seed in (0, 1, 2)
    ]
    ordering = np.argsort([item["field_energy_log10_rmse"] for item in summaries])
    return int(ordering[1]), summaries


def plot_field_energy(
    truth: np.ndarray, hp: dict[str, np.ndarray], fno: dict[str, np.ndarray], output_dir: Path,
) -> None:
    normalizer = float(truth[0])
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharex=True, sharey=True)
    for axis, rollout, color, title, label in (
        (axes[0], hp, "#0072B2", "Fluid + HP", "HP closure"),
        (axes[1], fno, "#D55E00", "Fluid + FNO", "FNO closure"),
    ):
        axis.semilogy(hp["time"], truth / normalizer, color="black", lw=2.0, label="Gkeyll truth")
        axis.semilogy(
            rollout["time"], rollout["field_energy"] / normalizer,
            color=color, lw=1.8, label=label,
        )
        axis.set_xlabel("time")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.22)
        axis.legend()
    axes[0].set_ylabel("normalized electric-field energy")
    figure.suptitle("Nonlinear Landau damping: paper-aligned closure comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "paper_field_energy_truth_hp_fno.png", dpi=210)
    plt.close(figure)


def plot_moments(
    time: np.ndarray, truth: np.ndarray, hp: np.ndarray, fno: np.ndarray, output_dir: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.3, 9.7), sharex=True, sharey=True)
    names = ("density n", "velocity u", "pressure p")
    columns = ((truth, "Gkeyll truth"), (hp, "Fluid + HP"), (fno, "Fluid + FNO"))
    for row, name in enumerate(names):
        combined = np.concatenate([value[:, row].ravel() for value, _ in columns])
        if row == 1:
            scale = max(float(np.quantile(np.abs(combined), 0.995)), 1.0e-8)
            limits, cmap = (-scale, scale), "RdBu_r"
        else:
            limits = tuple(np.quantile(combined, (0.005, 0.995)))
            cmap = "viridis"
        for column, (value, title) in enumerate(columns):
            image = axes[row, column].imshow(
                value[:, row], origin="lower", extent=(0, 1, time[0], time[-1]),
                aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1],
            )
            axes[row, column].set_title(f"{name}: {title}")
            figure.colorbar(image, ax=axes[row, column], shrink=0.78)
        axes[row, 0].set_ylabel("time")
    for axis in axes[-1]:
        axis.set_xlabel("x/L")
    figure.suptitle("Lower-order moments: Gkeyll, HP closure, and FNO closure")
    figure.tight_layout()
    figure.savefig(output_dir / "paper_moments_truth_hp_fno.png", dpi=195)
    plt.close(figure)


def plot_errors(
    time: np.ndarray, truth: np.ndarray, hp: np.ndarray, fno: np.ndarray, output_dir: Path,
) -> None:
    fno_error, hp_error = np.abs(fno - truth), np.abs(hp - truth)
    figure, axes = plt.subplots(3, 2, figsize=(8.8, 10.0), sharex=True, sharey=True)
    names = ("density n", "velocity u", "pressure p")
    for row, name in enumerate(names):
        scale = max(
            float(np.quantile(np.concatenate((fno_error[:, row].ravel(), hp_error[:, row].ravel())), 0.995)),
            1.0e-12,
        )
        for column, (value, title) in enumerate(
            ((fno_error, "|Fluid + FNO − truth|"), (hp_error, "|Fluid + HP − truth|"))
        ):
            image = axes[row, column].imshow(
                value[:, row], origin="lower", extent=(0, 1, time[0], time[-1]),
                aspect="auto", cmap="magma", vmin=0.0, vmax=scale,
            )
            axes[row, column].set_title(f"{name}: {title}")
            figure.colorbar(image, ax=axes[row, column], shrink=0.78)
        axes[row, 0].set_ylabel("time")
    for axis in axes[-1]:
        axis.set_xlabel("x/L")
    figure.suptitle("Absolute moment errors (paper Fig. 6 ordering)")
    figure.tight_layout()
    figure.savefig(output_dir / "paper_moment_errors_fno_hp.png", dpi=195)
    plt.close(figure)


@torch.no_grad()
def closure_fields(
    trajectory_path: Path, checkpoint_path: Path, hp_scale: float,
    deployment_mode: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory = load_huang_mat(trajectory_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    normalization = checkpoint["normalization"]
    mean = torch.tensor(normalization["input_mean"], device=device).reshape(1, 3, 1)
    std = torch.tensor(normalization["input_std"], device=device).reshape(1, 3, 1)
    target_mean = float(normalization["target_mean"])
    target_std = float(normalization["target_std"])
    truth = spectral_filter(
        torch.from_numpy(trajectory.heat_flux_gradient), deployment_mode
    ).numpy()
    hp_values, fno_values = [], []
    mode = torch.arange(
        trajectory.state.shape[-1] // 2 + 1, device=device, dtype=torch.float32
    )
    wave_number = trajectory.k * mode
    for start in range(0, len(trajectory.state), 256):
        physical = torch.from_numpy(trajectory.state[start:start + 256]).to(device)
        physical = spectral_filter(physical, deployment_mode)
        temperature = physical[:, 2] - physical[:, 0]
        hp = hp_scale * torch.fft.irfft(
            wave_number[None] * torch.fft.rfft(temperature.float(), dim=-1),
            n=temperature.shape[-1], dim=-1,
        )
        prediction = model(
            (physical - mean) / std, torch.zeros(len(physical), device=device)
        ) * target_std + target_mean
        hp_values.append(spectral_filter(hp, deployment_mode).cpu().numpy())
        fno_values.append(spectral_filter(prediction, deployment_mode).cpu().numpy())
    return trajectory.time, truth, np.concatenate(hp_values), np.concatenate(fno_values)


def plot_closures(
    time: np.ndarray, truth: np.ndarray, hp: np.ndarray, fno: np.ndarray, output_dir: Path,
) -> dict[str, dict[str, float]]:
    scale = max(
        float(np.quantile(np.abs(np.concatenate((truth.ravel(), hp.ravel(), fno.ravel()))), 0.995)),
        1.0e-12,
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), sharex=True, sharey=True)
    for axis, value, title in (
        (axes[0], truth, r"Gkeyll truth: $\partial_xq$"),
        (axes[1], hp, r"HP closure: $\partial_xq$"),
        (axes[2], fno, r"FNO closure: $\partial_xq$"),
    ):
        image = axis.imshow(
            value, origin="lower", extent=(0, 1, time[0], time[-1]), aspect="auto",
            cmap="RdBu_r", vmin=-scale, vmax=scale,
        )
        axis.set_title(title)
        axis.set_xlabel("x/L")
        figure.colorbar(image, ax=axis, shrink=0.8)
    axes[0].set_ylabel("time")
    figure.suptitle("Deployment-filtered heat-flux-gradient closures")
    figure.tight_layout()
    figure.savefig(output_dir / "paper_closure_truth_hp_fno.png", dpi=205)
    plt.close(figure)

    def values(prediction: np.ndarray) -> dict[str, float]:
        error = prediction - truth
        return {
            "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(truth)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "correlation": float(np.corrcoef(truth.ravel(), prediction.ravel())[0, 1]),
        }

    return {"hp": values(hp), "fno": values(fno)}


def plot_stability_screen(results_root: Path, round7_root: Path, output_dir: Path) -> None:
    t10_paths = {
        "baseline": [round7_root / "t10/w128_m32_seed0/summary.json"] + [
            round7_root / f"refine_t10/w128_m32_seed{seed}/summary.json" for seed in (1, 2)
        ],
        "C1 balanced": [results_root / f"t10/C1_balanced_seed{seed}/summary.json" for seed in (0, 1, 2)],
        "C2 deploy-focused": [results_root / f"t10/C2_deployment_focused_seed{seed}/summary.json" for seed in (0, 1, 2)],
        "C3 deploy-only": [results_root / f"t10/C3_deployment_only_seed{seed}/summary.json" for seed in (0, 1, 2)],
        "C4 work/RMS": [results_root / f"t10/C4_physics_work_seed{seed}/summary.json" for seed in (0, 1, 2)],
    }
    t40_paths = {
        "baseline": [round7_root / f"t40/w128_m32_seed{seed}/summary.json" for seed in (0, 1, 2)],
        "C1 balanced": [results_root / f"t40/C1_balanced_seed{seed}/summary.json" for seed in (0, 1, 2)],
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for axis, groups, title in (
        (axes[0], t10_paths, r"Short closed loop ($t\leq10$)"),
        (axes[1], t40_paths, r"Long closed loop ($t\leq40$)"),
    ):
        for index, (label, paths) in enumerate(groups.items()):
            values = np.asarray([
                read_json(path)["field_energy_log10_rmse"] for path in paths
            ])
            offsets = np.linspace(-0.10, 0.10, len(values))
            axis.scatter(index + offsets, values, s=52, color="#0072B2", zorder=3)
            axis.hlines(np.median(values), index - 0.20, index + 0.20, color="#D55E00", lw=2.0)
        axis.set_yscale("log")
        axis.set_xticks(range(len(groups)), list(groups), rotation=22, ha="right")
        axis.set_ylabel("field-energy log10 RMSE")
        axis.set_title(title)
        axis.grid(True, axis="y", which="both", alpha=0.22)
    figure.suptitle("Seed stability of width=128, modes=32 training constraints")
    figure.tight_layout()
    figure.savefig(output_dir / "stability_constraint_seed_comparison.png", dpi=205)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hp-scale", type=float, default=float(np.sqrt(8.0 / np.pi)))
    parser.add_argument("--deployment-mode", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--round7-root", type=Path)
    parser.add_argument("--fno-rollout", type=Path)
    parser.add_argument("--fno-checkpoint", type=Path)
    parser.add_argument("--fno-summary", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed, summaries = select_median_seed(args.results_root)
    hp = load_rollout(args.results_root / "hp_standard_t40" / "rollout.npz")
    if args.fno_rollout is None:
        fno_rollout = args.results_root / "t40" / f"C1_balanced_seed{seed}" / "rollout.npz"
        checkpoint = args.results_root / "training" / f"C1_balanced_seed{seed}" / "best.pt"
        selected_summary = summaries[seed]
        selection = "median C1 t40 field-energy error across seeds 0, 1, 2"
    else:
        if args.fno_checkpoint is None or args.fno_summary is None:
            parser.error("explicit --fno-rollout requires --fno-checkpoint and --fno-summary")
        fno_rollout = args.fno_rollout
        checkpoint = args.fno_checkpoint
        selected_summary = read_json(args.fno_summary)
        selection = "explicit retained best pure-supervised FNO"
    fno = load_rollout(fno_rollout)
    truth = fno["truth_field_energy"]
    plot_field_energy(truth, hp, fno, args.output_dir)
    plot_moments(fno["time"], fno["truth"], hp["prediction"], fno["prediction"], args.output_dir)
    plot_errors(fno["time"], fno["truth"], hp["prediction"], fno["prediction"], args.output_dir)
    closure_time, closure_truth, closure_hp, closure_fno = closure_fields(
        args.trajectory, checkpoint, args.hp_scale, args.deployment_mode,
        torch.device(args.device),
    )
    closure_metrics = plot_closures(
        closure_time, closure_truth, closure_hp, closure_fno, args.output_dir
    )
    if args.round7_root is not None:
        plot_stability_screen(args.results_root, args.round7_root, args.output_dir)
    summary = {
        "representative_fno_seed": seed if args.fno_rollout is None else None,
        "selection": selection,
        "selected_fno_t40": selected_summary,
        "fno_t40_by_seed": summaries,
        "hp_t40": read_json(args.results_root / "hp_standard_t40" / "summary.json"),
        "closure_metrics": closure_metrics,
        "hp_scale": args.hp_scale,
        "deployment_mode": args.deployment_mode,
        "rollout_finetuning_included": False,
    }
    (args.output_dir / "round8_paper_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
