"""Create the static Round-4 training, offline, rollout, and phase-space figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("k0p350_a0p075", "k0p400_a0p100")
LABELS = {"k0p350_a0p075": r"$k=0.35,\ A=0.075$", "k0p400_a0p100": r"$k=0.40,\ A=0.10$"}


def rel(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads((args.result_root / "regime_dual_seed0/history.json").read_text())
    summary = json.loads((args.result_root / "regime_dual_seed0/summary.json").read_text())
    old = json.loads(args.old_summary.read_text())

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0, 0].semilogy(epochs, [row["train_loss"] for row in history], color="C0")
    axes[0, 0].set(title="Regime-balanced training", xlabel="epoch", ylabel="training objective")
    axes[0, 1].plot(epochs, [row["validation_gradient_relative_l2"] for row in history], color="C1")
    best = summary["best_validation_gradient_relative_l2"]
    axes[0, 1].axhline(best, color="0.35", ls="--", lw=1, label=f"best={best:.4f}")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set(title="Held-out validation", xlabel="epoch", ylabel=r"relative $L_2(\partial_x q)$")
    old_test = old["metrics"]["test"]["relative_l2"]
    new_test = summary["metrics"]["test"]["gradient_relative_l2"]
    bars = axes[1, 0].bar(["Round 2\ndirect FNO", "Round 4\nenriched dual FNO"], [old_test, new_test], color=["0.65", "C2"])
    axes[1, 0].bar_label(bars, fmt="%.4f")
    axes[1, 0].set(title="Sealed two-case offline test", ylabel=r"relative $L_2(\partial_x q)$")
    regime_values = {value: [] for value in (0, 1, 2)}
    for identifier in CASES:
        with np.load(args.result_root / "regime_dual_seed0/evaluation/test" / f"{identifier}.npz") as values:
            for regime in regime_values:
                selected = values["regime"] == regime
                regime_values[regime].append(rel(values["prediction"][selected], values["target"][selected]))
    x = np.arange(3); width = 0.34
    for offset, identifier in zip((-width / 2, width / 2), CASES):
        case_index = CASES.index(identifier)
        axes[1, 1].bar(x + offset, [regime_values[r][case_index] for r in x], width, label=LABELS[identifier])
    axes[1, 1].set_xticks(x, ["damping", "turnover", "rebound"])
    axes[1, 1].set(title="Offline error by physical regime", ylabel=r"relative $L_2(\partial_x q)$")
    axes[1, 1].legend(frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.savefig(args.output_dir / "round4_training_and_offline.png", dpi=190)
    plt.close(figure)

    rollout_rows = {}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    for axis, identifier in zip(axes, CASES):
        rollout_dir = args.result_root / "t40_test" / identifier
        with np.load(rollout_dir / "rollout.npz") as values:
            time = values["time"]; predicted = values["field_energy"]; truth = values["truth_field_energy"]
        rollout_rows[identifier] = json.loads((rollout_dir / "summary.json").read_text())
        axis.semilogy(time, truth, color="k", lw=1.5, label="Gkeyll truth")
        axis.semilogy(time, predicted, color="C3", lw=1.1, label="FNO-fluid closure")
        axis.set(title=LABELS[identifier], xlabel="time", ylabel=r"field energy $\frac{1}{2}\int E^2dx$")
        axis.grid(alpha=0.2); axis.legend(frameon=False)
    figure.suptitle("Sealed-test long-time field-energy evolution")
    figure.savefig(args.output_dir / "round4_t40_field_energy.png", dpi=190)
    plt.close(figure)

    for identifier in CASES:
        with np.load(args.result_root / "t40_test" / identifier / "rollout.npz") as values:
            time = values["time"]; truth = values["truth"]; prediction = values["prediction"]
        fields = ("density", "velocity", "pressure")
        figure, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
        extent = (0.0, 1.0, float(time[-1]), float(time[0]))
        for row, name in enumerate(fields):
            scale = max(float(np.max(np.abs(truth[:, row] - np.mean(truth[:, row])))), 1.0e-8)
            error_scale = max(float(np.percentile(np.abs(prediction[:, row] - truth[:, row]), 99.5)), 1.0e-8)
            images = [
                axes[row, 0].imshow(truth[:, row], aspect="auto", extent=extent, cmap="RdBu_r", vmin=-scale if row == 1 else None, vmax=scale if row == 1 else None),
                axes[row, 1].imshow(prediction[:, row], aspect="auto", extent=extent, cmap="RdBu_r", vmin=-scale if row == 1 else None, vmax=scale if row == 1 else None),
                axes[row, 2].imshow(prediction[:, row] - truth[:, row], aspect="auto", extent=extent, cmap="RdBu_r", vmin=-error_scale, vmax=error_scale),
            ]
            axes[row, 0].set_ylabel(f"{name}\ntime")
            for column, image in enumerate(images):
                figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.02)
        for column, title in enumerate(("Gkeyll truth", "FNO-fluid", "error")):
            axes[0, column].set_title(title)
            axes[-1, column].set_xlabel(r"$x/L$")
        figure.suptitle(f"Long-time moment fields: {LABELS[identifier]}")
        figure.savefig(args.output_dir / f"round4_t40_moments_{identifier}.png", dpi=170)
        plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 6.2), constrained_layout=True, sharex=True)
    for row, identifier in enumerate(CASES):
        with h5py.File(args.dataset_root / "cases" / identifier / "processed/trajectory.h5", "r") as handle:
            phase_time = np.asarray(handle["phase/time"]); x = np.asarray(handle["phase/x"])
            velocity = np.asarray(handle["phase/v"]); distribution = np.asarray(handle["phase/f"])
            k = float(handle.attrs["k"])
        phase_velocity = np.sqrt(1.0 + 3.0 * k * k) / k
        selected_v = (velocity >= phase_velocity - 1.2) & (velocity <= phase_velocity + 1.2)
        for column, target_time in enumerate((20.0, 30.0, 40.0)):
            frame = int(np.argmin(np.abs(phase_time - target_time)))
            perturbation = distribution[frame, :, selected_v].T
            perturbation -= perturbation.mean(axis=1, keepdims=True)
            limit = max(float(np.percentile(np.abs(perturbation), 99.0)), 1.0e-8)
            image = axes[row, column].imshow(perturbation, origin="lower", aspect="auto",
                extent=(x[0] * k / (2*np.pi), x[-1] * k / (2*np.pi), velocity[selected_v][0], velocity[selected_v][-1]),
                cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[row, column].axhline(phase_velocity, color="k", ls="--", lw=0.8)
            axes[row, column].set_title(f"t={phase_time[frame]:.0f}")
            axes[row, column].set_xlabel(r"$x/L$")
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.02)
        axes[row, 0].set_ylabel(f"{LABELS[identifier]}\nvelocity")
    figure.suptitle("Gkeyll truth: resonant-window distribution perturbation (not model output)")
    figure.savefig(args.output_dir / "round4_phase_space_truth.png", dpi=190)
    plt.close(figure)

    result = {"old_test_gradient_relative_l2": old_test, "new_test_gradient_relative_l2": new_test,
              "relative_reduction": 1.0 - new_test / old_test,
              "best_validation_gradient_relative_l2": best, "rollouts": rollout_rows}
    (args.output_dir / "round4_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
