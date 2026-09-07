"""Generate the four publication-style result groups for closure research."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def representative_npz(directory: Path, preferred: str = "k0p350_a0p100") -> Path:
    selected = directory / f"{preferred}.npz"
    if selected.is_file():
        return selected
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(directory)
    return files[0]


def group1_closure(evaluation_dir: Path, output: Path) -> str:
    output.mkdir(parents=True, exist_ok=True)
    path = representative_npz(evaluation_dir)
    data = np.load(path)
    truth = data["target"]
    prediction = data["prediction"]
    time_values = data["time"]
    # Match the convention used by Huang et al.: time runs left-to-right and
    # the periodic spatial coordinate is vertical.  Arrays are stored [t, x].
    extent = [float(time_values[0]), float(time_values[-1]), 0.0, 1.0]
    scale = float(np.quantile(np.abs(truth), 0.995))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    signal_images = []
    for axis, value, title in zip(axes[:2], (truth, prediction), ("PIC truth", "FNO closure")):
        image = axis.imshow(
            value.T, origin="lower", aspect="auto", extent=extent,
            cmap="RdBu_r", vmin=-scale, vmax=scale,
        )
        signal_images.append(image)
        axis.set_title(title)
        axis.set_xlabel("time")
    absolute_error = np.abs(prediction - truth)
    error_scale = max(float(np.quantile(absolute_error, 0.995)), 1.0e-12)
    error_image = axes[2].imshow(
        absolute_error.T, origin="lower", aspect="auto", extent=extent,
        cmap="magma", vmin=0.0, vmax=error_scale,
    )
    axes[2].set_title("absolute error")
    axes[2].set_xlabel("time")
    axes[0].set_ylabel("x/L")
    fig.colorbar(signal_images[-1], ax=axes[:2], shrink=0.8, label=r"$\partial_x q$")
    fig.colorbar(error_image, ax=axes[2], shrink=0.8, label=r"$|\Delta\partial_x q|$")
    fig.subplots_adjust(wspace=0.18, right=0.96)
    fig.savefig(output / "dqdx_truth_prediction_error_xt.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for axis, target_time in zip(axes, (10.0, 30.0, 50.0)):
        index = int(np.argmin(np.abs(time_values - target_time)))
        axis.plot(np.arange(truth.shape[1]) / truth.shape[1], truth[index], label="PIC")
        axis.plot(np.arange(truth.shape[1]) / truth.shape[1], prediction[index], label="FNO")
        axis.set_title(f"t={time_values[index]:.1f}")
        axis.set_xlabel("x/L")
    axes[0].set_ylabel(r"$\partial_x q$")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(output / "dqdx_lineouts.png", dpi=180)
    plt.close(fig)
    return path.stem


def group2_rollout(model_dir: Path, hp_dir: Path, zero_dir: Path, output: Path, pair_id: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        "Fluid+FNO/HP": np.load(model_dir / f"{pair_id}.npz"),
        "Fluid+HP": np.load(hp_dir / f"{pair_id}.npz"),
        "Fluid+zero": np.load(zero_dir / f"{pair_id}.npz"),
    }
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    first = next(iter(datasets.values()))
    axis.semilogy(first["time"], first["truth_field_energy"], color="black", linewidth=2, label="PIC")
    for label, data in datasets.items():
        axis.semilogy(data["time"], data["field_energy"], label=label)
    axis.set_xlabel("time")
    axis.set_ylabel("electric-field energy")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "electric_energy_rollout.png", dpi=180)
    plt.close(fig)

    model = datasets["Fluid+FNO/HP"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    names = ("density", "velocity", "pressure")
    for row, name in enumerate(names):
        truth = model["truth"][:, row]
        prediction = model["prediction"][:, row]
        scale = float(np.quantile(np.abs(truth - np.mean(truth)), 0.995))
        center = float(np.mean(truth))
        axes[row, 0].imshow(truth, aspect="auto", cmap="RdBu_r", vmin=center-scale, vmax=center+scale)
        axes[row, 1].imshow(prediction, aspect="auto", cmap="RdBu_r", vmin=center-scale, vmax=center+scale)
        axes[row, 0].set_ylabel(name)
    axes[0, 0].set_title("PIC")
    axes[0, 1].set_title("Fluid+FNO/HP")
    fig.tight_layout()
    fig.savefig(output / "fluid_moments_xt.png", dpi=180)
    plt.close(fig)


def group3_stability(model_dir: Path, hp_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_rows = read_rows(model_dir / "pair_metrics.csv")
    hp_rows = read_rows(hp_dir / "pair_metrics.csv")
    labels = [row["pair_id"] for row in model_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x - width / 2, [float(row["field_energy_log10_rmse"]) for row in model_rows], width, label="FNO/HP")
    axis.bar(x + width / 2, [float(row["field_energy_log10_rmse"]) for row in hp_rows], width, label="HP")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("field-energy log10 RMSE")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "rollout_error_by_parameter.png", dpi=180)
    plt.close(fig)

    path = representative_npz(model_dir)
    data = np.load(path)
    drift = data["total_energy"] / data["total_energy"][0] - 1.0
    truth_drift = data["truth_total_energy"] / data["truth_total_energy"][0] - 1.0
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(data["time"], drift, label="Fluid+FNO")
    axis.plot(data["time"], truth_drift, label="PIC")
    axis.set_xlabel("time")
    axis.set_ylabel("relative total-energy change")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "total_energy_drift.png", dpi=180)
    plt.close(fig)


def group4_phase(run_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    matches = sorted((run_dir / "cases").glob("k0p350_a0p100_s00.h5"))
    if not matches:
        matches = sorted((run_dir / "cases").glob("*.h5"))[:1]
    with h5py.File(matches[0], "r") as handle:
        phase = np.asarray(handle["phase_space/f"], dtype=np.float32)
        phase_time = np.asarray(handle["grids/phase_time"], dtype=np.float64)
        velocity = np.asarray(handle["grids/phase_velocity"], dtype=np.float64)
        energy_time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
        field_energy = np.asarray(handle["energies/field_energy"], dtype=np.float64)
    selected_times = (0.0, 20.0, 30.0, 50.0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for axis, target_time in zip(axes, selected_times):
        index = int(np.argmin(np.abs(phase_time - target_time)))
        axis.imshow(
            phase[index].T, origin="lower", aspect="auto",
            extent=(0.0, 1.0, float(velocity[0]), float(velocity[-1])), cmap="magma",
        )
        axis.set_title(f"t={phase_time[index]:.0f}")
        axis.set_xlabel("x/L")
    axes[0].set_ylabel("v")
    fig.tight_layout()
    fig.savefig(output / "pic_phase_space_evolution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.semilogy(energy_time, field_energy, color="black")
    for target_time in selected_times:
        axis.axvline(target_time, linestyle="--", alpha=0.6)
    axis.set_xlabel("time")
    axis.set_ylabel("PIC electric-field energy")
    fig.tight_layout()
    fig.savefig(output / "phase_frames_on_energy_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--model-rollout-dir", type=Path, required=True)
    parser.add_argument("--hp-rollout-dir", type=Path, required=True)
    parser.add_argument("--zero-rollout-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pair_id = group1_closure(args.evaluation_dir, args.output_dir / "group1_closure")
    group2_rollout(args.model_rollout_dir, args.hp_rollout_dir, args.zero_rollout_dir, args.output_dir / "group2_dynamics", pair_id)
    group3_stability(args.model_rollout_dir, args.hp_rollout_dir, args.output_dir / "group3_stability")
    group4_phase(args.run_dir, args.output_dir / "group4_phase_space")
    summary = {"representative_pair": pair_id, "groups": ["group1_closure", "group2_dynamics", "group3_stability", "group4_phase_space"]}
    (args.output_dir / "visualization_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
