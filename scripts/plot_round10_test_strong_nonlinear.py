#!/usr/bin/env python3
"""Visualize every strong-nonlinear case in the sealed Round 10 test split."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from landau_surrogate.diagnostics.macrostep import field_energy


SELECTED = {
    "fno_single": 1,
    "fno_history4": 2,
    "unet_history4": 1,
}
LABELS = {
    "fno_single": "FNO · 1 frame",
    "fno_history4": "FNO · 4 frames",
    "unet_history4": "U-Net · 4 frames",
}
COLORS = {
    "fno_single": "#0072B2",
    "fno_history4": "#D55E00",
    "unet_history4": "#009E73",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(root: Path) -> dict[str, dict[str, Any]]:
    runs = {}
    for arm, seed in SELECTED.items():
        run_root = root / arm / f"seed{seed}" / "test_t80"
        summary_path = run_root / "summary.json"
        trajectory_path = run_root / "trajectories.npz"
        if not summary_path.exists() or not trajectory_path.exists():
            raise FileNotFoundError(f"missing sealed-test rollout: {run_root}")
        archive = np.load(trajectory_path, allow_pickle=False)
        runs[arm] = {
            "summary": load_json(summary_path),
            "arrays": {name: archive[name] for name in archive.files},
        }
    return runs


def strong_cases(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in runs["fno_single"]["summary"]["cases"]
        if row["regime"] == "strong_nonlinear"
    ]
    return sorted(rows, key=lambda row: (float(row["K"]), float(row["alpha"])))


def case_lookup(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in run["summary"]["cases"]}


def case_index(arrays: dict[str, np.ndarray], case_id: str) -> int:
    indices = np.flatnonzero(arrays["case_id"].astype(str) == case_id)
    if len(indices) != 1:
        raise ValueError(f"expected exactly one trajectory for {case_id}")
    return int(indices[0])


def normalized_log_field_energy(state: np.ndarray, k_value: float) -> np.ndarray:
    energy = field_energy(state, k_value)
    normalized = energy / max(float(energy[0]), 1.0e-30)
    return np.log10(np.maximum(normalized, 1.0e-10))


def plot_one_case(
    case: dict[str, Any], runs: dict[str, dict[str, Any]], output_dir: Path
) -> Path:
    case_id = case["case_id"]
    k_value = float(case["K"])
    reference = runs["fno_single"]["arrays"]
    index = case_index(reference, case_id)
    time = reference["time"][index]
    truth = reference["truth"][index]
    grid = np.arange(truth.shape[-1]) / truth.shape[-1]
    metrics = {arm: case_lookup(run)[case_id] for arm, run in runs.items()}

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    axes[0].plot(
        time,
        normalized_log_field_energy(truth, k_value),
        color="#111111",
        linewidth=2.0,
        label="Kinetic truth",
    )
    axes[1].plot(
        grid,
        truth[-1, 3],
        color="#111111",
        linewidth=2.0,
        label="Kinetic truth",
    )
    for arm, run in runs.items():
        arrays = run["arrays"]
        prediction = arrays["prediction"][case_index(arrays, case_id)]
        error = metrics[arm]["field_energy_log10_rmse"]
        axes[0].plot(
            time,
            normalized_log_field_energy(prediction, k_value),
            color=COLORS[arm],
            linewidth=1.15,
            label=f"{LABELS[arm]} · RMSE {error:.3f}",
        )
        axes[1].plot(
            grid,
            prediction[-1, 3],
            color=COLORS[arm],
            linewidth=1.15,
            label=LABELS[arm],
        )
    axes[0].set_title("Field-energy evolution", loc="left", fontweight="bold")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$\log_{10}[W_E(t)/W_E(0)]$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Electric field at t = 80", loc="left", fontweight="bold")
    axes[1].set_xlabel("Normalized position x/L")
    axes[1].set_ylabel("Electric field E")
    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Strong nonlinear test · {case_id} · K={k_value:.3f}, α={float(case['alpha']):.3f}",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output_dir / "cases" / f"{case_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_energy_atlas(
    cases: list[dict[str, Any]], runs: dict[str, dict[str, Any]], output_dir: Path
) -> list[Path]:
    columns = 4
    rows = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2.55 * rows), sharex=True)
    axes_flat = np.asarray(axes).ravel()
    for axis, case in zip(axes_flat, cases, strict=False):
        case_id = case["case_id"]
        k_value = float(case["K"])
        reference = runs["fno_single"]["arrays"]
        index = case_index(reference, case_id)
        time = reference["time"][index]
        truth = reference["truth"][index]
        axis.plot(
            time,
            normalized_log_field_energy(truth, k_value),
            color="#111111",
            linewidth=1.45,
            label="Kinetic truth",
        )
        for arm, run in runs.items():
            arrays = run["arrays"]
            prediction = arrays["prediction"][case_index(arrays, case_id)]
            axis.plot(
                time,
                normalized_log_field_energy(prediction, k_value),
                color=COLORS[arm],
                linewidth=0.8,
                label=LABELS[arm],
            )
        axis.set_title(
            f"{case_id}\nK={k_value:.3f}, α={float(case['alpha']):.3f}",
            fontsize=8.5,
            fontweight="bold",
        )
        axis.grid(alpha=0.16, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes_flat[len(cases) :]:
        axis.axis("off")
    for row_index in range(rows):
        axes[row_index, 0].set_ylabel(r"$\log_{10}(W_E/W_{E0})$")
    for axis in axes[-1]:
        axis.set_xlabel("Time")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"All strong-nonlinear sealed-test cases · field energy · n={len(cases)}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"strong_nonlinear_field_energy_atlas{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def plot_error_matrix(
    cases: list[dict[str, Any]], runs: dict[str, dict[str, Any]], output_dir: Path
) -> list[Path]:
    lookups = {arm: case_lookup(run) for arm, run in runs.items()}
    energy = np.asarray(
        [
            [lookups[arm][case["case_id"]]["field_energy_log10_rmse"] for arm in SELECTED]
            for case in cases
        ],
        dtype=np.float64,
    )
    perturbation = np.asarray(
        [
            [lookups[arm][case["case_id"]]["state_perturbation_relative_l2"] for arm in SELECTED]
            for case in cases
        ],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 10.5))
    for axis, values, title in (
        (axes[0], energy, "Field-energy log10-RMSE ↓"),
        (axes[1], perturbation, "Perturbation relative L2 ↓"),
    ):
        upper = float(np.nanpercentile(values, 95))
        image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=upper)
        axis.set_xticks(range(len(SELECTED)), [LABELS[arm] for arm in SELECTED], rotation=20)
        axis.set_yticks(range(len(cases)), [case["case_id"] for case in cases])
        axis.set_title(title, loc="left", fontweight="bold")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                normalized = values[row, column] / max(upper, 1.0e-30)
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="white" if normalized > 0.48 else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    figure.suptitle(
        "All strong-nonlinear sealed-test cases · error matrix",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"strong_nonlinear_error_matrix{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def write_metrics(
    cases: list[dict[str, Any]], runs: dict[str, dict[str, Any]], output_dir: Path
) -> Path:
    lookups = {arm: case_lookup(run) for arm, run in runs.items()}
    path = output_dir / "strong_nonlinear_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case_id",
                "K",
                "alpha",
                "model",
                "seed",
                "field_energy_log10_rmse",
                "state_relative_l2",
                "state_perturbation_relative_l2",
                "electric_mode_one_amplitude_relative_l2",
                "electric_mode_one_phase_mae_radians",
                "total_energy_max_relative_drift",
                "finite_to_final_time",
            ]
        )
        for case in cases:
            for arm, seed in SELECTED.items():
                row = lookups[arm][case["case_id"]]
                writer.writerow(
                    [
                        case["case_id"],
                        case["K"],
                        case["alpha"],
                        arm,
                        seed,
                        row["field_energy_log10_rmse"],
                        row["state_relative_l2"],
                        row["state_perturbation_relative_l2"],
                        row["electric_mode_one"]["amplitude_relative_l2"],
                        row["electric_mode_one"]["phase_mae_radians"],
                        row["total_energy_max_relative_drift"],
                        row["finite_to_final_time"],
                    ]
                )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/continuum_v1_macrostep_round10/formal"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir or root / "figures/test_strong_nonlinear"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 120})
    runs = load_runs(root)
    cases = strong_cases(runs)
    individual = [plot_one_case(case, runs, output_dir) for case in cases]
    atlas = plot_energy_atlas(cases, runs, output_dir)
    matrices = plot_error_matrix(cases, runs, output_dir)
    metrics_path = write_metrics(cases, runs, output_dir)
    manifest = {
        "stage": "round10_sealed_test_strong_nonlinear_visualization",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "split": "test",
        "test_opened_after_validation_selection": True,
        "selection": SELECTED,
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "outputs": {
            "individual_case_png": [str(path) for path in individual],
            "field_energy_atlas": [str(path) for path in atlas],
            "error_matrix": [str(path) for path in matrices],
            "metrics_csv": str(metrics_path),
        },
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
