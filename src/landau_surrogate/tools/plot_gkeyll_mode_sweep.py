"""Summarize the integer spectral-cutoff sweep for the strict Gkeyll rollout."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_through(path: Path, maximum_time: float = 20.0) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        keep = values["time"] <= maximum_time + 1.0e-9
        return {name: values[name][keep] for name in values.files}


def _metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    prediction = values["prediction"]
    truth = values["truth"]
    result = {}
    for index, name in enumerate(("density", "velocity", "pressure")):
        result[f"{name}_relative_l2"] = float(
            np.linalg.norm(prediction[:, index] - truth[:, index])
            / np.linalg.norm(truth[:, index])
        )
    field = values["field_energy"]
    truth_field = values["truth_field_energy"]
    floor = max(float(truth_field[0]) * 1.0e-10, 1.0e-14)
    result["field_energy_log10_rmse"] = float(
        np.sqrt(
            np.mean(
                (
                    np.log10(np.maximum(field, floor))
                    - np.log10(np.maximum(truth_field, floor))
                )
                ** 2
            )
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--mode24", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories: dict[int, dict[str, np.ndarray]] = {}
    for mode in range(8, 24):
        path = args.rollout_root / f"paper_like_ampere_primitive_dt0p002_mode{mode}_t20" / "rollout.npz"
        trajectories[mode] = _load_through(path)
    trajectories[24] = _load_through(args.mode24)
    rows = [{"maximum_mode": mode, **_metrics(values)} for mode, values in trajectories.items()]
    metric_names = (
        "density_relative_l2", "velocity_relative_l2", "pressure_relative_l2",
        "field_energy_log10_rmse",
    )
    best_by_metric = {
        name: min(rows, key=lambda row: row[name])["maximum_mode"] for name in metric_names
    }
    summary = {
        "time_window": [0.0, 20.0],
        "fixed_parameters": {
            "checkpoint": "paper_like_seed0/best.pt",
            "dt": 0.002,
            "field_solver": "ampere",
            "moment_formulation": "primitive",
            "initial_closure": "truth_first_step",
        },
        "best_by_metric": best_by_metric,
        "recommended_maximum_mode": 8 if set(best_by_metric.values()) == {8} else None,
        "results": rows,
    }
    (args.output_dir / "mode_sweep_8_24.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "mode_sweep_8_24.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("maximum_mode", *metric_names))
        writer.writeheader()
        writer.writerows(rows)

    modes = np.asarray([row["maximum_mode"] for row in rows])
    labels = ("density relative L2", "velocity relative L2", "pressure relative L2", "field-energy log10 RMSE")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=True)
    for axis, name, label in zip(axes.flat, metric_names, labels):
        values = np.asarray([row[name] for row in rows])
        axis.plot(modes, values, marker="o", markersize=3.7, linewidth=1.2)
        index = int(np.argmin(values))
        axis.scatter(modes[index], values[index], s=65, color="#b2182b", zorder=3, label=f"best: mode-{modes[index]}")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("maximum retained Fourier mode")
        axis.set_xticks(modes)
    fig.tight_layout()
    fig.savefig(args.output_dir / "mode_sweep_metrics_t20.png", dpi=180)
    plt.close(fig)

    selected = (8, 9, 14, 16, 20, 24)
    truth = trajectories[8]
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.4), sharex=True)
    axes[0].semilogy(truth["time"], truth["truth_field_energy"], color="black", linewidth=2.0, label="Gkeyll truth")
    floor = max(float(truth["truth_field_energy"][0]) * 1.0e-10, 1.0e-14)
    for mode in selected:
        values = trajectories[mode]
        width = 2.0 if mode == 8 else 1.0
        alpha = 1.0 if mode in (8, 24) else 0.75
        axes[0].semilogy(values["time"], values["field_energy"], linewidth=width, alpha=alpha, label=f"mode-{mode}")
        error = np.abs(
            np.log10(np.maximum(values["field_energy"], floor))
            - np.log10(np.maximum(values["truth_field_energy"], floor))
        )
        axes[1].plot(values["time"], error, linewidth=width, alpha=alpha, label=f"mode-{mode}")
    axes[0].set_ylabel(r"$\frac{1}{2}\int |E_x|^2\,dx$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8.5)
    axes[1].set(xlabel="time", ylabel=r"$|\Delta\log_{10} E|$")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "mode_sweep_field_energy_t20.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
