"""Visualize offline and closed-loop results for the strict Gkeyll case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--paper-evaluation", type=Path, required=True)
    parser.add_argument("--causal-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rollout = np.load(args.rollout)
    time = rollout["time"]
    prediction = rollout["prediction"]
    truth = rollout["truth"]
    field = rollout["field_energy"]
    truth_field = rollout["truth_field_energy"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].semilogy(time, truth_field, label="Gkeyll truth", linewidth=1.8)
    axes[0].semilogy(time, field, label="Fluid + FNO", linewidth=1.4)
    axes[0].set_ylabel(r"$\frac{1}{2}\int |E_x|^2 dx$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()
    axes[1].plot(time, np.abs(field - truth_field), color="#b2182b")
    axes[1].set(xlabel="time", ylabel="absolute field-energy error")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "closed_loop_field_energy.png", dpi=180)
    plt.close(fig)

    extent = (float(time[0]), float(time[-1]), 0.0, 1.0)
    names = ("density", "velocity", "pressure")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
    moment_metrics = {}
    for row, name in enumerate(names):
        offset = 1.0 if name in ("density", "pressure") else 0.0
        scale = float(np.quantile(np.abs(truth[:, row] - offset), 0.995))
        difference = prediction[:, row] - truth[:, row]
        error_scale = float(np.quantile(np.abs(difference), 0.995))
        for col, value in enumerate((truth[:, row] - offset, prediction[:, row] - offset, difference)):
            bound = scale if col < 2 else error_scale
            axes[row, col].imshow(
                value.T, origin="lower", extent=extent, aspect="auto",
                cmap="RdBu_r", vmin=-bound, vmax=bound,
            )
        axes[row, 0].set_ylabel(f"{name}\nx/L")
        moment_metrics[name] = {
            "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(truth[:, row])),
            "error_onset_gt_10pct_peak_time": None,
        }
        per_time = np.linalg.norm(difference, axis=1) / np.maximum(np.linalg.norm(truth[:, row] - offset, axis=1), 1e-12)
        hit = np.flatnonzero(per_time > 0.1)
        if len(hit):
            moment_metrics[name]["error_onset_gt_10pct_peak_time"] = float(time[hit[0]])
    for axis, title in zip(axes[0], ("Gkeyll truth", "Fluid + FNO", "prediction - truth")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("time")
    fig.tight_layout()
    fig.savefig(args.output_dir / "closed_loop_moments_xt.png", dpi=180)
    plt.close(fig)

    offline = {}
    for label, path in (("paper_like", args.paper_evaluation), ("causal", args.causal_evaluation)):
        values = np.load(path)
        target, pred = values["target"], values["prediction"]
        index = values["test_index"]
        offline[label] = {
            "test_relative_l2": float(np.linalg.norm(pred[index] - target[index]) / np.linalg.norm(target[index])),
            "test_correlation": float(np.corrcoef(pred[index].ravel(), target[index].ravel())[0, 1]),
        }
    summary = {"offline": offline, "closed_loop_moment_diagnostics": moment_metrics}
    (args.output_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
