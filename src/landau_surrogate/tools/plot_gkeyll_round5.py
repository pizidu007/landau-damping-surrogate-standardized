"""Plot Round-5 data-enrichment and long-rollout comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("k0p350_a0p075", "k0p400_a0p100")
LABELS = {
    "k0p350_a0p075": r"$k=0.35,\ A=0.075$",
    "k0p400_a0p100": r"$k=0.40,\ A=0.10$",
}


def window_relative(time: np.ndarray, prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers, values = [], []
    for left in np.arange(0.0, 40.0, 5.0):
        selected = (time >= left) & (time < left + 5.0 + 1.0e-12)
        centers.append(left + 2.5)
        error = prediction[selected] - truth[selected]
        values.append(float(np.linalg.norm(error) / max(np.linalg.norm(truth[selected]), 1.0e-12)))
    return np.asarray(centers), np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--round4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    supervised = args.result_root / "supervised_long_history_seed0"
    history = json.loads((supervised / "history.json").read_text())
    summary = json.loads((supervised / "summary.json").read_text())
    round4_summary = json.loads((args.round4_root / "regime_dual_seed0/summary.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].semilogy(epochs, [row["train_loss"] for row in history], color="C0")
    axes[0].set(title="Round-5 supervised objective", xlabel="epoch", ylabel="loss")
    axes[1].plot(epochs, [row["validation_gradient_relative_l2"] for row in history], color="C1")
    axes[1].set(title="Eight-case validation", xlabel="epoch", ylabel=r"relative $L_2(\partial_x q)$")
    old_test = round4_summary["metrics"]["test"]["gradient_relative_l2"]
    new_test = summary["metrics"]["test"]["gradient_relative_l2"]
    bars = axes[2].bar(["Round 4", "Round 5"], [old_test, new_test], color=["0.6", "C2"])
    axes[2].bar_label(bars, fmt="%.4f")
    axes[2].set(title="Sealed offline test", ylabel=r"relative $L_2(\partial_x q)$")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(args.output_dir / "round5_training_offline.png", dpi=190)
    plt.close(fig)

    rollout_metrics: dict[str, dict] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for axis, identifier in zip(axes, CASES):
        paths = {
            "Round 4": args.round4_root / "t40_test" / identifier / "rollout.npz",
            "Round 5 supervised": args.result_root / "t40_supervised" / identifier / "rollout.npz",
            "Round 5 long rollout": args.result_root / "t40_rollout" / identifier / "rollout.npz",
        }
        truth_drawn = False
        for label, path in paths.items():
            with np.load(path) as values:
                time = values["time"]
                prediction = values["field_energy"]
                truth = values["truth_field_energy"]
            if not truth_drawn:
                axis.semilogy(time, truth, color="k", lw=1.7, label="Gkeyll truth")
                truth_drawn = True
            axis.semilogy(time, prediction, lw=1.1, label=label)
        rollout_metrics[identifier] = {
            "supervised": json.loads((args.result_root / "t40_supervised" / identifier / "summary.json").read_text()),
            "long_rollout": json.loads((args.result_root / "t40_rollout" / identifier / "summary.json").read_text()),
        }
        axis.set(title=LABELS[identifier], xlabel="time", ylabel="field energy")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Sealed-test long-time field-energy evolution")
    fig.savefig(args.output_dir / "round5_field_energy.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True, sharex=True)
    names = ("density", "velocity", "pressure")
    for row, identifier in enumerate(CASES):
        with np.load(args.result_root / "t40_rollout" / identifier / "rollout.npz") as values:
            time = values["time"]
            prediction = values["prediction"]
            truth = values["truth"]
        for channel, name in enumerate(names):
            center, error = window_relative(time, prediction[:, channel], truth[:, channel])
            axes[row, channel].semilogy(center, error, marker="o", ms=3)
            axes[row, channel].set(title=name, xlabel="time-window center", ylabel=f"{LABELS[identifier]}\nrelative error")
            axes[row, channel].grid(alpha=0.2)
    fig.suptitle("Round-5 long-rollout state-error growth")
    fig.savefig(args.output_dir / "round5_error_growth.png", dpi=190)
    plt.close(fig)

    result = {
        "round4_test_gradient_relative_l2": old_test,
        "round5_test_gradient_relative_l2": new_test,
        "offline_relative_reduction": 1.0 - new_test / old_test,
        "round5_validation_gradient_relative_l2": summary["best_validation_gradient_relative_l2"],
        "rollouts": rollout_metrics,
    }
    (args.output_dir / "round5_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
