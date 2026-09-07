"""Plot Round-3 history/q/rollout-curriculum comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = ("k0p350_a0p075", "k0p400_a0p100")
LABELS = {
    "k0p350_a0p075": r"$k=0.35,\ A=0.075$",
    "k0p400_a0p100": r"$k=0.40,\ A=0.10$",
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_field_energy(old_root: Path, new_root: Path, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True, constrained_layout=True)
    for axis, identifier in zip(axes, CASES):
        direct = np.load(old_root / "supervised" / identifier / "rollout.npz")
        history = np.load(new_root / "supervised_history" / identifier / "rollout.npz")
        curriculum = np.load(new_root / "curriculum_h2" / identifier / "rollout.npz")
        stride = max(1, len(direct["time"]) // 2000)
        selection = slice(None, None, stride)
        axis.semilogy(
            direct["time"][selection],
            np.maximum(direct["truth_field_energy"][selection], 1.0e-12),
            color="black", linewidth=1.8, label="Gkeyll truth",
        )
        axis.semilogy(
            direct["time"][selection],
            np.maximum(direct["field_energy"][selection], 1.0e-12),
            linewidth=1.2, label=r"direct $\partial_xq$ FNO",
        )
        axis.semilogy(
            history["time"][selection],
            np.maximum(history["field_energy"][selection], 1.0e-12),
            linewidth=1.2, label="history q-FNO",
        )
        axis.semilogy(
            curriculum["time"][selection],
            np.maximum(curriculum["field_energy"][selection], 1.0e-12),
            linewidth=1.2, label="history q-FNO + curriculum",
        )
        axis.set_title(LABELS[identifier])
        axis.set_xlabel("physical time")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel(r"field energy $\int E_x^2\,dx$")
    axes[0].legend(frameon=False, fontsize=8.5)
    figure.suptitle("Round 3: held-out long-time field-energy comparison")
    figure.savefig(output_dir / "round3_long_time_field_energy.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_metrics(
    old_summary: Path,
    nohistory_summary: Path,
    history_summary: Path,
    old_root: Path,
    new_root: Path,
    output_dir: Path,
) -> None:
    direct = load_summary(old_summary)["metrics"]["test"]["cases"]
    nohistory = load_summary(nohistory_summary)["metrics"]["test"]["cases"]
    history = load_summary(history_summary)["metrics"]["test"]["cases"]
    identifiers = [row["case_id"] for row in direct]
    offline = {
        r"direct $\partial_xq$": [row["relative_l2"] for row in direct],
        "q, no history": [row["gradient_relative_l2"] for row in nohistory],
        "q, 4-state history": [row["gradient_relative_l2"] for row in history],
    }
    long = {r"direct $\partial_xq$": [], "history q": [], "history q + curriculum": []}
    for identifier in identifiers:
        long[r"direct $\partial_xq$"].append(
            load_summary(old_root / "supervised" / identifier / "summary.json")["field_energy_log10_rmse"]
        )
        long["history q"].append(
            load_summary(new_root / "supervised_history" / identifier / "summary.json")["field_energy_log10_rmse"]
        )
        long["history q + curriculum"].append(
            load_summary(new_root / "curriculum_h2" / identifier / "summary.json")["field_energy_log10_rmse"]
        )
    x = np.arange(len(identifiers))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    width = 0.24
    for index, (name, values) in enumerate(offline.items()):
        offset = (index - 1) * width
        axes[0].bar(x + offset, values, width, label=name)
        for position, value in enumerate(values):
            axes[0].text(position + offset, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Instantaneous closure on held-out cases")
    axes[0].set_ylabel(r"relative L2 error of $\partial_xq$")
    axes[0].set_xticks(x, [LABELS[value] for value in identifiers])
    axes[0].legend(frameon=False, fontsize=8)
    for index, (name, values) in enumerate(long.items()):
        offset = (index - 1) * width
        axes[1].bar(x + offset, values, width, label=name)
        for position, value in enumerate(values):
            axes[1].text(position + offset, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title(r"Closed-loop field energy to $t=40$")
    axes[1].set_ylabel("field-energy log10 RMSE")
    axes[1].set_xticks(x, [LABELS[value] for value in identifiers])
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Round 3: history and heat-flux target ablation")
    figure.savefig(output_dir / "round3_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_paper_anchor(anchor_root: Path, output_dir: Path) -> None:
    direct = np.load(anchor_root / "direct_gradient" / "rollout.npz")
    history = np.load(anchor_root / "supervised_history" / "rollout.npz")
    curriculum = np.load(anchor_root / "curriculum_h2" / "rollout.npz")
    stride = max(1, len(direct["time"]) // 2500)
    selection = slice(None, None, stride)
    figure, axis = plt.subplots(figsize=(7.5, 4.3), constrained_layout=True)
    axis.semilogy(
        direct["time"][selection],
        np.maximum(direct["truth_field_energy"][selection], 1.0e-12),
        color="black", linewidth=1.8, label="Gkeyll truth",
    )
    axis.semilogy(
        direct["time"][selection], np.maximum(direct["field_energy"][selection], 1.0e-12),
        linewidth=1.2, label=r"direct $\partial_xq$ FNO",
    )
    axis.semilogy(
        history["time"][selection], np.maximum(history["field_energy"][selection], 1.0e-12),
        linewidth=1.2, label="history q-FNO",
    )
    axis.semilogy(
        curriculum["time"][selection], np.maximum(curriculum["field_energy"][selection], 1.0e-12),
        linewidth=1.2, label="history q-FNO + curriculum",
    )
    axis.set(
        title=r"Paper anchor $k=0.35,\ A=0.10$",
        xlabel="physical time",
        ylabel=r"field energy $\int E_x^2\,dx$",
    )
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(output_dir / "round3_paper_anchor_field_energy.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--nohistory-summary", type=Path, required=True)
    parser.add_argument("--history-summary", type=Path, required=True)
    parser.add_argument("--old-rollout-root", type=Path, required=True)
    parser.add_argument("--new-rollout-root", type=Path, required=True)
    parser.add_argument("--paper-anchor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_field_energy(args.old_rollout_root, args.new_rollout_root, args.output_dir)
    plot_metrics(
        args.old_summary,
        args.nohistory_summary,
        args.history_summary,
        args.old_rollout_root,
        args.new_rollout_root,
        args.output_dir,
    )
    plot_paper_anchor(args.paper_anchor_root, args.output_dir)


if __name__ == "__main__":
    main()
