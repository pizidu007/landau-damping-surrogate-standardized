"""Plot PDE-oracle and closed-loop feedback error decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("k0p350_a0p075", "k0p400_a0p100")
LABELS = (r"$k=.35,A=.075$", r"$k=.40,A=.10$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audits = [json.loads((args.root / f"{case}.json").read_text()) for case in CASES]
    feedback = [json.loads((args.root / f"{case}_feedback.json").read_text()) for case in CASES]
    energy = json.loads((args.root / "energy_conservation.json").read_text())
    fno = [json.loads((args.root / f"fno_{case}_ampere_dt0p01/summary.json").read_text()) for case in CASES]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    x = np.arange(2); width = 0.36
    oracle = []
    for audit in audits:
        row = next(value for value in audit["oracle_rollouts"] if value["maximum_mode"] == 8
                   and value["field_solver"] == "ampere" and value["dt"] == 0.01)
        oracle.append(row["field_energy_log10_rmse"])
    axes[0].bar(x-width/2, oracle, width, label="truth-q oracle")
    axes[0].bar(x+width/2, [row["field_energy_log10_rmse"] for row in fno], width, label="FNO closure")
    axes[0].set_yscale("log"); axes[0].set_xticks(x, LABELS)
    axes[0].set(ylabel="field-energy log RMSE", title="PDE oracle vs learned closure")
    axes[0].legend(frameon=False); axes[0].grid(alpha=.2, axis="y")

    windows = ("0-10", "10-20", "20-30", "30-40")
    xw = np.arange(len(windows)); colors = ("C0", "C1")
    for index, (item, label, color) in enumerate(zip(feedback, LABELS, colors)):
        truth = [item["windows"][window]["truth_manifold_closure_relative_l2"] for window in windows]
        rollout = [item["windows"][window]["rollout_path_vs_truth_time_closure_relative_l2"] for window in windows]
        axes[1].plot(xw, truth, marker="o", color=color, ls="--", label=f"{label} truth path")
        axes[1].plot(xw, rollout, marker="s", color=color, label=f"{label} self rollout")
    axes[1].set_yscale("log"); axes[1].set_xticks(xw, windows)
    axes[1].set(xlabel="time window", ylabel=r"relative $L_2(\partial_xq)$",
                title="Closure leaves the truth manifold")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(alpha=.2)

    truth_span = [energy[case]["truth_total_energy_relative_span"] for case in CASES]
    fno_span = [energy[case]["fno_total_energy_relative_span"] for case in CASES]
    axes[2].bar(x-width/2, truth_span, width, label="Gkeyll truth")
    axes[2].bar(x+width/2, fno_span, width, label="FNO-fluid")
    axes[2].set_yscale("log"); axes[2].set_xticks(x, LABELS)
    axes[2].set(ylabel="relative total-energy span", title="Energy conservation remains good")
    axes[2].legend(frameon=False); axes[2].grid(alpha=.2, axis="y")
    figure.savefig(args.output, dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
