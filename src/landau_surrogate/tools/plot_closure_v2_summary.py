"""Build a compact summary figure for the second closure experiment round."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root
    paper = read_json(root / "stage1_huang_reproduction/paper_like_dt0005_seed0/summary.json")
    causal = read_json(root / "stage1_huang_reproduction/causal_dt0005_seed0/summary.json")
    history_rows = []
    for name, label in (
        ("nupk_q_filtered_seed0", "1 frame"),
        ("nupk_q_history3_seed0", "3 frames"),
        ("nupk_q_history5_seed0", "5 frames"),
    ):
        data = read_json(root / f"stage4_filtered_history/{name}/validation_summary.json")
        history_rows.append((label, data["metrics"]["relative_l2"]))
    audit = read_json(root / "stage3_pic_gkeyll_audit/summary.json")
    rollout = read_json(args.rollout_dir / "summary.json")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axis = axes[0, 0]
    values = [paper["test"]["relative_l2"], causal["test"]["relative_l2"]]
    bars = axis.bar(["interleaved\nwithin trajectory", "causal\nt=30–40"], values, color=["#4C78A8", "#E45756"])
    axis.set_yscale("log")
    axis.set_ylabel("relative L2 error")
    axis.set_title("Gkeyll protocol sensitivity")
    axis.bar_label(bars, labels=[f"{value:.3g}" for value in values], padding=3)

    axis = axes[0, 1]
    labels, values = zip(*history_rows)
    bars = axis.bar(labels, values, color=["#B0B0B0", "#72B7B2", "#54A24B"])
    axis.set_ylabel("validation relative L2")
    axis.set_title("PIC memory ablation (filtered q)")
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)

    axis = axes[1, 0]
    phases = ["0–15", "15–30", "30–40"]
    phase_keys = ["linear_0_15", "transition_15_30", "nonlinear_30_40"]
    for field, label in (("density", "density"), ("pressure", "pressure"), ("heat_flux_gradient", r"$\partial_x q$")):
        axis.plot(phases, [audit["field_metrics"][field][key]["correlation"] for key in phase_keys], marker="o", label=label)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("PIC–Gkeyll correlation")
    axis.set_title("Matched simulation comparison")
    axis.legend()

    axis = axes[1, 1]
    files = sorted(args.rollout_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(args.rollout_dir)
    data = np.load(files[0])
    axis.semilogy(data["time"], data["truth_field_energy"], color="black", linewidth=2, label="PIC")
    axis.semilogy(data["time"], data["field_energy"], color="#F58518", label=rollout["closure"])
    axis.set_xlabel("time")
    axis.set_ylabel("electric-field energy")
    axis.set_title(f"Selected validation rollout (macro log RMSE={rollout['macro']['field_energy_log10_rmse']:.3f})")
    axis.legend()

    figure.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "closure_v2_summary.png", dpi=190)
    plt.close(figure)
    manifest = {
        "paper_like_test_relative_l2": paper["test"]["relative_l2"],
        "causal_test_relative_l2": causal["test"]["relative_l2"],
        "history_validation": dict(history_rows),
        "rollout": rollout,
    }
    (args.output_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
