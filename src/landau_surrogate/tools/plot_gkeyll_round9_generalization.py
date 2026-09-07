"""Summarize strict temporal and whole-trajectory Gkeyll generalization tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (0, 1, 2)
TEST_CASES = ("k0p350_a0p075", "k0p400_a0p100")
CASE_LABELS = {
    "k0p350_a0p075": r"held out: $k=0.35, A=0.075$",
    "k0p400_a0p100": r"held out: $k=0.40, A=0.10$",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def smooth(value: np.ndarray, width: int = 101) -> np.ndarray:
    if len(value) < width:
        return value
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(value, (width // 2, width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(value)]


def timewise_error(evaluation: dict[str, np.ndarray]) -> np.ndarray:
    target = evaluation["target"]
    scale = max(float(np.sqrt(np.mean(target**2))), 1.0e-12)
    return smooth(np.sqrt(np.mean((evaluation["prediction"] - target) ** 2, axis=-1)) / scale)


def add_split_regions(axis: plt.Axes) -> None:
    axis.axvspan(0.0, 24.0, color="#009E73", alpha=0.08, label="train: 0–24")
    axis.axvspan(24.0, 30.0, color="#E69F00", alpha=0.09, label="validation: 24–30")
    axis.axvspan(30.0, 40.0, color="#D55E00", alpha=0.07, label="test: 30–40")
    axis.axvline(24.0, color="0.45", lw=0.8, ls="--")
    axis.axvline(30.0, color="0.45", lw=0.8, ls="--")


def plot_single_temporal(root: Path, output_dir: Path) -> dict:
    evaluations = [load_npz(root / f"training/seed{seed}/evaluation.npz") for seed in SEEDS]
    rollouts = [load_npz(root / f"t40/seed{seed}/rollout.npz") for seed in SEEDS]
    summaries = [read_json(root / f"t40/seed{seed}/summary.json") for seed in SEEDS]
    training = [read_json(root / f"training/seed{seed}/summary.json") for seed in SEEDS]

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    errors = np.stack([timewise_error(value) for value in evaluations])
    time = evaluations[0]["time"]
    add_split_regions(axes[0])
    axes[0].fill_between(time, errors.min(axis=0), errors.max(axis=0), color="#56B4E9", alpha=0.25)
    axes[0].semilogy(time, np.median(errors, axis=0), color="#0072B2", lw=1.5, label="3-seed median")
    axes[0].set(title="Teacher-forced closure error", xlabel="time", ylabel="RMSE / global target RMS")
    axes[0].grid(True, which="both", alpha=0.22)
    axes[0].legend(fontsize=8, ncol=2)

    longest = int(np.argmax([len(value["time"]) for value in rollouts]))
    truth = rollouts[longest]["truth_field_energy"]
    truth_time = rollouts[longest]["time"]
    normalizer = float(truth[0])
    add_split_regions(axes[1])
    colors = ("#0072B2", "#D55E00", "#CC79A7")
    for seed, rollout, color in zip(SEEDS, rollouts, colors):
        axes[1].semilogy(
            rollout["time"], rollout["field_energy"] / normalizer,
            color=color, lw=1.25, alpha=0.9, label=f"FNO seed {seed}",
        )
    axes[1].semilogy(truth_time, truth / normalizer, color="black", lw=2.0, label="Gkeyll truth", zorder=5)
    axes[1].set(title="Closed-loop temporal extrapolation", xlabel="time", ylabel="normalized field energy")
    axes[1].set_ylim(1.0e-7, 1.0e1)
    diverged = [(seed, item) for seed, item in zip(SEEDS, summaries) if item["completed_time"] < 39.999]
    if diverged:
        diverged_seed, diverged_summary = diverged[0]
        axes[1].text(
            0.985, 0.96,
            f"seed {diverged_seed} diverged at t={diverged_summary['completed_time']:.3f}\n(explosion clipped)",
            transform=axes[1].transAxes, ha="right", va="top", fontsize=8,
            color="#0072B2", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
    axes[1].grid(True, which="both", alpha=0.22)
    axes[1].legend(fontsize=8, ncol=2)
    figure.suptitle(r"Single-case causal test: train $t<24$, test nonlinear future")
    figure.tight_layout()
    figure.savefig(output_dir / "single_case_temporal_extrapolation.png", dpi=210)
    plt.close(figure)

    segmented = []
    for seed, rollout in zip(SEEDS, rollouts):
        floor = max(float(rollout["truth_field_energy"][0]) * 1.0e-10, 1.0e-14)
        row = {"seed": seed}
        for name, low, high in (("train_time", 0.0, 24.0), ("validation_time", 24.0, 30.0), ("test_time", 30.0, 40.0001)):
            selected = (rollout["time"] >= low) & (rollout["time"] < high)
            difference = (
                np.log10(np.maximum(rollout["field_energy"][selected], floor))
                - np.log10(np.maximum(rollout["truth_field_energy"][selected], floor))
            )
            row[name + "_field_log10_rmse"] = float(np.sqrt(np.mean(difference**2)))
        segmented.append(row)

    return {
        "split": {"train": [0.0, 24.0], "validation": [24.0, 30.0], "test": [30.0, 40.0]},
        "offline_by_seed": [
            {
                "seed": seed,
                "train_relative_l2": item["train"]["relative_l2"],
                "validation_relative_l2": item["validation"]["relative_l2"],
                "test_relative_l2": item["test"]["relative_l2"],
                "test_correlation": item["test"]["correlation"],
            }
            for seed, item in zip(SEEDS, training)
        ],
        "rollout_by_seed": [
            {"seed": seed, **summary, **segment}
            for seed, summary, segment in zip(SEEDS, summaries, segmented)
        ],
    }


def plot_cross_case(root: Path, output_dir: Path) -> dict:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    result: dict[str, object] = {"cases": {}}
    for row, case_id in enumerate(TEST_CASES):
        evaluations = [
            load_npz(root / f"training/seed{seed}/evaluation/test/{case_id}.npz")
            for seed in SEEDS
        ]
        rollouts = [load_npz(root / f"t40/{case_id}/seed{seed}/rollout.npz") for seed in SEEDS]
        summaries = [read_json(root / f"t40/{case_id}/seed{seed}/summary.json") for seed in SEEDS]
        training = [read_json(root / f"training/seed{seed}/summary.json") for seed in SEEDS]

        errors = np.stack([timewise_error(value) for value in evaluations])
        time = evaluations[0]["time"]
        axes[row, 0].fill_between(time, errors.min(axis=0), errors.max(axis=0), color="#56B4E9", alpha=0.25)
        axes[row, 0].semilogy(time, np.median(errors, axis=0), color="#0072B2", lw=1.5, label="3-seed median")
        axes[row, 0].set(title=CASE_LABELS[case_id] + ": closure error", xlabel="time", ylabel="RMSE / global target RMS")
        axes[row, 0].grid(True, which="both", alpha=0.22)
        axes[row, 0].legend(fontsize=8)

        truth = rollouts[0]["truth_field_energy"]
        normalizer = float(truth[0])
        prediction = np.stack([value["field_energy"] / normalizer for value in rollouts])
        axes[row, 1].fill_between(
            rollouts[0]["time"], prediction.min(axis=0), prediction.max(axis=0),
            color="#56B4E9", alpha=0.24, label="3-seed range",
        )
        axes[row, 1].semilogy(
            rollouts[0]["time"], np.median(prediction, axis=0),
            color="#0072B2", lw=1.45, ls="--", label="3-seed median",
        )
        axes[row, 1].semilogy(
            rollouts[0]["time"], truth / normalizer,
            color="black", lw=2.0, label="Gkeyll truth", zorder=5,
        )
        axes[row, 1].set(title=CASE_LABELS[case_id] + ": closed loop", xlabel="time", ylabel="normalized field energy")
        axes[row, 1].grid(True, which="both", alpha=0.22)
        axes[row, 1].legend(fontsize=8)

        offline_rows = []
        for seed, item in zip(SEEDS, training):
            case_metrics = next(value for value in item["metrics"]["test"]["cases"] if value["case_id"] == case_id)
            offline_rows.append({"seed": seed, **case_metrics})
        result["cases"][case_id] = {
            "offline_by_seed": offline_rows,
            "rollout_by_seed": [{"seed": seed, **item} for seed, item in zip(SEEDS, summaries)],
        }

    figure.suptitle("Whole-trajectory held-out generalization: pure supervised FNO 128/32")
    figure.tight_layout()
    figure.savefig(output_dir / "cross_case_generalization.png", dpi=210)
    plt.close(figure)
    return result


def plot_protocol_summary(summary: dict, round7_root: Path, output_dir: Path) -> None:
    in_sample = [
        read_json(round7_root / f"t40/w128_m32_seed{seed}/summary.json")["field_energy_log10_rmse"]
        for seed in SEEDS
    ]
    temporal = [item["field_energy_log10_rmse"] for item in summary["single_case_temporal"]["rollout_by_seed"]]
    cross = [
        item["field_energy_log10_rmse"]
        for case in summary["cross_case"]["cases"].values()
        for item in case["rollout_by_seed"]
    ]
    groups = (in_sample, temporal, cross)
    labels = ("same trajectory\n3 seeds", "temporal extrapolation\n3 seeds", "held-out cases\n2 cases × 3 seeds")
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    for index, values in enumerate(groups):
        values = np.asarray(values)
        offsets = np.linspace(-0.10, 0.10, len(values))
        axis.scatter(index + offsets, values, color="#0072B2", s=48, zorder=3)
        axis.hlines(np.median(values), index - 0.22, index + 0.22, color="#D55E00", lw=2.5)
    axis.set_yscale("log")
    axis.set_xticks(range(3), labels)
    axis.set_ylabel("closed-loop field-energy log10 RMSE")
    axis.set_title("Evaluation protocol changes the apparent FNO performance")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "protocol_generalization_gap.png", dpi=210)
    plt.close(figure)
    summary["same_trajectory_reference"] = {
        "field_energy_log10_rmse_by_seed": in_sample,
        "median": float(np.median(in_sample)),
    }


def add_aggregates(summary: dict) -> None:
    temporal_offline = np.asarray([item["test_relative_l2"] for item in summary["single_case_temporal"]["offline_by_seed"]])
    temporal_rollout = np.asarray([item["field_energy_log10_rmse"] for item in summary["single_case_temporal"]["rollout_by_seed"]])
    summary["single_case_temporal"]["aggregate"] = {
        "offline_test_relative_l2_median": float(np.median(temporal_offline)),
        "offline_test_relative_l2_range": [float(temporal_offline.min()), float(temporal_offline.max())],
        "rollout_field_log10_rmse_median": float(np.median(temporal_rollout)),
        "rollout_field_log10_rmse_range": [float(temporal_rollout.min()), float(temporal_rollout.max())],
        "completed_t40": int(sum(item["completed_time"] >= 39.999 for item in summary["single_case_temporal"]["rollout_by_seed"])),
    }
    for case in summary["cross_case"]["cases"].values():
        offline = np.asarray([item["relative_l2"] for item in case["offline_by_seed"]])
        rollout = np.asarray([item["field_energy_log10_rmse"] for item in case["rollout_by_seed"]])
        case["aggregate"] = {
            "offline_relative_l2_median": float(np.median(offline)),
            "offline_relative_l2_range": [float(offline.min()), float(offline.max())],
            "rollout_field_log10_rmse_median": float(np.median(rollout)),
            "rollout_field_log10_rmse_range": [float(rollout.min()), float(rollout.max())],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--round7-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": {"width": 128, "modes": 32, "layers": 4, "rollout_finetuning": False},
        "single_case_temporal": plot_single_temporal(args.results_root / "single_causal", args.output_dir),
        "cross_case": plot_cross_case(args.results_root / "cross_case", args.output_dir),
    }
    add_aggregates(summary)
    plot_protocol_summary(summary, args.round7_root, args.output_dir)
    (args.output_dir / "round9_generalization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
