#!/usr/bin/env python3
"""Create publication-ready Round 10 validation figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from landau_surrogate.diagnostics.macrostep import field_energy


ARMS = ("fno_single", "fno_history4", "unet_history4")
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
MARKERS = {"fno_single": "o", "fno_history4": "s", "unet_history4": "^"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )


def seed_values(summary: dict[str, Any], arm: str, metric: str) -> np.ndarray:
    return np.asarray(
        [row[metric] for row in summary["arms"][arm]["seeds"]],
        dtype=np.float64,
    )


def metric_panel(
    axis: plt.Axes,
    summary: dict[str, Any],
    metric: str,
    title: str,
    ylabel: str,
    *,
    reference: float | None = None,
    reference_label: str | None = None,
    percent: bool = False,
) -> None:
    for x_index, arm in enumerate(ARMS):
        values = seed_values(summary, arm, metric)
        if percent:
            values = 100.0 * values
        x = x_index + np.linspace(-0.09, 0.09, len(values))
        axis.scatter(
            x,
            values,
            s=35,
            color=COLORS[arm],
            marker=MARKERS[arm],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        median = float(np.median(values))
        axis.plot(
            [x_index - 0.19, x_index + 0.19],
            [median, median],
            color=COLORS[arm],
            linewidth=2.6,
            solid_capstyle="round",
            zorder=2,
        )
        axis.annotate(
            f"{median:.3g}",
            (x_index, median),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=COLORS[arm],
            fontsize=8,
        )
    if reference is not None:
        if percent:
            reference *= 100.0
        axis.axhline(reference, color="#555555", linestyle="--", linewidth=1.1)
        axis.text(
            2.42,
            reference,
            reference_label or f"reference {reference:.3g}",
            ha="right",
            va="bottom",
            color="#555555",
            fontsize=8,
        )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(len(ARMS)), [LABELS[arm] for arm in ARMS], rotation=12)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)


def save_figure(figure: plt.Figure, stem: Path) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight")
        paths.append(str(path.resolve()))
    return paths


def overview_figure(summary: dict[str, Any], output_dir: Path) -> list[str]:
    closure = summary["closure_validation_benchmark"]
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.4))
    metric_panel(
        axes[0, 0],
        summary,
        "short_validation_relative_l2",
        "A  Short-horizon validation",
        "Normalized state relative L2 ↓",
    )
    metric_panel(
        axes[0, 1],
        summary,
        "state_perturbation_relative_l2_median",
        "B  Free rollout to t = 80",
        "Perturbation relative L2 ↓",
    )
    metric_panel(
        axes[0, 2],
        summary,
        "field_energy_log10_rmse_median",
        "C  Field-energy trajectory",
        "Median log10-RMSE ↓",
        reference=float(closure["field_energy_log10_rmse_median"]),
        reference_label="small-step closure",
    )
    metric_panel(
        axes[1, 0],
        summary,
        "total_energy_max_relative_drift_median",
        "D  Total-energy drift",
        "Median maximum drift (%) ↓",
        percent=True,
    )
    metric_panel(
        axes[1, 1],
        summary,
        "wall_speedup_vs_closure",
        "E  Validation wall-time speedup",
        "Speedup vs small-step closure (×) ↑",
        reference=50.0,
        reference_label="50× target",
    )

    axis = axes[1, 2]
    width = 0.18
    categories = ("closure", *ARMS)
    labels = ("Small-step\nclosure", *(LABELS[arm] for arm in ARMS))
    x = np.arange(len(categories))
    complete = [float(closure["complete_case_count"])]
    safe = [float(closure["violation_free_case_count"])]
    for arm in ARMS:
        rows = summary["arms"][arm]["seeds"]
        complete.append(float(np.min([row["complete_case_count"] for row in rows])))
        safe.append(float(np.min([row["violation_free_case_count"] for row in rows])))
    axis.bar(x - width / 2, complete, width, color="#777777", label="Reached t=80")
    axis.bar(x + width / 2, safe, width, color="#56B4E9", label="Constraint-safe")
    for xpos, first, second in zip(x, complete, safe, strict=True):
        axis.text(xpos - width / 2, first + 0.25, f"{first:.0f}", ha="center", fontsize=8)
        axis.text(xpos + width / 2, second + 0.25, f"{second:.0f}", ha="center", fontsize=8)
    axis.set_ylim(0, 22.5)
    axis.set_yticks((0, 5, 10, 15, 20))
    axis.set_xticks(x, labels, rotation=12)
    axis.set_ylabel("Worst-seed cases (of 20) ↑")
    axis.set_title("F  Stability and physical admissibility", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)

    handles = [
        plt.Line2D(
            [],
            [],
            color=COLORS[arm],
            marker=MARKERS[arm],
            linestyle="-",
            label=LABELS[arm],
        )
        for arm in ARMS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Round 10 · Conservative macro-step validation (three seeds)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    paths = save_figure(figure, output_dir / "round10_metrics_overview")
    plt.close(figure)
    return paths


def representative_seed(summary: dict[str, Any], arm: str) -> int:
    rows = summary["arms"][arm]["seeds"]
    values = np.asarray(
        [row["field_energy_log10_rmse_median"] for row in rows], dtype=np.float64
    )
    target = float(np.median(values))
    return int(rows[int(np.argmin(np.abs(values - target)))]["seed"])


def load_representative_runs(
    root: Path, summary: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    runs: dict[str, dict[str, Any]] = {}
    seeds = {}
    for arm in ARMS:
        seed = representative_seed(summary, arm)
        seeds[arm] = seed
        run_root = root / arm / f"seed{seed}" / "validation_t80"
        archive = np.load(run_root / "trajectories.npz", allow_pickle=False)
        runs[arm] = {
            "arrays": {name: archive[name] for name in archive.files},
            "summary": load_json(run_root / "summary.json"),
        }
    return runs, seeds


def choose_cases(runs: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    rows_by_arm = {
        arm: {row["case_id"]: row for row in runs[arm]["summary"]["cases"]}
        for arm in ARMS
    }
    first_rows = runs[ARMS[0]]["summary"]["cases"]
    selected = []
    for regime in ("strong_nonlinear", "transition", "weak"):
        candidates = [row["case_id"] for row in first_rows if row["regime"] == regime]
        scores = np.asarray(
            [
                np.median(
                    [rows_by_arm[arm][case_id]["field_energy_log10_rmse"] for arm in ARMS]
                )
                for case_id in candidates
            ]
        )
        target = float(np.median(scores))
        selected.append((regime, candidates[int(np.argmin(np.abs(scores - target)))]))
    return selected


def rollout_figure(
    root: Path, summary: dict[str, Any], output_dir: Path
) -> tuple[list[str], dict[str, int], list[tuple[str, str]]]:
    runs, seeds = load_representative_runs(root, summary)
    cases = choose_cases(runs)
    figure, axes = plt.subplots(3, 2, figsize=(12.8, 10.0))
    regime_labels = {
        "strong_nonlinear": "strong nonlinear",
        "transition": "transition",
        "weak": "weak",
    }
    for row_index, (regime, case_id) in enumerate(cases):
        reference = runs[ARMS[0]]["arrays"]
        case_ids = reference["case_id"].astype(str)
        index = int(np.flatnonzero(case_ids == case_id)[0])
        time = reference["time"][index]
        k_value = float(reference["K"][index])
        truth = reference["truth"][index]
        truth_energy = field_energy(truth, k_value)
        truth_energy = truth_energy / max(float(truth_energy[0]), 1.0e-30)
        axes[row_index, 0].plot(
            time,
            np.log10(np.maximum(truth_energy, 1.0e-10)),
            color="#111111",
            linewidth=2.0,
            label="Kinetic truth",
            zorder=4,
        )
        grid = np.arange(truth.shape[-1]) / truth.shape[-1]
        axes[row_index, 1].plot(
            grid,
            truth[-1, 3],
            color="#111111",
            linewidth=2.0,
            label="Kinetic truth",
            zorder=4,
        )
        for arm in ARMS:
            arrays = runs[arm]["arrays"]
            prediction = arrays["prediction"][index]
            prediction_energy = field_energy(prediction, k_value)
            prediction_energy /= max(float(prediction_energy[0]), 1.0e-30)
            axes[row_index, 0].plot(
                time,
                np.log10(np.maximum(prediction_energy, 1.0e-10)),
                color=COLORS[arm],
                linewidth=1.15,
                alpha=0.92,
                label=f"{LABELS[arm]} (s{seeds[arm]})",
            )
            axes[row_index, 1].plot(
                grid,
                prediction[-1, 3],
                color=COLORS[arm],
                linewidth=1.15,
                alpha=0.92,
                label=f"{LABELS[arm]} (s{seeds[arm]})",
            )
        axes[row_index, 0].set_title(
            f"{regime_labels[regime]} · {case_id} · field energy",
            loc="left",
            fontweight="bold",
        )
        axes[row_index, 1].set_title(
            f"{regime_labels[regime]} · {case_id} · final E(x)",
            loc="left",
            fontweight="bold",
        )
        axes[row_index, 0].set_ylabel(r"$\log_{10}[W_E(t)/W_E(0)]$")
        axes[row_index, 1].set_ylabel("Electric field E")
        axes[row_index, 0].grid(alpha=0.18, linewidth=0.7)
        axes[row_index, 1].grid(alpha=0.18, linewidth=0.7)
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Normalized position x/L")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Representative free rollouts to t = 80 (median seed per model)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    paths = save_figure(figure, output_dir / "round10_representative_rollouts")
    plt.close(figure)
    return paths, seeds, cases


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
    output_dir = (args.output_dir or root / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = root / "round10_validation_summary.json"
    summary = load_json(summary_path)
    style()
    overview_paths = overview_figure(summary, output_dir)
    rollout_paths, seeds, cases = rollout_figure(root, summary, output_dir)
    manifest = {
        "stage": "continuum_v1_macrostep_round10_visualization",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_summary": str(summary_path),
        "diagnostic_test_opened": False,
        "representative_seed_rule": "closest to each arm's median field-energy log10-RMSE",
        "representative_seeds": seeds,
        "representative_case_rule": "closest to the within-regime median across-arm error",
        "representative_cases": [
            {"regime": regime, "case_id": case_id} for regime, case_id in cases
        ],
        "outputs": overview_paths + rollout_paths,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
