#!/usr/bin/env python3
"""Audit and plot nonlinear rebound across continuum_v1 production cases."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.signal import find_peaks


DEFAULT_ROOT = Path(
    "/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def label_energy(
    case_id: str, K: float, alpha: float, time: np.ndarray, energy: np.ndarray
) -> dict[str, Any]:
    """Apply the frozen continuum_v1 rebound definition used in design selection."""
    energy = np.maximum(np.asarray(energy, dtype=np.float64), np.finfo(float).tiny)
    time = np.asarray(time, dtype=np.float64)
    amplitude = np.sqrt(energy)
    dt = float(np.median(np.diff(time)))
    peaks, _ = find_peaks(energy, distance=max(1, int(round(0.5 / dt))))
    peaks = peaks[(time[peaks] >= 1.0) & (time[peaks] <= time[-1] - dt)]
    eligible_peaks = peaks[time[peaks] <= time[-1] - 10.0]
    if len(eligible_peaks) < 3:
        raise RuntimeError(f"too few eligible peaks for {case_id}")
    minimum = int(eligible_peaks[np.argmin(energy[eligible_peaks])])
    later = peaks[peaks > minimum]
    post_peak = int(later[np.argmax(energy[later])]) if len(later) else minimum

    rebound_energy_ratio = float(energy[post_peak] / energy[minimum])
    rebound_amplitude_ratio = float(amplitude[post_peak] / amplitude[minimum])
    post_peak_fraction_initial = float(energy[post_peak] / energy[0])
    domain_length = 2.0 * np.pi / K
    electric_mode_amplitude = np.sqrt(2.0 * energy / domain_length)
    bounce_frequency = np.sqrt(K * np.maximum(electric_mode_amplitude, 0.0))
    post = time >= time[minimum]
    bounce_cycles_postminimum = float(
        np.trapz(bounce_frequency[post], time[post]) / (2.0 * np.pi)
    )

    early_peaks = peaks[
        (time[peaks] >= 1.0) & (time[peaks] <= min(15.0, time[minimum]))
    ]
    damping_rate = (
        float(np.polyfit(time[early_peaks], np.log(amplitude[early_peaks]), 1)[0])
        if len(early_peaks) >= 3
        else None
    )
    reliable_rebound = bool(
        rebound_energy_ratio >= 1.5
        and (bounce_cycles_postminimum >= 0.75 or rebound_energy_ratio >= 3.0)
        and post_peak_fraction_initial >= 1.0e-8
    )
    if reliable_rebound:
        regime = "strong_nonlinear"
    elif rebound_energy_ratio >= 1.25 or bounce_cycles_postminimum >= 0.5:
        regime = "transition"
    else:
        regime = "weak"

    return {
        "case_id": case_id,
        "K": float(K),
        "alpha": float(alpha),
        "regime": regime,
        "strong_nonlinear": reliable_rebound,
        "early_amplitude_damping_rate": damping_rate,
        "minimum_index": minimum,
        "minimum_time": float(time[minimum]),
        "minimum_energy_fraction_initial": float(energy[minimum] / energy[0]),
        "rebound_energy_ratio": rebound_energy_ratio,
        "rebound_amplitude_ratio": rebound_amplitude_ratio,
        "post_peak_index": post_peak,
        "post_peak_time": float(time[post_peak]),
        "post_peak_energy_fraction_initial": post_peak_fraction_initial,
        "bounce_cycles_postminimum": bounce_cycles_postminimum,
    }


def load_cases(root: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    eligibility = json.loads(
        (root / "manifests" / "training_eligibility_v1.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {item["case_id"] for item in eligibility["excluded_cases"]}
    paths = sorted(root.glob("profiles/production/cases/*/processed/trajectory.h5"))
    rows: list[dict[str, Any]] = []
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            case_id = str(handle.attrs["case_id"])
            K = float(handle.attrs["K"])
            alpha = float(handle.attrs["alpha"])
            time = np.asarray(handle["diagnostics/field_energy_time"][:], dtype=float)
            values = np.asarray(handle["diagnostics/field_energy"][:], dtype=float)
            energy = values[:, 0] if values.ndim == 2 else values
        row = label_energy(case_id, K, alpha, time, energy)
        row["training_eligible"] = case_id not in excluded
        row["trajectory"] = str(path)
        rows.append(row)
        series[case_id] = (time, energy)
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 production trajectories, found {len(rows)}")
    return rows, series


COLORS = {
    "weak": "#7f8c8d",
    "transition": "#f39c12",
    "strong_nonlinear": "#c0392b",
}
LABELS = {
    "weak": "Weak",
    "transition": "Transition",
    "strong_nonlinear": "Strong nonlinear",
}


def counts_text(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["regime"] for row in rows)
    return " | ".join(
        f"{LABELS[name]}: {counts[name]} ({100.0 * counts[name] / len(rows):.1f}%)"
        for name in ("weak", "transition", "strong_nonlinear")
    )


def plot_overview(
    rows: list[dict[str, Any]],
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax_map = figure.add_subplot(grid[0, 0])
    ax_hist = figure.add_subplot(grid[0, 1])
    ax_heat = figure.add_subplot(grid[1, :])

    for regime in ("weak", "transition", "strong_nonlinear"):
        subset = [row for row in rows if row["regime"] == regime]
        ax_map.scatter(
            [row["K"] for row in subset],
            [row["alpha"] for row in subset],
            s=42,
            c=COLORS[regime],
            edgecolors="white",
            linewidths=0.45,
            label=f"{LABELS[regime]} (n={len(subset)})",
            zorder=2,
        )
    excluded = [row for row in rows if not row["training_eligible"]]
    ax_map.scatter(
        [row["K"] for row in excluded],
        [row["alpha"] for row in excluded],
        marker="x",
        s=85,
        c="black",
        linewidths=1.5,
        label="QC-isolated (n=5)",
        zorder=4,
    )
    ax_map.set(xlabel=r"$K=k\lambda_D$", ylabel=r"Perturbation amplitude $\alpha$", title="Rebound regimes in parameter space")
    ax_map.legend(frameon=True, fontsize=9, loc="best")

    bins = np.logspace(
        np.log10(max(1.0, min(row["rebound_energy_ratio"] for row in rows))),
        np.log10(max(row["rebound_energy_ratio"] for row in rows) * 1.05),
        34,
    )
    for regime in ("weak", "transition", "strong_nonlinear"):
        values = [
            row["rebound_energy_ratio"] for row in rows if row["regime"] == regime
        ]
        ax_hist.hist(values, bins=bins, color=COLORS[regime], alpha=0.82, label=LABELS[regime])
    ax_hist.axvline(1.25, color="#f39c12", linestyle="--", linewidth=1.5, label="transition ratio = 1.25")
    ax_hist.axvline(1.5, color="#c0392b", linestyle="--", linewidth=1.5, label="strong ratio = 1.5")
    ax_hist.set_xscale("log")
    ax_hist.set(xlabel=r"Rebound energy ratio $W_{post}/W_{min}$", ylabel="Number of cases", title="Rebound-ratio distribution")
    ax_hist.legend(frameon=True, fontsize=8)

    regime_order = {"weak": 0, "transition": 1, "strong_nonlinear": 2}
    ordered = sorted(rows, key=lambda row: (regime_order[row["regime"]], row["rebound_energy_ratio"]))
    target_time = np.linspace(0.0, 80.0, 801)
    heat = []
    for row in ordered:
        time, energy = series[row["case_id"]]
        normalized = np.maximum(energy / energy[0], 1.0e-14)
        heat.append(np.log10(np.interp(target_time, time, normalized)))
    image = ax_heat.imshow(
        np.asarray(heat),
        origin="lower",
        aspect="auto",
        extent=(target_time[0], target_time[-1], 0, len(ordered)),
        cmap="viridis",
        vmin=-12,
        vmax=0,
        interpolation="nearest",
    )
    cumulative = 0
    counts = Counter(row["regime"] for row in ordered)
    tick_positions = []
    tick_labels = []
    for regime in ("weak", "transition", "strong_nonlinear"):
        count = counts[regime]
        if count:
            tick_positions.append(cumulative + count / 2)
            tick_labels.append(f"{LABELS[regime]} ({count})")
            cumulative += count
            if cumulative < len(ordered):
                ax_heat.axhline(cumulative, color="white", linewidth=1.5)
    ax_heat.set_yticks(tick_positions, tick_labels)
    ax_heat.set(xlabel=r"Normalized time $\tau=\omega_{pe}t$", ylabel="Cases sorted by regime and rebound ratio", title=r"All 200 trajectories: $\log_{10}[W_E(\tau)/W_E(0)]$")
    colorbar = figure.colorbar(image, ax=ax_heat, pad=0.01, fraction=0.025)
    colorbar.set_label(r"$\log_{10}[W_E/W_E(0)]$")
    figure.suptitle("continuum_v1 nonlinear Landau-rebound audit\n" + counts_text(rows), fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for regime in ("weak", "transition", "strong_nonlinear"):
        eligible = sorted(
            (row for row in rows if row["regime"] == regime and row["training_eligible"]),
            key=lambda row: row["rebound_energy_ratio"],
        )
        indices = sorted({int(round(q * (len(eligible) - 1))) for q in (0.2, 0.5, 0.8)})
        chosen.extend(eligible[index] for index in indices)
    return chosen


def plot_representatives(
    rows: list[dict[str, Any]],
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    chosen = representative_rows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, regime in zip(axes, ("weak", "transition", "strong_nonlinear")):
        subset = [row for row in chosen if row["regime"] == regime]
        for row in subset:
            time, energy = series[row["case_id"]]
            normalized = np.maximum(energy / energy[0], 1.0e-14)
            label = (
                f"{row['case_id']}\n"
                f"R={row['rebound_energy_ratio']:.2g}, Nb={row['bounce_cycles_postminimum']:.2f}"
            )
            line, = axis.semilogy(time, normalized, linewidth=1.25, alpha=0.9, label=label)
            imin = row["minimum_index"]
            ipost = row["post_peak_index"]
            axis.scatter(time[imin], normalized[imin], marker="o", s=28, color=line.get_color(), edgecolors="black", linewidths=0.4, zorder=3)
            axis.scatter(time[ipost], normalized[ipost], marker="^", s=36, color=line.get_color(), edgecolors="black", linewidths=0.4, zorder=3)
        axis.set_title(f"{LABELS[regime]} (quantile examples)", color=COLORS[regime], fontweight="bold")
        axis.set_xlabel(r"$\tau=\omega_{pe}t$")
        axis.set_xlim(0, 80)
        axis.set_ylim(1.0e-12, 2.0)
        axis.legend(fontsize=7.8, frameon=True, loc="lower left")
    axes[0].set_ylabel(r"Normalized field energy $W_E(\tau)/W_E(0)$")
    figure.suptitle("Representative trajectories selected by within-regime rebound-ratio quantiles\nCircles: envelope minimum; triangles: strongest later peak", fontsize=14, fontweight="bold")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output_dir = args.output_dir or root / "figures" / "rebound_audit_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, series = load_cases(root)
    plot_overview(rows, series, output_dir / "rebound_overview.png")
    plot_representatives(rows, series, output_dir / "representative_field_energy.png")

    public_rows = [
        {key: value for key, value in row.items() if key not in {"minimum_index", "post_peak_index"}}
        for row in sorted(rows, key=lambda item: item["case_id"])
    ]
    all_counts = Counter(row["regime"] for row in rows)
    eligible_rows = [row for row in rows if row["training_eligible"]]
    eligible_counts = Counter(row["regime"] for row in eligible_rows)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "definition": {
            "strong_nonlinear": "rebound_energy_ratio >= 1.5 AND post_peak_energy_fraction_initial >= 1e-8 AND (bounce_cycles_postminimum >= 0.75 OR rebound_energy_ratio >= 3.0)",
            "transition": "not strong AND (rebound_energy_ratio >= 1.25 OR bounce_cycles_postminimum >= 0.5)",
            "weak": "otherwise",
            "note": "Ratios use field-energy peaks; amplitude ratio is the square root of the energy ratio.",
        },
        "counts_all": {name: all_counts[name] for name in ("weak", "transition", "strong_nonlinear")},
        "counts_training_eligible": {name: eligible_counts[name] for name in ("weak", "transition", "strong_nonlinear")},
        "cases": public_rows,
    }
    atomic_json(output_dir / "production_rebound_labels.json", payload)
    print(json.dumps({"output_dir": str(output_dir), "counts_all": payload["counts_all"], "counts_training_eligible": payload["counts_training_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
