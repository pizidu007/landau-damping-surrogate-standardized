"""Create the Round-6 strict-reproduction diagnostic figures and summary."""
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


def load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    for variant in range(8):
        for seed in (0, 1):
            offline = read_json(
                args.results_root / "stage2" / f"A{variant}_seed{seed}" / "summary.json"
            )
            rollout = read_json(
                args.results_root / "stage2_rollout" / f"A{variant}_seed{seed}_t10" / "summary.json"
            )
            variants.append({
                "variant": f"A{variant}", "seed": seed,
                "offline": offline["test"]["relative_l2"],
                "field": rollout["field_energy_log10_rmse"],
                "clamps": rollout["clamp_count"],
            })
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for row in variants:
        marker = "x" if row["clamps"] else "o"
        axis.scatter(row["offline"], row["field"], marker=marker, s=55)
        axis.annotate(
            f'{row["variant"]}/s{row["seed"]}',
            (row["offline"], row["field"]), xytext=(4, 3),
            textcoords="offset points", fontsize=7,
        )
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel(r"offline relative $L_2$ of $\partial_x q$")
    axis.set_ylabel("t≤10 field-energy log10 RMSE")
    axis.set_title("Round 6 architecture screen: offline accuracy is not stability")
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "stage2_offline_vs_closed_loop.png", dpi=190)
    plt.close(fig)

    best_path = args.results_root / "stage4/mode_sweep/A7_seed2_mode8_t40/rollout.npz"
    old_path = args.old_root / "rollout/paper_like_ampere_primitive_dt0p002_mode8_t40/rollout.npz"
    b4_path = args.results_root / "stage5_eval/B4_h200_mode8_t40/rollout.npz"
    best, old, b4 = map(load_rollout, (best_path, old_path, b4_path))
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    normalizer = float(best["truth_field_energy"][0])
    axis.semilogy(best["time"], best["truth_field_energy"] / normalizer, "k", lw=2, label="Gkeyll truth")
    axis.semilogy(old["time"], old["field_energy"] / normalizer, color="#777777", lw=1.3, label="old GroupNorm/GELU")
    axis.semilogy(best["time"], best["field_energy"] / normalizer, color="#0072B2", lw=1.7, label="Round 6 ReLU, mode 8")
    axis.semilogy(b4["time"], b4["field_energy"] / normalizer, color="#D55E00", lw=1.2, label="H200 rollout fine-tune")
    axis.set_xlabel("time"); axis.set_ylabel("normalized electric-field energy")
    axis.set_title("Strict single-case long-time closed loop")
    axis.grid(True, which="both", alpha=0.22); axis.legend(ncol=2)
    fig.tight_layout(); fig.savefig(args.output_dir / "best_long_time_field_energy.png", dpi=200)
    plt.close(fig)

    time = best["time"]
    truth, prediction = best["truth"], best["prediction"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)
    names = ("density n", "velocity u", "pressure p")
    for row, name in enumerate(names):
        scale = float(np.quantile(np.abs(truth[:, row]), 0.995))
        if row != 0:
            scale = max(scale, 1.0e-8)
            limits = (-scale, scale)
            cmap = "RdBu_r"
        else:
            limits = (float(np.quantile(truth[:, row], 0.005)), float(np.quantile(truth[:, row], 0.995)))
            cmap = "viridis"
        error = np.abs(prediction[:, row] - truth[:, row])
        for column, value, title in (
            (0, truth[:, row], "Gkeyll truth"),
            (1, prediction[:, row], "FNO-fluid"),
        ):
            image = axes[row, column].imshow(
                value.T, origin="lower", extent=(time[0], time[-1], 0, 1),
                aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1],
            )
            axes[row, column].set_title(f"{name}: {title}")
            fig.colorbar(image, ax=axes[row, column], shrink=0.78)
        error_image = axes[row, 2].imshow(
            error.T, origin="lower", extent=(time[0], time[-1], 0, 1),
            aspect="auto", cmap="magma", vmin=0,
            vmax=float(np.quantile(error, 0.995)),
        )
        axes[row, 2].set_title(f"{name}: absolute error")
        fig.colorbar(error_image, ax=axes[row, 2], shrink=0.78)
        axes[row, 0].set_ylabel("x/L")
    for axis in axes[-1]: axis.set_xlabel("time")
    fig.tight_layout(); fig.savefig(args.output_dir / "best_moments_truth_prediction_error.png", dpi=185)
    plt.close(fig)

    modes, mode_field = [], []
    for mode in (8, 12, 16, 24):
        path = (
            args.results_root / "stage4/native/A7_seed2_t40/summary.json"
            if mode == 16 else
            args.results_root / f"stage4/mode_sweep/A7_seed2_mode{mode}_t40/summary.json"
        )
        modes.append(mode); mode_field.append(read_json(path)["field_energy_log10_rmse"])
    stages = ["supervised", "H10", "H50", "H200", "H200+spectral"]
    stage_paths = [
        args.results_root / "stage4/mode_sweep/A7_seed2_mode8_t40/summary.json",
        args.results_root / "stage5_eval/B2_h10_mode8_t40/summary.json",
        args.results_root / "stage5_eval/B3_h50_mode8_t40/summary.json",
        args.results_root / "stage5_eval/B4_h200_mode8_t40/summary.json",
        args.results_root / "stage5_eval/B5_h200_spectral_mode8_t40/summary.json",
    ]
    stage_field = [read_json(path)["field_energy_log10_rmse"] for path in stage_paths]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar([str(value) for value in modes], mode_field, color="#0072B2")
    axes[0].axhline(0.15, color="k", ls="--", lw=1, label="target 0.15")
    axes[0].set_xlabel("deployment maximum mode"); axes[0].set_ylabel("field-energy log10 RMSE")
    axes[0].set_title("Spectral truncation"); axes[0].legend()
    axes[1].bar(stages, stage_field, color=["#0072B2"] + ["#D55E00"] * 4)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("field-energy log10 RMSE"); axes[1].set_title("Rollout curriculum ablation")
    fig.tight_layout(); fig.savefig(args.output_dir / "mode_and_rollout_ablation.png", dpi=190)
    plt.close(fig)

    starts = list(range(0, 40, 5)); horizons = [1, 2, 5]
    restart = np.full((len(horizons), len(starts)), np.nan, dtype=np.float64)
    restart_rows = []
    for column, start in enumerate(starts):
        for row, horizon in enumerate(horizons):
            summary = read_json(
                args.results_root / "stage4/restarts" / f"start{start}_h{horizon}" / "summary.json"
            )
            value = summary["field_energy_log10_rmse"]
            restart[row, column] = value
            restart_rows.append({"start": start, "horizon": horizon, **summary})
    fig, axis = plt.subplots(figsize=(9, 3.5))
    image = axis.imshow(np.log10(np.maximum(restart, 1.0e-8)), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(starts)), starts); axis.set_yticks(range(len(horizons)), horizons)
    axis.set_xlabel("restart time"); axis.set_ylabel("prediction horizon")
    axis.set_title("log10 field-energy RMSE after restarting from truth")
    fig.colorbar(image, ax=axis, label="log10 RMSE")
    fig.tight_layout(); fig.savefig(args.output_dir / "truth_restart_error_heatmap.png", dpi=190)
    plt.close(fig)

    generalization = {}
    for identifier in ("k0p350_a0p075", "k0p400_a0p100"):
        path = args.results_root / "stage6/generalization" / identifier / "summary.json"
        if path.exists(): generalization[identifier] = read_json(path)
    summary = {
        "stage2": variants,
        "best_checkpoint": str(args.results_root / "stage3/A7_full_seed2/best.pt"),
        "best_deployment_mode": 8,
        "best_t40": read_json(args.results_root / "stage4/mode_sweep/A7_seed2_mode8_t40/summary.json"),
        "mode_ablation": dict(zip(map(str, modes), mode_field)),
        "rollout_ablation": dict(zip(stages, stage_field)),
        "restart_evaluations": restart_rows,
        "generalization": generalization,
    }
    (args.output_dir / "round6_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"figures": 5, "best_field_error": mode_field[0]}, indent=2))


if __name__ == "__main__":
    main()
