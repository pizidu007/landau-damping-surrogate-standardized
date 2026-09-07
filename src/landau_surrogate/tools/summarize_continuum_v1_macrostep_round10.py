"""Aggregate Round 10 training and full-validation rollout results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("fno_single", "fno_history4", "unet_history4")
ROLLOUT_AGGREGATE_METRICS = (
    "field_energy_log10_rmse_median",
    "field_energy_log10_rmse_mean",
    "state_relative_l2_median",
    "state_relative_l2_mean",
    "state_perturbation_relative_l2_median",
    "state_perturbation_relative_l2_mean",
)
DERIVED_CASE_METRICS = (
    "electric_mode_one_amplitude_relative_l2_median",
    "electric_mode_one_phase_mae_radians_median",
    "total_energy_max_relative_drift_median",
)
LONG_METRICS = (*ROLLOUT_AGGREGATE_METRICS, *DERIVED_CASE_METRICS)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_values(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": values,
        "median": float(np.nanmedian(array)),
        "mean": float(np.nanmean(array)),
        "minimum": float(np.nanmin(array)),
        "maximum": float(np.nanmax(array)),
    }


def collect_arm(root: Path, arm: str) -> dict[str, Any]:
    seeds = []
    arm_root = root / arm
    for seed_dir in sorted(arm_root.glob("seed*")):
        train_path = seed_dir / "summary.json"
        rollout_path = seed_dir / "validation_t80" / "summary.json"
        if not train_path.exists() or not rollout_path.exists():
            continue
        train = load_json(train_path)
        rollout = load_json(rollout_path)
        cases = rollout["cases"]
        derived = {
            "electric_mode_one_amplitude_relative_l2_median": float(
                np.nanmedian(
                    [
                        row["electric_mode_one"]["amplitude_relative_l2"]
                        for row in cases
                    ]
                )
            ),
            "electric_mode_one_phase_mae_radians_median": float(
                np.nanmedian(
                    [row["electric_mode_one"]["phase_mae_radians"] for row in cases]
                )
            ),
            "total_energy_max_relative_drift_median": float(
                np.nanmedian(
                    [row["total_energy_max_relative_drift"] for row in cases]
                )
            ),
        }
        per_regime = {}
        for regime in sorted({row["regime"] for row in rollout["cases"]}):
            regime_rows = [
                row for row in rollout["cases"] if row["regime"] == regime
            ]
            per_regime[regime] = {
                "case_count": len(regime_rows),
                "field_energy_log10_rmse_median": float(
                    np.nanmedian(
                        [row["field_energy_log10_rmse"] for row in regime_rows]
                    )
                ),
                "state_relative_l2_median": float(
                    np.nanmedian([row["state_relative_l2"] for row in regime_rows])
                ),
                "state_perturbation_relative_l2_median": float(
                    np.nanmedian(
                        [
                            row["state_perturbation_relative_l2"]
                            for row in regime_rows
                        ]
                    )
                ),
            }
        seeds.append(
            {
                "seed": int(train["seed"]),
                "best_epoch": int(train["best_epoch"]),
                "short_validation_relative_l2": float(
                    train["best_validation_relative_l2"]
                ),
                "parameter_count": int(train["parameter_count"]),
                "training_wall_seconds": float(train["elapsed_seconds"]),
                "rollout_wall_seconds": float(rollout["wall_seconds"]),
                "model_calls": int(rollout["model_calls"]),
                "complete_case_count": int(
                    rollout["aggregate"]["complete_case_count"]
                ),
                "violation_free_case_count": int(
                    rollout["aggregate"]["violation_free_case_count"]
                ),
                "per_regime": per_regime,
                **{
                    key: float(rollout["aggregate"][key])
                    for key in ROLLOUT_AGGREGATE_METRICS
                },
                **derived,
            }
        )
    aggregate = {}
    if seeds:
        for key in ("short_validation_relative_l2", *LONG_METRICS):
            aggregate[key] = summarize_values([row[key] for row in seeds])
        aggregate["all_complete"] = all(
            row["complete_case_count"] == 20 for row in seeds
        )
        aggregate["all_violation_free"] = all(
            row["violation_free_case_count"] == 20 for row in seeds
        )
    return {"completed_seed_count": len(seeds), "seeds": seeds, "aggregate": aggregate}


def paired_comparison(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    first_by_seed = {row["seed"]: row for row in first["seeds"]}
    second_by_seed = {row["seed"]: row for row in second["seeds"]}
    common = sorted(set(first_by_seed) & set(second_by_seed))
    metrics = {}
    for key in ("short_validation_relative_l2", *LONG_METRICS):
        rows = []
        for seed in common:
            baseline = first_by_seed[seed][key]
            candidate = second_by_seed[seed][key]
            rows.append(
                {
                    "seed": seed,
                    "baseline": baseline,
                    "candidate": candidate,
                    "relative_improvement": (baseline - candidate)
                    / max(abs(baseline), 1.0e-30),
                }
            )
        metrics[key] = {
            "pairs": rows,
            "candidate_better_count": sum(
                row["relative_improvement"] > 0.0 for row in rows
            ),
        }
    return {"common_seeds": common, "metrics": metrics}


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/continuum_v1_macrostep_round10/formal"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = {arm: collect_arm(args.root, arm) for arm in ARMS}
    closure_path = args.root / "closure_validation_benchmark" / "summary.json"
    closure = load_json(closure_path) if closure_path.exists() else None
    if closure is not None:
        closure_wall = float(closure["wall_seconds"])
        closure_calls = int(closure["model_calls"])
        closure_energy = float(closure["field_energy_log10_rmse_median"])
        for arm in arms.values():
            for row in arm["seeds"]:
                row["wall_speedup_vs_closure"] = closure_wall / row[
                    "rollout_wall_seconds"
                ]
                row["model_call_reduction_vs_closure"] = closure_calls / row[
                    "model_calls"
                ]
                row["field_energy_improvement_vs_closure"] = (
                    closure_energy - row["field_energy_log10_rmse_median"]
                ) / closure_energy
    summary = {
        "stage": "continuum_v1_macrostep_round10_validation_summary",
        "selection_split": "validation",
        "diagnostic_test_opened": False,
        "required_seed_count": 3,
        "arms": arms,
        "history_fno_vs_single_fno": paired_comparison(
            arms["fno_single"], arms["fno_history4"]
        ),
        "closure_validation_benchmark": closure,
    }
    output = args.output or args.root / "round10_validation_summary.json"
    atomic_write(output, json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
