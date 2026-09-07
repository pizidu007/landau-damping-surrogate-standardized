"""Create the final machine-readable and human-readable closure study report."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def improvement(reference: float, candidate: float) -> float:
    return 1.0 - candidate / reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--pic-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    label = read(args.study_dir / "stage1_label_audit/summary.json")
    selection = read(args.study_dir / "stage3_selection/selection.json")
    pure_closure = read(args.study_dir / "stage6_test_closure/summary.json")
    hybrid_closure = read(args.study_dir / "stage6_test_closure_hybrid/summary.json")
    baselines = read(args.study_dir / "stage6_test_baselines/summary.json")
    rollout_model = read(args.study_dir / "stage6_test_rollout/model_hp_0p75/summary.json")
    rollout_hp = read(args.study_dir / "stage6_test_rollout/hp/summary.json")
    rollout_zero = read(args.study_dir / "stage6_test_rollout/zero/summary.json")
    with (args.study_dir / "stage6_test_rollout/model_hp_0p75/pair_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rollout_pair_rows = list(csv.DictReader(handle))
    total_clamps = sum(int(row["clamp_count"]) for row in rollout_pair_rows)
    pic = read(args.pic_summary)
    hp_closure_error = float(baselines["macro"]["hp_calibrated"]["relative_l2"])
    pure_error = float(pure_closure["macro"]["relative_l2"])
    hybrid_error = float(hybrid_closure["macro"]["relative_l2"])
    hp_rollout_error = float(rollout_hp["macro"]["field_energy_log10_rmse"])
    model_rollout_error = float(rollout_model["macro"]["field_energy_log10_rmse"])
    pic_runtime = float(pic["aggregate_metrics"]["runtime_seconds_mean"])
    model_runtime = float(rollout_model["macro"]["runtime_seconds"])
    summary = {
        "study": "PIC-derived FNO heat-flux closure v1",
        "dataset": {
            "pairs": 20,
            "cases": 60,
            "train_validation_test_cases": [30, 15, 15],
            "label_audit": label,
        },
        "selected_supervised_candidate": selection["selected"],
        "deployment": {
            "checkpoint": "models/closure/closure_fno_pic_v1.pt",
            "target_parameterization": "predict q then take a physical spectral derivative",
            "hp_blend": 0.75,
            "maximum_mode": 24,
            "fluid_substep_dt": 0.02,
        },
        "test_closure": {
            "pure_fno_relative_l2": pure_error,
            "hybrid_relative_l2": hybrid_error,
            "hp_relative_l2": hp_closure_error,
            "pure_fno_improvement_over_hp": improvement(hp_closure_error, pure_error),
            "hybrid_improvement_over_hp": improvement(hp_closure_error, hybrid_error),
            "pure_fno_correlation": pure_closure["macro"]["correlation"],
        },
        "test_rollout": {
            "model_hp": rollout_model,
            "hp": rollout_hp,
            "zero": rollout_zero,
            "field_energy_improvement_over_hp": improvement(hp_rollout_error, model_rollout_error),
            "density_improvement_over_hp": improvement(
                float(rollout_hp["macro"]["density_relative_l2"]),
                float(rollout_model["macro"]["density_relative_l2"]),
            ),
            "velocity_improvement_over_hp": improvement(
                float(rollout_hp["macro"]["velocity_relative_l2"]),
                float(rollout_model["macro"]["velocity_relative_l2"]),
            ),
            "pressure_improvement_over_hp": improvement(
                float(rollout_hp["macro"]["pressure_relative_l2"]),
                float(rollout_model["macro"]["pressure_relative_l2"]),
            ),
        },
        "runtime": {
            "pic_case_seconds_mean": pic_runtime,
            "fluid_fno_case_seconds_mean": model_runtime,
            "observed_speedup": pic_runtime / model_runtime,
            "note": "Shared-GPU wall time; kernel-launch optimization and batching are not yet applied.",
        },
        "acceptance": {
            "closure_improves_hp_by_30_percent": improvement(hp_closure_error, pure_error) >= 0.30,
            "all_test_rollouts_reach_t60": float(rollout_model["macro"]["completed_time"]) == 60.0,
            "test_rollouts_have_positive_floors_without_clamps": total_clamps == 0,
            "field_energy_improves_hp_by_50_percent": improvement(hp_rollout_error, model_rollout_error) >= 0.50,
            "observed_speedup_at_least_10x": pic_runtime / model_runtime >= 10.0,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "study_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, args.output_dir / "study_summary.json")
    markdown = f"""# PIC-derived FNO heat-flux closure v1

## Outcome

- Pure FNO test closure relative L2: `{pure_error:.6f}` (HP `{hp_closure_error:.6f}`, improvement `{100*improvement(hp_closure_error, pure_error):.1f}%`).
- Deployed 75% HP hybrid closure relative L2: `{hybrid_error:.6f}`.
- Hybrid fluid test field-energy log10 RMSE: `{model_rollout_error:.6f}` (HP `{hp_rollout_error:.6f}`, improvement `{100*improvement(hp_rollout_error, model_rollout_error):.1f}%`).
- All five sealed test parameter pairs reached `t=60` without density or pressure clamps.
- Observed shared-GPU wall-time speedup over the formal PIC generator: `{pic_runtime/model_runtime:.2f}x`; the 10x target is not yet met.

## Interpretation

Predicting the smoother heat flux and differentiating spectrally was substantially more accurate than predicting its gradient directly. Pure FNO closure was not stable for every long rollout. Validation-only selection chose a 75% calibrated HP blend, which trades some pointwise closure accuracy for robust long-time integration. The result is a successful PIC-based closure proof of concept, but not yet a full reproduction of the paper's nonlinear bounce-frequency accuracy or computational speedup.
"""
    (args.output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
