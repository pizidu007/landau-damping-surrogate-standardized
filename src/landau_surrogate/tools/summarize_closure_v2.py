"""Freeze the machine-readable conclusions of closure experiment round two."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root
    paper = read(root / "stage1_huang_reproduction/paper_like_dt0005_seed0/summary.json")
    causal = read(root / "stage1_huang_reproduction/causal_dt0005_seed0/summary.json")
    audit = read(root / "stage3_pic_gkeyll_audit/summary.json")
    validation = {}
    for label in ("pure", "hp025", "hp050", "hp075"):
        name = f"history5_poisson_dt002_t60_{label}"
        validation[label] = read(root / f"stage5_rollout_validation/{name}/summary.json")
    short_rollouts = {
        "poisson_dt_0p02": read(root / "stage5_rollout_validation/history5_poisson_dt002_t10/summary.json"),
        "poisson_dt_0p005": read(root / "stage5_rollout_validation/history5_poisson_dt0005_t10/summary.json"),
        "ampere_dt_0p02": read(root / "stage5_rollout_validation/history5_ampere_dt002_t10/summary.json"),
    }
    result = {
        "round": "closure_fno_v2",
        "huang_public_data": {
            "snapshot_dt_inferred_from_continuity": audit["gkeyll_continuity_inferred_dt"],
            "arrays_interpreted_as": {"p_new": "raw M2", "q_new": "raw M3"},
            "raw_moment_equation_audit": audit["gkeyll_raw_moment_equation_audit_t0_t24"],
            "paper_like_test": paper["test"],
            "causal_nonlinear_test": causal["test"],
            "raw_moment_closed_loop": read(root / "stage1_huang_reproduction/paper_like_dt0005_rollout_raw_poisson_mode8/summary.json"),
        },
        "matched_pic_gkeyll": {
            "replicas": audit["pic_replica_count"],
            "central_moment_metrics": audit["field_metrics"],
            "dqdx_spectral_metrics": audit["heat_flux_gradient_spectral_metrics"],
        },
        "pic_offline_closure": {
            "single_frame_validation": read(root / "stage4_filtered_history/nupk_q_filtered_seed0/validation_summary.json")["metrics"],
            "history3_validation": read(root / "stage4_filtered_history/nupk_q_history3_seed0/validation_summary.json")["metrics"],
            "history5_validation": read(root / "stage4_filtered_history/nupk_q_history5_seed0/validation_summary.json")["metrics"],
            "history5_reused_test": read(root / "stage4_filtered_history/test_history5/summary.json")["macro"],
        },
        "pic_rollout": {
            "short_ablation": short_rollouts,
            "long_validation": validation,
            "selected_v2_stable_candidate": "hp075",
            "deployment_decision": "retain_v1_hp075",
            "reason": "The v2 history model improves offline closure but its stable hp075 validation field error is worse than v1 (1.550 vs 1.393).",
        },
        "rollout_finetuning": {
            "supervised_t10_field_log_rmse": short_rollouts["poisson_dt_0p02"]["macro"]["field_energy_log10_rmse"],
            "window5_t10_field_log_rmse": read(root / "stage6_rollout_finetune/window5_validation_t10/summary.json")["macro"]["field_energy_log10_rmse"],
            "window20_t10_field_log_rmse": read(root / "stage6_rollout_finetune/window20_validation_t10/summary.json")["macro"]["field_energy_log10_rmse"],
            "window5_field_weight_t10_field_log_rmse": read(root / "stage6_rollout_finetune/window5_field_validation_t10/summary.json")["macro"]["field_energy_log10_rmse"],
            "decision": "reject_all_rollout_finetuned_candidates",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
