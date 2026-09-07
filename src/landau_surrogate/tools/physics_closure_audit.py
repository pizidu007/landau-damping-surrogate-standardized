#!/usr/bin/env python3
"""Stage 10A formal audit: density -> Poisson -> E -> energy closure."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from landau_surrogate.data.rollout_cache import GROUP_CODE_TO_NAME, load_case_sequence, split_case_indices
from landau_surrogate.diagnostics.rollout import build_snapshot_model, build_stepper, rollout_candidate
from landau_surrogate.physics.closure import (
    aggregate_rows,
    closure_from_normalized_delta,
    match_time_indices,
    model_closure_metrics,
    relative_l2,
    resolve_mean_anchor_beta,
    sha256_file,
    solve_periodic_poisson_from_density,
)

EXPECTED = {
    "mother_sha256": "f92b58323b627ed526c29028abc0da0869173f6c380e8df4b76f424f89d1c722",
    "cache_sha256": "84fd51e09e3898555a690aef3df57625f48bc374dab3a27fa5856cf750801a2c",
    "stage8d2c_sha256": "047b5a032c0ffc34ef7c4edbe742135ccd7e742ab753504081bece99ecaea8e8",
    "stage9b_sha256": "b94434678c1a1cdd8d22e5e94b4bfa2ba2ddfc2ba8b7fc4e0acded76c9077dd5",
    "gpu_source_sha256": "e642abff2be4cc88da7866bc16288cfbe63c49944688efab18aad1b0b6538847",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--mother", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stage8d2c", type=Path, required=True)
    parser.add_argument("--stage9b", type=Path, required=True)
    parser.add_argument("--gpu-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_models(path: Path, device: torch.device) -> tuple[dict[str, Any], Any, Any, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "model_config", "cache_contract",
        "snapshot_model_state_dict", "snapshot_model_config",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing}")
    stepper = build_stepper(checkpoint, device)
    snapshot = build_snapshot_model(
        {
            "model_config": checkpoint["snapshot_model_config"],
            "model_state_dict": checkpoint["snapshot_model_state_dict"],
            "cache_contract": checkpoint["cache_contract"],
        },
        device,
    )
    beta, beta_source = resolve_mean_anchor_beta(checkpoint)
    checkpoint["_stage10a_resolved_mean_anchor_beta"] = beta
    checkpoint["_stage10a_mean_anchor_beta_source"] = beta_source
    return checkpoint, stepper, snapshot, beta


def truth_closure_audit(
    mother_path: Path,
    cache_path: Path,
    case_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decompose closure into two independently auditable contracts.

    1. Mother density -> Poisson -> mother electric verifies the frozen Poisson
       sign, coordinate scale, FFT convention, and energy definition.
    2. Cache phase-space -> density -> electric measures whether the strided
       128x193 cache is sufficiently conservative for field/energy closure.

    The full-density L2 is intentionally retained for continuity, but the
    physically relevant density metric is the zero-mean charge relative L2:
    electric field depends on rho - <rho>, not on the O(1) density background.
    """
    rows: list[dict[str, Any]] = []
    worst: dict[str, Any] = {}
    with h5py.File(cache_path, "r") as cache, h5py.File(mother_path, "r") as mother:
        phase_time = np.asarray(cache["grids/phase_time"], dtype=np.float64)
        velocity = np.asarray(cache["grids/velocity"], dtype=np.float64)
        f0_train = np.asarray(cache["grids/f0_train"], dtype=np.float64)
        delta_scale = float(cache.attrs["delta_global_rms"])
        phase_count = phase_time.size
        for case_index in case_indices:
            index = int(case_index)
            identifier = decode(cache["cases/case_id"][index])
            k_value = float(cache["cases/k"][index])
            alpha = float(cache["cases/alpha"][index])
            split_code = int(cache["cases/split_code"][index])
            group_name = GROUP_CODE_TO_NAME[int(cache["cases/group_code"][index])]
            start = index * phase_count
            stop = start + phase_count
            normalized = np.asarray(cache["samples/field"][start:stop], dtype=np.float64)
            cache_closure = closure_from_normalized_delta(
                normalized, f0_train, delta_scale, velocity, k_value
            )

            group = mother["cases"][identifier]
            scalar_time = np.asarray(group["scalar_time"], dtype=np.float64)
            time_indices = match_time_indices(scalar_time, phase_time)
            mother_density = np.asarray(group["density"][time_indices], dtype=np.float64)
            mother_electric = np.asarray(group["electric"][time_indices], dtype=np.float64)
            mother_field_energy = np.asarray(group["field_energy"][time_indices], dtype=np.float64)
            mother_total_energy = np.asarray(group["total_energy"][time_indices], dtype=np.float64)
            mother_kinetic = mother_total_energy - mother_field_energy

            # Direct contract: solve Poisson from the already deposited mother
            # density. This bypasses the strided velocity quadrature entirely.
            direct_poisson = solve_periodic_poisson_from_density(mother_density, k_value)
            direct_electric = np.asarray(direct_poisson["electric_field"], dtype=np.float64)
            direct_field_energy = 0.5 * float(direct_poisson["dx"]) * np.sum(
                direct_electric * direct_electric, axis=-1
            )

            mother_charge = 1.0 - mother_density
            cache_charge = np.asarray(cache_closure["charge_density"], dtype=np.float64)
            mother_charge_solvable = mother_charge - np.mean(mother_charge, axis=-1, keepdims=True)
            cache_charge_solvable = cache_charge - np.mean(cache_charge, axis=-1, keepdims=True)

            row = {
                "case_index": index,
                "case_id": identifier,
                "k": k_value,
                "alpha": alpha,
                "split_code": split_code,
                "evaluation_group": group_name,

                # Mother-density Poisson contract.
                "mother_density_to_electric_relative_l2": relative_l2(
                    direct_electric, mother_electric
                ),
                "mother_density_to_field_energy_relative_l2": relative_l2(
                    direct_field_energy, mother_field_energy
                ),
                "mother_density_poisson_residual_absolute_max": float(
                    np.max(np.abs(direct_poisson["poisson_residual"]))
                ),

                # Cache phase-space adequacy.
                "density_relative_l2": relative_l2(
                    cache_closure["electron_density"], mother_density
                ),
                "charge_solvable_relative_l2": relative_l2(
                    cache_charge_solvable, mother_charge_solvable
                ),
                "electric_field_relative_l2": relative_l2(
                    cache_closure["electric_field"], mother_electric
                ),
                "field_energy_relative_l2": relative_l2(
                    cache_closure["field_energy"], mother_field_energy
                ),
                "kinetic_energy_relative_l2": relative_l2(
                    cache_closure["kinetic_energy"], mother_kinetic
                ),
                "total_energy_relative_l2": relative_l2(
                    cache_closure["total_energy"], mother_total_energy
                ),
                "density_absolute_max_error": float(
                    np.max(np.abs(cache_closure["electron_density"] - mother_density))
                ),
                "charge_solvable_absolute_max_error": float(
                    np.max(np.abs(cache_charge_solvable - mother_charge_solvable))
                ),
                "electric_absolute_max_error": float(
                    np.max(np.abs(cache_closure["electric_field"] - mother_electric))
                ),
                "field_energy_relative_max_error": float(
                    np.max(
                        np.abs(np.asarray(cache_closure["field_energy"]) - mother_field_energy)
                        / np.maximum(np.abs(mother_field_energy), 1.0e-30)
                    )
                ),
                "total_energy_relative_max_error": float(
                    np.max(
                        np.abs(np.asarray(cache_closure["total_energy"]) - mother_total_energy)
                        / np.maximum(np.abs(mother_total_energy), 1.0e-30)
                    )
                ),
                "poisson_residual_absolute_max": float(
                    np.max(np.abs(cache_closure["poisson_residual"]))
                ),
                "mean_charge_absolute_max": float(
                    np.max(np.abs(np.mean(cache_charge, axis=-1)))
                ),
                "finite": bool(
                    all(
                        np.isfinite(np.asarray(cache_closure[key])).all()
                        for key in (
                            "electron_density", "electric_field", "field_energy",
                            "kinetic_energy", "total_energy",
                        )
                    )
                    and np.isfinite(direct_electric).all()
                    and np.isfinite(direct_field_energy).all()
                ),
            }

            row["poisson_contract_passed"] = bool(
                row["finite"]
                and row["mother_density_to_electric_relative_l2"] <= 1.0e-4
                and row["mother_density_to_field_energy_relative_l2"] <= 1.0e-4
                and row["mother_density_poisson_residual_absolute_max"] <= 1.0e-9
            )
            row["cache_closure_passed"] = bool(
                row["finite"]
                and row["electric_field_relative_l2"] <= 0.02
                and row["field_energy_relative_l2"] <= 0.05
                and row["total_energy_relative_l2"] <= 0.02
                and row["poisson_residual_absolute_max"] <= 1.0e-9
                and row["mean_charge_absolute_max"] <= 5.0e-3
            )
            # Backward-compatible alias: an audit case passes if the fundamental
            # Poisson contract is correct. Cache inadequacy is a limitation, not
            # a code/physics-contract failure.
            row["passed"] = row["poisson_contract_passed"]
            rows.append(row)

    if rows:
        for metric in (
            "mother_density_to_electric_relative_l2",
            "mother_density_to_field_energy_relative_l2",
            "charge_solvable_relative_l2",
            "density_relative_l2",
            "electric_field_relative_l2",
            "field_energy_relative_l2",
            "kinetic_energy_relative_l2",
            "total_energy_relative_l2",
        ):
            selected = max(rows, key=lambda item: float(item[metric]))
            worst[metric] = {"case_id": selected["case_id"], "value": selected[metric]}
    return rows, worst

def _mother_target_closure(
    group: h5py.Group,
    phase_time: np.ndarray,
    k_value: float,
) -> dict[str, Any]:
    scalar_time = np.asarray(group["scalar_time"], dtype=np.float64)
    indices = match_time_indices(scalar_time, phase_time)
    density = np.asarray(group["density"][indices], dtype=np.float64)
    electric = np.asarray(group["electric"][indices], dtype=np.float64)
    field_energy = np.asarray(group["field_energy"][indices], dtype=np.float64)
    total_energy = np.asarray(group["total_energy"][indices], dtype=np.float64)
    kinetic_energy = total_energy - field_energy
    electric_hat = np.fft.rfft(electric, axis=-1)
    density_hat = np.fft.rfft(density, axis=-1)
    nx = density.shape[-1]
    direct_poisson = solve_periodic_poisson_from_density(density, k_value)
    return {
        "electron_density": density,
        "charge_density": 1.0 - density,
        "electric_field": electric,
        "electric_mode_complex": 2.0 * electric_hat[:, 1] / nx,
        "density_mode_complex": 2.0 * density_hat[:, 1] / nx,
        "field_energy": field_energy,
        "kinetic_energy": kinetic_energy,
        "total_energy": total_energy,
        "poisson_residual": np.asarray(direct_poisson["poisson_residual"]),
    }


def evaluate_models(
    mother_path: Path,
    cache_path: Path,
    model_paths: dict[str, Path],
    selected_indices: dict[str, np.ndarray],
    device: torch.device,
    amp: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    with h5py.File(cache_path, "r") as handle:
        f0_train = np.asarray(handle["grids/f0_train"], dtype=np.float64)
        velocity = np.asarray(handle["grids/velocity"], dtype=np.float64)
        phase_time = np.asarray(handle["grids/phase_time"], dtype=np.float64)
        delta_scale = float(handle.attrs["delta_global_rms"])
        bounds = {
            "k_min": float(np.min(handle["cases/k"])),
            "k_max": float(np.max(handle["cases/k"])),
            "alpha_min": float(np.min(handle["cases/alpha"])),
            "alpha_max": float(np.max(handle["cases/alpha"])),
            "time_min": float(phase_time[0]),
            "time_max": float(phase_time[-1]),
        }
    dt = float(phase_time[1] - phase_time[0])
    rows: list[dict[str, Any]] = []
    validation_summary: dict[str, dict[str, Any]] = {}
    representative: dict[str, Any] = {"phase_time": phase_time}

    with h5py.File(mother_path, "r") as mother:
        for model_name, checkpoint_path in model_paths.items():
            checkpoint, stepper, snapshot, beta = checkpoint_models(checkpoint_path, device)
            validation_rows: list[dict[str, Any]] = []
            for split_name, indices in selected_indices.items():
                sequences = [load_case_sequence(cache_path, int(index)) for index in indices]
                if not sequences:
                    continue
                initial = np.stack([item["field"][0] for item in sequences]).astype(np.float32)
                k_values = np.asarray([item["k"] for item in sequences], dtype=np.float32)
                alpha_values = np.asarray([item["alpha"] for item in sequences], dtype=np.float32)
                rollout = rollout_candidate(
                    stepper, initial, k_values, alpha_values, float(phase_time[0]),
                    len(phase_time) - 1, dt, bounds, device,
                    snapshot_model=snapshot, mean_anchor_beta=beta, amp=amp,
                )
                for local, sequence in enumerate(sequences):
                    truth_normalized = np.asarray(sequence["field"], dtype=np.float64)
                    pred_closure = closure_from_normalized_delta(
                        rollout[local], f0_train, delta_scale, velocity, float(sequence["k"])
                    )
                    cache_truth_closure = closure_from_normalized_delta(
                        truth_normalized, f0_train, delta_scale, velocity, float(sequence["k"])
                    )
                    mother_truth_closure = _mother_target_closure(
                        mother["cases"][sequence["case_id"]], phase_time, float(sequence["k"])
                    )

                    # Main physical metrics are against the mother PIC outputs.
                    metrics = model_closure_metrics(
                        pred_closure, mother_truth_closure, phase_time
                    )
                    pred_charge = np.asarray(pred_closure["charge_density"], dtype=np.float64)
                    mother_charge = np.asarray(mother_truth_closure["charge_density"], dtype=np.float64)
                    pred_charge_solvable = pred_charge - np.mean(pred_charge, axis=-1, keepdims=True)
                    mother_charge_solvable = mother_charge - np.mean(mother_charge, axis=-1, keepdims=True)
                    metrics["charge_solvable_relative_l2"] = relative_l2(
                        pred_charge_solvable, mother_charge_solvable
                    )

                    # Internal-cache metrics isolate model error from the fixed
                    # 128x193 velocity-quadrature floor.
                    internal_metrics = model_closure_metrics(
                        pred_closure, cache_truth_closure, phase_time
                    )
                    cache_charge = np.asarray(cache_truth_closure["charge_density"], dtype=np.float64)
                    cache_charge_solvable = cache_charge - np.mean(cache_charge, axis=-1, keepdims=True)
                    internal_metrics["charge_solvable_relative_l2"] = relative_l2(
                        pred_charge_solvable, cache_charge_solvable
                    )
                    row = {
                        "model": model_name,
                        "split": split_name,
                        "case_index": int(sequence["case_index"]),
                        "case_id": sequence["case_id"],
                        "k": float(sequence["k"]),
                        "alpha": float(sequence["alpha"]),
                        "evaluation_group": GROUP_CODE_TO_NAME[int(sequence["group_code"])],
                        "phase_space_relative_l2": relative_l2(
                            rollout[local], truth_normalized
                        ),
                        **metrics,
                        **{f"internal_cache_{key}": value for key, value in internal_metrics.items()},
                        "finite": bool(np.isfinite(rollout[local]).all()),
                    }
                    rows.append(row)
                    if split_name == "val":
                        validation_rows.append(row)
                    if split_name == "test" and local < 3:
                        prefix = f"{model_name}__{sequence['case_id']}"
                        representative[f"{prefix}__field_energy"] = np.asarray(
                            pred_closure["field_energy"], dtype=np.float32
                        )
                        representative[f"truth__{sequence['case_id']}__field_energy"] = np.asarray(
                            mother_truth_closure["field_energy"], dtype=np.float32
                        )
                        representative[f"cache_truth__{sequence['case_id']}__field_energy"] = np.asarray(
                            cache_truth_closure["field_energy"], dtype=np.float32
                        )
                        representative[f"{prefix}__electric_mode_real"] = np.real(
                            pred_closure["electric_mode_complex"]
                        ).astype(np.float32)
                        representative[f"{prefix}__electric_mode_imag"] = np.imag(
                            pred_closure["electric_mode_complex"]
                        ).astype(np.float32)
                        representative[f"truth__{sequence['case_id']}__electric_mode_real"] = np.real(
                            mother_truth_closure["electric_mode_complex"]
                        ).astype(np.float32)
                        representative[f"truth__{sequence['case_id']}__electric_mode_imag"] = np.imag(
                            mother_truth_closure["electric_mode_complex"]
                        ).astype(np.float32)

            if not validation_rows:
                raise RuntimeError(f"No validation rows for {model_name}")
            validation_summary[model_name] = {
                "case_count": len(validation_rows),
                "target": "mother PIC density/electric/field_energy/total_energy",
                "density_relative_l2_macro": float(np.mean([
                    row["density_relative_l2"] for row in validation_rows
                ])),
                "charge_solvable_relative_l2_macro": float(np.mean([
                    row["charge_solvable_relative_l2"] for row in validation_rows
                ])),
                "electric_field_relative_l2_macro": float(np.mean([
                    row["electric_field_relative_l2"] for row in validation_rows
                ])),
                "field_energy_log10_rmse_macro": float(np.mean([
                    row["field_energy_log10_rmse"] for row in validation_rows
                ])),
                "total_energy_drift_rmse_macro": float(np.mean([
                    row["total_energy_drift_rmse_over_truth_field0"]
                    for row in validation_rows
                ])),
                "internal_cache_electric_field_relative_l2_macro": float(np.mean([
                    row["internal_cache_electric_field_relative_l2"]
                    for row in validation_rows
                ])),
            }
            summary = validation_summary[model_name]
            summary["closure_score"] = (
                0.30 * summary["charge_solvable_relative_l2_macro"]
                + 0.35 * summary["electric_field_relative_l2_macro"]
                + 0.25 * summary["field_energy_log10_rmse_macro"]
                + 0.10 * summary["total_energy_drift_rmse_macro"]
            )
            del stepper, snapshot, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    recommendation = min(
        validation_summary,
        key=lambda name: validation_summary[name]["closure_score"],
    )
    return rows, validation_summary, {
        "recommended_for_stage10b_initialization": recommendation,
        "selection_basis": "validation-only closure score against mother PIC outputs; density term uses zero-mean charge relative L2",
        "validation": validation_summary,
        "representative": representative,
    }

def plot_representative(output: Path, representative: dict[str, Any]) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    time_array = np.asarray(representative["phase_time"])
    case_ids = sorted({key.split("__")[1] for key in representative if key.startswith("truth__") and key.endswith("__field_energy")})
    for case_id in case_ids:
        truth = np.asarray(representative[f"truth__{case_id}__field_energy"])
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        axis.plot(time_array, truth / max(float(truth[0]), 1.0e-30), label="truth")
        for model_name in ("stage8d2c", "stage9b"):
            key = f"{model_name}__{case_id}__field_energy"
            if key in representative:
                values = np.asarray(representative[key])
                axis.plot(time_array, values / max(float(values[0]), 1.0e-30), label=model_name)
        axis.set_yscale("log")
        axis.set_xlabel("time")
        axis.set_ylabel("normalized electric-field energy")
        axis.set_title(case_id)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / f"{case_id}_field_energy.png", dpi=160)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    for path in (args.mother, args.cache, args.stage8d2c, args.stage9b, args.gpu_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        torch.cuda.set_device(int(device.index or 0))
    else:
        device = torch.device("cpu")

    actual_hashes = {
        "mother_sha256": sha256_file(args.mother),
        "cache_sha256": sha256_file(args.cache),
        "stage8d2c_sha256": sha256_file(args.stage8d2c),
        "stage9b_sha256": sha256_file(args.stage9b),
        "gpu_source_sha256": sha256_file(args.gpu_source),
    }
    hash_match = {key: actual_hashes[key] == EXPECTED[key] for key in EXPECTED}
    source_text = args.gpu_source.read_text(encoding="utf-8", errors="replace")
    source_markers = {
        "charge_density_1_minus_density": ("1.0 -" in source_text and "density" in source_text),
        "fft_poisson_divide_by_i_k": ("1j" in source_text and "wavenumber" in source_text),
        "zero_electric_mode": ("electric_hat" in source_text and "[0]" in source_text),
        "field_energy_half_dx_sum_e2": ("field_energy" in source_text and "0.5" in source_text and "electric" in source_text),
        "total_energy_field_plus_kinetic": ("total_energy" in source_text and "kinetic" in source_text),
    }
    source_audit = {
        "expected": EXPECTED,
        "actual": actual_hashes,
        "hash_match": hash_match,
        "source_markers": source_markers,
        "passed": bool(all(hash_match.values()) and all(source_markers.values())),
    }
    atomic_json(args.output / "source_audit.json", source_audit)

    with h5py.File(args.cache, "r") as cache:
        case_count = int(cache["cases/case_id"].shape[0])
        all_indices = np.arange(case_count, dtype=np.int64)
    truth_indices = all_indices if args.mode == "formal" else all_indices[:5]
    truth_rows, truth_worst = truth_closure_audit(args.mother, args.cache, truth_indices)
    truth_groups = aggregate_rows(truth_rows, ("evaluation_group",))
    write_csv(args.output / "truth_closure_case_metrics.csv", truth_rows)
    write_csv(args.output / "truth_closure_group_metrics.csv", truth_groups)

    val_indices = split_case_indices(args.cache, "val")
    test_indices = split_case_indices(args.cache, "test")
    if args.mode == "smoke":
        val_indices = val_indices[:2]
        test_indices = test_indices[:3]
    model_rows, validation_summary, recommendation_payload = evaluate_models(
        args.mother,
        args.cache,
        {"stage8d2c": args.stage8d2c, "stage9b": args.stage9b},
        {"val": val_indices, "test": test_indices},
        device,
        args.amp,
    )
    representative = recommendation_payload.pop("representative")
    model_groups = aggregate_rows(model_rows, ("model", "split", "evaluation_group"))
    model_alpha_groups = aggregate_rows(model_rows, ("model", "split", "alpha"))
    write_csv(args.output / "model_closure_case_metrics.csv", model_rows)
    write_csv(args.output / "model_closure_group_metrics.csv", model_groups)
    write_csv(args.output / "model_closure_alpha_metrics.csv", model_alpha_groups)
    atomic_json(args.output / "model_recommendation.json", recommendation_payload)
    np.savez_compressed(args.output / "representative_closure_rollouts.npz", **representative)
    plot_representative(args.output, representative)

    physics_contract = {
        "stage": "stage10a_density_poisson_energy_closure",
        "coordinate": "normalized_x xi=x/L, L=2*pi/k",
        "electron_density": "n_e(x,t)=integral f(x,v,t) dv",
        "charge_density": "rho=1-n_e",
        "poisson": "dE/dx=rho with periodic E_hat[0]=0",
        "fourier_solution": "E_hat[m]=rho_hat[m]/(i*k_m), k_m=m*k",
        "field_energy": "W_E=0.5*dx*sum_x E^2",
        "kinetic_energy": "W_K=L*mean_x integral 0.5*v^2*f dv",
        "total_energy": "W_total=W_K+W_E",
        "phase_grid": {"nx": 128, "nv": 193, "phase_count": 31},
        "important_limit": "The cache is an exact strided subset in v, not conservative averaging. Mother-density Poisson correctness and cache phase-space adequacy are audited separately.",
    }
    atomic_json(args.output / "physics_closure_contract.json", physics_contract)

    poisson_contract_pass = bool(all(row["poisson_contract_passed"] for row in truth_rows))
    cache_closure_pass = bool(all(row["cache_closure_passed"] for row in truth_rows))
    model_finite = bool(all(row["finite"] for row in model_rows))
    passed = bool(source_audit["passed"] and poisson_contract_pass and model_finite)
    if passed and cache_closure_pass:
        status = "PASS_AUDIT"
        next_stage = "Stage 10B field-aware/energy-aware loss design may proceed on the 128x193 cache."
    elif passed:
        status = "PASS_AUDIT_CACHE_LIMITED"
        next_stage = (
            "Poisson contract is valid, but the 128x193 strided velocity cache is not "
            "accurate enough for mother-PIC electric/field-energy closure. Build and audit "
            "a higher-v or conservative closure cache before Stage 10B energy-aware training."
        )
    else:
        status = "FAIL_CONTRACT"
        next_stage = "Stop before Stage 10B and repair the Poisson/source contract."
    acceptance = {
        "stage": "Stage 10A",
        "version": "stage10a_closure_audit_v4",
        "mode": args.mode,
        "status": status,
        "passed": passed,
        "scope": "physics contract audit and baseline closure evaluation; no gradient training and no model promotion",
        "source_audit_passed": source_audit["passed"],
        "poisson_contract_passed": poisson_contract_pass,
        "cache_closure_passed": cache_closure_pass,
        "truth_closure_passed": cache_closure_pass,
        "truth_case_count": len(truth_rows),
        "model_case_row_count": len(model_rows),
        "model_finite": model_finite,
        "truth_worst": truth_worst,
        "validation_only_recommendation": recommendation_payload,
        "runtime_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        "next_stage": next_stage,
    }
    atomic_json(args.output / "acceptance.json", acceptance)
    print(json.dumps(json_safe(acceptance), indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
