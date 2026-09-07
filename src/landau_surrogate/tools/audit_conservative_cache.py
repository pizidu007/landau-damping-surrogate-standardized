#!/usr/bin/env python3
"""Audit and select Stage 10A-2 closure-cache candidates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from landau_surrogate.physics.closure import (
    closure_from_normalized_delta,
    match_time_indices,
    relative_l2,
    solve_periodic_poisson_from_density,
)
from landau_surrogate.data.conservative import (
    atomic_json,
    group_rows,
    json_safe,
    sha256_file,
    trapezoid_weights,
)

THRESHOLDS = {
    "density_relative_l2": 0.002,
    "charge_solvable_relative_l2": 0.10,
    "electric_field_relative_l2": 0.02,
    "field_energy_relative_l2": 0.05,
    "kinetic_energy_relative_l2": 0.01,
    "total_energy_relative_l2": 0.01,
    "poisson_residual_absolute_max": 1.0e-9,
    "mean_charge_absolute_max": 5.0e-3,
}


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--build-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def cache_paths_from_summary(summary: dict[str, Any], baseline: Path) -> dict[str, Path]:
    paths = {"x128_v193_stride8_baseline": baseline}
    candidates = summary.get("candidates", {})
    if not isinstance(candidates, dict) or not candidates:
        raise RuntimeError("build summary has no candidate caches")
    for name, payload in candidates.items():
        path = Path(payload["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != payload.get("sha256"):
            raise RuntimeError(f"candidate cache SHA mismatch: {name}")
        paths[str(name)] = path
    return paths


def audit_cache(
    cache_name: str,
    cache_path: Path,
    mother: h5py.File,
    allowed_case_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(cache_path, "r") as cache:
        if decode(cache.attrs.get("status", "")) != "COMPLETE":
            raise RuntimeError(f"cache is not COMPLETE: {cache_path}")
        velocity = np.asarray(cache["grids/velocity"], dtype=np.float64)
        stored_weights = (
            np.asarray(cache["grids/quadrature_weights"], dtype=np.float64)
            if "quadrature_weights" in cache["grids"]
            else trapezoid_weights(velocity)
        )
        expected_weights = trapezoid_weights(velocity)
        quadrature_weight_error = float(np.max(np.abs(stored_weights - expected_weights)))
        if quadrature_weight_error > 1.0e-12:
            raise RuntimeError(f"unsupported non-trapezoidal quadrature in {cache_name}")
        phase_time = np.asarray(cache["grids/phase_time"], dtype=np.float64)
        f0_train = np.asarray(cache["grids/f0_train"], dtype=np.float64)
        delta_scale = float(cache.attrs["delta_global_rms"])
        case_ids = [decode(value) for value in cache["cases/case_id"]]
        phase_count = phase_time.size
        sample_count = int(cache.attrs["sample_count"])
        if sample_count != len(case_ids) * phase_count:
            raise RuntimeError(f"sample count contract mismatch in {cache_name}")

        for case_index, identifier in enumerate(case_ids):
            if allowed_case_ids is not None and identifier not in allowed_case_ids:
                continue
            k_value = float(cache["cases/k"][case_index])
            alpha = float(cache["cases/alpha"][case_index])
            split_code = int(cache["cases/split_code"][case_index])
            group_code = int(cache["cases/group_code"][case_index])
            start = case_index * phase_count
            stop = start + phase_count
            normalized = np.asarray(cache["samples/field"][start:stop], dtype=np.float64)
            closure = closure_from_normalized_delta(
                normalized,
                f0_train,
                delta_scale,
                velocity,
                k_value,
            )

            group = mother["cases"][identifier]
            scalar_time = np.asarray(group["scalar_time"], dtype=np.float64)
            indices = match_time_indices(scalar_time, phase_time)
            mother_density = np.asarray(group["density"][indices], dtype=np.float64)
            mother_electric = np.asarray(group["electric"][indices], dtype=np.float64)
            mother_field_energy = np.asarray(group["field_energy"][indices], dtype=np.float64)
            mother_total_energy = np.asarray(group["total_energy"][indices], dtype=np.float64)
            mother_kinetic = mother_total_energy - mother_field_energy

            mother_charge = 1.0 - mother_density
            candidate_charge = np.asarray(closure["charge_density"], dtype=np.float64)
            mother_solvable = mother_charge - np.mean(mother_charge, axis=-1, keepdims=True)
            candidate_solvable = candidate_charge - np.mean(
                candidate_charge, axis=-1, keepdims=True
            )

            row: dict[str, Any] = {
                "cache_name": cache_name,
                "case_index": case_index,
                "case_id": identifier,
                "k": k_value,
                "alpha": alpha,
                "split_code": split_code,
                "group_code": group_code,
                "density_relative_l2": relative_l2(
                    closure["electron_density"], mother_density
                ),
                "charge_solvable_relative_l2": relative_l2(
                    candidate_solvable, mother_solvable
                ),
                "electric_field_relative_l2": relative_l2(
                    closure["electric_field"], mother_electric
                ),
                "field_energy_relative_l2": relative_l2(
                    closure["field_energy"], mother_field_energy
                ),
                "kinetic_energy_relative_l2": relative_l2(
                    closure["kinetic_energy"], mother_kinetic
                ),
                "total_energy_relative_l2": relative_l2(
                    closure["total_energy"], mother_total_energy
                ),
                "density_absolute_max_error": float(
                    np.max(np.abs(np.asarray(closure["electron_density"]) - mother_density))
                ),
                "electric_absolute_max_error": float(
                    np.max(np.abs(np.asarray(closure["electric_field"]) - mother_electric))
                ),
                "poisson_residual_absolute_max": float(
                    np.max(np.abs(np.asarray(closure["poisson_residual"])))
                ),
                "mean_charge_absolute_max": float(
                    np.max(np.abs(np.mean(candidate_charge, axis=-1)))
                ),
                "finite": bool(
                    all(
                        np.isfinite(np.asarray(closure[key])).all()
                        for key in (
                            "electron_density",
                            "electric_field",
                            "field_energy",
                            "kinetic_energy",
                            "total_energy",
                        )
                    )
                ),
            }
            row["field_ready"] = bool(
                row["finite"]
                and row["density_relative_l2"] <= THRESHOLDS["density_relative_l2"]
                and row["electric_field_relative_l2"] <= THRESHOLDS["electric_field_relative_l2"]
                and row["field_energy_relative_l2"] <= THRESHOLDS["field_energy_relative_l2"]
                and row["poisson_residual_absolute_max"]
                <= THRESHOLDS["poisson_residual_absolute_max"]
                and row["mean_charge_absolute_max"] <= THRESHOLDS["mean_charge_absolute_max"]
            )
            row["closure_ready"] = bool(
                row["field_ready"]
                and row["charge_solvable_relative_l2"]
                <= THRESHOLDS["charge_solvable_relative_l2"]
                and row["kinetic_energy_relative_l2"]
                <= THRESHOLDS["kinetic_energy_relative_l2"]
                and row["total_energy_relative_l2"]
                <= THRESHOLDS["total_energy_relative_l2"]
            )
            rows.append(row)

    worst: dict[str, Any] = {}
    for metric in (
        "density_relative_l2",
        "charge_solvable_relative_l2",
        "electric_field_relative_l2",
        "field_energy_relative_l2",
        "kinetic_energy_relative_l2",
        "total_energy_relative_l2",
        "poisson_residual_absolute_max",
        "mean_charge_absolute_max",
    ):
        selected = max(rows, key=lambda item: float(item[metric]))
        worst[metric] = {
            "case_id": selected["case_id"],
            "alpha": selected["alpha"],
            "value": selected[metric],
        }
    summary = {
        "cache_name": cache_name,
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "size_bytes": cache_path.stat().st_size,
        "case_count": len(rows),
        "field_ready_case_count": int(sum(bool(row["field_ready"]) for row in rows)),
        "closure_ready_case_count": int(sum(bool(row["closure_ready"]) for row in rows)),
        "all_cases_field_ready": bool(all(bool(row["field_ready"]) for row in rows)),
        "all_cases_closure_ready": bool(all(bool(row["closure_ready"]) for row in rows)),
        "quadrature_weight_max_error": quadrature_weight_error,
        "worst": worst,
    }
    return rows, summary


def choose_candidate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    new_candidates = [
        payload
        for name, payload in summaries.items()
        if name != "x128_v193_stride8_baseline"
    ]
    eligible = [item for item in new_candidates if item["all_cases_closure_ready"]]
    if eligible:
        eligible.sort(
            key=lambda item: (
                int(item["size_bytes"]),
                float(item["worst"]["electric_field_relative_l2"]["value"]),
                float(item["worst"]["field_energy_relative_l2"]["value"]),
            )
        )
        selected = eligible[0]
        return {
            "status": "PASS_SELECTED",
            "passed": True,
            "selected_cache": selected["cache_name"],
            "selection_basis": (
                "all audited cases pass density/charge/electric/field-energy/kinetic/"
                "total-energy closure thresholds; choose smallest eligible cache"
            ),
            "next_stage": "Stage 10B field-aware and total-energy-aware training baseline",
        }

    field_only = [item for item in new_candidates if item["all_cases_field_ready"]]
    if field_only:
        field_only.sort(
            key=lambda item: (
                int(item["size_bytes"]),
                float(item["worst"]["electric_field_relative_l2"]["value"]),
            )
        )
        selected = field_only[0]
        return {
            "status": "PASS_SELECTED_FIELD_ONLY",
            "passed": True,
            "selected_cache": selected["cache_name"],
            "selection_basis": (
                "all cases pass density/electric/field-energy closure but at least one "
                "charge or kinetic/total-energy threshold remains unmet"
            ),
            "next_stage": (
                "Use the selected cache only for field-aware diagnostics; build x128_v769_stride2 "
                "before total-energy-aware Stage 10B training"
            ),
        }

    return {
        "status": "PASS_AUDIT_NO_ELIGIBLE_CACHE",
        "passed": True,
        "selected_cache": None,
        "selection_basis": "no new candidate passes all-case field closure",
        "next_stage": "Build and audit x128_v769_stride2 before Stage 10B",
    }


def save_representative(
    output: Path,
    mother_path: Path,
    caches: dict[str, Path],
    case_ids: list[str],
) -> None:
    payload: dict[str, np.ndarray] = {}
    first_cache = next(iter(caches.values()))
    with h5py.File(first_cache, "r") as handle:
        reference_phase_time = np.asarray(handle["grids/phase_time"], dtype=np.float64)
    payload["phase_time"] = reference_phase_time.astype(np.float32)
    with h5py.File(mother_path, "r") as mother:
        for identifier in case_ids:
            group = mother["cases"][identifier]
            scalar_time = np.asarray(group["scalar_time"], dtype=np.float64)
            indices = match_time_indices(scalar_time, reference_phase_time)
            payload[f"{identifier}__mother_field_energy"] = np.asarray(
                group["field_energy"][indices], dtype=np.float32
            )
        for cache_name, cache_path in caches.items():
            with h5py.File(cache_path, "r") as cache:
                ids = [decode(value) for value in cache["cases/case_id"]]
                velocity = np.asarray(cache["grids/velocity"], dtype=np.float64)
                f0 = np.asarray(cache["grids/f0_train"], dtype=np.float64)
                phase_time = np.asarray(cache["grids/phase_time"], dtype=np.float64)
                scale = float(cache.attrs["delta_global_rms"])
                for identifier in case_ids:
                    if identifier not in ids:
                        continue
                    index = ids.index(identifier)
                    start_index = index * phase_time.size
                    stop_index = start_index + phase_time.size
                    normalized = np.asarray(
                        cache["samples/field"][start_index:stop_index], dtype=np.float64
                    )
                    k_value = float(cache["cases/k"][index])
                    closure = closure_from_normalized_delta(
                        normalized, f0, scale, velocity, k_value
                    )
                    payload[f"{identifier}__{cache_name}__field_energy"] = np.asarray(
                        closure["field_energy"], dtype=np.float32
                    )
    np.savez_compressed(output, **payload)


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)

    args.output.mkdir(parents=True, exist_ok=True)
    audit_marker = args.output / "acceptance.json"
    if audit_marker.exists() and not args.overwrite:
        raise FileExistsError(f"audit output exists: {audit_marker}")
    if args.overwrite:
        for name in (
            "acceptance.json",
            "selected_cache_reference.json",
            "cache_audit_case_metrics.csv",
            "cache_audit_group_metrics.csv",
            "cache_audit_alpha_metrics.csv",
            "representative_cache_closure.npz",
            "cache_artifact_references.json",
            "cache_candidate_summaries.json",
            "selection_summary.json",
            "closure_thresholds.json",
        ):
            target = args.output / name
            if target.exists():
                target.unlink()
    build_summary = json.loads(args.build_summary.read_text(encoding="utf-8"))
    selected_case_ids = set(str(value) for value in build_summary.get("case_ids", []))
    if not selected_case_ids:
        raise RuntimeError("build summary does not contain case_ids")
    caches = cache_paths_from_summary(build_summary, args.baseline_cache)

    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    with h5py.File(args.mother, "r") as mother:
        for cache_name, cache_path in caches.items():
            rows, summary = audit_cache(
                cache_name, cache_path, mother, allowed_case_ids=selected_case_ids
            )
            all_rows.extend(rows)
            summaries[cache_name] = summary
            print(
                f"audit {cache_name}: closure_ready="
                f"{summary['closure_ready_case_count']}/{summary['case_count']}"
            )

    selection = choose_candidate(summaries)
    selected_name = selection.get("selected_cache")
    selected_reference = None
    if selected_name is not None:
        selected_reference = summaries[selected_name]
        atomic_json(args.output / "selected_cache_reference.json", selected_reference)

    write_csv(args.output / "cache_audit_case_metrics.csv", all_rows)
    write_csv(
        args.output / "cache_audit_group_metrics.csv",
        group_rows(all_rows, ("cache_name", "split_code", "group_code")),
    )
    write_csv(
        args.output / "cache_audit_alpha_metrics.csv",
        group_rows(all_rows, ("cache_name", "alpha")),
    )

    # Representative low/mid/high-alpha cases from the frozen parameter grid.
    with h5py.File(args.mother, "r") as mother_for_ids:
        available_case_ids = set(mother_for_ids["cases"].keys())
    representative_ids = [
        identifier
        for identifier in ("k0p45_a0p005", "k0p45_a0p025", "k0p45_a0p050")
        if identifier in available_case_ids
    ]
    save_representative(
        args.output / "representative_cache_closure.npz",
        args.mother,
        caches,
        representative_ids,
    )

    artifact_references = {
        name: {
            "path": str(path),
            "sha256": summaries[name]["sha256"],
            "size_bytes": summaries[name]["size_bytes"],
            "included_in_acceptance_archive": False,
        }
        for name, path in caches.items()
    }
    atomic_json(args.output / "cache_artifact_references.json", artifact_references)
    atomic_json(args.output / "cache_candidate_summaries.json", summaries)
    atomic_json(args.output / "selection_summary.json", selection)
    atomic_json(args.output / "closure_thresholds.json", THRESHOLDS)

    acceptance = {
        "stage": "Stage 10A-2",
        "version": "stage10a2_closure_cache_v2",
        "mode": args.mode,
        "status": selection["status"],
        "passed": selection["passed"],
        "scope": (
            "build and audit higher-velocity and conservative closure caches; "
            "no gradient training and no model promotion"
        ),
        "case_count_per_cache": next(iter(summaries.values()))["case_count"],
        "cache_count": len(caches),
        "baseline_cache": "x128_v193_stride8_baseline",
        "candidate_summaries": summaries,
        "selection": selection,
        "selected_cache_reference": selected_reference,
        "thresholds": THRESHOLDS,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output / "acceptance.json", acceptance)
    print(json.dumps(json_safe(acceptance), indent=2))


if __name__ == "__main__":
    main()
