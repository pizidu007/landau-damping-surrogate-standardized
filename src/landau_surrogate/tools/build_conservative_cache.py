#!/usr/bin/env python3
"""Build Stage 10A-2 higher-v and conservative closure-cache candidates."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from landau_surrogate.data.conservative import (
    atomic_json,
    conservative_m02_compress,
    json_safe,
    periodic_target_indices_for_stride,
    sha256_file,
    target_indices_for_stride,
    trapezoid_weights,
)

VERSION = "stage10a2_closure_cache_v2"
EXPECTED_MOTHER_SHA = "f92b58323b627ed526c29028abc0da0869173f6c380e8df4b76f424f89d1c722"
EXPECTED_REFERENCE_CACHE_SHA = "84fd51e09e3898555a690aef3df57625f48bc374dab3a27fa5856cf750801a2c"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--stage10a-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: float, lower: float, upper: float) -> float:
    return 2.0 * (value - lower) / (upper - lower) - 1.0


def create_candidate_file(
    path: Path,
    *,
    candidate_name: str,
    case_count: int,
    sample_count: int,
    nx: int,
    nv: int,
    compression: str,
) -> h5py.File:
    temporary = path.with_suffix(path.suffix + ".building")
    if path.exists():
        path.unlink()
    if temporary.exists():
        temporary.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(temporary, "w", libver="latest")
    handle.attrs["dataset_version"] = VERSION
    handle.attrs["status"] = "BUILDING"
    handle.attrs["created_at_utc"] = utc_now()
    handle.attrs["candidate_name"] = candidate_name
    handle.attrs["representation"] = "delta_f0"
    handle.attrs["normalization"] = "frozen_train_only_global_rms"
    handle.attrs["case_count"] = case_count
    handle.attrs["sample_count"] = sample_count
    handle.attrs["field_shape_json"] = json.dumps([nx, nv])

    selected_compression = None if compression == "none" else compression
    compression_opts = 4 if compression == "gzip" else None
    string_dtype = h5py.string_dtype("utf-8")

    cases = handle.create_group("cases")
    cases.create_dataset("case_id", shape=(case_count,), dtype=string_dtype)
    cases.create_dataset("parameter_origin", shape=(case_count,), dtype=string_dtype)
    for name, dtype in (
        ("k", np.float32),
        ("alpha", np.float32),
        ("phase_velocity", np.float32),
        ("split_code", np.uint8),
        ("group_code", np.uint8),
    ):
        cases.create_dataset(name, shape=(case_count,), dtype=dtype)

    samples = handle.create_group("samples")
    samples.create_dataset(
        "field",
        shape=(sample_count, nx, nv),
        dtype=np.float32,
        chunks=(1, nx, nv),
        compression=selected_compression,
        compression_opts=compression_opts,
        shuffle=selected_compression is not None,
    )
    samples.create_dataset("condition", shape=(sample_count, 3), dtype=np.float32)
    samples.create_dataset("physical_condition", shape=(sample_count, 3), dtype=np.float32)
    samples.create_dataset("case_index", shape=(sample_count,), dtype=np.int16)
    samples.create_dataset("time_index", shape=(sample_count,), dtype=np.int16)
    samples.create_dataset("split_code", shape=(sample_count,), dtype=np.uint8)
    samples.create_dataset("group_code", shape=(sample_count,), dtype=np.uint8)
    return handle


def finalize_candidate(path: Path, handle: h5py.File) -> None:
    temporary = Path(handle.filename)
    handle.attrs["status"] = "COMPLETE"
    handle.flush()
    handle.close()
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)

    acceptance = load_json(args.stage10a_acceptance)
    if acceptance.get("status") != "PASS_AUDIT_CACHE_LIMITED":
        raise RuntimeError("Stage 10A acceptance must be PASS_AUDIT_CACHE_LIMITED")
    if not acceptance.get("poisson_contract_passed", False):
        raise RuntimeError("Stage 10A Poisson contract did not pass")

    args.output.mkdir(parents=True, exist_ok=True)
    candidate_root = args.output / "caches"
    if candidate_root.exists() and any(candidate_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Candidate cache directory exists: {candidate_root}")
    if args.overwrite and candidate_root.exists():
        for child in candidate_root.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in sorted(candidate_root.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
    candidate_root.mkdir(parents=True, exist_ok=True)

    reference_sha = sha256_file(args.reference_cache)
    if reference_sha != EXPECTED_REFERENCE_CACHE_SHA:
        raise RuntimeError(f"Reference cache SHA mismatch: {reference_sha}")

    started = time.perf_counter()
    mother_sha = None
    if args.mode == "formal":
        mother_sha = sha256_file(args.mother)
        if mother_sha != EXPECTED_MOTHER_SHA:
            raise RuntimeError(f"Mother HDF5 SHA mismatch: {mother_sha}")

    with h5py.File(args.reference_cache, "r") as reference:
        normalization_path = Path(decode(reference.attrs["normalization_stats"]))
        normalization_sha = decode(reference.attrs["normalization_sha256"])
        delta_scale = float(reference.attrs["delta_global_rms"])
        reference_case_ids = [decode(value) for value in reference["cases/case_id"]]
        reference_phase_time = np.asarray(reference["grids/phase_time"], dtype=np.float64)
        reference_conditions = np.asarray(reference["samples/condition"], dtype=np.float32)
        reference_physical_conditions = np.asarray(
            reference["samples/physical_condition"], dtype=np.float32
        )
        reference_split = np.asarray(reference["samples/split_code"], dtype=np.uint8)
        reference_group = np.asarray(reference["samples/group_code"], dtype=np.uint8)
        reference_case_index = np.asarray(reference["samples/case_index"], dtype=np.int16)
        reference_time_index = np.asarray(reference["samples/time_index"], dtype=np.int16)
        reference_case_data = {
            name: np.asarray(reference[f"cases/{name}"])
            for name in ("k", "alpha", "phase_velocity", "split_code", "group_code")
        }
        reference_origins = [decode(value) for value in reference["cases/parameter_origin"]]

    actual_norm_sha = sha256_file(normalization_path)
    if actual_norm_sha != normalization_sha:
        raise RuntimeError("Normalization NPZ SHA mismatch")
    with np.load(normalization_path, allow_pickle=False) as archive:
        source_x_norm = np.asarray(archive["normalized_x"], dtype=np.float64)
        source_velocity = np.asarray(archive["velocity"], dtype=np.float64)
        source_phase_time = np.asarray(archive["phase_time"], dtype=np.float64)
        f0_full = np.asarray(archive["f0_train"], dtype=np.float64)
        k_min = float(np.asarray(archive["k_min"]).reshape(-1)[0])
        k_max = float(np.asarray(archive["k_max"]).reshape(-1)[0])
        alpha_min = float(np.asarray(archive["alpha_min"]).reshape(-1)[0])
        alpha_max = float(np.asarray(archive["alpha_max"]).reshape(-1)[0])

    if not np.allclose(source_phase_time, reference_phase_time):
        raise RuntimeError("Normalization and reference phase-time grids differ")
    if source_velocity.size != 1537 or source_x_norm.size != 512:
        raise RuntimeError("Unexpected frozen mother grid")
    if not math.isfinite(delta_scale) or delta_scale <= 0.0:
        raise RuntimeError("Invalid delta_global_rms")

    case_limit = 5 if args.mode == "smoke" else len(reference_case_ids)
    selected_case_ids = reference_case_ids[:case_limit]
    phase_count = source_phase_time.size
    sample_count = case_limit * phase_count
    x_indices = periodic_target_indices_for_stride(source_x_norm.size, 4)
    v385_indices = target_indices_for_stride(source_velocity.size, 4)
    v193_indices = target_indices_for_stride(source_velocity.size, 8)
    if x_indices.size != 128 or x_indices[0] != 0 or x_indices[-1] != 508:
        raise RuntimeError(
            f"Unexpected periodic x coarsening: size={x_indices.size}, "
            f"first={x_indices[0]}, last={x_indices[-1]}"
        )

    stride_name = "x128_v385_stride4_v2"
    conservative_name = "x128_v193_conservative_m02_v2"
    stride_path = candidate_root / stride_name / "landau_deltaf_cache_x128_v385_stride4_v2.h5"
    conservative_path = (
        candidate_root
        / conservative_name
        / "landau_deltaf_cache_x128_v193_conservative_m02_v2.h5"
    )

    stride_handle = create_candidate_file(
        stride_path,
        candidate_name=stride_name,
        case_count=case_limit,
        sample_count=sample_count,
        nx=x_indices.size,
        nv=v385_indices.size,
        compression=args.compression,
    )
    conservative_handle = create_candidate_file(
        conservative_path,
        candidate_name=conservative_name,
        case_count=case_limit,
        sample_count=sample_count,
        nx=x_indices.size,
        nv=v193_indices.size,
        compression=args.compression,
    )

    handles = {
        stride_name: stride_handle,
        conservative_name: conservative_handle,
    }
    paths = {
        stride_name: stride_path,
        conservative_name: conservative_path,
    }

    try:
        for name, handle in handles.items():
            handle.attrs["mode"] = args.mode
            handle.attrs["source_dataset"] = str(args.mother)
            handle.attrs["source_dataset_sha256"] = EXPECTED_MOTHER_SHA
            handle.attrs["reference_cache"] = str(args.reference_cache)
            handle.attrs["reference_cache_sha256"] = reference_sha
            handle.attrs["normalization_stats"] = str(normalization_path)
            handle.attrs["normalization_sha256"] = actual_norm_sha
            handle.attrs["delta_global_rms"] = delta_scale
            handle.attrs["x_stride"] = 4

        stride_handle.attrs["v_stride"] = 4
        stride_handle.attrs["coarsening_semantics"] = (
            "exact strided subset of the frozen mother velocity grid"
        )
        stride_handle.attrs["m0_preservation"] = "approximate trapezoidal quadrature"
        stride_handle.attrs["m2_preservation"] = "approximate trapezoidal quadrature"

        conservative_handle.attrs["v_stride"] = 8
        conservative_handle.attrs["coarsening_semantics"] = (
            "control-volume averages on stride-8 nodes plus smooth M2 projection; "
            "coarse trapezoidal quadrature preserves source M0 and M2 per x/frame"
        )
        conservative_handle.attrs["m0_preservation"] = "exact within floating-point"
        conservative_handle.attrs["m2_preservation"] = "exact within floating-point"

        stride_grids = stride_handle.create_group("grids")
        conservative_grids = conservative_handle.create_group("grids")
        for grids, v_indices in (
            (stride_grids, v385_indices),
            (conservative_grids, v193_indices),
        ):
            grids.create_dataset("normalized_x", data=source_x_norm[x_indices].astype(np.float32))
            grids.create_dataset("velocity", data=source_velocity[v_indices].astype(np.float32))
            grids.create_dataset("phase_time", data=source_phase_time.astype(np.float32))
            grids.create_dataset("source_x_index", data=x_indices.astype(np.int32))
            grids.create_dataset("source_v_index", data=v_indices.astype(np.int32))
            grids.create_dataset(
                "quadrature_weights",
                data=trapezoid_weights(source_velocity[v_indices]).astype(np.float64),
            )

        f0_stride = f0_full[v385_indices]
        f0_conservative, conservative_metadata = conservative_m02_compress(
            f0_full, source_velocity, stride=8
        )
        stride_grids.create_dataset("f0_train", data=f0_stride.astype(np.float32))
        conservative_grids.create_dataset("f0_train", data=f0_conservative.astype(np.float32))
        conservative_grids.create_dataset(
            "control_volume_left_index",
            data=np.asarray(conservative_metadata["left_indices"], dtype=np.int32),
        )
        conservative_grids.create_dataset(
            "control_volume_right_index",
            data=np.asarray(conservative_metadata["right_indices"], dtype=np.int32),
        )
        conservative_grids.create_dataset(
            "m2_projection_direction",
            data=np.asarray(conservative_metadata["projection_direction"], dtype=np.float64),
        )

        # Copy frozen conditions and case metadata from the validated reference cache.
        source_case_lookup = {identifier: index for index, identifier in enumerate(reference_case_ids)}
        for local_index, identifier in enumerate(selected_case_ids):
            source_index = source_case_lookup[identifier]
            for handle in handles.values():
                handle["cases/case_id"][local_index] = identifier
                handle["cases/parameter_origin"][local_index] = reference_origins[source_index]
                for name in reference_case_data:
                    handle[f"cases/{name}"][local_index] = reference_case_data[name][source_index]

        field_statistics = {
            name: {
                "sum": 0.0,
                "sumsq": 0.0,
                "count": 0,
                "minimum": math.inf,
                "maximum": -math.inf,
                "m0_residual_max": 0.0,
                "m2_residual_max": 0.0,
            }
            for name in handles
        }

        with h5py.File(args.mother, "r") as mother:
            mother_x = np.asarray(mother["grids/normalized_x"], dtype=np.float64)
            mother_v = np.asarray(mother["grids/velocity"], dtype=np.float64)
            mother_t = np.asarray(mother["grids/phase_time"], dtype=np.float64)
            if not (
                np.allclose(mother_x, source_x_norm)
                and np.allclose(mother_v, source_velocity)
                and np.allclose(mother_t, source_phase_time)
            ):
                raise RuntimeError("Mother and normalization grids differ")

            sample_cursor = 0
            for local_case, identifier in enumerate(selected_case_ids):
                source_case = source_case_lookup[identifier]
                group = mother["cases"][identifier]
                for time_index, time_value in enumerate(source_phase_time):
                    frame_full = np.asarray(
                        group["f_phase"][time_index, ::4, :], dtype=np.float64
                    )
                    frame_stride = frame_full[:, v385_indices]
                    frame_conservative, metadata = conservative_m02_compress(
                        frame_full, source_velocity, stride=8
                    )
                    normalized_stride = (
                        frame_stride - f0_stride[None, :]
                    ) / delta_scale
                    normalized_conservative = (
                        frame_conservative - f0_conservative[None, :]
                    ) / delta_scale
                    candidate_fields = {
                        stride_name: normalized_stride,
                        conservative_name: normalized_conservative,
                    }
                    for name, field in candidate_fields.items():
                        if not np.isfinite(field).all():
                            raise RuntimeError(f"{identifier} t={time_index}: non-finite {name}")
                        handle = handles[name]
                        handle["samples/field"][sample_cursor] = field.astype(np.float32)
                        source_sample = source_case * phase_count + time_index
                        handle["samples/condition"][sample_cursor] = reference_conditions[source_sample]
                        handle["samples/physical_condition"][sample_cursor] = (
                            reference_physical_conditions[source_sample]
                        )
                        handle["samples/case_index"][sample_cursor] = local_case
                        handle["samples/time_index"][sample_cursor] = reference_time_index[source_sample]
                        handle["samples/split_code"][sample_cursor] = reference_split[source_sample]
                        handle["samples/group_code"][sample_cursor] = reference_group[source_sample]
                        values = field.astype(np.float64, copy=False)
                        stats = field_statistics[name]
                        stats["sum"] += float(np.sum(values))
                        stats["sumsq"] += float(np.sum(values * values))
                        stats["count"] += int(values.size)
                        stats["minimum"] = min(stats["minimum"], float(np.min(values)))
                        stats["maximum"] = max(stats["maximum"], float(np.max(values)))
                    field_statistics[conservative_name]["m0_residual_max"] = max(
                        field_statistics[conservative_name]["m0_residual_max"],
                        float(metadata["m0_absolute_residual_max"]),
                    )
                    field_statistics[conservative_name]["m2_residual_max"] = max(
                        field_statistics[conservative_name]["m2_residual_max"],
                        float(metadata["m2_absolute_residual_max"]),
                    )
                    sample_cursor += 1
                print(f"build {local_case + 1}/{case_limit} {identifier}")

        if sample_cursor != sample_count:
            raise RuntimeError("sample count mismatch")

        for name, handle in handles.items():
            stats = field_statistics[name]
            handle.attrs["field_statistics_json"] = json.dumps(
                {
                    "mean": stats["sum"] / stats["count"],
                    "rms": math.sqrt(stats["sumsq"] / stats["count"]),
                    "minimum": stats["minimum"],
                    "maximum": stats["maximum"],
                    "finite": True,
                },
                sort_keys=True,
            )
            handle.attrs["projection_m0_absolute_residual_max"] = stats["m0_residual_max"]
            handle.attrs["projection_m2_absolute_residual_max"] = stats["m2_residual_max"]

        for name in list(handles):
            finalize_candidate(paths[name], handles[name])
            handles.pop(name, None)

    except BaseException:
        for handle in list(handles.values()):
            try:
                filename = Path(handle.filename)
                handle.close()
                if filename.exists():
                    filename.unlink()
            except Exception:
                pass
        raise

    candidates: dict[str, Any] = {}
    for name, path in paths.items():
        cache_sha = sha256_file(path)
        with h5py.File(path, "r") as handle:
            candidates[name] = {
                "path": str(path),
                "sha256": cache_sha,
                "size_bytes": path.stat().st_size,
                "field_shape": json.loads(decode(handle.attrs["field_shape_json"])),
                "case_count": int(handle.attrs["case_count"]),
                "sample_count": int(handle.attrs["sample_count"]),
                "coarsening_semantics": decode(handle.attrs["coarsening_semantics"]),
                "m0_preservation": decode(handle.attrs["m0_preservation"]),
                "m2_preservation": decode(handle.attrs["m2_preservation"]),
                "projection_m0_absolute_residual_max": float(
                    handle.attrs["projection_m0_absolute_residual_max"]
                ),
                "projection_m2_absolute_residual_max": float(
                    handle.attrs["projection_m2_absolute_residual_max"]
                ),
            }

    summary = {
        "stage": "Stage 10A-2",
        "version": VERSION,
        "mode": args.mode,
        "status": "PASS_BUILD",
        "passed": True,
        "source_dataset": str(args.mother),
        "source_dataset_sha256": mother_sha or EXPECTED_MOTHER_SHA,
        "reference_cache": str(args.reference_cache),
        "reference_cache_sha256": reference_sha,
        "normalization_stats": str(normalization_path),
        "normalization_sha256": actual_norm_sha,
        "delta_global_rms": delta_scale,
        "case_count": case_limit,
        "sample_count": sample_count,
        "case_ids": selected_case_ids,
        "candidates": candidates,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output / "build_summary.json", summary)
    print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
