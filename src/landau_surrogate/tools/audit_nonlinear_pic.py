"""Audit independently written nonlinear CUDA-PIC case files."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from landau_surrogate.tools.generate_nonlinear_pic import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/nonlinear_pic_v1.json"))
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "paper_match", "formal"), required=True
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    solver, expected = load_config(args.config, args.profile)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    for case in expected:
        path = args.run_dir / "cases" / f"{case['case_id']}.h5"
        if not path.is_file():
            missing.append(case["case_id"])
            continue
        try:
            with h5py.File(path, "r") as handle:
                status = str(handle.attrs.get("status", ""))
                total = np.asarray(handle["energies/total_energy"], dtype=np.float64)
                overflow = np.asarray(
                    handle.get("phase_space/overflow_fraction", np.asarray([0.0])),
                    dtype=np.float64,
                )
                heat_flux_gradient = np.asarray(
                    handle["fluid/heat_flux_gradient"], dtype=np.float64
                )
                pressure = np.asarray(handle["fluid/pressure"], dtype=np.float64)
                finite = bool(
                    np.isfinite(total).all()
                    and np.isfinite(heat_flux_gradient).all()
                    and np.isfinite(pressure).all()
                )
                row = {
                    "case_id": case["case_id"],
                    "k": case["k"],
                    "alpha": case["alpha"],
                    "seed": case.get("seed", 0),
                    "split": case["split"],
                    "status": status,
                    "size_bytes": path.stat().st_size,
                    "runtime_seconds": float(handle.attrs["runtime_seconds"]),
                    "peak_memory_gib": int(handle.attrs["peak_memory_bytes"]) / 1024**3,
                    "total_energy_relative_span": float(np.ptp(total / total[0] - 1.0)),
                    "maximum_phase_overflow_fraction": float(np.max(overflow)),
                    "heat_flux_gradient_mean_abs_max": float(
                        np.max(np.abs(np.mean(heat_flux_gradient, axis=-1)))
                    ),
                    "pressure_minimum": float(np.min(pressure)),
                    "finite": finite,
                }
                row["passed"] = bool(
                    status == "COMPLETE"
                    and finite
                    and row["maximum_phase_overflow_fraction"] <= 1.0e-6
                    and row["heat_flux_gradient_mean_abs_max"] <= 1.0e-6
                    and row["pressure_minimum"] >= 0.0
                )
                if not row["passed"]:
                    failed.append(case["case_id"])
                rows.append(row)
        except (OSError, KeyError, ValueError) as error:
            failed.append(case["case_id"])
            rows.append({"case_id": case["case_id"], "passed": False, "error": repr(error)})

    args.run_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    csv_path = args.run_dir / "case_metrics.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    valid_rows = [row for row in rows if "runtime_seconds" in row]
    split_counts = {
        split: sum(row.get("split") == split for row in valid_rows)
        for split in ("train", "validation", "test")
    }
    aggregate_metrics = {}
    if valid_rows:
        aggregate_metrics = {
            "total_case_bytes": sum(int(row["size_bytes"]) for row in valid_rows),
            "runtime_seconds_sum": sum(float(row["runtime_seconds"]) for row in valid_rows),
            "runtime_seconds_mean": float(
                np.mean([float(row["runtime_seconds"]) for row in valid_rows])
            ),
            "runtime_seconds_min": min(float(row["runtime_seconds"]) for row in valid_rows),
            "runtime_seconds_max": max(float(row["runtime_seconds"]) for row in valid_rows),
            "peak_memory_gib_max": max(float(row["peak_memory_gib"]) for row in valid_rows),
            "total_energy_relative_span_max": max(
                float(row["total_energy_relative_span"]) for row in valid_rows
            ),
            "maximum_phase_overflow_fraction": max(
                float(row["maximum_phase_overflow_fraction"]) for row in valid_rows
            ),
            "heat_flux_gradient_mean_abs_max": max(
                float(row["heat_flux_gradient_mean_abs_max"]) for row in valid_rows
            ),
            "pressure_minimum": min(float(row["pressure_minimum"]) for row in valid_rows),
        }

    summary = {
        "dataset_version": "nonlinear_cuda_pic_m03_v1",
        "profile": args.profile,
        "expected_case_count": len(expected),
        "completed_case_count": len(rows),
        "passed_case_count": sum(bool(row.get("passed")) for row in rows),
        "missing_case_ids": missing,
        "failed_case_ids": sorted(set(failed)),
        "complete": not missing and not failed,
        "split_counts": split_counts,
        "aggregate_metrics": aggregate_metrics,
        "solver": solver.__dict__,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.require_complete and not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
