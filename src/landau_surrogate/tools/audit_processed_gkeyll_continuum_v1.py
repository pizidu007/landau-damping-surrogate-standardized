"""Audit processed continuum_v1 trajectories against calibrated QC gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


DEFAULT_ROOT = Path(
    "/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1"
)
QC_VERSION = "continuum_v1_anchor_calibrated_2026-08-31"
BASE_LIMITS = {
    "mass_relative_drift": 1.0e-8,
    "momentum_absolute_max": 1.0e-6,
    "total_energy_relative_drift": 1.0e-4,
    "negative_sample_mass_fraction_max": 1.0e-3,
    "velocity_edge_sample_mass_fraction_max": 1.0e-8,
}
L2_LIMITS = {
    "smoke": 5.0e-2,
    "scout": 5.0e-2,
    "anchors_low": 2.0e-2,
    "anchors": 1.0e-2,
    "anchors_high": 5.0e-3,
    "production": 5.0e-3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def audit_profile(root: Path, profile: str, verify_checksums: bool) -> dict[str, Any]:
    plan = json.loads(
        (root / "manifests" / f"{profile}_plan.json").read_text(encoding="utf-8")
    )
    rows = []
    counts = {"complete": 0, "missing": 0, "invalid": 0}
    maxima: dict[str, float] = {}
    limits = {**BASE_LIMITS, "integrated_l2_relative_change": L2_LIMITS[profile]}
    expected_diagnostic_frames = int(plan["profile_settings"]["num_frames"]) + 1
    expected_kinetic_frames = (
        int(plan["profile_settings"]["num_frames"])
        // int(plan["profile_settings"]["distribution_frame_stride"])
        + 1
    )
    for item in plan["cases"]:
        case_root = root / "profiles" / profile / "cases" / item["case_id"]
        status_path = case_root / "processed" / "COMPLETE.json"
        trajectory = case_root / "processed" / "trajectory.h5"
        row: dict[str, Any] = {"case_id": item["case_id"], "errors": []}
        if not status_path.exists() or not trajectory.exists():
            row["status"] = "missing"
            counts["missing"] += 1
            rows.append(row)
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            row["errors"].append("processed status is not complete")
        if status.get("source_solver_use_gpu") != 1:
            row["errors"].append("processed source_solver_use_gpu is not 1")
        if status.get("diagnostic_frames") != expected_diagnostic_frames:
            row["errors"].append("diagnostic frame count differs from plan")
        if status.get("kinetic_frames") != expected_kinetic_frames:
            row["errors"].append("kinetic frame count differs from plan")
        if verify_checksums and _sha256(trajectory) != status.get("trajectory_sha256"):
            row["errors"].append("trajectory SHA256 differs")
        with h5py.File(trajectory, "r") as handle:
            if int(handle.attrs.get("source_solver_use_gpu", 0)) != 1:
                row["errors"].append("HDF5 source_solver_use_gpu is not 1")
            if str(handle.attrs.get("case_id", "")) != item["case_id"]:
                row["errors"].append("HDF5 case_id differs from plan")
            if int(handle["kinetic/valid"][:].min()) != 1:
                row["errors"].append("one or more kinetic frames are invalid")
        metrics = {name: float(status[name]) for name in limits}
        row["metrics"] = metrics
        for name, value in metrics.items():
            maxima[name] = max(maxima.get(name, 0.0), value)
            if value > limits[name]:
                row["errors"].append(
                    f"{name}={value:.6g} exceeds calibrated limit {limits[name]:.6g}"
                )
        if row["errors"]:
            row["status"] = "invalid"
            counts["invalid"] += 1
        else:
            row["status"] = "complete"
            counts["complete"] += 1
        rows.append(row)
    return {
        "profile": profile,
        "planned": len(plan["cases"]),
        "counts": counts,
        "limits": limits,
        "observed_maxima": maxima,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    root = args.dataset_root.resolve()
    reports = [audit_profile(root, profile, args.verify_checksums) for profile in args.profile]
    result = {
        "schema_version": 1,
        "qc_version": QC_VERSION,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "profiles": reports,
    }
    _atomic_json(root / "audit" / "processed_status.json", result)
    for report in reports:
        print(
            f"{report['profile']}: planned={report['planned']} "
            f"counts={json.dumps(report['counts'], sort_keys=True)}"
        )
    invalid = sum(report["counts"]["invalid"] for report in reports)
    missing = sum(report["counts"]["missing"] for report in reports)
    if invalid or (args.require_complete and missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
