"""Audit GPU provenance and completion state for continuum_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def audit_profile(
    dataset_root: Path, profile: str, verify_checksums: bool
) -> dict[str, Any]:
    plan_path = dataset_root / "manifests" / f"{profile}_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    counts = {"complete": 0, "running": 0, "failed": 0, "missing": 0, "invalid": 0}
    for case in plan["cases"]:
        identifier = case["case_id"]
        case_root = dataset_root / "profiles" / profile / "cases" / identifier
        provenance = case_root / "provenance"
        complete_path = provenance / "COMPLETE.json"
        row: dict[str, Any] = {"case_id": identifier, "errors": []}
        if complete_path.exists():
            record = json.loads(complete_path.read_text(encoding="utf-8"))
            if record.get("status") != "complete":
                row["errors"].append("COMPLETE.json status is not complete")
            if record.get("cpu_fallback") is not False:
                row["errors"].append("CPU fallback flag is not false")
            if record.get("gkeyll_stat", {}).get("use_gpu") != 1:
                row["errors"].append("Gkeyll use_gpu is not 1")
            if record.get("parameters", {}).get("K") != case.get("K"):
                row["errors"].append("K differs from immutable plan")
            if record.get("parameters", {}).get("alpha") != case.get("alpha"):
                row["errors"].append("alpha differs from immutable plan")
            checksum_path = provenance / "raw_files.sha256"
            if not checksum_path.exists():
                row["errors"].append("raw checksum manifest is missing")
            elif _sha256(checksum_path) != record.get("raw_checksums_sha256"):
                row["errors"].append("raw checksum manifest hash differs")
            if verify_checksums and checksum_path.exists():
                for line in checksum_path.read_text(encoding="utf-8").splitlines():
                    expected, relative = line.split(maxsplit=1)
                    raw_path = case_root / relative
                    if not raw_path.exists() or _sha256(raw_path) != expected:
                        row["errors"].append(f"checksum failed: {relative}")
            if row["errors"]:
                row["status"] = "invalid"
                counts["invalid"] += 1
            else:
                row["status"] = "complete"
                row["solver_seconds"] = record["gkeyll_stat"]["total_tm"]
                row["raw_gkyl_file_count"] = record["raw_gkyl_file_count"]
                counts["complete"] += 1
        elif (provenance / "RUNNING.lock").exists():
            row["status"] = "running"
            counts["running"] += 1
        elif (provenance / "FAILED.json").exists():
            row["status"] = "failed"
            counts["failed"] += 1
        else:
            row["status"] = "missing"
            counts["missing"] += 1
        rows.append(row)
    return {
        "profile": profile,
        "planned": len(plan["cases"]),
        "counts": counts,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    root = args.dataset_root.resolve()
    profiles = args.profile or sorted(
        path.name.removesuffix("_plan.json")
        for path in (root / "manifests").glob("*_plan.json")
    )
    reports = [audit_profile(root, name, args.verify_checksums) for name in profiles]
    result = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "profiles": reports,
    }
    _atomic_json(root / "audit" / "latest_status.json", result)
    for report in reports:
        print(
            f"{report['profile']}: planned={report['planned']} "
            f"counts={json.dumps(report['counts'], sort_keys=True)}"
        )
    invalid = sum(item["counts"]["invalid"] for item in reports)
    failed = sum(item["counts"]["failed"] for item in reports)
    incomplete = sum(
        item["counts"]["missing"] + item["counts"]["running"] for item in reports
    )
    if invalid or failed or (args.require_complete and incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
