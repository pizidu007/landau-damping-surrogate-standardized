#!/usr/bin/env python3
"""Verify organized data/model assets against the canonical JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/assets.json"),
    )
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "LANDAU_DATA_ROOT",
                "/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized",
            )
        ),
        help="Canonical external root used by manifest external_path entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    project_root = manifest_path.parents[2]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = []
    failed = False
    records = (
        payload["assets"]
        + payload.get("reference_results", [])
        + payload.get("source_references", [])
    )
    for asset in records:
        path = (
            args.data_root.resolve() / asset["external_path"]
            if "external_path" in asset
            else project_root / asset["path"]
        )
        if not path.is_file():
            status = "MISSING"
            failed = failed or args.require_all
            actual_size = None
            actual_sha256 = None
        else:
            actual_size = path.stat().st_size
            actual_sha256 = sha256_file(path)
            size_ok = actual_size == int(asset["size_bytes"])
            hash_ok = actual_sha256 == asset["sha256"]
            status = "OK" if size_ok and hash_ok else "MISMATCH"
            failed = failed or status != "OK"
        rows.append(
            {
                "id": asset["id"],
                "status": status,
                "path": str(path),
                "size_bytes": actual_size,
                "sha256": actual_sha256,
            }
        )

    print(json.dumps({"passed": not failed, "assets": rows}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
