#!/usr/bin/env python3
"""Download selected public checkpoints, verify SHA256, and preserve local files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/manifests/public_release_2026-09-09.json"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, asset: dict) -> bool:
    return (path.is_file() and path.stat().st_size == asset["size_bytes"]
            and digest_file(path) == asset["sha256"])


def destination(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe destination path: {relative}")
    # Existing symlinked directories must not redirect writes into shared assets.
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"Destination escapes output root: {relative}") from error
    return path


def download_asset(asset: dict, output_root: Path) -> str:
    path = destination(output_root, asset["path"])
    if path.exists() or path.is_symlink():
        if verified(path, asset):
            return "already verified"
        raise FileExistsError(f"Different local file exists; preserved without changes: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = Request(asset["download_url"], headers={"User-Agent": "landau-checkpoint-downloader"})
            with urlopen(request, timeout=60) as response:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > asset["size_bytes"]:
                        raise ValueError(f"Download exceeds expected size: {asset['id']}")
                    output.write(block)
        if not verified(temporary, asset):
            raise ValueError(f"Size or SHA256 mismatch: {asset['id']}")
        # Creating a hard link fails atomically if another process created path.
        # The temporary file is on the same filesystem as the destination.
        os.link(temporary, path)
        return "downloaded and verified"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="List assets without downloading.")
    selection.add_argument("--asset", nargs="+", help="One or more exact asset IDs from --list.")
    selection.add_argument("--group", choices=["legacy", "continuum_baseline", "round10", "round11", "all"])
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="Restore manifest-relative paths under this directory (default: repository).")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = manifest["checkpoints"]
    by_id = {asset["id"]: asset for asset in assets}
    if len(by_id) != len(assets):
        parser.error("Manifest contains duplicate IDs")
    if args.list:
        print(f"Release: {manifest['release_url']}")
        for asset in assets:
            print(f"{asset['id']:32} {asset['size_bytes']/2**20:7.2f} MiB  "
                  f"{asset['group']:19} {asset['status']}")
        return
    if args.asset:
        unknown = set(args.asset) - by_id.keys()
        if unknown:
            parser.error(f"Unknown asset IDs: {', '.join(sorted(unknown))}")
        selected = [by_id[name] for name in dict.fromkeys(args.asset)]
    else:
        selected = [asset for asset in assets if args.group == "all" or asset["group"] == args.group]
    if not selected:
        parser.error("No checkpoints selected")
    # Validate every destination before downloading the first asset.
    for asset in selected:
        destination(args.output_root, asset["path"])
    print(f"Selected {len(selected)} checkpoints, {sum(a['size_bytes'] for a in selected)/2**20:.2f} MiB")
    for asset in selected:
        for attempt in range(3):
            try:
                status = download_asset(asset, args.output_root)
                print(f"{asset['id']}: {status} -> {asset['path']}", flush=True)
                break
            except FileExistsError:
                raise
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)


if __name__ == "__main__":
    main()
