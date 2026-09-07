
#!/usr/bin/env python3
"""Inspect the compact HDF5 cache contract without loading all fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from landau_surrogate.data.snapshot_cache import cache_metadata
from landau_surrogate.data.rollout_cache import cache_rollout_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    payload = {
        "snapshot_contract": cache_metadata(args.cache),
        "rollout_contract": cache_rollout_contract(args.cache),
    }
    with h5py.File(args.cache, "r") as handle:
        payload["root_attributes"] = {str(k): str(v) for k, v in handle.attrs.items()}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
