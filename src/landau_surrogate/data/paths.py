"""Canonical external data locations for the Landau-surrogate project."""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    "/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized"
)


def data_root() -> Path:
    """Return the canonical data root, optionally overridden for portability."""
    value = os.environ.get("LANDAU_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else DEFAULT_DATA_ROOT


def historical_cache_path() -> Path:
    return (
        data_root()
        / "processed"
        / "historical_exact"
        / "landau_deltaf_cache_x128_v193_v1.h5"
    )


def conservative_cache_path() -> Path:
    return (
        data_root()
        / "processed"
        / "conservative_m02"
        / "landau_deltaf_cache_x128_v193_conservative_m02_v2.h5"
    )


def mother_dataset_path() -> Path:
    return data_root() / "raw" / "pic" / "landau_xv_master_110_v1.h5"


def nonlinear_runs_root() -> Path:
    return data_root() / "nonlinear" / "runs"
