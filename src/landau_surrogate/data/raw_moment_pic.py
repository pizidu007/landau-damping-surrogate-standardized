"""Adapters for single-pair raw-moment closure experiments on CUDA-PIC data."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class RawMomentPICTrajectory:
    pair_id: str
    k: float
    alpha: float
    time: np.ndarray
    x_over_l: np.ndarray
    public_state: np.ndarray  # [time, (M0, u, M2), x]
    raw_state: np.ndarray  # [time, (M0, M1, M2), x]
    third_moment: np.ndarray
    third_moment_gradient: np.ndarray
    electric: np.ndarray
    field_energy: np.ndarray
    replica_count: int
    source_paths: tuple[str, ...]


def spectral_derivative(value: np.ndarray, k: float) -> np.ndarray:
    modes = np.arange(value.shape[-1] // 2 + 1, dtype=np.float64) * float(k)
    return np.fft.irfft(
        1j * modes * np.fft.rfft(value, axis=-1), n=value.shape[-1], axis=-1
    )


def spectral_filter(value: np.ndarray, maximum_mode: int | None) -> np.ndarray:
    if maximum_mode is None:
        return value
    transformed = np.fft.rfft(value, axis=-1)
    transformed[..., maximum_mode + 1 :] = 0.0
    return np.fft.irfft(transformed, n=value.shape[-1], axis=-1)


def load_raw_moment_pic_pair(
    case_dir: str | Path,
    pair_id: str,
    *,
    maximum_target_mode: int | None = 24,
) -> RawMomentPICTrajectory:
    case_dir = Path(case_dir)
    paths = sorted(case_dir.glob(f"{pair_id}_s*.h5"))
    if not paths:
        raise FileNotFoundError(f"No CUDA-PIC replicas for {pair_id} in {case_dir}")
    raw_values: list[np.ndarray] = []
    electrics: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    time = None
    x_over_l = None
    k_value = None
    alpha = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("status", "")) != "COMPLETE":
                raise ValueError(f"Incomplete PIC case: {path}")
            case = json.loads(str(handle.attrs["case_json"]))
            current_time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
            current_x = np.asarray(handle["grids/normalized_x"], dtype=np.float64)
            if time is None:
                time, x_over_l = current_time, current_x
                k_value, alpha = float(case["k"]), float(case["alpha"])
            elif not np.array_equal(time, current_time) or not np.array_equal(x_over_l, current_x):
                raise ValueError(f"Replica grid mismatch in {path}")
            raw_values.append(np.stack([
                np.asarray(handle[f"raw_moments/m{order}"], dtype=np.float64)
                for order in range(4)
            ], axis=1))
            electrics.append(np.asarray(handle["fields/electric"], dtype=np.float64))
            energies.append(np.asarray(handle["energies/field_energy"], dtype=np.float64))
    assert time is not None and x_over_l is not None and k_value is not None and alpha is not None
    raw = np.mean(np.stack(raw_values), axis=0)
    electric = np.mean(np.stack(electrics), axis=0)
    field_energy = np.mean(np.stack(energies), axis=0)
    safe_density = np.maximum(raw[:, 0], np.finfo(np.float64).eps)
    public_state = np.stack((raw[:, 0], raw[:, 1] / safe_density, raw[:, 2]), axis=1)
    gradient = spectral_derivative(raw[:, 3], k_value)
    gradient = spectral_filter(gradient, maximum_target_mode)
    return RawMomentPICTrajectory(
        pair_id=pair_id, k=k_value, alpha=alpha, time=time, x_over_l=x_over_l,
        public_state=public_state.astype(np.float32), raw_state=raw[:, :3].astype(np.float32),
        third_moment=raw[:, 3].astype(np.float32),
        third_moment_gradient=gradient.astype(np.float32),
        electric=electric.astype(np.float32), field_energy=field_energy,
        replica_count=len(paths), source_paths=tuple(str(path) for path in paths),
    )


def interleaved_indices(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.arange(count, dtype=np.int64)
    phase = order % 10
    return order[phase < 8], order[phase == 8], order[phase == 9]


__all__ = [
    "RawMomentPICTrajectory", "interleaved_indices", "load_raw_moment_pic_pair",
    "spectral_derivative", "spectral_filter",
]
