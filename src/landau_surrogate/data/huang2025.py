"""Adapter for the public Huang et al. (2025) Gkeyll/Vlasov MATLAB data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py
from scipy.io import loadmat


@dataclass(frozen=True)
class HuangTrajectory:
    state: np.ndarray
    heat_flux: np.ndarray
    heat_flux_gradient: np.ndarray
    time: np.ndarray
    x_over_l: np.ndarray
    k: float
    alpha: float
    source_kind: str
    moment_definition: str


def spectral_derivative(value: np.ndarray, k: float) -> np.ndarray:
    nx = value.shape[-1]
    modes = np.arange(nx // 2 + 1, dtype=np.float64) * float(k)
    return np.fft.irfft(1j * modes * np.fft.rfft(value, axis=-1), n=nx, axis=-1)


def load_huang_mat(
    path: str | Path,
    *,
    k: float = 0.35,
    alpha: float = 0.1,
    dt: float = 0.005,
) -> HuangTrajectory:
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as handle:
            density = np.asarray(handle["central_moments/density"], dtype=np.float32)
            velocity = np.asarray(handle["central_moments/velocity"], dtype=np.float32)
            pressure = np.asarray(handle["central_moments/pressure"], dtype=np.float32)
            heat_flux = np.asarray(handle["central_moments/heat_flux"], dtype=np.float32)
            time = np.asarray(handle["time"], dtype=np.float64)
            x = np.asarray(handle["x"], dtype=np.float64)
            k_value = float(handle.attrs.get("k", k))
            alpha_value = float(handle.attrs.get("alpha", alpha))
        shape = density.shape
        if len(shape) != 2 or any(value.shape != shape for value in (velocity, pressure, heat_flux)):
            raise ValueError(f"Expected matching [time, x] HDF5 moments, got {shape}")
        if time.shape != (shape[0],) or x.shape != (shape[1],):
            raise ValueError("HDF5 time/x coordinates do not match moment arrays")
        domain_length = 2.0 * np.pi / k_value
        return HuangTrajectory(
            state=np.stack((density, velocity, pressure), axis=1),
            heat_flux=heat_flux,
            heat_flux_gradient=spectral_derivative(heat_flux.astype(np.float64), k_value).astype(np.float32),
            time=time,
            x_over_l=x / domain_length,
            k=k_value,
            alpha=alpha_value,
            source_kind="gkeyll_hdf5",
            moment_definition="central",
        )
    data = loadmat(path)
    required = ("n_new", "u_new", "p_new", "q_new")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing Huang variables {missing} in {path}")
    arrays = [np.asarray(data[name], dtype=np.float32) for name in required]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise ValueError(f"Expected matching [time, x] arrays, got {[a.shape for a in arrays]}")
    density, velocity, pressure, heat_flux = arrays
    gradient = spectral_derivative(heat_flux.astype(np.float64), k).astype(np.float32)
    return HuangTrajectory(
        state=np.stack((density, velocity, pressure), axis=1),
        heat_flux=heat_flux,
        heat_flux_gradient=gradient,
        time=np.arange(shape[0], dtype=np.float64) * float(dt),
        x_over_l=(np.arange(shape[1], dtype=np.float64) + 0.5) / shape[1],
        k=float(k),
        alpha=float(alpha),
        source_kind="public_mat",
        moment_definition="exported_raw_M2_M3",
    )


def paper_like_indices(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interleave train/validation/test within the paper's first 8000 frames."""
    order = np.arange(min(count, 8_000), dtype=np.int64)
    phase = order % 10
    train = order[phase < 8]
    validation = order[phase == 8]
    test = order[phase == 9]
    return train, validation, test


def causal_indices(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within t=0..40, reserve t=30..40 for nonlinear extrapolation."""
    paper_count = min(count, 8_000)
    train_end = int(round(paper_count * 0.60))
    validation_end = int(round(paper_count * 0.75))
    order = np.arange(paper_count, dtype=np.int64)
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


__all__ = [
    "HuangTrajectory",
    "causal_indices",
    "load_huang_mat",
    "paper_like_indices",
    "spectral_derivative",
]
