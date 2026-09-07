"""Adapter for the normalized ``continuum_v1`` Gkeyll dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np


REGIME_TO_INDEX = {"weak": 0, "transition": 1, "strong_nonlinear": 2}


@dataclass(frozen=True)
class ContinuumCase:
    case_id: str
    K: float
    alpha: float
    split: str
    regime: str
    path: Path


@dataclass(frozen=True)
class ContinuumTrajectory:
    state: np.ndarray
    heat_flux_gradient: np.ndarray
    time: np.ndarray
    x_over_l: np.ndarray
    K: float
    alpha: float
    case_id: str


def spectral_lowpass(value: np.ndarray, maximum_mode: int | None) -> np.ndarray:
    """Low-pass periodic values along their final dimension."""
    value = np.asarray(value)
    if maximum_mode is None:
        return value.astype(np.float32, copy=False)
    transformed = np.fft.rfft(value, axis=-1)
    transformed[..., int(maximum_mode) + 1 :] = 0.0
    return np.fft.irfft(
        transformed, n=value.shape[-1], axis=-1
    ).astype(np.float32)


def spectral_derivative(
    value: np.ndarray,
    fundamental_wavenumber: float,
    maximum_mode: int | None = None,
) -> np.ndarray:
    """Differentiate a periodic field defined on one wavelength."""
    value = np.asarray(value, dtype=np.float64)
    transformed = np.fft.rfft(value, axis=-1)
    modes = np.arange(transformed.shape[-1], dtype=np.float64)
    transformed *= 1j * float(fundamental_wavenumber) * modes
    if maximum_mode is not None:
        transformed[..., int(maximum_mode) + 1 :] = 0.0
    return np.fft.irfft(
        transformed, n=value.shape[-1], axis=-1
    ).astype(np.float32)


def load_continuum_case(
    case: ContinuumCase,
    *,
    input_maximum_mode: int | None = None,
    target_maximum_mode: int | None = None,
    temporal_stride: int = 1,
) -> ContinuumTrajectory:
    """Load ``[n,u,p]`` and construct ``d q / d x`` for one case.

    The stored channel contract is ``[density, velocity, temperature,
    central_heat_flux]``.  Pressure is therefore reconstructed as ``n*T``.
    Stored time coordinates are retained because output-trigger times differ
    slightly between trajectories.
    """
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")
    with h5py.File(case.path, "r") as handle:
        if str(handle.attrs.get("dataset_version", "")) != "continuum_v1":
            raise ValueError(f"Not a continuum_v1 trajectory: {case.path}")
        if str(handle.attrs.get("case_id", "")) != case.case_id:
            raise ValueError(f"case_id metadata mismatch in {case.path}")
        if int(handle.attrs.get("source_solver_use_gpu", 0)) != 1:
            raise ValueError(f"Non-CUDA trajectory is not eligible: {case.path}")
        K = float(handle.attrs["K"])
        alpha = float(handle.attrs["alpha"])
        if not np.isclose(K, case.K) or not np.isclose(alpha, case.alpha):
            raise ValueError(f"parameter metadata mismatch in {case.path}")
        central = np.asarray(
            handle["diagnostics/central_moments"][::temporal_stride],
            dtype=np.float32,
        )
        time = np.asarray(
            handle["diagnostics/time"][::temporal_stride], dtype=np.float64
        )
        x = np.asarray(handle["coordinates/x_cell"][:], dtype=np.float64)

    if central.ndim != 3 or central.shape[-1] != 4:
        raise ValueError(
            f"Expected central moments [time,x,4], got {central.shape}"
        )
    if time.shape != (central.shape[0],) or x.shape != (central.shape[1],):
        raise ValueError(f"coordinate shape mismatch in {case.path}")
    if not np.isfinite(central).all() or not np.isfinite(time).all():
        raise ValueError(f"non-finite diagnostic data in {case.path}")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError(f"non-monotonic time coordinate in {case.path}")

    density = central[..., 0]
    velocity = central[..., 1]
    temperature = central[..., 2]
    heat_flux = central[..., 3]
    if float(density.min()) <= 0.0 or float(temperature.min()) <= 0.0:
        raise ValueError(f"non-positive density or temperature in {case.path}")
    pressure = density * temperature
    state = np.stack((density, velocity, pressure), axis=1)
    state = spectral_lowpass(state, input_maximum_mode)
    gradient = spectral_derivative(heat_flux, K, target_maximum_mode)
    domain_length = 2.0 * np.pi / K
    x_over_l = x / domain_length
    return ContinuumTrajectory(
        state=np.ascontiguousarray(state, dtype=np.float32),
        heat_flux_gradient=np.ascontiguousarray(gradient, dtype=np.float32),
        time=time,
        x_over_l=x_over_l,
        K=K,
        alpha=alpha,
        case_id=case.case_id,
    )


def load_continuum_case_index(dataset_root: str | Path) -> list[ContinuumCase]:
    """Load the frozen whole-trajectory split after applying QC eligibility."""
    root = Path(dataset_root).resolve()
    split_path = root / "manifests" / "splits_v1.json"
    eligibility_path = root / "manifests" / "training_eligibility_v1.json"
    label_path = (
        root
        / "figures"
        / "rebound_audit_v1"
        / "production_rebound_labels.json"
    )
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    excluded = {item["case_id"] for item in eligibility["excluded_cases"]}
    regime_by_case = {item["case_id"]: item["regime"] for item in labels["cases"]}

    cases: list[ContinuumCase] = []
    for item in splits["assignments"]:
        identifier = str(item["case_id"])
        if identifier in excluded:
            continue
        regime = regime_by_case.get(identifier)
        if regime not in REGIME_TO_INDEX:
            raise ValueError(f"Missing or invalid regime for {identifier}")
        path = (
            root
            / "profiles"
            / "production"
            / "cases"
            / identifier
            / "processed"
            / "trajectory.h5"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        cases.append(
            ContinuumCase(
                case_id=identifier,
                K=float(item["K"]),
                alpha=float(item["alpha"]),
                split=str(item["canonical_split"]),
                regime=regime,
                path=path,
            )
        )

    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate eligible case IDs")
    expected = int(eligibility["counts"]["eligible"])
    if len(cases) != expected:
        raise ValueError(f"Expected {expected} eligible cases, found {len(cases)}")
    for split in ("train", "validation", "test"):
        expected_split = int(eligibility["eligible_by_canonical_split"][split])
        actual = sum(case.split == split for case in cases)
        if actual != expected_split:
            raise ValueError(
                f"Expected {expected_split} {split} cases, found {actual}"
            )
    return cases


__all__ = [
    "ContinuumCase",
    "ContinuumTrajectory",
    "REGIME_TO_INDEX",
    "load_continuum_case",
    "load_continuum_case_index",
    "spectral_derivative",
    "spectral_lowpass",
]
