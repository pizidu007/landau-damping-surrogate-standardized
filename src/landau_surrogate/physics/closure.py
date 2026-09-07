#!/usr/bin/env python3
"""Core density--Poisson--electric-field--energy closure utilities for Stage 10A."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np



def resolve_mean_anchor_beta(checkpoint: dict[str, Any]) -> tuple[float, str]:
    """Resolve mean-anchor beta across frozen checkpoint schema versions.

    Stage 8D-2C stores the value in ``candidate_contract`` while later
    checkpoints also promote it to a top-level key.  All discovered explicit
    values must agree; otherwise the checkpoint is rejected rather than
    silently choosing one.
    """
    candidates: list[tuple[str, float]] = []

    def add(path: str, container: Any) -> None:
        if isinstance(container, dict) and "mean_anchor_beta" in container:
            raw = container["mean_anchor_beta"]
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid {path}.mean_anchor_beta={raw!r}") from exc
            candidates.append((f"{path}.mean_anchor_beta", value))

    if "mean_anchor_beta" in checkpoint:
        try:
            candidates.append(("mean_anchor_beta", float(checkpoint["mean_anchor_beta"])))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid mean_anchor_beta={checkpoint['mean_anchor_beta']!r}"
            ) from exc

    for key in ("candidate_contract", "candidate_config", "model_contract"):
        add(key, checkpoint.get(key))

    # Last-resort compatibility for an otherwise self-contained historical
    # checkpoint whose selected-candidate name encodes the frozen beta.
    if not candidates:
        selected = str(checkpoint.get("selected_candidate", "")).strip().lower()
        match = re.fullmatch(r"mean_anchor_(\d{3})", selected)
        if match is not None:
            candidates.append(("selected_candidate", int(match.group(1)) / 100.0))

    if not candidates:
        raise RuntimeError(
            "Checkpoint does not define mean_anchor_beta in any supported schema: "
            "top-level, candidate_contract, candidate_config, model_contract, "
            "or selected_candidate=mean_anchor_XXX."
        )

    reference_path, reference = candidates[0]
    if not math.isfinite(reference) or not 0.0 <= reference <= 1.0:
        raise RuntimeError(f"Resolved {reference_path}={reference!r} is outside [0, 1].")
    for path, value in candidates[1:]:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Resolved {path}={value!r} is outside [0, 1].")
        if abs(value - reference) > 1.0e-12:
            details = ", ".join(f"{name}={item}" for name, item in candidates)
            raise RuntimeError(f"Conflicting mean_anchor_beta values: {details}")

    return float(reference), reference_path

def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray, axis: int) -> np.ndarray:
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(values, x=coordinates, axis=axis)


def relative_l2(prediction: np.ndarray, target: np.ndarray, floor: float = 1.0e-30) -> float:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    difference = prediction_array - target_array
    numerator = float(np.sum(np.abs(difference) ** 2, dtype=np.float64))
    denominator = max(
        float(np.sum(np.abs(target_array) ** 2, dtype=np.float64)),
        float(floor),
    )
    return float(math.sqrt(numerator / denominator))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(difference * difference)))


def phase_to_density(f_phase: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    values = np.asarray(f_phase, dtype=np.float64)
    velocity_array = np.asarray(velocity, dtype=np.float64)
    if values.shape[-1] != velocity_array.size:
        raise ValueError("Velocity axis does not match f_phase.")
    return trapezoidal_integral(values, velocity_array, axis=-1)


def solve_periodic_poisson_from_density(
    electron_density: np.ndarray,
    k_fundamental: float,
) -> dict[str, np.ndarray | float]:
    """Frozen normalized 1D contract: rho=1-ne, dE/dx=rho, E_0=0."""
    density = np.asarray(electron_density, dtype=np.float64)
    if density.ndim < 1:
        raise ValueError("electron_density must have an x axis.")
    nx = int(density.shape[-1])
    if nx < 2 or k_fundamental <= 0.0:
        raise ValueError("Invalid periodic grid or k.")
    domain_length = 2.0 * math.pi / float(k_fundamental)
    dx = domain_length / nx
    charge = 1.0 - density
    charge_hat = np.fft.rfft(charge, axis=-1)
    wave_numbers = 2.0 * math.pi * np.fft.rfftfreq(nx, d=dx)
    electric_hat = np.zeros_like(charge_hat, dtype=np.complex128)
    electric_hat[..., 1:] = charge_hat[..., 1:] / (1j * wave_numbers[1:])
    electric_hat[..., 0] = 0.0
    electric = np.fft.irfft(electric_hat, n=nx, axis=-1)
    derivative_hat = 1j * wave_numbers * electric_hat
    solvable_charge_hat = charge_hat.copy()
    solvable_charge_hat[..., 0] = 0.0
    poisson_residual = np.fft.irfft(
        derivative_hat - solvable_charge_hat, n=nx, axis=-1
    )
    return {
        "domain_length": float(domain_length),
        "dx": float(dx),
        "charge_density": charge,
        "charge_hat": charge_hat,
        "wave_numbers": wave_numbers,
        "electric_hat": electric_hat,
        "electric_field": electric,
        "poisson_residual": poisson_residual,
    }


def closure_from_f_phase(
    f_phase: np.ndarray,
    velocity: np.ndarray,
    k_fundamental: float,
) -> dict[str, np.ndarray | float]:
    phase = np.asarray(f_phase, dtype=np.float64)
    if phase.ndim == 2:
        phase = phase[None, ...]
    if phase.ndim != 3:
        raise ValueError(f"Expected [T,Nx,Nv], got {phase.shape}")
    velocity_array = np.asarray(velocity, dtype=np.float64)
    density = phase_to_density(phase, velocity_array)
    poisson = solve_periodic_poisson_from_density(density, k_fundamental)
    electric = np.asarray(poisson["electric_field"], dtype=np.float64)
    electric_hat = np.asarray(poisson["electric_hat"], dtype=np.complex128)
    domain_length = float(poisson["domain_length"])
    dx = float(poisson["dx"])
    field_energy = 0.5 * dx * np.sum(electric * electric, axis=-1)
    velocity_energy_density = trapezoidal_integral(
        0.5 * velocity_array[None, None, :] ** 2 * phase,
        velocity_array,
        axis=-1,
    )
    kinetic_energy = domain_length * np.mean(velocity_energy_density, axis=-1)
    total_energy = kinetic_energy + field_energy
    density_hat = np.fft.rfft(density, axis=-1)
    nx = phase.shape[1]
    electric_mode_complex = 2.0 * electric_hat[:, 1] / nx
    density_mode_complex = 2.0 * density_hat[:, 1] / nx
    return {
        "f_phase": phase,
        "electron_density": density,
        "charge_density": np.asarray(poisson["charge_density"]),
        "electric_field": electric,
        "electric_hat": electric_hat,
        "density_hat": density_hat,
        "electric_mode_complex": electric_mode_complex,
        "density_mode_complex": density_mode_complex,
        "electric_mode_source_scalar": -2.0 * np.imag(electric_hat[:, 1]) / nx,
        "density_mode_source_scalar": 2.0 * np.real(density_hat[:, 1]) / nx,
        "field_energy": field_energy,
        "kinetic_energy": kinetic_energy,
        "total_energy": total_energy,
        "poisson_residual": np.asarray(poisson["poisson_residual"]),
        "domain_length": domain_length,
        "dx": dx,
    }


def closure_from_normalized_delta(
    normalized_delta: np.ndarray,
    f0_train: np.ndarray,
    delta_global_rms: float,
    velocity: np.ndarray,
    k_fundamental: float,
) -> dict[str, np.ndarray | float]:
    normalized = np.asarray(normalized_delta, dtype=np.float64)
    if normalized.ndim == 2:
        normalized = normalized[None, ...]
    f0 = np.asarray(f0_train, dtype=np.float64)
    delta = normalized * float(delta_global_rms)
    f_phase = delta + f0[None, None, :]
    output = closure_from_f_phase(f_phase, velocity, k_fundamental)
    output["normalized_delta_f0"] = normalized
    output["delta_f0"] = delta
    return output


def match_time_indices(source_time: np.ndarray, requested_time: np.ndarray, atol: float = 1.0e-8) -> np.ndarray:
    source = np.asarray(source_time, dtype=np.float64)
    requested = np.asarray(requested_time, dtype=np.float64)
    indices = np.empty(requested.size, dtype=np.int64)
    for index, value in enumerate(requested):
        nearest = int(np.argmin(np.abs(source - value)))
        if abs(float(source[nearest] - value)) > atol:
            raise RuntimeError(f"Unable to match time {value}: nearest={source[nearest]}")
        indices[index] = nearest
    return indices


def phase_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_array = np.asarray(prediction, dtype=np.complex128)
    target_array = np.asarray(target, dtype=np.complex128)
    amplitude = np.abs(target_array)
    threshold = max(float(np.max(amplitude)) * 1.0e-6, 1.0e-14)
    mask = amplitude >= threshold
    if not np.any(mask):
        return 0.0
    difference = np.angle(prediction_array[mask] * np.conjugate(target_array[mask]))
    return float(np.mean(np.abs(np.angle(np.exp(1j * difference)))))


def normalized_curve(values: np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    denominator = max(abs(float(array[0])), floor)
    return array / denominator


def log_curve_rmse(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1.0e-12) -> float:
    pred = normalized_curve(prediction)
    truth = normalized_curve(target)
    return rmse(np.log10(np.maximum(pred, 0.0) + epsilon), np.log10(np.maximum(truth, 0.0) + epsilon))


def fit_effective_gamma(time: np.ndarray, field_energy: np.ndarray, start: float, stop: float) -> dict[str, float]:
    time_array = np.asarray(time, dtype=np.float64)
    energy = np.asarray(field_energy, dtype=np.float64)
    mask = (time_array >= start) & (time_array <= stop) & np.isfinite(energy) & (energy > 0.0)
    if np.count_nonzero(mask) < 3:
        return {"gamma": float("nan"), "r2": float("nan"), "count": int(np.count_nonzero(mask))}
    x = time_array[mask]
    y = 0.5 * np.log(energy[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / max(denominator, 1.0e-30)
    return {"gamma": float(slope), "r2": float(r2), "count": int(x.size)}


def model_closure_metrics(
    prediction: dict[str, Any],
    target: dict[str, Any],
    phase_time: np.ndarray,
) -> dict[str, float]:
    pred_field = np.asarray(prediction["field_energy"], dtype=np.float64)
    true_field = np.asarray(target["field_energy"], dtype=np.float64)
    pred_kinetic = np.asarray(prediction["kinetic_energy"], dtype=np.float64)
    true_kinetic = np.asarray(target["kinetic_energy"], dtype=np.float64)
    pred_total = np.asarray(prediction["total_energy"], dtype=np.float64)
    true_total = np.asarray(target["total_energy"], dtype=np.float64)
    field0 = max(abs(float(true_field[0])), 1.0e-12)
    pred_exchange = (pred_kinetic - pred_kinetic[0]) + (pred_field - pred_field[0])
    true_exchange = (true_kinetic - true_kinetic[0]) + (true_field - true_field[0])
    pred_total_drift = (pred_total - pred_total[0]) / field0
    true_total_drift = (true_total - true_total[0]) / field0
    pred_early = fit_effective_gamma(phase_time, pred_field, 0.5, 5.0)
    true_early = fit_effective_gamma(phase_time, true_field, 0.5, 5.0)
    pred_full = fit_effective_gamma(phase_time, pred_field, 0.5, 15.0)
    true_full = fit_effective_gamma(phase_time, true_field, 0.5, 15.0)
    return {
        "density_relative_l2": relative_l2(prediction["electron_density"], target["electron_density"]),
        "electric_field_relative_l2": relative_l2(prediction["electric_field"], target["electric_field"]),
        "electric_mode_complex_relative_l2": relative_l2(prediction["electric_mode_complex"], target["electric_mode_complex"]),
        "electric_mode_phase_mae": phase_mae(prediction["electric_mode_complex"], target["electric_mode_complex"]),
        "field_energy_relative_l2": relative_l2(pred_field, true_field),
        "field_energy_normalized_relative_l2": relative_l2(normalized_curve(pred_field), normalized_curve(true_field)),
        "field_energy_log10_rmse": log_curve_rmse(pred_field, true_field),
        "kinetic_energy_relative_l2": relative_l2(pred_kinetic, true_kinetic),
        "kinetic_energy_delta_relative_l2": relative_l2(pred_kinetic - pred_kinetic[0], true_kinetic - true_kinetic[0], floor=field0 * field0),
        "total_energy_relative_l2": relative_l2(pred_total, true_total),
        "total_energy_drift_rmse_over_truth_field0": rmse(pred_total_drift, true_total_drift),
        "energy_exchange_residual_rmse_over_truth_field0": rmse(pred_exchange / field0, true_exchange / field0),
        "pred_total_energy_relative_span": float((np.max(pred_total) - np.min(pred_total)) / max(abs(float(pred_total[0])), 1.0e-30)),
        "truth_total_energy_relative_span": float((np.max(true_total) - np.min(true_total)) / max(abs(float(true_total[0])), 1.0e-30)),
        "pred_mean_charge_absolute_max": float(np.max(np.abs(np.mean(prediction["charge_density"], axis=-1)))),
        "pred_poisson_residual_absolute_max": float(np.max(np.abs(prediction["poisson_residual"]))),
        "early_gamma_pred": pred_early["gamma"],
        "early_gamma_truth": true_early["gamma"],
        "early_gamma_absolute_error": abs(pred_early["gamma"] - true_early["gamma"]),
        "full_gamma_pred": pred_full["gamma"],
        "full_gamma_truth": true_full["gamma"],
        "full_gamma_absolute_error": abs(pred_full["gamma"] - true_full["gamma"]),
    }


def aggregate_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows_list = list(rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows_list:
        group_key = tuple(row.get(key) for key in keys)
        groups.setdefault(group_key, []).append(row)
    output: list[dict[str, Any]] = []
    excluded = set(keys) | {"case_id", "case_index"}
    for group_key, members in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        record = {key: value for key, value in zip(keys, group_key)}
        record["case_count"] = len(members)
        metric_names = sorted(set().union(*(member.keys() for member in members)) - excluded)
        for name in metric_names:
            values = []
            for member in members:
                value = member.get(name)
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    values.append(float(value))
            if values:
                record[f"{name}_mean"] = float(np.mean(values))
                record[f"{name}_max"] = float(np.max(values))
        output.append(record)
    return output
