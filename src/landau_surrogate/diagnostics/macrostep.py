"""Diagnostics for full conservative-field macro-step rollouts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


CHANNEL_NAMES = ("M0", "M1", "M2", "E")


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(
        np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1.0e-30)
    )


def field_energy(state: np.ndarray, k_value: float) -> np.ndarray:
    """Return ``0.5*integral(E^2 dx)`` for ``[...,4,Nx]`` states."""
    state = np.asarray(state, dtype=np.float64)
    if state.ndim < 2 or state.shape[-2] != 4:
        raise ValueError(f"Expected [...,4,Nx], got {state.shape}")
    domain_length = 2.0 * np.pi / float(k_value)
    return 0.5 * domain_length * np.mean(state[..., 3, :] ** 2, axis=-1)


def total_energy(state: np.ndarray, k_value: float) -> np.ndarray:
    """Return electron kinetic plus electric energy in normalized units."""
    state = np.asarray(state, dtype=np.float64)
    domain_length = 2.0 * np.pi / float(k_value)
    return 0.5 * domain_length * np.mean(
        state[..., 2, :] + state[..., 3, :] ** 2, axis=-1
    )


def normalized_log10_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not np.any(valid):
        return float("nan")
    prediction = prediction / max(float(prediction.flat[0]), 1.0e-30)
    truth = truth / max(float(truth.flat[0]), 1.0e-30)
    floor = 1.0e-10
    difference = np.log10(np.maximum(prediction[valid], floor)) - np.log10(
        np.maximum(truth[valid], floor)
    )
    return float(np.sqrt(np.mean(difference**2)))


def _wrapped_phase_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (first - second)))


def mode_one_metrics(
    prediction: np.ndarray, truth: np.ndarray, channel: int = 3
) -> dict[str, float]:
    prediction_mode = np.fft.rfft(prediction[:, channel], axis=-1)[:, 1]
    truth_mode = np.fft.rfft(truth[:, channel], axis=-1)[:, 1]
    amplitude_error = relative_l2(np.abs(prediction_mode), np.abs(truth_mode))
    threshold = max(float(np.max(np.abs(truth_mode))) * 1.0e-6, 1.0e-12)
    valid = np.abs(truth_mode) > threshold
    phase = _wrapped_phase_difference(
        np.angle(prediction_mode[valid]), np.angle(truth_mode[valid])
    )
    return {
        "amplitude_relative_l2": amplitude_error,
        "phase_mae_radians": float(np.mean(np.abs(phase))) if len(phase) else 0.0,
    }


def trajectory_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    time: np.ndarray,
    k_value: float,
    density_floor: float = 1.0e-4,
    pressure_floor: float = 1.0e-5,
    divergence_limit: float = 1.0e6,
) -> dict[str, Any]:
    """Evaluate one complete free rollout without silently clamping states."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and truth must both be [time,4,Nx]")
    if prediction.shape[1] != 4 or time.shape != (prediction.shape[0],):
        raise ValueError("invalid state or time shape")
    finite_frame = np.isfinite(prediction).all(axis=(1, 2))
    bounded_frame = np.max(np.abs(prediction), axis=(1, 2)) < divergence_limit
    valid_frame = finite_frame & bounded_frame
    first_invalid = np.flatnonzero(~valid_frame)
    valid_count = int(first_invalid[0]) if len(first_invalid) else len(time)
    if valid_count == 0:
        return {
            "completed_time": float(time[0]),
            "finite_to_final_time": False,
            "valid_frame_count": 0,
        }
    prediction_valid = prediction[:valid_count]
    truth_valid = truth[:valid_count]
    time_valid = time[:valid_count]
    density = prediction_valid[:, 0]
    pressure = (
        prediction_valid[:, 2]
        - prediction_valid[:, 1] ** 2 / np.maximum(density, 1.0e-30)
    )
    pred_field_energy = field_energy(prediction_valid, k_value)
    truth_field_energy = field_energy(truth_valid, k_value)
    pred_total_energy = total_energy(prediction_valid, k_value)
    channel_metrics = {
        name: relative_l2(prediction_valid[:, index], truth_valid[:, index])
        for index, name in enumerate(CHANNEL_NAMES)
    }
    equilibrium = np.zeros_like(truth_valid)
    equilibrium[:, 0] = 1.0
    equilibrium[:, 2] = 1.0
    channel_perturbation_metrics = {
        name: float(
            np.linalg.norm(prediction_valid[:, index] - truth_valid[:, index])
            / max(
                np.linalg.norm(truth_valid[:, index] - equilibrium[:, index]),
                1.0e-30,
            )
        )
        for index, name in enumerate(CHANNEL_NAMES)
    }
    return {
        "completed_time": float(time_valid[-1]),
        "finite_to_final_time": bool(valid_count == len(time)),
        "valid_frame_count": valid_count,
        "state_relative_l2": relative_l2(prediction_valid, truth_valid),
        "state_perturbation_relative_l2": relative_l2(
            prediction_valid - equilibrium, truth_valid - equilibrium
        ),
        "channel_relative_l2": channel_metrics,
        "channel_perturbation_relative_l2": channel_perturbation_metrics,
        "field_energy_log10_rmse": normalized_log10_rmse(
            pred_field_energy, truth_field_energy
        ),
        "electric_mode_one": mode_one_metrics(
            prediction_valid, truth_valid, channel=3
        ),
        "mass_max_relative_drift": float(
            np.max(np.abs(density.mean(axis=-1) - density[0].mean()))
            / max(abs(float(density[0].mean())), 1.0e-30)
        ),
        "momentum_max_absolute_drift": float(
            np.max(
                np.abs(
                    prediction_valid[:, 1].mean(axis=-1)
                    - prediction_valid[0, 1].mean()
                )
            )
        ),
        "total_energy_max_relative_drift": float(
            np.max(np.abs(pred_total_energy - pred_total_energy[0]))
            / max(abs(float(pred_total_energy[0])), 1.0e-30)
        ),
        "minimum_density": float(np.min(density)),
        "minimum_pressure": float(np.min(pressure)),
        "density_violation_count": int(np.sum(density < density_floor)),
        "pressure_violation_count": int(np.sum(pressure < pressure_floor)),
    }


def aggregate_case_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("case metrics are empty")
    complete = [bool(row.get("finite_to_final_time", False)) for row in rows]
    energy = np.asarray(
        [float(row["field_energy_log10_rmse"]) for row in rows], dtype=np.float64
    )
    state = np.asarray(
        [float(row["state_relative_l2"]) for row in rows], dtype=np.float64
    )
    perturbation = np.asarray(
        [float(row["state_perturbation_relative_l2"]) for row in rows],
        dtype=np.float64,
    )
    violation_free = [
        int(row.get("density_violation_count", 0)) == 0
        and int(row.get("pressure_violation_count", 0)) == 0
        for row in rows
    ]
    return {
        "case_count": len(rows),
        "complete_case_count": int(sum(complete)),
        "violation_free_case_count": int(sum(violation_free)),
        "field_energy_log10_rmse_median": float(np.nanmedian(energy)),
        "field_energy_log10_rmse_mean": float(np.nanmean(energy)),
        "state_relative_l2_median": float(np.nanmedian(state)),
        "state_relative_l2_mean": float(np.nanmean(state)),
        "state_perturbation_relative_l2_median": float(
            np.nanmedian(perturbation)
        ),
        "state_perturbation_relative_l2_mean": float(np.nanmean(perturbation)),
    }


__all__ = [
    "aggregate_case_metrics",
    "field_energy",
    "mode_one_metrics",
    "normalized_log10_rmse",
    "relative_l2",
    "total_energy",
    "trajectory_metrics",
]
