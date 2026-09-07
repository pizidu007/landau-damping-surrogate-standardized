"""Metrics shared by heat-flux closure training, baselines, and rollout."""
from __future__ import annotations

from typing import Any

import numpy as np


def relative_l2(prediction: np.ndarray, target: np.ndarray, floor: float = 1.0e-12) -> float:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    error = np.linalg.norm(prediction_array - target_array)
    scale = max(float(np.linalg.norm(target_array)), floor)
    return float(error / scale)


def closure_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    difference = prediction - target
    pred_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    correlation = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    pred_spectrum = np.fft.rfft(prediction, axis=-1)
    target_spectrum = np.fft.rfft(target, axis=-1)
    return {
        "relative_l2": relative_l2(prediction, target),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "correlation": correlation,
        "spectral_relative_l2": relative_l2(pred_spectrum, target_spectrum),
        "prediction_mean_abs_max": float(np.max(np.abs(np.mean(prediction, axis=-1)))),
    }


def macro_average(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float]:
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }


__all__ = ["closure_metrics", "macro_average", "relative_l2"]
