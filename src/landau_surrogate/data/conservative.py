#!/usr/bin/env python3
"""Core compression and audit utilities for Stage 10A-2 closure caches."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    x = np.asarray(coordinates, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("coordinates must be one-dimensional with at least two points")
    differences = np.diff(x)
    if not np.all(np.isfinite(differences)) or np.any(differences <= 0.0):
        raise ValueError("coordinates must be strictly increasing and finite")
    weights = np.empty_like(x)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    return weights


def weighted_integral(values: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    quadrature = np.asarray(weights, dtype=np.float64)
    if array.shape[axis] != quadrature.size:
        raise ValueError("quadrature length does not match integration axis")
    shape = [1] * array.ndim
    shape[axis] = quadrature.size
    return np.sum(array * quadrature.reshape(shape), axis=axis, dtype=np.float64)


def target_indices_for_stride(point_count: int, stride: int) -> np.ndarray:
    if point_count < 2 or stride < 1:
        raise ValueError("invalid point_count or stride")
    indices = np.arange(0, point_count, stride, dtype=np.int64)
    if indices[-1] != point_count - 1:
        raise ValueError(
            f"stride={stride} does not include final source point for count={point_count}"
        )
    return indices



def periodic_target_indices_for_stride(point_count: int, stride: int) -> np.ndarray:
    """Return exact stride indices for a periodic grid without a duplicate endpoint.

    A periodic nodal grid with ``point_count`` samples represents [0, L) and must
    not include an additional sample at L. Therefore a 512-point periodic grid
    with stride 4 correctly maps to indices 0, 4, ..., 508 (128 points).
    """
    if point_count < 1 or stride < 1:
        raise ValueError("invalid point_count or stride")
    if point_count % stride != 0:
        raise ValueError(
            f"periodic point_count={point_count} is not divisible by stride={stride}"
        )
    indices = np.arange(0, point_count, stride, dtype=np.int64)
    expected_count = point_count // stride
    if indices.size != expected_count or indices[-1] != point_count - stride:
        raise RuntimeError("periodic stride-index contract failure")
    return indices

def control_volume_index_bounds(point_count: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride % 2 != 0:
        raise ValueError("control-volume compression requires an even stride")
    target = target_indices_for_stride(point_count, stride)
    half = stride // 2
    left = target - half
    right = target + half
    left[0] = 0
    right[-1] = point_count - 1
    if np.any(left < 0) or np.any(right >= point_count):
        raise RuntimeError("invalid control-volume bounds")
    if np.any(left[1:] != right[:-1]):
        raise RuntimeError("control volumes do not partition the source domain")
    return left.astype(np.int64), right.astype(np.int64)


def cumulative_trapezoid(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    x = np.asarray(coordinates, dtype=np.float64)
    if array.shape[-1] != x.size:
        raise ValueError("velocity axis mismatch")
    increments = 0.5 * (array[..., 1:] + array[..., :-1]) * np.diff(x)
    zeros = np.zeros(array.shape[:-1] + (1,), dtype=np.float64)
    return np.concatenate([zeros, np.cumsum(increments, axis=-1)], axis=-1)


def control_volume_average(
    values: np.ndarray,
    coordinates: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compress nodal data to coarse control-volume averages.

    For a uniform source grid and an even stride, the coarse trapezoidal weights
    equal the control-volume widths. Therefore integrating the returned values
    on the coarse nodes preserves the source trapezoidal integral exactly.
    """
    array = np.asarray(values, dtype=np.float64)
    x = np.asarray(coordinates, dtype=np.float64)
    if array.shape[-1] != x.size:
        raise ValueError("source velocity axis mismatch")
    differences = np.diff(x)
    if not np.allclose(differences, differences[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("control-volume compressor currently requires a uniform grid")

    target_indices = target_indices_for_stride(x.size, stride)
    left, right = control_volume_index_bounds(x.size, stride)
    cumulative = cumulative_trapezoid(array, x)
    integrals = cumulative[..., right] - cumulative[..., left]
    widths = x[right] - x[left]
    averaged = integrals / widths

    target_x = x[target_indices]
    coarse_weights = trapezoid_weights(target_x)
    if not np.allclose(coarse_weights, widths, rtol=1.0e-11, atol=1.0e-13):
        raise RuntimeError("coarse trapezoid weights do not match control-volume widths")

    metadata = {
        "target_indices": target_indices,
        "left_indices": left,
        "right_indices": right,
        "target_coordinates": target_x,
        "quadrature_weights": coarse_weights,
    }
    return averaged, metadata


def conservative_m02_compress(
    values: np.ndarray,
    velocity: np.ndarray,
    stride: int = 8,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Control-volume compression with exact M0 and M2 projection.

    M0 is already exact after control-volume averaging. A smooth correction in
    the direction ``0.5*v**2 - weighted_mean(0.5*v**2)`` preserves M0 while
    making the kinetic moment M2 exact under the coarse trapezoidal rule.
    """
    source = np.asarray(values, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    coarse, metadata = control_volume_average(source, v, stride)
    target_v = np.asarray(metadata["target_coordinates"], dtype=np.float64)
    weights = np.asarray(metadata["quadrature_weights"], dtype=np.float64)

    fine_weights = trapezoid_weights(v)
    fine_mass = weighted_integral(source, fine_weights)
    energy_fine = 0.5 * v * v
    fine_kinetic = weighted_integral(source * energy_fine, fine_weights)

    energy_coarse = 0.5 * target_v * target_v
    weight_sum = float(np.sum(weights))
    energy_mean = float(np.sum(weights * energy_coarse) / weight_sum)
    direction = energy_coarse - energy_mean
    denominator = float(np.sum(weights * energy_coarse * direction))
    if not math.isfinite(denominator) or abs(denominator) < 1.0e-30:
        raise RuntimeError("degenerate M2 projection denominator")

    coarse_kinetic = weighted_integral(coarse * energy_coarse, weights)
    correction_scale = (fine_kinetic - coarse_kinetic) / denominator
    corrected = coarse + correction_scale[..., None] * direction

    corrected_mass = weighted_integral(corrected, weights)
    corrected_kinetic = weighted_integral(corrected * energy_coarse, weights)
    mass_residual = corrected_mass - fine_mass
    kinetic_residual = corrected_kinetic - fine_kinetic

    output_metadata: dict[str, np.ndarray | float] = dict(metadata)
    output_metadata.update(
        {
            "m0_absolute_residual_max": float(np.max(np.abs(mass_residual))),
            "m2_absolute_residual_max": float(np.max(np.abs(kinetic_residual))),
            "projection_direction": direction,
        }
    )
    return corrected, output_metadata


def relative_l2(prediction: np.ndarray, target: np.ndarray, floor: float = 1.0e-30) -> float:
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    numerator = float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
    denominator = max(float(np.sum(np.abs(truth) ** 2, dtype=np.float64)), floor)
    return float(math.sqrt(numerator / denominator))


def group_rows(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        summary: dict[str, Any] = {field: value for field, value in zip(key_fields, key)}
        summary["case_count"] = len(items)
        numeric_fields: list[str] = []
        for item in items:
            for field, value in item.items():
                if field in key_fields or field in numeric_fields:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
                    numeric_fields.append(field)
        for field in numeric_fields:
            values = np.asarray([float(item[field]) for item in items], dtype=np.float64)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_max"] = float(np.max(values))
        output.append(summary)
    return output
