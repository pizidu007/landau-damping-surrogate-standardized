#!/usr/bin/env python3
"""Positivity losses and diagnostics for Stage 9B.

The model predicts normalized delta_f0. Physical full distribution values are
reconstructed as

    f_phase = f0_train + delta_global_rms * normalized_delta_f0.

The target data can contain tiny negative values from discretization, so the
module reports both raw predicted negativity and excess negativity relative to
truth. Training candidates use smooth soft penalties; no hard clipping is
applied to the production rollout.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch


CANDIDATE_CONFIGS: dict[str, dict[str, float]] = {
    "absolute_hinge": {
        "absolute_weight": 2.0,
        "tail_weight": 0.0,
        "tolerance_fraction": 0.0,
        "tail_floor_fraction": 0.02,
    },
    "tail_relative": {
        "absolute_weight": 0.0,
        "tail_weight": 0.05,
        "tolerance_fraction": 1.0e-4,
        "tail_floor_fraction": 0.02,
    },
    "hybrid_tolerant": {
        "absolute_weight": 1.0,
        "tail_weight": 0.025,
        "tolerance_fraction": 1.0e-4,
        "tail_floor_fraction": 0.02,
    },
}


def reconstruct_full_distribution_torch(
    normalized_delta: torch.Tensor,
    f0_train: torch.Tensor,
    delta_global_rms: float,
) -> torch.Tensor:
    if normalized_delta.ndim != 3:
        raise ValueError(
            f"Expected [batch,x,v], got {tuple(normalized_delta.shape)}"
        )
    if f0_train.ndim != 1:
        raise ValueError(f"Expected 1D f0, got {tuple(f0_train.shape)}")
    if normalized_delta.shape[-1] != f0_train.numel():
        raise ValueError("Velocity dimension does not match f0_train.")
    return (
        normalized_delta.float() * float(delta_global_rms)
        + f0_train.float()[None, None, :]
    )


def positivity_losses(
    normalized_prediction: torch.Tensor,
    f0_train: torch.Tensor,
    delta_global_rms: float,
    tolerance_fraction: float = 0.0,
    tail_floor_fraction: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return absolute and tail-scaled soft negativity penalties.

    absolute loss:
        mean(ReLU(-(f + tolerance)) / delta_global_rms)

    tail-scaled loss:
        mean(log1p(ReLU(-(f + tolerance)) /
                   (f0 + tail_floor_fraction * delta_global_rms)))

    The logarithm prevents the Maxwellian tails from dominating while still
    strongly penalizing violations relative to the local equilibrium scale.
    """
    if delta_global_rms <= 0.0:
        raise ValueError(delta_global_rms)
    if tolerance_fraction < 0.0:
        raise ValueError(tolerance_fraction)
    if tail_floor_fraction <= 0.0:
        raise ValueError(tail_floor_fraction)

    full = reconstruct_full_distribution_torch(
        normalized_prediction,
        f0_train,
        delta_global_rms,
    )
    tolerance = float(tolerance_fraction) * float(delta_global_rms)
    violation = torch.relu(-(full + tolerance))
    absolute = torch.mean(violation / float(delta_global_rms))

    local_scale = (
        f0_train.float()[None, None, :]
        + float(tail_floor_fraction) * float(delta_global_rms)
    )
    tail_scaled = torch.mean(torch.log1p(violation / local_scale))

    negative_mask = full < 0.0
    metrics = {
        "absolute": absolute,
        "tail_scaled": tail_scaled,
        "negative_fraction": torch.mean(negative_mask.float()),
        "negative_mass_normalized": torch.mean(
            torch.relu(-full) / float(delta_global_rms)
        ),
        "negative_depth_rms_normalized": torch.sqrt(
            torch.mean((torch.relu(-full) / float(delta_global_rms)) ** 2)
            + 1.0e-30
        ),
        "minimum_full_distribution": torch.amin(full),
    }
    return absolute, tail_scaled, metrics


def reconstruct_full_distribution_numpy(
    normalized_delta: np.ndarray,
    f0_train: np.ndarray,
    delta_global_rms: float,
) -> np.ndarray:
    values = np.asarray(normalized_delta, dtype=np.float64)
    f0 = np.asarray(f0_train, dtype=np.float64)
    if values.shape[-1] != f0.shape[0]:
        raise ValueError("Velocity dimension does not match f0_train.")
    return values * float(delta_global_rms) + f0


def _relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    floor: float = 1.0e-30,
) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    error = pred - truth
    return math.sqrt(
        float(np.sum(error * error))
        / max(float(np.sum(truth * truth)), floor)
    )


def positivity_case_metrics(
    model_name: str,
    prediction: np.ndarray,
    sequence_data: dict[str, Any],
    f0_train: np.ndarray,
    delta_global_rms: float,
) -> list[dict[str, Any]]:
    target = np.asarray(sequence_data["field"], dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction {pred.shape} != target {target.shape}")

    rows: list[dict[str, Any]] = []
    split = str(sequence_data["split"])
    for index, case_id in enumerate(sequence_data["case_ids"]):
        # Exclude the shared initial state from rollout-quality diagnostics.
        pred_full = reconstruct_full_distribution_numpy(
            pred[index, 1:], f0_train, delta_global_rms
        )
        truth_full = reconstruct_full_distribution_numpy(
            target[index, 1:], f0_train, delta_global_rms
        )
        pred_negative = pred_full < 0.0
        truth_negative = truth_full < 0.0

        pred_fraction_by_time = np.mean(pred_negative, axis=(1, 2))
        truth_fraction_by_time = np.mean(truth_negative, axis=(1, 2))
        fraction_excess_by_time = np.maximum(
            pred_fraction_by_time - truth_fraction_by_time, 0.0
        )

        pred_negative_mass = np.mean(
            np.maximum(-pred_full, 0.0), axis=(1, 2)
        ) / float(delta_global_rms)
        truth_negative_mass = np.mean(
            np.maximum(-truth_full, 0.0), axis=(1, 2)
        ) / float(delta_global_rms)
        mass_excess = np.maximum(
            pred_negative_mass - truth_negative_mass, 0.0
        )

        pred_negative_depth_rms = np.sqrt(
            np.mean(np.maximum(-pred_full, 0.0) ** 2, axis=(1, 2))
        ) / float(delta_global_rms)
        truth_negative_depth_rms = np.sqrt(
            np.mean(np.maximum(-truth_full, 0.0) ** 2, axis=(1, 2))
        ) / float(delta_global_rms)
        depth_excess = np.maximum(
            pred_negative_depth_rms - truth_negative_depth_rms, 0.0
        )

        clamped = np.maximum(pred_full, 0.0)
        clamp_change = _relative_l2(clamped, pred_full)
        clamp_mass_added = float(np.mean(clamped - pred_full))

        evaluation_group = (
            "validation"
            if split == "val"
            else str(int(sequence_data["group_code"][index]))
        )
        row = {
            "model": model_name,
            "split": split,
            "evaluation_group": evaluation_group,
            "case_index": int(sequence_data["case_indices"][index]),
            "case_id": str(case_id),
            "k": float(sequence_data["k"][index]),
            "alpha": float(sequence_data["alpha"][index]),
            "pred_negative_fraction_mean": float(
                np.mean(pred_fraction_by_time)
            ),
            "pred_negative_fraction_max": float(
                np.max(pred_fraction_by_time)
            ),
            "truth_negative_fraction_mean": float(
                np.mean(truth_fraction_by_time)
            ),
            "truth_negative_fraction_max": float(
                np.max(truth_fraction_by_time)
            ),
            "negative_fraction_excess_mean": float(
                np.mean(fraction_excess_by_time)
            ),
            "negative_fraction_excess_max": float(
                np.max(fraction_excess_by_time)
            ),
            "pred_negative_mass_normalized_mean": float(
                np.mean(pred_negative_mass)
            ),
            "pred_negative_mass_normalized_max": float(
                np.max(pred_negative_mass)
            ),
            "truth_negative_mass_normalized_mean": float(
                np.mean(truth_negative_mass)
            ),
            "truth_negative_mass_normalized_max": float(
                np.max(truth_negative_mass)
            ),
            "negative_mass_excess_normalized_mean": float(
                np.mean(mass_excess)
            ),
            "negative_mass_excess_normalized_max": float(
                np.max(mass_excess)
            ),
            "pred_negative_depth_rms_normalized_mean": float(
                np.mean(pred_negative_depth_rms)
            ),
            "pred_negative_depth_rms_normalized_max": float(
                np.max(pred_negative_depth_rms)
            ),
            "truth_negative_depth_rms_normalized_mean": float(
                np.mean(truth_negative_depth_rms)
            ),
            "truth_negative_depth_rms_normalized_max": float(
                np.max(truth_negative_depth_rms)
            ),
            "negative_depth_excess_normalized_mean": float(
                np.mean(depth_excess)
            ),
            "negative_depth_excess_normalized_max": float(
                np.max(depth_excess)
            ),
            "pred_minimum_full_distribution": float(np.min(pred_full)),
            "truth_minimum_full_distribution": float(np.min(truth_full)),
            "hard_clamp_relative_change_l2": float(clamp_change),
            "hard_clamp_mean_mass_added": clamp_mass_added,
        }
        numeric_values = [
            float(value)
            for key, value in row.items()
            if key not in {"model", "split", "evaluation_group", "case_id"}
        ]
        if not np.isfinite(numeric_values).all():
            raise RuntimeError(f"Non-finite positivity metric for {case_id}")
        rows.append(row)
    return rows


def aggregate_positivity_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No positivity rows.")
    excluded = {
        "model",
        "split",
        "evaluation_group",
        "case_index",
        "case_id",
        "k",
        "alpha",
    }
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float, np.number))
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row["model"])
        split = str(row["split"])
        if split == "val":
            grouped[(model, "all_validation")].append(row)
        else:
            grouped[(model, "all_test")].append(row)
            grouped[(model, f"test_group_{row['evaluation_group']}")].append(row)
        grouped[(model, f"{split}_alpha_{float(row['alpha']):.3f}")].append(row)

    output: list[dict[str, Any]] = []
    for (model, group), selected in sorted(grouped.items()):
        record: dict[str, Any] = {
            "model": model,
            "positivity_group": group,
            "case_count": len(selected),
        }
        for key in numeric_keys:
            values = np.asarray(
                [float(row[key]) for row in selected], dtype=np.float64
            )
            record[f"{key}_mean"] = float(np.mean(values))
            record[f"{key}_max"] = float(np.max(values))
        output.append(record)
    return output


def find_positivity_group(
    rows: list[dict[str, Any]],
    model_name: str,
    group_name: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["model"] == model_name
        and row["positivity_group"] == group_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one positivity row for {model_name}/{group_name}, "
            f"got {len(matches)}"
        )
    return matches[0]


__all__ = [
    "CANDIDATE_CONFIGS",
    "aggregate_positivity_metrics",
    "find_positivity_group",
    "positivity_case_metrics",
    "positivity_losses",
    "reconstruct_full_distribution_numpy",
    "reconstruct_full_distribution_torch",
]
