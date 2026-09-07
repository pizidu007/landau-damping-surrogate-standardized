#!/usr/bin/env python3
"""Spectral training objectives and evaluation metrics for Stage 9A."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch


BANDS = {
    "low": (1, 5),
    "mid": (5, 9),
    "high": (9, None),
}


def _band_slice(
    transformed: torch.Tensor,
    lower: int,
    upper: int | None,
) -> torch.Tensor:
    stop = transformed.shape[1] if upper is None else min(
        upper, transformed.shape[1]
    )
    start = min(lower, stop)
    return transformed[:, start:stop, :]


def multiband_spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor_fraction: float = 0.005,
    low_weight: float = 0.10,
    mid_weight: float = 0.30,
    high_weight: float = 0.60,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Relative complex-coefficient loss with a total-energy floor."""
    pred_fft = torch.fft.rfft(
        prediction.float(), dim=1, norm="forward"
    )
    target_fft = torch.fft.rfft(
        target.float(), dim=1, norm="forward"
    )
    total_nonzero_energy = torch.sum(
        torch.abs(target_fft[:, 1:, :]) ** 2,
        dim=(1, 2),
    )
    denominator_floor = (
        float(floor_fraction) * total_nonzero_energy
        + 1.0e-12
    )
    losses: dict[str, torch.Tensor] = {}
    for name, (lower, upper) in BANDS.items():
        pred_band = _band_slice(pred_fft, lower, upper)
        target_band = _band_slice(target_fft, lower, upper)
        if pred_band.numel() == 0:
            loss = torch.zeros(
                (), device=prediction.device
            )
        else:
            error_energy = torch.sum(
                torch.abs(pred_band - target_band) ** 2,
                dim=(1, 2),
            )
            target_energy = torch.sum(
                torch.abs(target_band) ** 2,
                dim=(1, 2),
            )
            loss = torch.mean(
                error_energy
                / torch.maximum(
                    target_energy, denominator_floor
                )
            )
        losses[name] = loss
    total = (
        float(low_weight) * losses["low"]
        + float(mid_weight) * losses["mid"]
        + float(high_weight) * losses["high"]
    )
    return total, losses


def periodic_x_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor_fraction: float = 0.005,
) -> torch.Tensor:
    pred_gradient = torch.roll(
        prediction.float(), shifts=-1, dims=1
    ) - prediction.float()
    target_gradient = torch.roll(
        target.float(), shifts=-1, dims=1
    ) - target.float()
    error_energy = torch.mean(
        (pred_gradient - target_gradient) ** 2,
        dim=(1, 2),
    )
    target_energy = torch.mean(
        target_gradient**2, dim=(1, 2)
    )
    field_energy = torch.mean(
        target.float() ** 2, dim=(1, 2)
    )
    floor = (
        float(floor_fraction) * field_energy + 1.0e-12
    )
    return torch.mean(
        error_energy / torch.maximum(target_energy, floor)
    )


def correction_regularization(
    correction: torch.Tensor,
    target: torch.Tensor,
    floor_fraction: float = 0.005,
) -> torch.Tensor:
    correction_energy = torch.mean(
        correction.float() ** 2, dim=(1, 2)
    )
    target_energy = torch.mean(
        target.float() ** 2, dim=(1, 2)
    )
    return torch.mean(
        correction_energy
        / (
            target_energy
            + float(floor_fraction)
            * torch.mean(target_energy).detach()
            + 1.0e-12
        )
    )


def _relative_l2_complex(
    prediction: np.ndarray,
    target: np.ndarray,
    floor: float = 1.0e-30,
) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    truth = np.asarray(target)
    return math.sqrt(
        float(np.sum(np.abs(error) ** 2))
        / max(float(np.sum(np.abs(truth) ** 2)), floor)
    )


def _relative_l2_real(
    prediction: np.ndarray,
    target: np.ndarray,
    floor: float = 1.0e-30,
) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    truth = np.asarray(target, dtype=np.float64)
    return math.sqrt(
        float(np.sum(error * error))
        / max(float(np.sum(truth * truth)), floor)
    )


def spectral_case_metrics(
    model_name: str,
    prediction: np.ndarray,
    sequence_data: dict[str, Any],
    cutoff_mode: int = 9,
) -> list[dict[str, Any]]:
    target = np.asarray(
        sequence_data["field"], dtype=np.float64
    )
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction {pred.shape} != target {target.shape}"
        )
    split = str(sequence_data["split"])
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(
        sequence_data["case_ids"]
    ):
        truth = target[index, 1:]
        values = pred[index, 1:]
        truth_fft = np.fft.rfft(
            truth, axis=1, norm="forward"
        )
        pred_fft = np.fft.rfft(
            values, axis=1, norm="forward"
        )
        metrics: dict[str, float] = {}
        for name, (lower, upper) in BANDS.items():
            stop = (
                truth_fft.shape[1]
                if upper is None
                else min(upper, truth_fft.shape[1])
            )
            start = min(lower, stop)
            metrics[f"{name}_band_complex_relative_l2"] = (
                _relative_l2_complex(
                    pred_fft[:, start:stop],
                    truth_fft[:, start:stop],
                )
                if stop > start
                else 0.0
            )

        truth_energy = np.sum(
            np.abs(truth_fft) ** 2, axis=2
        )
        pred_energy = np.sum(
            np.abs(pred_fft) ** 2, axis=2
        )
        truth_nonzero = np.sum(
            truth_energy[:, 1:], axis=1
        )
        pred_nonzero = np.sum(
            pred_energy[:, 1:], axis=1
        )
        truth_high = np.sum(
            truth_energy[:, cutoff_mode:], axis=1
        ) / np.maximum(truth_nonzero, 1.0e-30)
        pred_high = np.sum(
            pred_energy[:, cutoff_mode:], axis=1
        ) / np.maximum(pred_nonzero, 1.0e-30)

        truth_gradient = np.roll(
            truth, -1, axis=1
        ) - truth
        pred_gradient = np.roll(
            values, -1, axis=1
        ) - values

        modes = np.arange(
            truth_energy.shape[1], dtype=np.float64
        )[None, :]
        truth_centroid = np.sum(
            truth_energy[:, 1:] * modes[:, 1:], axis=1
        ) / np.maximum(truth_nonzero, 1.0e-30)
        pred_centroid = np.sum(
            pred_energy[:, 1:] * modes[:, 1:], axis=1
        ) / np.maximum(pred_nonzero, 1.0e-30)

        evaluation_group = (
            "validation"
            if split == "val"
            else str(int(sequence_data["group_code"][index]))
        )
        row = {
            "model": model_name,
            "split": split,
            "evaluation_group": evaluation_group,
            "case_index": int(
                sequence_data["case_indices"][index]
            ),
            "case_id": str(case_id),
            "k": float(sequence_data["k"][index]),
            "alpha": float(
                sequence_data["alpha"][index]
            ),
            **metrics,
            "x_gradient_relative_l2": _relative_l2_real(
                pred_gradient, truth_gradient
            ),
            "high_energy_ratio_relative_l2": (
                _relative_l2_real(pred_high, truth_high)
            ),
            "high_energy_ratio_mean_true": float(
                np.mean(truth_high)
            ),
            "high_energy_ratio_mean_pred": float(
                np.mean(pred_high)
            ),
            "high_energy_ratio_final_true": float(
                truth_high[-1]
            ),
            "high_energy_ratio_final_pred": float(
                pred_high[-1]
            ),
            "high_energy_recovery_final": float(
                pred_high[-1]
                / max(float(truth_high[-1]), 1.0e-30)
            ),
            "spectral_centroid_relative_l2": (
                _relative_l2_real(
                    pred_centroid, truth_centroid
                )
            ),
            "spectral_centroid_final_true": float(
                truth_centroid[-1]
            ),
            "spectral_centroid_final_pred": float(
                pred_centroid[-1]
            ),
        }
        if not all(
            np.isfinite(float(value))
            for key, value in row.items()
            if key not in {"model", "split", "evaluation_group", "case_id"}
        ):
            raise RuntimeError(
                f"Non-finite spectral metric for {case_id}."
            )
        rows.append(row)
    return rows


def aggregate_spectral_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No spectral rows.")
    metric_keys = [
        key
        for key, value in rows[0].items()
        if key
        not in {
            "model",
            "split",
            "evaluation_group",
            "case_index",
            "case_id",
            "k",
            "alpha",
        }
        and isinstance(value, (int, float, np.number))
    ]
    buckets: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        split_group = (
            "all_validation"
            if row["split"] == "val"
            else "all_test"
        )
        keys = [
            (row["model"], split_group),
            (
                row["model"],
                "validation"
                if row["split"] == "val"
                else f"group_{row['evaluation_group']}",
            ),
            (
                row["model"],
                f"alpha_{float(row['alpha']):.3f}",
            ),
        ]
        for key in keys:
            buckets[key].append(row)

    output: list[dict[str, Any]] = []
    for (model_name, group_name), selected in sorted(
        buckets.items()
    ):
        record: dict[str, Any] = {
            "model": model_name,
            "spectral_group": group_name,
            "case_count": len(selected),
        }
        for metric in metric_keys:
            values = np.asarray(
                [float(row[metric]) for row in selected],
                dtype=np.float64,
            )
            record[f"{metric}_mean"] = float(
                np.mean(values)
            )
            record[f"{metric}_max"] = float(
                np.max(values)
            )
        output.append(record)
    return output


def find_spectral_group(
    rows: list[dict[str, Any]],
    model_name: str,
    group_name: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["model"] == model_name
        and row["spectral_group"] == group_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one spectral group for {model_name}/{group_name}."
        )
    return matches[0]


__all__ = [
    "BANDS",
    "aggregate_spectral_metrics",
    "correction_regularization",
    "find_spectral_group",
    "multiband_spectral_loss",
    "periodic_x_gradient_loss",
    "spectral_case_metrics",
]
