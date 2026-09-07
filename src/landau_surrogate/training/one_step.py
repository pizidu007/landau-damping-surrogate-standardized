#!/usr/bin/env python3
"""
Train and evaluate one Stage 8D-2A one-step FNO variant.

Training uses teacher-forced one-step pairs from the 70 training cases.
Checkpoint selection uses full 30-step autoregressive rollout on the 14
validation cases, starting only from the true t=0 state.

Test cases are evaluated after restoring the best validation checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from landau_surrogate.data.rollout_cache import (
    GROUP_CODE_TO_NAME,
    Stage8D2PairDataset,
    load_case_sequence,
    split_case_indices,
)
from landau_surrogate.models.stepper_fno import (
    ConditionalStepperFNO2d,
    StepperLossFloors,
    StepperLossWeights,
    VALID_VARIANTS,
    compute_stepper_loss,
)


HORIZONS = (1, 2, 4, 8, 16, 30)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_loader(
    dataset: Stage8D2PairDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def move_pair_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "case_index",
        "time_index",
        "current",
        "target",
        "increment",
        "condition",
        "next_condition",
        "physical_condition",
        "next_physical_condition",
        "phase_velocity",
        "dt",
    ):
        if isinstance(moved.get(key), torch.Tensor):
            moved[key] = moved[key].to(
                device,
                non_blocking=device.type == "cuda",
            )
    return moved


def autocast_context(
    device: torch.device,
    amp: str,
):
    enabled = device.type == "cuda" and amp != "none"
    dtype = (
        torch.bfloat16
        if amp == "bf16"
        else torch.float16
        if amp == "fp16"
        else torch.float32
    )
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def compute_train_only_floors(
    loader: DataLoader,
    velocity: torch.Tensor,
    device: torch.device,
    quantile: float,
) -> StepperLossFloors:
    frame_values: list[np.ndarray] = []
    mode1_values: list[np.ndarray] = []
    resonance_values: list[np.ndarray] = []
    mean_values: list[np.ndarray] = []
    increment_values: list[np.ndarray] = []

    for batch in loader:
        batch = move_pair_batch(batch, device)
        target = batch["target"].float()
        current = batch["current"].float()
        increment = target - current

        frame_values.append(
            torch.mean(
                target * target, dim=(1, 2)
            ).cpu().numpy()
        )
        target_fft = torch.fft.rfft(
            target, dim=1, norm="forward"
        )
        mode1_values.append(
            torch.mean(
                torch.abs(target_fft[:, 1]) ** 2,
                dim=1,
            ).cpu().numpy()
        )

        mask = (
            torch.abs(
                velocity[None, :]
                - batch["phase_velocity"][:, None]
            )
            <= 0.5
        ).float()
        count = torch.clamp(
            torch.sum(mask, dim=1) * target.shape[1],
            min=1.0,
        )
        resonance_values.append(
            (
                torch.sum(
                    target * target * mask[:, None, :],
                    dim=(1, 2),
                )
                / count
            ).cpu().numpy()
        )
        target_mean = torch.mean(target, dim=1)
        mean_values.append(
            torch.mean(
                target_mean * target_mean, dim=1
            ).cpu().numpy()
        )
        increment_values.append(
            torch.mean(
                increment * increment, dim=(1, 2)
            ).cpu().numpy()
        )

    def floor_from(values: list[np.ndarray]) -> float:
        combined = np.concatenate(values).astype(np.float64)
        positive = combined[
            np.isfinite(combined) & (combined > 0.0)
        ]
        if positive.size == 0:
            raise RuntimeError("No positive train energies.")
        return max(
            float(np.quantile(positive, quantile)),
            float(np.max(positive)) * 1.0e-8,
            1.0e-12,
        )

    return StepperLossFloors(
        relative_mse=floor_from(frame_values),
        mode1_energy=floor_from(mode1_values),
        resonance_mse=floor_from(resonance_values),
        mean_delta_mse=floor_from(mean_values),
        increment_mse=floor_from(increment_values),
    )


def load_split_sequences(
    cache_path: Path,
    split: str,
) -> dict[str, Any]:
    indices = split_case_indices(cache_path, split)
    sequences = [
        load_case_sequence(cache_path, int(index))
        for index in indices
    ]
    return {
        "split": split,
        "case_indices": indices,
        "case_ids": [item["case_id"] for item in sequences],
        "k": np.asarray(
            [item["k"] for item in sequences],
            dtype=np.float32,
        ),
        "alpha": np.asarray(
            [item["alpha"] for item in sequences],
            dtype=np.float32,
        ),
        "phase_velocity": np.asarray(
            [item["phase_velocity"] for item in sequences],
            dtype=np.float32,
        ),
        "group_code": np.asarray(
            [item["group_code"] for item in sequences],
            dtype=np.uint8,
        ),
        "phase_time": sequences[0]["phase_time"],
        "field": np.stack(
            [item["field"] for item in sequences],
            axis=0,
        ).astype(np.float32),
        "condition": np.stack(
            [item["condition"] for item in sequences],
            axis=0,
        ).astype(np.float32),
        "physical_condition": np.stack(
            [item["physical_condition"] for item in sequences],
            axis=0,
        ).astype(np.float32),
        "normalized_x": sequences[0]["normalized_x"],
        "velocity": sequences[0]["velocity"],
    }


def autoregressive_rollout(
    model: ConditionalStepperFNO2d,
    sequence_data: dict[str, Any],
    device: torch.device,
    amp: str,
) -> np.ndarray:
    true_field = np.asarray(
        sequence_data["field"], dtype=np.float32
    )
    condition = torch.from_numpy(
        sequence_data["condition"]
    ).to(device)
    physical = torch.from_numpy(
        sequence_data["physical_condition"]
    ).to(device)
    current = torch.from_numpy(
        true_field[:, 0]
    ).to(device)
    predictions = [current.float().cpu().numpy()]

    model.eval()
    with torch.no_grad():
        for time_index in range(true_field.shape[1] - 1):
            with autocast_context(device, amp):
                output = model(
                    current,
                    condition[:, time_index],
                    physical[:, time_index],
                )
            current = output["field"].float()
            if not torch.isfinite(current).all():
                raise RuntimeError(
                    f"Non-finite rollout at step {time_index + 1}"
                )
            predictions.append(current.cpu().numpy())
    return np.stack(predictions, axis=1).astype(np.float32)


def validation_rollout_macro(
    model: ConditionalStepperFNO2d,
    sequence_data: dict[str, Any],
    device: torch.device,
    amp: str,
) -> float:
    prediction = autoregressive_rollout(
        model, sequence_data, device, amp
    )
    target = np.asarray(sequence_data["field"], dtype=np.float64)
    prediction64 = prediction.astype(np.float64)
    error = prediction64[:, 1:] - target[:, 1:]
    numerator = np.sum(error * error, axis=(1, 2, 3))
    denominator = np.sum(
        target[:, 1:] * target[:, 1:],
        axis=(1, 2, 3),
    )
    values = np.sqrt(
        numerator / np.maximum(denominator, 1.0e-30)
    )
    return float(np.mean(values))


def wrapped_phase_difference(
    predicted: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    return np.angle(predicted * np.conj(target))


def trapezoidal_integral(
    values: np.ndarray,
    coordinates: np.ndarray,
    axis: int,
) -> np.ndarray:
    """NumPy-version-compatible trapezoidal integration."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(values, x=coordinates, axis=axis)


def evaluate_sequence_predictions(
    prediction: np.ndarray,
    sequence_data: dict[str, Any],
    model_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Evaluate complete sequences. Frame 0 is the true supplied initial state and
    is excluded from rollout error aggregation.
    """
    target = np.asarray(
        sequence_data["field"], dtype=np.float64
    )
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} != target {target.shape}"
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{model_name}: non-finite sequence.")

    velocity = np.asarray(
        sequence_data["velocity"], dtype=np.float64
    )
    phase_time = np.asarray(
        sequence_data["phase_time"], dtype=np.float64
    )
    split = str(sequence_data["split"])

    case_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    accum_by_group: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for local_index, case_id in enumerate(
        sequence_data["case_ids"]
    ):
        truth = target[local_index, 1:]
        pred = prediction[local_index, 1:]
        error = pred - truth

        truth_mean = np.mean(truth, axis=1)
        pred_mean = np.mean(pred, axis=1)
        truth_nonzero = truth - truth_mean[:, None, :]
        pred_nonzero = pred - pred_mean[:, None, :]

        truth_fft = np.fft.rfft(
            truth, axis=1, norm="forward"
        )
        pred_fft = np.fft.rfft(
            pred, axis=1, norm="forward"
        )
        mode1_error = pred_fft[:, 1] - truth_fft[:, 1]

        phase_velocity = float(
            sequence_data["phase_velocity"][local_index]
        )
        resonance_mask = (
            np.abs(velocity - phase_velocity) <= 0.5
        )
        res_error = error[:, :, resonance_mask]
        res_truth = truth[:, :, resonance_mask]

        # Density mode-1 is derived only from f by trapezoidal integration in v.
        truth_density = trapezoidal_integral(
            truth, coordinates=velocity, axis=2
        )
        pred_density = trapezoidal_integral(
            pred, coordinates=velocity, axis=2
        )
        truth_density_mode1 = np.fft.rfft(
            truth_density, axis=1, norm="forward"
        )[:, 1]
        pred_density_mode1 = np.fft.rfft(
            pred_density, axis=1, norm="forward"
        )[:, 1]
        density_mode_error = (
            pred_density_mode1 - truth_density_mode1
        )
        amplitude = np.abs(truth_density_mode1)
        phase_mask = amplitude > max(
            float(np.max(amplitude)) * 1.0e-3,
            1.0e-12,
        )
        phase_mae = (
            float(
                np.mean(
                    np.abs(
                        wrapped_phase_difference(
                            pred_density_mode1[phase_mask],
                            truth_density_mode1[phase_mask],
                        )
                    )
                )
            )
            if np.any(phase_mask)
            else 0.0
        )

        target_norm_by_frame = np.sqrt(
            np.sum(truth * truth, axis=(1, 2))
        )
        error_norm_by_frame = np.sqrt(
            np.sum(error * error, axis=(1, 2))
        )
        frame_relative = (
            error_norm_by_frame
            / np.maximum(target_norm_by_frame, 1.0e-30)
        )
        predicted_norm_by_frame = np.sqrt(
            np.sum(pred * pred, axis=(1, 2))
        )
        norm_ratio = predicted_norm_by_frame / np.maximum(
            target_norm_by_frame, 1.0e-30
        )

        group_name = (
            "validation"
            if split == "val"
            else GROUP_CODE_TO_NAME[
                int(sequence_data["group_code"][local_index])
            ]
        )
        row = {
            "model": model_name,
            "split": split,
            "evaluation_group": group_name,
            "case_index": int(
                sequence_data["case_indices"][local_index]
            ),
            "case_id": case_id,
            "k": float(sequence_data["k"][local_index]),
            "alpha": float(
                sequence_data["alpha"][local_index]
            ),
            "trajectory_relative_l2": math.sqrt(
                float(np.sum(error * error))
                / max(float(np.sum(truth * truth)), 1.0e-30)
            ),
            "trajectory_case_mean_frame_relative_l2": float(
                np.mean(frame_relative)
            ),
            "trajectory_case_max_frame_relative_l2": float(
                np.max(frame_relative)
            ),
            "horizon30_relative_l2": float(
                frame_relative[-1]
            ),
            "mean_delta_relative_l2": math.sqrt(
                float(
                    np.sum(
                        (pred_mean - truth_mean) ** 2
                    )
                )
                / max(
                    float(np.sum(truth_mean * truth_mean)),
                    1.0e-30,
                )
            ),
            "nonzero_relative_l2": math.sqrt(
                float(
                    np.sum(
                        (pred_nonzero - truth_nonzero) ** 2
                    )
                )
                / max(
                    float(
                        np.sum(
                            truth_nonzero * truth_nonzero
                        )
                    ),
                    1.0e-30,
                )
            ),
            "mode1_relative_l2": math.sqrt(
                float(np.sum(np.abs(mode1_error) ** 2))
                / max(
                    float(
                        np.sum(
                            np.abs(truth_fft[:, 1]) ** 2
                        )
                    ),
                    1.0e-30,
                )
            ),
            "resonance_relative_l2": math.sqrt(
                float(np.sum(res_error * res_error))
                / max(
                    float(np.sum(res_truth * res_truth)),
                    1.0e-30,
                )
            ),
            "density_mode1_complex_relative_l2": math.sqrt(
                float(
                    np.sum(
                        np.abs(density_mode_error) ** 2
                    )
                )
                / max(
                    float(
                        np.sum(
                            np.abs(
                                truth_density_mode1
                            )
                            ** 2
                        )
                    ),
                    1.0e-30,
                )
            ),
            "density_mode1_phase_mae_rad": phase_mae,
            "max_predicted_to_true_norm_ratio": float(
                np.max(norm_ratio)
            ),
        }
        case_rows.append(row)

        for horizon in HORIZONS:
            index = min(horizon, truth.shape[0]) - 1
            horizon_rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "evaluation_group": group_name,
                    "case_id": case_id,
                    "horizon": horizon,
                    "phase_time": float(
                        phase_time[min(horizon, len(phase_time) - 1)]
                    ),
                    "relative_l2": float(
                        frame_relative[index]
                    ),
                    "predicted_to_true_norm_ratio": float(
                        norm_ratio[index]
                    ),
                }
            )

        for aggregate_name in (
            group_name,
            "all_validation" if split == "val" else "all_test",
        ):
            bucket = accum_by_group[aggregate_name]
            bucket["error_sq"] += float(
                np.sum(error * error)
            )
            bucket["target_sq"] += float(
                np.sum(truth * truth)
            )
            bucket["mean_error_sq"] += float(
                np.sum((pred_mean - truth_mean) ** 2)
            )
            bucket["mean_target_sq"] += float(
                np.sum(truth_mean * truth_mean)
            )
            bucket["nonzero_error_sq"] += float(
                np.sum(
                    (pred_nonzero - truth_nonzero) ** 2
                )
            )
            bucket["nonzero_target_sq"] += float(
                np.sum(truth_nonzero * truth_nonzero)
            )
            bucket["mode1_error_sq"] += float(
                np.sum(np.abs(mode1_error) ** 2)
            )
            bucket["mode1_target_sq"] += float(
                np.sum(np.abs(truth_fft[:, 1]) ** 2)
            )
            bucket["res_error_sq"] += float(
                np.sum(res_error * res_error)
            )
            bucket["res_target_sq"] += float(
                np.sum(res_truth * res_truth)
            )
            bucket["density_error_sq"] += float(
                np.sum(np.abs(density_mode_error) ** 2)
            )
            bucket["density_target_sq"] += float(
                np.sum(
                    np.abs(truth_density_mode1) ** 2
                )
            )
            bucket["phase_abs_sum"] += (
                phase_mae * int(np.sum(phase_mask))
            )
            bucket["phase_count"] += int(
                np.sum(phase_mask)
            )
            bucket["case_count"] += 1
            bucket["case_relative_sum"] += row[
                "trajectory_relative_l2"
            ]
            bucket["case_frame_mean_sum"] += row[
                "trajectory_case_mean_frame_relative_l2"
            ]
            bucket["case_max_sum"] += row[
                "trajectory_case_max_frame_relative_l2"
            ]
            bucket["horizon30_sum"] += row[
                "horizon30_relative_l2"
            ]
            bucket["max_norm_ratio"] = max(
                bucket["max_norm_ratio"],
                row["max_predicted_to_true_norm_ratio"],
            )

    group_rows: list[dict[str, Any]] = []
    for group_name, bucket in sorted(
        accum_by_group.items()
    ):
        count = max(int(bucket["case_count"]), 1)
        group_rows.append(
            {
                "model": model_name,
                "split_group": group_name,
                "case_count": count,
                "trajectory_weighted_relative_l2": math.sqrt(
                    bucket["error_sq"]
                    / max(bucket["target_sq"], 1.0e-30)
                ),
                "trajectory_case_macro_relative_l2": (
                    bucket["case_relative_sum"] / count
                ),
                "trajectory_case_macro_mean_frame_relative_l2": (
                    bucket["case_frame_mean_sum"] / count
                ),
                "trajectory_case_macro_max_frame_relative_l2": (
                    bucket["case_max_sum"] / count
                ),
                "horizon30_case_macro_relative_l2": (
                    bucket["horizon30_sum"] / count
                ),
                "mean_delta_relative_l2": math.sqrt(
                    bucket["mean_error_sq"]
                    / max(
                        bucket["mean_target_sq"], 1.0e-30
                    )
                ),
                "nonzero_relative_l2": math.sqrt(
                    bucket["nonzero_error_sq"]
                    / max(
                        bucket["nonzero_target_sq"], 1.0e-30
                    )
                ),
                "mode1_relative_l2": math.sqrt(
                    bucket["mode1_error_sq"]
                    / max(
                        bucket["mode1_target_sq"], 1.0e-30
                    )
                ),
                "resonance_relative_l2": math.sqrt(
                    bucket["res_error_sq"]
                    / max(
                        bucket["res_target_sq"], 1.0e-30
                    )
                ),
                "density_mode1_complex_relative_l2": math.sqrt(
                    bucket["density_error_sq"]
                    / max(
                        bucket["density_target_sq"], 1.0e-30
                    )
                ),
                "density_mode1_phase_mae_rad": (
                    bucket["phase_abs_sum"]
                    / max(bucket["phase_count"], 1.0)
                ),
                "max_predicted_to_true_norm_ratio": bucket[
                    "max_norm_ratio"
                ],
            }
        )
    return case_rows, group_rows, horizon_rows


def evaluate_teacher_forced_one_step(
    model: ConditionalStepperFNO2d,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    split: str,
) -> list[dict[str, Any]]:
    accum: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_pair_batch(batch, device)
            with autocast_context(device, amp):
                output = model(
                    batch["current"],
                    batch["condition"],
                    batch["physical_condition"],
                )
            prediction = output["field"].float()
            target = batch["target"].float()
            error = prediction - target
            for local in range(target.shape[0]):
                case_index = int(
                    batch["case_index"][local].item()
                )
                bucket = accum[case_index]
                bucket["error_sq"] += float(
                    torch.sum(error[local] ** 2).item()
                )
                bucket["target_sq"] += float(
                    torch.sum(target[local] ** 2).item()
                )
                bucket["increment_error_sq"] += float(
                    torch.sum(
                        (
                            output["increment"][local]
                            - batch["increment"][local]
                        )
                        ** 2
                    ).item()
                )
                bucket["increment_target_sq"] += float(
                    torch.sum(
                        batch["increment"][local] ** 2
                    ).item()
                )
    return [
        {
            "model": model.variant,
            "split": split,
            "case_index": case_index,
            "one_step_relative_l2": math.sqrt(
                bucket["error_sq"]
                / max(bucket["target_sq"], 1.0e-30)
            ),
            "increment_relative_l2": math.sqrt(
                bucket["increment_error_sq"]
                / max(
                    bucket["increment_target_sq"], 1.0e-30
                )
            ),
        }
        for case_index, bucket in sorted(accum.items())
    ]


def make_plots(
    output_dir: Path,
    history: list[dict[str, float]],
    horizon_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        [row["epoch"] for row in history],
        [row["train_total_loss"] for row in history],
        label="train composite",
    )
    plt.semilogy(
        [row["epoch"] for row in history],
        [row["val_rollout_case_macro_l2"] for row in history],
        label="validation 30-step rollout macro",
    )
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "training_and_rollout_validation.png",
        dpi=180,
    )
    plt.close()

    test_horizon = [
        row for row in horizon_rows if row["split"] == "test"
    ]
    horizons = sorted({row["horizon"] for row in test_horizon})
    macro = [
        np.mean(
            [
                row["relative_l2"]
                for row in test_horizon
                if row["horizon"] == horizon
            ]
        )
        for horizon in horizons
    ]
    plt.figure(figsize=(8, 5))
    plt.plot(horizons, macro, marker="o")
    plt.xlabel("rollout horizon")
    plt.ylabel("test case-macro relative L2")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "test_error_by_horizon.png", dpi=180
    )
    plt.close()

    test_cases = [
        row for row in case_rows if row["split"] == "test"
    ]
    plt.figure(figsize=(8, 5))
    plt.scatter(
        [row["alpha"] for row in test_cases],
        [row["trajectory_relative_l2"] for row in test_cases],
    )
    plt.xlabel("alpha")
    plt.ylabel("trajectory relative L2")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "test_trajectory_error_by_alpha.png",
        dpi=180,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=sorted(VALID_VARIANTS),
        required=True,
    )
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--modes-v", type=int, default=32)
    parser.add_argument("--v-padding", type=int, default=16)
    parser.add_argument("--condition-channels", type=int, default=16)
    parser.add_argument(
        "--condition-hidden-dim", type=int, default=128
    )
    parser.add_argument(
        "--condition-hidden-layers", type=int, default=2
    )
    parser.add_argument("--time-harmonics", type=int, default=8)
    parser.add_argument("--phase-harmonics", type=int, default=8)
    parser.add_argument(
        "--x-coordinate-harmonics", type=int, default=2
    )
    parser.add_argument("--lambda-relative", type=float, default=0.20)
    parser.add_argument("--lambda-mode1", type=float, default=0.10)
    parser.add_argument("--lambda-resonance", type=float, default=0.05)
    parser.add_argument("--lambda-mean-delta", type=float, default=0.10)
    parser.add_argument("--lambda-increment", type=float, default=0.10)
    parser.add_argument("--floor-quantile", type=float, default=0.10)
    parser.add_argument(
        "--amp", choices=("none", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    device = (
        torch.device(args.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        cuda_index = (
            int(device.index)
            if device.index is not None
            else 0
        )
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    else:
        cuda_index = None

    train_dataset = Stage8D2PairDataset(
        args.cache, split="train"
    )
    val_dataset = Stage8D2PairDataset(
        args.cache, split="val"
    )
    test_dataset = Stage8D2PairDataset(
        args.cache, split="test"
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, device
    )
    floor_loader = make_loader(
        train_dataset, max(args.batch_size, 64), False, device
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, device
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, False, device
    )
    val_sequences = load_split_sequences(args.cache, "val")
    test_sequences = load_split_sequences(args.cache, "test")

    model = ConditionalStepperFNO2d(
        normalized_x=train_dataset.normalized_x,
        velocity=train_dataset.velocity,
        variant=args.variant,
        width=args.width,
        layers=args.layers,
        modes_x=args.modes_x,
        modes_v=args.modes_v,
        v_padding=args.v_padding,
        condition_channels=args.condition_channels,
        condition_hidden_dim=args.condition_hidden_dim,
        condition_hidden_layers=args.condition_hidden_layers,
        time_harmonics=args.time_harmonics,
        phase_harmonics=args.phase_harmonics,
        x_coordinate_harmonics=args.x_coordinate_harmonics,
    ).to(device)

    floors = compute_train_only_floors(
        floor_loader,
        model.velocity,
        device,
        quantile=args.floor_quantile,
    )
    weights = StepperLossWeights(
        relative=args.lambda_relative,
        mode1=args.lambda_mode1,
        resonance=args.lambda_resonance,
        mean_delta=args.lambda_mean_delta,
        increment=args.lambda_increment,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.10,
    )
    scaler_enabled = (
        device.type == "cuda" and args.amp == "fp16"
    )
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=scaler_enabled
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=scaler_enabled
        )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_rollout = math.inf
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals = defaultdict(float)
        sample_count = 0

        for batch in train_loader:
            batch = move_pair_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                prediction = model(
                    batch["current"],
                    batch["condition"],
                    batch["physical_condition"],
                )
                loss, components = compute_stepper_loss(
                    prediction,
                    batch["current"],
                    batch["target"],
                    batch["phase_velocity"],
                    model.velocity,
                    floors,
                    weights,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite stepper loss.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()

            count = int(batch["current"].shape[0])
            sample_count += count
            for name, value in components.items():
                totals[name] += float(value.item()) * count

        val_rollout = validation_rollout_macro(
            model,
            val_sequences,
            device,
            args.amp,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"]
            / max(sample_count, 1),
            "train_global_mse": totals["global_mse"]
            / max(sample_count, 1),
            "train_relative_loss": totals["relative"]
            / max(sample_count, 1),
            "train_mode1_loss": totals["mode1"]
            / max(sample_count, 1),
            "train_resonance_loss": totals["resonance"]
            / max(sample_count, 1),
            "train_mean_delta_loss": totals["mean_delta"]
            / max(sample_count, 1),
            "train_increment_loss": totals["increment"]
            / max(sample_count, 1),
            "val_rollout_case_macro_l2": val_rollout,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "epoch_seconds": time.perf_counter()
            - epoch_started,
        }
        history.append(row)
        print(
            f"{args.variant} epoch {epoch:03d}/{args.epochs} "
            f"loss={row['train_total_loss']:.6e} "
            f"val_rollout={val_rollout:.6e} "
            f"seconds={row['epoch_seconds']:.2f}"
        )

        if val_rollout < best_val_rollout:
            best_val_rollout = val_rollout
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    one_step_rows = (
        evaluate_teacher_forced_one_step(
            model, val_loader, device, args.amp, "val"
        )
        + evaluate_teacher_forced_one_step(
            model, test_loader, device, args.amp, "test"
        )
    )
    val_prediction = autoregressive_rollout(
        model, val_sequences, device, args.amp
    )
    test_prediction = autoregressive_rollout(
        model, test_sequences, device, args.amp
    )
    val_case, val_group, val_horizon = (
        evaluate_sequence_predictions(
            val_prediction,
            val_sequences,
            model_name=args.variant,
        )
    )
    test_case, test_group, test_horizon = (
        evaluate_sequence_predictions(
            test_prediction,
            test_sequences,
            model_name=args.variant,
        )
    )
    case_rows = val_case + test_case
    group_rows = val_group + test_group
    horizon_rows = val_horizon + test_horizon

    checkpoint_path = args.output_dir / "best.pt"
    torch.save(
        {
            "stage": "stage8d2a_stepper",
            "variant": args.variant,
            "model_state_dict": model.state_dict(),
            "model_contract": model.contract(),
            "loss_floors": floors.as_dict(),
            "loss_weights": weights.as_dict(),
            "best_epoch": best_epoch,
            "best_val_rollout_case_macro_l2": (
                best_val_rollout
            ),
            "cache": str(args.cache),
            "cache_sha256": sha256_file(args.cache),
            "dt": train_dataset.dt,
            "config": vars(args),
        },
        checkpoint_path,
    )

    write_csv(
        args.output_dir / "training_history.csv", history
    )
    write_csv(
        args.output_dir / "one_step_case_metrics.csv",
        one_step_rows,
    )
    write_csv(
        args.output_dir / "rollout_case_metrics.csv",
        case_rows,
    )
    write_csv(
        args.output_dir / "rollout_group_metrics.csv",
        group_rows,
    )
    write_csv(
        args.output_dir / "rollout_horizon_metrics.csv",
        horizon_rows,
    )

    def group(name: str) -> dict[str, Any]:
        matches = [
            row for row in group_rows
            if row["split_group"] == name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one group row for {name}."
            )
        return matches[0]

    val_summary = group("all_validation")
    test_summary = group("all_test")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(cuda_index))
        if cuda_index is not None
        else 0
    )
    rollout_limit = 100.0 if args.mode == "smoke" else 20.0
    norm_ratio_limit = 1000.0 if args.mode == "smoke" else 100.0
    checks = {
        "training_history_length": len(history) == args.epochs,
        "training_finite": all(
            np.isfinite(row["train_total_loss"])
            and np.isfinite(
                row["val_rollout_case_macro_l2"]
            )
            for row in history
        ),
        "rollout_metrics_finite": all(
            np.isfinite(float(row[key]))
            for row in group_rows
            for key in (
                "trajectory_weighted_relative_l2",
                "trajectory_case_macro_relative_l2",
                "horizon30_case_macro_relative_l2",
                "mean_delta_relative_l2",
                "nonzero_relative_l2",
                "mode1_relative_l2",
                "resonance_relative_l2",
                "density_mode1_complex_relative_l2",
                "density_mode1_phase_mae_rad",
                "max_predicted_to_true_norm_ratio",
            )
        ),
        "validation_rollout_bounded": (
            float(
                val_summary[
                    "trajectory_case_macro_relative_l2"
                ]
            )
            < rollout_limit
        ),
        "test_rollout_bounded": (
            float(
                test_summary[
                    "trajectory_case_macro_relative_l2"
                ]
            )
            < rollout_limit
        ),
        "no_catastrophic_norm_explosion": (
            float(
                test_summary[
                    "max_predicted_to_true_norm_ratio"
                ]
            )
            < norm_ratio_limit
        ),
        "peak_gpu_memory_under_20_gib": (
            peak_memory < 20 * 1024**3
        ),
    }
    passed = all(checks.values())

    acceptance = {
        "stage": "stage8d2a_stepper_variant",
        "variant": args.variant,
        "mode": args.mode,
        "status": "PASS" if passed else "FAILED",
        "passed": passed,
        "checks": checks,
        "best_epoch": best_epoch,
        "best_val_rollout_case_macro_l2": (
            best_val_rollout
        ),
        "all_validation": val_summary,
        "all_test": test_summary,
        "quality_flags": {
            "test_trajectory_macro_below_one": (
                float(
                    test_summary[
                        "trajectory_case_macro_relative_l2"
                    ]
                )
                < 1.0
            ),
            "test_horizon30_macro_below_one": (
                float(
                    test_summary[
                        "horizon30_case_macro_relative_l2"
                    ]
                )
                < 1.0
            ),
            "test_max_norm_ratio_below_two": (
                float(
                    test_summary[
                        "max_predicted_to_true_norm_ratio"
                    ]
                )
                < 2.0
            ),
        },
        "peak_gpu_memory_bytes": peak_memory,
        "peak_gpu_memory_gib": peak_memory / 1024**3,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }

    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["model_contract"] = model.contract()
    config["loss_floors"] = floors.as_dict()
    config["loss_weights"] = weights.as_dict()
    environment = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
    }
    atomic_json(args.output_dir / "config.json", config)
    atomic_json(
        args.output_dir / "environment.json", environment
    )
    atomic_json(
        args.output_dir / "acceptance.json", acceptance
    )

    make_plots(
        args.output_dir,
        history,
        horizon_rows,
        case_rows,
    )

    artifact_lines = []
    for path in sorted(args.output_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_sha256.txt"
        ):
            artifact_lines.append(
                f"{sha256_file(path)}  "
                f"{path.relative_to(args.output_dir)}"
            )
    (
        args.output_dir / "artifact_sha256.txt"
    ).write_text(
        "\n".join(artifact_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(json_safe(acceptance), indent=2))
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
