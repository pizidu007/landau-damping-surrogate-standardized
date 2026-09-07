#!/usr/bin/env python3
"""
Train one Stage 8D-1C conditional FNO variant.

Both variants use the same phase-aware condition encoding and the same
composite loss. The controlled comparison isolates the architectural effect of
an explicit mean_delta/nonzero dual output head.

Model selection uses validation case-macro relative L2 only. Test metrics are
computed after restoring the best validation checkpoint.
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

from landau_surrogate.data.snapshot_cache import Stage8D1CacheDataset
from landau_surrogate.models.snapshot_fno import (
    ConditionalFNO2d,
    LossFloors,
    LossWeights,
    VALID_CONDITION_MODES,
    VALID_VARIANTS,
    compute_training_loss,
)


GROUP_CODE_TO_NAME = {
    0: "none",
    1: "validation",
    2: "legacy",
    3: "interstitial",
    4: "boundary",
}


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


def read_idw_case_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("model") == "parameter_idw"
                and int(row.get("rank", "0")) == 0
            ):
                result[row["case_id"]] = {
                    key: float(row[key])
                    for key in (
                        "relative_l2",
                        "normalized_rmse",
                        "normalized_mae",
                        "resonance_relative_l2",
                        "mode0_relative_l2",
                        "mode1_relative_l2",
                    )
                }
    if len(result) != 40:
        raise RuntimeError(
            f"Expected 40 IDW validation/test cases, got {len(result)}"
        )
    return result


def load_case_metadata(cache_path: Path) -> dict[str, Any]:
    with h5py.File(cache_path, "r") as handle:
        cases = handle["cases"]
        return {
            "case_id": [
                item.decode("utf-8")
                if isinstance(item, bytes)
                else str(item)
                for item in cases["case_id"][...]
            ],
            "k": np.asarray(cases["k"][...], dtype=np.float64),
            "alpha": np.asarray(
                cases["alpha"][...], dtype=np.float64
            ),
            "phase_velocity": np.asarray(
                cases["phase_velocity"][...], dtype=np.float64
            ),
            "group_code": np.asarray(
                cases["group_code"][...], dtype=np.uint8
            ),
        }


def make_loader(
    dataset: Stage8D1CacheDataset,
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


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "condition",
        "physical_condition",
        "field",
        "phase_velocity",
        "case_index",
        "time_index",
    ):
        if isinstance(moved.get(key), torch.Tensor):
            moved[key] = moved[key].to(
                device,
                non_blocking=device.type == "cuda",
            )
    return moved


def compute_train_only_floors(
    loader: DataLoader,
    velocity: torch.Tensor,
    device: torch.device,
    quantile: float,
) -> LossFloors:
    frame_values: list[np.ndarray] = []
    mode1_values: list[np.ndarray] = []
    resonance_values: list[np.ndarray] = []
    mean_values: list[np.ndarray] = []

    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["field"].float()

        frame_mse = torch.mean(target * target, dim=(1, 2))
        target_fft = torch.fft.rfft(
            target, dim=1, norm="forward"
        )
        mode1 = torch.mean(
            torch.abs(target_fft[:, 1]) ** 2, dim=1
        )

        resonance_mask = (
            torch.abs(
                velocity[None, :]
                - batch["phase_velocity"][:, None]
            )
            <= 0.5
        ).float()
        resonance_count = torch.clamp(
            torch.sum(resonance_mask, dim=1)
            * target.shape[1],
            min=1.0,
        )
        resonance = torch.sum(
            target
            * target
            * resonance_mask[:, None, :],
            dim=(1, 2),
        ) / resonance_count

        mean_delta = torch.mean(target, dim=1)
        mean_mse = torch.mean(
            mean_delta * mean_delta, dim=1
        )

        frame_values.append(frame_mse.cpu().numpy())
        mode1_values.append(mode1.cpu().numpy())
        resonance_values.append(resonance.cpu().numpy())
        mean_values.append(mean_mse.cpu().numpy())

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

    return LossFloors(
        relative_mse=floor_from(frame_values),
        mode1_energy=floor_from(mode1_values),
        resonance_mse=floor_from(resonance_values),
        mean_delta_mse=floor_from(mean_values),
    )


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


def alpha_sample_weights(
    physical_condition: torch.Tensor,
    gamma: float,
    alpha_reference: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor | None:
    """Return bounded amplitude-aware weights without changing targets."""
    if gamma <= 0.0:
        return None
    alpha = torch.clamp(
        physical_condition[:, 1].float(), min=1.0e-6
    )
    raw = (float(alpha_reference) / alpha) ** float(gamma)
    return torch.clamp(
        raw,
        min=float(minimum),
        max=float(maximum),
    )


def validation_macro_relative(
    model: ConditionalFNO2d,
    loader: DataLoader,
    device: torch.device,
    amp: str,
) -> float:
    accum: dict[int, list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(device, amp):
                prediction = model(
                    batch["condition"],
                    batch["physical_condition"],
                )["field"]
            prediction = prediction.float()
            target = batch["field"].float()
            error = prediction - target
            error_sq = torch.sum(
                error * error, dim=(1, 2)
            ).cpu().numpy()
            target_sq = torch.sum(
                target * target, dim=(1, 2)
            ).cpu().numpy()
            case_indices = batch["case_index"].cpu().numpy()
            for case_index, numerator, denominator in zip(
                case_indices, error_sq, target_sq
            ):
                bucket = accum[int(case_index)]
                bucket[0] += float(numerator)
                bucket[1] += float(denominator)

    case_values = [
        math.sqrt(numerator / max(denominator, 1.0e-30))
        for numerator, denominator in accum.values()
    ]
    return float(np.mean(case_values))


def rich_evaluation(
    model: ConditionalFNO2d,
    loader: DataLoader,
    split: str,
    device: torch.device,
    amp: str,
    metadata: dict[str, Any],
    idw_metrics: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accum: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    velocity = model.velocity.float()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(device, amp):
                prediction_dict = model(
                    batch["condition"],
                    batch["physical_condition"],
                )
            prediction = prediction_dict["field"].float()
            target = batch["field"].float()
            error = prediction - target

            predicted_mean = prediction_dict[
                "mean_delta"
            ].float()
            target_mean = torch.mean(target, dim=1)
            mean_error = predicted_mean - target_mean

            predicted_nonzero = prediction_dict[
                "nonzero"
            ].float()
            target_nonzero = target - target_mean[:, None, :]
            nonzero_error = predicted_nonzero - target_nonzero

            pred_fft = torch.fft.rfft(
                prediction, dim=1, norm="forward"
            )
            target_fft = torch.fft.rfft(
                target, dim=1, norm="forward"
            )
            fft_error = pred_fft - target_fft

            resonance_mask = (
                torch.abs(
                    velocity[None, :]
                    - batch["phase_velocity"][:, None]
                )
                <= 0.5
            ).float()

            for local in range(target.shape[0]):
                case_index = int(
                    batch["case_index"][local].item()
                )
                bucket = accum[case_index]
                err = error[local]
                truth = target[local]

                bucket["error_sq"] += float(
                    torch.sum(err * err).item()
                )
                bucket["target_sq"] += float(
                    torch.sum(truth * truth).item()
                )
                bucket["abs_error"] += float(
                    torch.sum(torch.abs(err)).item()
                )
                bucket["count"] += float(err.numel())

                mask = resonance_mask[local][None, :]
                res_err = err * mask
                res_truth = truth * mask
                bucket["res_error_sq"] += float(
                    torch.sum(res_err * res_err).item()
                )
                bucket["res_target_sq"] += float(
                    torch.sum(res_truth * res_truth).item()
                )

                mean_err = mean_error[local]
                mean_truth = target_mean[local]
                bucket["mean_error_sq"] += float(
                    torch.sum(mean_err * mean_err).item()
                )
                bucket["mean_target_sq"] += float(
                    torch.sum(mean_truth * mean_truth).item()
                )

                nz_err = nonzero_error[local]
                nz_truth = target_nonzero[local]
                bucket["nonzero_error_sq"] += float(
                    torch.sum(nz_err * nz_err).item()
                )
                bucket["nonzero_target_sq"] += float(
                    torch.sum(nz_truth * nz_truth).item()
                )

                for mode_index, prefix in (
                    (0, "mode0"),
                    (1, "mode1"),
                ):
                    mode_error = fft_error[local, mode_index]
                    mode_target = target_fft[local, mode_index]
                    bucket[f"{prefix}_error_sq"] += float(
                        torch.sum(
                            torch.abs(mode_error) ** 2
                        ).item()
                    )
                    bucket[f"{prefix}_target_sq"] += float(
                        torch.sum(
                            torch.abs(mode_target) ** 2
                        ).item()
                    )

    case_rows: list[dict[str, Any]] = []
    for case_index in sorted(accum):
        bucket = accum[case_index]
        case_id = metadata["case_id"][case_index]
        group_name = (
            "validation"
            if split == "val"
            else GROUP_CODE_TO_NAME[
                int(metadata["group_code"][case_index])
            ]
        )
        relative = math.sqrt(
            bucket["error_sq"]
            / max(bucket["target_sq"], 1.0e-30)
        )
        row = {
            "variant": model.variant,
            "split": split,
            "evaluation_group": group_name,
            "case_index": case_index,
            "case_id": case_id,
            "k": float(metadata["k"][case_index]),
            "alpha": float(metadata["alpha"][case_index]),
            "relative_l2": relative,
            "normalized_rmse": math.sqrt(
                bucket["error_sq"]
                / max(bucket["count"], 1.0)
            ),
            "normalized_mae": (
                bucket["abs_error"]
                / max(bucket["count"], 1.0)
            ),
            "resonance_relative_l2": math.sqrt(
                bucket["res_error_sq"]
                / max(bucket["res_target_sq"], 1.0e-30)
            ),
            "mean_delta_relative_l2": math.sqrt(
                bucket["mean_error_sq"]
                / max(bucket["mean_target_sq"], 1.0e-30)
            ),
            "nonzero_relative_l2": math.sqrt(
                bucket["nonzero_error_sq"]
                / max(bucket["nonzero_target_sq"], 1.0e-30)
            ),
            "mode0_relative_l2": math.sqrt(
                bucket["mode0_error_sq"]
                / max(bucket["mode0_target_sq"], 1.0e-30)
            ),
            "mode1_relative_l2": math.sqrt(
                bucket["mode1_error_sq"]
                / max(bucket["mode1_target_sq"], 1.0e-30)
            ),
        }
        reference = idw_metrics[case_id]
        row["idw_relative_l2"] = reference["relative_l2"]
        row["beats_idw"] = relative < reference["relative_l2"]
        row["relative_l2_minus_idw"] = (
            relative - reference["relative_l2"]
        )
        case_rows.append(row)

    group_rows: list[dict[str, Any]] = []
    groups = sorted(
        {row["evaluation_group"] for row in case_rows}
    )
    for group_name in groups + [
        "all_validation" if split == "val" else "all_test"
    ]:
        selected = (
            case_rows
            if group_name.startswith("all_")
            else [
                row
                for row in case_rows
                if row["evaluation_group"] == group_name
            ]
        )
        selected_indices = {
            int(row["case_index"]) for row in selected
        }
        aggregate = defaultdict(float)
        for case_index in selected_indices:
            for key, value in accum[case_index].items():
                aggregate[key] += value

        relative_values = np.asarray(
            [row["relative_l2"] for row in selected],
            dtype=np.float64,
        )
        idw_values = np.asarray(
            [row["idw_relative_l2"] for row in selected],
            dtype=np.float64,
        )
        group_rows.append(
            {
                "variant": model.variant,
                "split_group": group_name,
                "case_count": len(selected),
                "relative_l2": math.sqrt(
                    aggregate["error_sq"]
                    / max(aggregate["target_sq"], 1.0e-30)
                ),
                "case_macro_mean_relative_l2": float(
                    np.mean(relative_values)
                ),
                "case_median_relative_l2": float(
                    np.median(relative_values)
                ),
                "case_max_relative_l2": float(
                    np.max(relative_values)
                ),
                "case_min_relative_l2": float(
                    np.min(relative_values)
                ),
                "case_win_rate_vs_idw": float(
                    np.mean(relative_values < idw_values)
                ),
                "idw_case_macro_mean_relative_l2": float(
                    np.mean(idw_values)
                ),
                "normalized_rmse": math.sqrt(
                    aggregate["error_sq"]
                    / max(aggregate["count"], 1.0)
                ),
                "normalized_mae": (
                    aggregate["abs_error"]
                    / max(aggregate["count"], 1.0)
                ),
                "resonance_relative_l2": math.sqrt(
                    aggregate["res_error_sq"]
                    / max(
                        aggregate["res_target_sq"], 1.0e-30
                    )
                ),
                "mean_delta_relative_l2": math.sqrt(
                    aggregate["mean_error_sq"]
                    / max(
                        aggregate["mean_target_sq"], 1.0e-30
                    )
                ),
                "nonzero_relative_l2": math.sqrt(
                    aggregate["nonzero_error_sq"]
                    / max(
                        aggregate["nonzero_target_sq"], 1.0e-30
                    )
                ),
                "mode0_relative_l2": math.sqrt(
                    aggregate["mode0_error_sq"]
                    / max(
                        aggregate["mode0_target_sq"], 1.0e-30
                    )
                ),
                "mode1_relative_l2": math.sqrt(
                    aggregate["mode1_error_sq"]
                    / max(
                        aggregate["mode1_target_sq"], 1.0e-30
                    )
                ),
            }
        )

    alpha_rows: list[dict[str, Any]] = []
    for alpha in sorted({row["alpha"] for row in case_rows}):
        selected = [
            row for row in case_rows
            if abs(row["alpha"] - alpha) < 1.0e-12
        ]
        relative_values = np.asarray(
            [row["relative_l2"] for row in selected],
            dtype=np.float64,
        )
        idw_values = np.asarray(
            [row["idw_relative_l2"] for row in selected],
            dtype=np.float64,
        )
        alpha_rows.append(
            {
                "variant": model.variant,
                "split": split,
                "alpha": alpha,
                "case_count": len(selected),
                "case_macro_mean_relative_l2": float(
                    np.mean(relative_values)
                ),
                "case_median_relative_l2": float(
                    np.median(relative_values)
                ),
                "case_max_relative_l2": float(
                    np.max(relative_values)
                ),
                "case_win_rate_vs_idw": float(
                    np.mean(relative_values < idw_values)
                ),
                "idw_case_macro_mean_relative_l2": float(
                    np.mean(idw_values)
                ),
                "mean_delta_macro_mean_relative_l2": float(
                    np.mean(
                        [
                            row["mean_delta_relative_l2"]
                            for row in selected
                        ]
                    )
                ),
                "mode1_macro_mean_relative_l2": float(
                    np.mean(
                        [
                            row["mode1_relative_l2"]
                            for row in selected
                        ]
                    )
                ),
            }
        )

    return case_rows, group_rows, alpha_rows


def make_plots(
    output_dir: Path,
    history: list[dict[str, float]],
    case_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
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
        [row["val_case_macro_relative_l2"] for row in history],
        label="validation macro relative L2",
    )
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "training_and_validation_curve.png",
        dpi=180,
    )
    plt.close()

    test_rows = [
        row for row in case_rows if row["split"] == "test"
    ]
    plt.figure(figsize=(6, 6))
    plt.scatter(
        [row["idw_relative_l2"] for row in test_rows],
        [row["relative_l2"] for row in test_rows],
    )
    maximum = max(
        max(row["idw_relative_l2"] for row in test_rows),
        max(row["relative_l2"] for row in test_rows),
    )
    plt.plot([0, maximum], [0, maximum], linestyle="--")
    plt.xlabel("IDW test case relative L2")
    plt.ylabel(f"{test_rows[0]['variant']} relative L2")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "test_case_error_vs_idw.png", dpi=180
    )
    plt.close()

    test_alpha = [
        row for row in alpha_rows if row["split"] == "test"
    ]
    plt.figure(figsize=(8, 5))
    plt.plot(
        [row["alpha"] for row in test_alpha],
        [
            row["case_macro_mean_relative_l2"]
            for row in test_alpha
        ],
        marker="o",
        label="FNO",
    )
    plt.plot(
        [row["alpha"] for row in test_alpha],
        [
            row["idw_case_macro_mean_relative_l2"]
            for row in test_alpha
        ],
        marker="s",
        label="IDW",
    )
    plt.xlabel("alpha")
    plt.ylabel("test case-macro relative L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "test_error_by_alpha.png", dpi=180
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--stage8d1a-case-metrics", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument(
        "--variant",
        choices=sorted(VALID_VARIANTS),
        default="dual_head",
    )
    parser.add_argument(
        "--condition-mode",
        choices=sorted(VALID_CONDITION_MODES),
        default="phase_aware",
    )
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--modes-v", type=int, default=24)
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
    parser.add_argument("--floor-quantile", type=float, default=0.10)
    parser.add_argument("--alpha-weight-gamma", type=float, default=0.0)
    parser.add_argument("--alpha-weight-reference", type=float, default=0.025)
    parser.add_argument("--alpha-weight-min", type=float, default=0.75)
    parser.add_argument("--alpha-weight-max", type=float, default=2.50)
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

    idw_metrics = read_idw_case_metrics(
        args.stage8d1a_case_metrics
    )
    metadata = load_case_metadata(args.cache)

    train_dataset = Stage8D1CacheDataset(
        args.cache, split="train"
    )
    val_dataset = Stage8D1CacheDataset(args.cache, split="val")
    test_dataset = Stage8D1CacheDataset(
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

    model = ConditionalFNO2d(
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
        condition_mode=args.condition_mode,
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
    weights = LossWeights(
        relative=args.lambda_relative,
        mode1=args.lambda_mode1,
        resonance=args.lambda_resonance,
        mean_delta=args.lambda_mean_delta,
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
    best_val_macro = math.inf
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals = defaultdict(float)
        sample_count = 0

        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                prediction = model(
                    batch["condition"],
                    batch["physical_condition"],
                )
                sample_weights = alpha_sample_weights(
                    batch["physical_condition"],
                    gamma=args.alpha_weight_gamma,
                    alpha_reference=args.alpha_weight_reference,
                    minimum=args.alpha_weight_min,
                    maximum=args.alpha_weight_max,
                )
                total_loss, components = compute_training_loss(
                    prediction,
                    batch["field"],
                    batch["phase_velocity"],
                    model.velocity,
                    floors,
                    weights,
                    sample_weights=sample_weights,
                )
            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite FNO loss.")

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()

            count = int(batch["field"].shape[0])
            sample_count += count
            for name, value in components.items():
                totals[name] += float(value.item()) * count

        val_macro = validation_macro_relative(
            model, val_loader, device, args.amp
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
            "val_case_macro_relative_l2": val_macro,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "epoch_seconds": time.perf_counter()
            - epoch_started,
        }
        history.append(row)
        print(
            f"{args.candidate_name} epoch {epoch:03d}/{args.epochs} "
            f"loss={row['train_total_loss']:.6e} "
            f"val_macro={val_macro:.6e} "
            f"seconds={row['epoch_seconds']:.2f}"
        )

        if val_macro < best_val_macro:
            best_val_macro = val_macro
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_case_rows, val_group_rows, val_alpha_rows = (
        rich_evaluation(
            model,
            val_loader,
            "val",
            device,
            args.amp,
            metadata,
            idw_metrics,
        )
    )
    test_case_rows, test_group_rows, test_alpha_rows = (
        rich_evaluation(
            model,
            test_loader,
            "test",
            device,
            args.amp,
            metadata,
            idw_metrics,
        )
    )
    case_rows = val_case_rows + test_case_rows
    group_rows = val_group_rows + test_group_rows
    alpha_rows = val_alpha_rows + test_alpha_rows
    for rows in (case_rows, group_rows, alpha_rows):
        for row in rows:
            row["candidate"] = args.candidate_name
            row["condition_mode"] = args.condition_mode
            row["alpha_weight_gamma"] = args.alpha_weight_gamma

    checkpoint_path = args.output_dir / "best.pt"
    torch.save(
        {
            "stage": "stage8d1d_fno_ablation",
            "candidate_name": args.candidate_name,
            "variant": args.variant,
            "model_state_dict": model.state_dict(),
            "model_contract": model.contract(),
            "floors": floors.as_dict(),
            "loss_weights": weights.as_dict(),
            "best_epoch": best_epoch,
            "best_val_case_macro_relative_l2": best_val_macro,
            "cache": str(args.cache),
            "cache_sha256": sha256_file(args.cache),
            "config": vars(args),
        },
        checkpoint_path,
    )

    write_csv(args.output_dir / "training_history.csv", history)
    write_csv(args.output_dir / "case_metrics.csv", case_rows)
    write_csv(args.output_dir / "group_metrics.csv", group_rows)
    write_csv(args.output_dir / "alpha_metrics.csv", alpha_rows)

    def group_metric(name: str) -> dict[str, Any]:
        matches = [
            row for row in group_rows
            if row["split_group"] == name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one group row for {name}, got {len(matches)}"
            )
        return matches[0]

    all_validation = group_metric("all_validation")
    all_test = group_metric("all_test")
    expected_groups = {
        "validation",
        "legacy",
        "interstitial",
        "boundary",
        "all_validation",
        "all_test",
    }
    observed_groups = {
        row["split_group"] for row in group_rows
    }
    metric_keys = (
        "relative_l2",
        "case_macro_mean_relative_l2",
        "case_median_relative_l2",
        "case_max_relative_l2",
        "normalized_rmse",
        "normalized_mae",
        "resonance_relative_l2",
        "mean_delta_relative_l2",
        "nonzero_relative_l2",
        "mode0_relative_l2",
        "mode1_relative_l2",
    )
    all_finite = all(
        np.isfinite(float(row[key]))
        for row in group_rows
        for key in metric_keys
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(cuda_index))
        if cuda_index is not None
        else 0
    )
    quality_threshold = 5.0 if args.mode == "smoke" else 1.0
    checks = {
        "training_history_length": len(history) == args.epochs,
        "training_finite": all(
            np.isfinite(row["train_total_loss"])
            and np.isfinite(
                row["val_case_macro_relative_l2"]
            )
            for row in history
        ),
        "metrics_finite": all_finite,
        "evaluation_groups_complete": expected_groups.issubset(
            observed_groups
        ),
        "validation_macro_better_than_zero": (
            float(
                all_validation[
                    "case_macro_mean_relative_l2"
                ]
            )
            < quality_threshold
        ),
        "test_weighted_better_than_zero": (
            float(all_test["relative_l2"])
            < quality_threshold
        ),
        "peak_gpu_memory_under_20_gib": (
            peak_memory < 20 * 1024**3
        ),
    }
    passed = all(checks.values())

    acceptance = {
        "stage": "stage8d1d_fno_ablation",
        "candidate_name": args.candidate_name,
        "condition_mode": args.condition_mode,
        "alpha_weight_gamma": args.alpha_weight_gamma,
        "variant": args.variant,
        "mode": args.mode,
        "status": "PASS" if passed else "FAILED",
        "passed": passed,
        "checks": checks,
        "best_epoch": best_epoch,
        "best_val_case_macro_relative_l2": best_val_macro,
        "all_validation": all_validation,
        "all_test": all_test,
        "quality_flags": {
            "beats_idw_all_test_weighted": (
                float(all_test["relative_l2"]) < 0.156895
            ),
            "beats_idw_all_test_macro": (
                float(
                    all_test[
                        "case_macro_mean_relative_l2"
                    ]
                )
                < float(
                    all_test[
                        "idw_case_macro_mean_relative_l2"
                    ]
                )
            ),
            "majority_case_win_rate": (
                float(all_test["case_win_rate_vs_idw"])
                >= 0.5
            ),
            "mean_delta_relative_below_one": (
                float(all_test["mean_delta_relative_l2"]) < 1.0
            ),
        },
        "loss_floors": floors.as_dict(),
        "loss_weights": weights.as_dict(),
        "ablation_contract": {
            "candidate_name": args.candidate_name,
            "condition_mode": args.condition_mode,
            "modes_v": args.modes_v,
            "v_padding": args.v_padding,
            "lambda_mean_delta": args.lambda_mean_delta,
            "alpha_weight_gamma": args.alpha_weight_gamma,
            "alpha_weight_reference": args.alpha_weight_reference,
            "alpha_weight_min": args.alpha_weight_min,
            "alpha_weight_max": args.alpha_weight_max,
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

    make_plots(args.output_dir, history, case_rows, alpha_rows)

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
