#!/usr/bin/env python3
"""
Rollout-aware multi-step fine-tuning for Stage 8D-2B.

Both candidates warm-start from the frozen Stage 8D-2A residual stepper.

Curriculum
----------
epochs  1-10: horizon 2
epochs 11-20: horizon 4
epochs 21-30: horizon 8

For short smoke runs, the available epochs are divided as evenly as possible
across horizons 2, 4, and 8.

Training uses pure autoregressive inputs after the true first frame. Gradients
are truncated every `tbptt_steps` rollout steps. The formal configuration uses
micro-batch 8 and gradient accumulation 4, giving effective batch size 32 while
remaining safely below the 20 GiB GPU-memory limit.

Candidates
----------
uniform
    Equal loss weight at every unrolled step.

late_weighted
    Linearly increasing weights from early to late steps. Weights are
    normalized to sum to one for every horizon.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from landau_surrogate.data.rollout_cache import Stage8D2PairDataset
from landau_surrogate.models.stepper_fno import (
    ConditionalStepperFNO2d,
    StepperLossFloors,
    StepperLossWeights,
    compute_stepper_loss,
)
from landau_surrogate.training.one_step import (
    autoregressive_rollout,
    autocast_context,
    evaluate_sequence_predictions,
    evaluate_teacher_forced_one_step,
    load_split_sequences,
)
from landau_surrogate.data.rollout_windows import Stage8D2BWindowDataset


VALID_CANDIDATES = {"uniform", "late_weighted"}
CURRICULUM_HORIZONS = (2, 4, 8)


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
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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


def curriculum_horizon(
    epoch: int,
    total_epochs: int,
) -> int:
    if epoch < 1 or epoch > total_epochs:
        raise ValueError(
            f"epoch={epoch}, total_epochs={total_epochs}"
        )
    # Divide [1, total_epochs] into three monotone curriculum stages.
    stage = min(
        2,
        int((epoch - 1) * 3 / max(total_epochs, 1)),
    )
    return CURRICULUM_HORIZONS[stage]


def normalized_step_weights(
    candidate: str,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    if candidate not in VALID_CANDIDATES:
        raise ValueError(candidate)
    if horizon < 1:
        raise ValueError(horizon)
    if candidate == "uniform":
        values = torch.ones(
            horizon, dtype=torch.float32, device=device
        )
    else:
        values = torch.linspace(
            1.0,
            2.0,
            horizon,
            dtype=torch.float32,
            device=device,
        )
    return values / torch.sum(values)


def sampled_window_indices(
    dataset: Stage8D2BWindowDataset,
    windows_per_case: int,
    seed: int,
    epoch: int,
) -> list[int]:
    if windows_per_case < 1:
        raise ValueError("windows_per_case must be positive.")
    start_count = dataset.phase_count - dataset.horizon
    count = min(windows_per_case, start_count)
    rng = np.random.default_rng(seed + 1009 * epoch)
    selected: list[int] = []
    for case_position in range(
        len(dataset.selected_case_indices)
    ):
        starts = rng.choice(
            start_count,
            size=count,
            replace=False,
        )
        selected.extend(
            case_position * start_count + int(start)
            for start in starts
        )
    rng.shuffle(selected)
    return selected


def make_window_loader(
    dataset: Stage8D2BWindowDataset,
    indices: list[int],
    micro_batch_size: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def make_pair_loader(
    dataset: Stage8D2PairDataset,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def move_window_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "case_index",
        "start_time_index",
        "frames",
        "condition",
        "physical_condition",
        "phase_velocity",
        "dt",
    ):
        if isinstance(moved.get(key), torch.Tensor):
            moved[key] = moved[key].to(
                device,
                non_blocking=device.type == "cuda",
            )
    return moved


def model_config_from_checkpoint(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    config = checkpoint["model_config"]
    return {
        "variant": str(config["variant"]),
        "width": int(config["width"]),
        "layers": int(config["layers"]),
        "modes_x": int(config["modes_x"]),
        "modes_v": int(config["modes_v"]),
        "v_padding": int(config["v_padding"]),
        "condition_channels": int(config["condition_channels"]),
        "condition_hidden_dim": int(
            config["condition_hidden_dim"]
        ),
        "condition_hidden_layers": int(
            config["condition_hidden_layers"]
        ),
        "time_harmonics": int(config["time_harmonics"]),
        "phase_harmonics": int(config["phase_harmonics"]),
        "x_coordinate_harmonics": int(
            config["x_coordinate_harmonics"]
        ),
    }


def build_warm_started_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> ConditionalStepperFNO2d:
    if checkpoint.get("stage") != "stage8d2a_final_stepper":
        raise RuntimeError(
            "Parent checkpoint is not Stage 8D-2A final stepper."
        )
    if checkpoint.get("selected_variant") != "residual":
        raise RuntimeError(
            "Stage 8D-2B requires the residual parent stepper."
        )
    contract = checkpoint["cache_contract"]
    model = ConditionalStepperFNO2d(
        normalized_x=contract["normalized_x"].cpu().numpy(),
        velocity=contract["velocity"].cpu().numpy(),
        **model_config_from_checkpoint(checkpoint),
    )
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    return model.to(device)


def train_one_epoch(
    model: ConditionalStepperFNO2d,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp: str,
    candidate: str,
    floors: StepperLossFloors,
    weights: StepperLossWeights,
    gradient_clip: float,
    accumulation_steps: int,
    tbptt_steps: int,
) -> dict[str, float]:
    if accumulation_steps < 1:
        raise ValueError(accumulation_steps)
    if tbptt_steps < 1:
        raise ValueError(tbptt_steps)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = defaultdict(float)
    sample_count = 0
    optimizer_steps = 0
    batch_count = len(loader)

    for batch_number, raw_batch in enumerate(loader, start=1):
        group_start = (
            (batch_number - 1) // accumulation_steps
        ) * accumulation_steps
        group_end = min(
            group_start + accumulation_steps,
            batch_count,
        )
        accumulation_divisor = float(
            group_end - group_start
        )

        batch = move_window_batch(raw_batch, device)
        frames = batch["frames"].float()
        horizon = int(frames.shape[1] - 1)
        step_weights = normalized_step_weights(
            candidate, horizon, device
        )
        current = frames[:, 0]
        chunk_loss: torch.Tensor | None = None
        chunk_step_count = 0
        batch_components = defaultdict(float)

        for step in range(horizon):
            with autocast_context(device, amp):
                prediction = model(
                    current,
                    batch["condition"][:, step],
                    batch["physical_condition"][:, step],
                )
                step_loss, components = compute_stepper_loss(
                    prediction,
                    current,
                    frames[:, step + 1],
                    batch["phase_velocity"],
                    model.velocity,
                    floors,
                    weights,
                )
                weighted_loss = (
                    step_weights[step] * step_loss
                )

            chunk_loss = (
                weighted_loss
                if chunk_loss is None
                else chunk_loss + weighted_loss
            )
            chunk_step_count += 1
            for name, value in components.items():
                batch_components[name] += (
                    float(step_weights[step].item())
                    * float(value.item())
                )

            current = prediction["field"]
            chunk_boundary = (
                chunk_step_count >= tbptt_steps
                or step == horizon - 1
            )
            if chunk_boundary:
                if chunk_loss is None:
                    raise RuntimeError("Missing TBPTT chunk loss.")
                if not torch.isfinite(chunk_loss):
                    raise RuntimeError(
                        "Non-finite multi-step loss."
                    )
                scaler.scale(
                    chunk_loss / accumulation_divisor
                ).backward()
                current = current.detach()
                chunk_loss = None
                chunk_step_count = 0

        batch_size = int(frames.shape[0])
        sample_count += batch_size
        for name, value in batch_components.items():
            totals[name] += value * batch_size

        should_step = (
            batch_number % accumulation_steps == 0
            or batch_number == batch_count
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    result = {
        f"train_{name}_loss": value
        / max(sample_count, 1)
        for name, value in totals.items()
    }
    result["train_sample_count"] = float(sample_count)
    result["optimizer_steps"] = float(optimizer_steps)
    return result


def group_row(
    rows: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row["split_group"] == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one group row for {name}."
        )
    return matches[0]


def make_plots(
    output_dir: Path,
    history: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.semilogy(
        [row["epoch"] for row in history],
        [row["train_total_loss"] for row in history],
        label="train multi-step loss",
    )
    plt.semilogy(
        [row["epoch"] for row in history],
        [row["val_rollout_case_macro_l2"] for row in history],
        label="validation 30-step macro",
    )
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "training_and_validation_rollout.png",
        dpi=180,
    )
    plt.close()

    test_rows = [
        row for row in horizon_rows if row["split"] == "test"
    ]
    horizons = sorted(
        {int(row["horizon"]) for row in test_rows}
    )
    values = [
        float(
            np.mean(
                [
                    row["relative_l2"]
                    for row in test_rows
                    if int(row["horizon"]) == horizon
                ]
            )
        )
        for horizon in horizons
    ]
    plt.figure(figsize=(8, 5))
    plt.plot(horizons, values, marker="o")
    plt.xlabel("rollout horizon")
    plt.ylabel("test case-macro relative L2")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "test_rollout_error_by_horizon.png",
        dpi=180,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--parent-checkpoint", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=sorted(VALID_CANDIDATES),
        required=True,
    )
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--micro-batch-size", type=int, default=8
    )
    parser.add_argument(
        "--effective-batch-size", type=int, default=32
    )
    parser.add_argument(
        "--windows-per-case", type=int, default=4
    )
    parser.add_argument("--tbptt-steps", type=int, default=2)
    parser.add_argument(
        "--learning-rate", type=float, default=1.0e-4
    )
    parser.add_argument(
        "--minimum-learning-rate", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--amp", choices=("none", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.effective_batch_size
        % args.micro_batch_size
        != 0
    ):
        raise ValueError(
            "effective_batch_size must be divisible by "
            "micro_batch_size."
        )
    accumulation_steps = (
        args.effective_batch_size
        // args.micro_batch_size
    )

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

    parent = torch.load(
        args.parent_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = build_warm_started_model(parent, device)
    floors = StepperLossFloors(
        **{
            key: float(value)
            for key, value in parent["loss_floors"].items()
        }
    )
    weights = StepperLossWeights(
        **{
            key: float(value)
            for key, value in parent["loss_weights"].items()
            if key != "global_mse"
        }
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.minimum_learning_rate,
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

    val_sequences = load_split_sequences(args.cache, "val")
    test_sequences = load_split_sequences(args.cache, "test")
    pair_val = Stage8D2PairDataset(args.cache, split="val")
    pair_test = Stage8D2PairDataset(args.cache, split="test")
    pair_val_loader = make_pair_loader(
        pair_val, 32, device
    )
    pair_test_loader = make_pair_loader(
        pair_test, 32, device
    )

    datasets: dict[int, Stage8D2BWindowDataset] = {}
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_rollout = math.inf
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        horizon = curriculum_horizon(epoch, args.epochs)
        if horizon not in datasets:
            datasets[horizon] = Stage8D2BWindowDataset(
                args.cache,
                split="train",
                horizon=horizon,
            )
        dataset = datasets[horizon]
        indices = sampled_window_indices(
            dataset,
            windows_per_case=args.windows_per_case,
            seed=args.seed,
            epoch=epoch,
        )
        loader = make_window_loader(
            dataset,
            indices,
            args.micro_batch_size,
            device,
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=args.amp,
            candidate=args.candidate,
            floors=floors,
            weights=weights,
            gradient_clip=args.gradient_clip,
            accumulation_steps=accumulation_steps,
            tbptt_steps=args.tbptt_steps,
        )

        val_prediction = autoregressive_rollout(
            model,
            val_sequences,
            device,
            args.amp,
        )
        _, val_group_rows, _ = evaluate_sequence_predictions(
            val_prediction,
            val_sequences,
            model_name=args.candidate,
        )
        val_summary = group_row(
            val_group_rows, "all_validation"
        )
        val_rollout = float(
            val_summary[
                "trajectory_case_macro_relative_l2"
            ]
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "curriculum_horizon": horizon,
            "candidate": args.candidate,
            "micro_batch_size": args.micro_batch_size,
            "effective_batch_size": (
                args.effective_batch_size
            ),
            "accumulation_steps": accumulation_steps,
            "tbptt_steps": args.tbptt_steps,
            "windows_per_case": args.windows_per_case,
            "sampled_window_count": len(indices),
            "val_rollout_case_macro_l2": val_rollout,
            "val_horizon30_case_macro_l2": float(
                val_summary[
                    "horizon30_case_macro_relative_l2"
                ]
            ),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "epoch_seconds": time.perf_counter()
            - epoch_started,
            **train_metrics,
        }
        history.append(row)
        print(
            f"{args.candidate} epoch {epoch:03d}/{args.epochs} "
            f"horizon={horizon} "
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
            model,
            pair_val_loader,
            device,
            args.amp,
            "val",
        )
        + evaluate_teacher_forced_one_step(
            model,
            pair_test_loader,
            device,
            args.amp,
            "test",
        )
    )
    for row in one_step_rows:
        row["model"] = args.candidate
        row["candidate"] = args.candidate

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
            model_name=args.candidate,
        )
    )
    test_case, test_group, test_horizon = (
        evaluate_sequence_predictions(
            test_prediction,
            test_sequences,
            model_name=args.candidate,
        )
    )
    case_rows = val_case + test_case
    group_rows = val_group + test_group
    horizon_rows = val_horizon + test_horizon
    val_summary = group_row(group_rows, "all_validation")
    test_summary = group_row(group_rows, "all_test")

    checkpoint_path = args.output_dir / "best.pt"
    torch.save(
        {
            "stage": "stage8d2b_multistep_candidate",
            "candidate": args.candidate,
            "model_config": model_config_from_checkpoint(parent),
            "model_contract": model.contract(),
            "model_state_dict": model.state_dict(),
            "loss_floors": floors.as_dict(),
            "loss_weights": weights.as_dict(),
            "best_epoch": best_epoch,
            "best_val_rollout_case_macro_l2": (
                best_val_rollout
            ),
            "curriculum_horizons": list(
                CURRICULUM_HORIZONS
            ),
            "tbptt_steps": args.tbptt_steps,
            "micro_batch_size": args.micro_batch_size,
            "effective_batch_size": args.effective_batch_size,
            "windows_per_case": args.windows_per_case,
            "parent_checkpoint": str(
                args.parent_checkpoint
            ),
            "parent_checkpoint_sha256": sha256_file(
                args.parent_checkpoint
            ),
            "cache": str(args.cache),
            "cache_sha256": sha256_file(args.cache),
            "dt": float(parent["dt"]),
            "config": vars(args),
        },
        checkpoint_path,
    )

    write_csv(
        args.output_dir / "training_history.csv",
        history,
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

    peak_memory = (
        int(torch.cuda.max_memory_allocated(cuda_index))
        if cuda_index is not None
        else 0
    )
    rollout_limit = (
        1000.0 if args.mode == "smoke" else 20.0
    )
    norm_ratio_limit = (
        10000.0 if args.mode == "smoke" else 100.0
    )
    checks = {
        "parent_stage8d2a_contract_valid": (
            parent.get("stage") == "stage8d2a_final_stepper"
            and parent.get("selected_variant") == "residual"
        ),
        "training_history_length": len(history) == args.epochs,
        "all_curriculum_horizons_used": (
            set(
                int(row["curriculum_horizon"])
                for row in history
            )
            == set(CURRICULUM_HORIZONS)
        ),
        "training_finite": all(
            np.isfinite(float(row["train_total_loss"]))
            and np.isfinite(
                float(row["val_rollout_case_macro_l2"])
            )
            for row in history
        ),
        "metrics_finite": all(
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
        "stage": "stage8d2b_multistep_candidate",
        "candidate": args.candidate,
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
            "test_rollout_macro_below_parent": (
                float(
                    test_summary[
                        "trajectory_case_macro_relative_l2"
                    ]
                )
                < float(
                    parent["test_metrics"][
                        "trajectory_case_macro_relative_l2"
                    ]
                )
            ),
            "test_horizon30_below_parent": (
                float(
                    test_summary[
                        "horizon30_case_macro_relative_l2"
                    ]
                )
                < float(
                    parent["test_metrics"][
                        "horizon30_case_macro_relative_l2"
                    ]
                )
            ),
            "test_rollout_macro_below_one": (
                float(
                    test_summary[
                        "trajectory_case_macro_relative_l2"
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
    config.update(
        {
            "device_resolved": str(device),
            "accumulation_steps": accumulation_steps,
            "curriculum_horizons": list(
                CURRICULUM_HORIZONS
            ),
            "model_contract": model.contract(),
            "loss_floors": floors.as_dict(),
            "loss_weights": weights.as_dict(),
        }
    )
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
        args.output_dir / "environment.json",
        environment,
    )
    atomic_json(
        args.output_dir / "acceptance.json",
        acceptance,
    )

    make_plots(
        args.output_dir,
        history,
        horizon_rows,
    )

    artifact_lines = []
    for artifact in sorted(args.output_dir.rglob("*")):
        if (
            artifact.is_file()
            and artifact.name != "artifact_sha256.txt"
        ):
            artifact_lines.append(
                f"{sha256_file(artifact)}  "
                f"{artifact.relative_to(args.output_dir)}"
            )
    (
        args.output_dir / "artifact_sha256.txt"
    ).write_text(
        "\n".join(artifact_lines) + "\n",
        encoding="utf-8",
    )

    for dataset in datasets.values():
        dataset.close()
    pair_val.close()
    pair_test.close()

    print(json.dumps(json_safe(acceptance), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
