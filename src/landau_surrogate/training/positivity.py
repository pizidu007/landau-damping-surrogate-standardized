#!/usr/bin/env python3
"""Train one Stage 9B distribution-positivity candidate."""

from __future__ import annotations

import argparse
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

from landau_surrogate.models.snapshot_fno import ConditionalFNO2d
from landau_surrogate.models.stepper_fno import (
    ConditionalStepperFNO2d,
    StepperLossFloors,
    StepperLossWeights,
    compute_stepper_loss,
)
from landau_surrogate.training.one_step import (
    evaluate_sequence_predictions,
    load_split_sequences,
)
from landau_surrogate.data.rollout_windows import Stage8D2BWindowDataset
from landau_surrogate.diagnostics.rollout import (
    build_snapshot_model,
    build_stepper,
    rollout_candidate,
)
from landau_surrogate.losses.spectral import (
    aggregate_spectral_metrics,
    find_spectral_group,
    spectral_case_metrics,
)
from landau_surrogate.losses.positivity import (
    CANDIDATE_CONFIGS,
    aggregate_positivity_metrics,
    find_positivity_group,
    positivity_case_metrics,
    positivity_losses,
)


CURRICULUM_HORIZONS = (2, 4, 8)


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
        raise ValueError(f"Empty CSV: {path}")
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


def curriculum_horizon(epoch: int, total_epochs: int) -> int:
    stage = min(2, int((epoch - 1) * 3 / max(total_epochs, 1)))
    return CURRICULUM_HORIZONS[stage]


def sampled_window_indices(
    dataset: Stage8D2BWindowDataset,
    windows_per_case: int,
    seed: int,
    epoch: int,
) -> list[int]:
    starts_per_case = dataset.phase_count - dataset.horizon
    count = min(windows_per_case, starts_per_case)
    rng = np.random.default_rng(seed + 1009 * epoch)
    selected: list[int] = []
    for case_position in range(len(dataset.selected_case_indices)):
        starts = rng.choice(starts_per_case, size=count, replace=False)
        selected.extend(
            case_position * starts_per_case + int(start)
            for start in starts
        )
    rng.shuffle(selected)
    return selected


def make_loader(
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


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    output = dict(batch)
    for key in (
        "frames",
        "condition",
        "physical_condition",
        "phase_velocity",
        "dt",
    ):
        output[key] = output[key].to(
            device, non_blocking=device.type == "cuda"
        )
    return output


def autocast_context(device: torch.device, amp: str):
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


def snapshot_payload_from_parent(
    parent: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_config": parent["snapshot_model_config"],
        "model_state_dict": parent["snapshot_model_state_dict"],
        "cache_contract": parent["cache_contract"],
    }


def next_conditions(
    normalized: torch.Tensor,
    physical: torch.Tensor,
    dt: float,
    bounds: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_next = physical.float().clone()
    physical_next[:, 2] += float(dt)
    normalized_next = normalized.float().clone()
    normalized_next[:, 2] = (
        2.0
        * (physical_next[:, 2] - float(bounds["time_min"]))
        / float(bounds["time_max"] - bounds["time_min"])
        - 1.0
    )
    return normalized_next, physical_next


def apply_mean_anchor(
    prediction: dict[str, torch.Tensor],
    current: torch.Tensor,
    snapshot_model: ConditionalFNO2d,
    normalized_condition: torch.Tensor,
    physical_condition: torch.Tensor,
    dt: float,
    bounds: dict[str, float],
    beta: float,
    device: torch.device,
    amp: str,
) -> dict[str, torch.Tensor]:
    normalized_next, physical_next = next_conditions(
        normalized_condition,
        physical_condition,
        dt,
        bounds,
    )
    with torch.no_grad(), autocast_context(device, amp):
        snapshot = snapshot_model(normalized_next, physical_next)
    snapshot_mean = torch.mean(snapshot["field"].float(), dim=1)
    corrected_mean = (
        (1.0 - float(beta)) * prediction["mean_delta"].float()
        + float(beta) * snapshot_mean
    )
    nonzero = prediction["nonzero"].float()
    field = nonzero + corrected_mean[:, None, :]
    anchored = dict(prediction)
    anchored.update(
        {
            "field": field,
            "mean_delta": corrected_mean,
            "nonzero": nonzero,
            "increment": field - current.float(),
            "increment_mean": corrected_mean
            - torch.mean(current.float(), dim=1),
        }
    )
    return anchored


def train_one_epoch(
    model: ConditionalStepperFNO2d,
    snapshot_model: ConditionalFNO2d,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp: str,
    candidate_config: dict[str, float],
    floors: StepperLossFloors,
    weights: StepperLossWeights,
    bounds: dict[str, float],
    mean_anchor_beta: float,
    f0_train: torch.Tensor,
    delta_global_rms: float,
    accumulation_steps: int,
    tbptt_steps: int,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    snapshot_model.eval()
    optimizer.zero_grad(set_to_none=True)
    totals = defaultdict(float)
    sample_count = 0
    optimizer_steps = 0
    batch_count = len(loader)

    for batch_number, raw_batch in enumerate(loader, start=1):
        group_start = (
            (batch_number - 1) // accumulation_steps
        ) * accumulation_steps
        group_end = min(group_start + accumulation_steps, batch_count)
        accumulation_divisor = float(group_end - group_start)
        batch = move_batch(raw_batch, device)
        frames = batch["frames"].float()
        horizon = int(frames.shape[1] - 1)
        step_weight = 1.0 / float(horizon)
        current = frames[:, 0]
        chunk_loss: torch.Tensor | None = None
        chunk_steps = 0
        batch_components = defaultdict(float)

        for step in range(horizon):
            with autocast_context(device, amp):
                raw_prediction = model(
                    current,
                    batch["condition"][:, step],
                    batch["physical_condition"][:, step],
                )
                prediction = apply_mean_anchor(
                    raw_prediction,
                    current,
                    snapshot_model,
                    batch["condition"][:, step],
                    batch["physical_condition"][:, step],
                    float(batch["dt"][0].item()),
                    bounds,
                    mean_anchor_beta,
                    device,
                    amp,
                )
                base_loss, base_components = compute_stepper_loss(
                    prediction,
                    current,
                    frames[:, step + 1],
                    batch["phase_velocity"],
                    model.velocity,
                    floors,
                    weights,
                )
                absolute_loss, tail_loss, positivity_components = (
                    positivity_losses(
                        prediction["field"],
                        f0_train,
                        delta_global_rms,
                        tolerance_fraction=float(
                            candidate_config["tolerance_fraction"]
                        ),
                        tail_floor_fraction=float(
                            candidate_config["tail_floor_fraction"]
                        ),
                    )
                )
                total_step = (
                    base_loss
                    + float(candidate_config["absolute_weight"])
                    * absolute_loss
                    + float(candidate_config["tail_weight"])
                    * tail_loss
                )
                weighted = step_weight * total_step

            chunk_loss = weighted if chunk_loss is None else chunk_loss + weighted
            chunk_steps += 1
            for name, value in base_components.items():
                batch_components[f"base_{name}"] += (
                    step_weight * float(value.item())
                )
            batch_components["positivity_absolute"] += (
                step_weight * float(absolute_loss.item())
            )
            batch_components["positivity_tail"] += (
                step_weight * float(tail_loss.item())
            )
            for name, value in positivity_components.items():
                batch_components[f"positivity_{name}"] += (
                    step_weight * float(value.item())
                )
            batch_components["total"] += (
                step_weight * float(total_step.item())
            )

            current = prediction["field"]
            if chunk_steps >= tbptt_steps or step == horizon - 1:
                if chunk_loss is None or not torch.isfinite(chunk_loss):
                    raise RuntimeError("Non-finite Stage 9B loss.")
                scaler.scale(chunk_loss / accumulation_divisor).backward()
                current = current.detach()
                chunk_loss = None
                chunk_steps = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    result = {
        f"train_{name}_loss": value / max(sample_count, 1)
        for name, value in totals.items()
    }
    result["train_sample_count"] = float(sample_count)
    result["optimizer_steps"] = float(optimizer_steps)
    return result


def group_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [row for row in rows if row["split_group"] == name]
    if len(matches) != 1:
        raise RuntimeError(name)
    return matches[0]


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--loss-contract-checkpoint", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=sorted(CANDIDATE_CONFIGS),
        required=True,
    )
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--windows-per-case", type=int, default=4)
    parser.add_argument("--tbptt-steps", type=int, default=2)
    parser.add_argument(
        "--learning-rate", type=float, default=5.0e-5
    )
    parser.add_argument(
        "--minimum-learning-rate", type=float, default=5.0e-6
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--amp", choices=("none", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--trajectory-gate-relative", type=float, default=0.02
    )
    parser.add_argument(
        "--mode1-gate-relative", type=float, default=0.05
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.effective_batch_size % args.micro_batch_size != 0:
        raise ValueError("Effective batch must divide by micro batch.")
    accumulation_steps = args.effective_batch_size // args.micro_batch_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)

    device = (
        torch.device(args.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        cuda_index = int(device.index or 0)
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    else:
        cuda_index = None

    parent = torch.load(
        args.parent_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if parent.get("stage") != "stage8d2c_final_rollout":
        raise RuntimeError("Parent is not Stage 8D-2C final rollout.")
    if float(parent["candidate_contract"]["mean_anchor_beta"]) != 0.5:
        raise RuntimeError("Stage 9B expects mean_anchor_beta=0.5.")

    loss_contract = torch.load(
        args.loss_contract_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    expected_uniform_sha = parent["source_checkpoints"][
        "uniform_final"
    ]["sha256"]
    if sha256_file(args.loss_contract_checkpoint) != expected_uniform_sha:
        raise RuntimeError("Uniform loss-contract checkpoint SHA mismatch.")

    candidate_config = CANDIDATE_CONFIGS[args.candidate]
    model = build_stepper(parent, device)
    snapshot_model = build_snapshot_model(
        snapshot_payload_from_parent(parent), device
    )
    snapshot_model.requires_grad_(False)

    floors = StepperLossFloors(
        **{
            key: float(value)
            for key, value in loss_contract["loss_floors"].items()
        }
    )
    weights = StepperLossWeights(
        **{
            key: float(value)
            for key, value in loss_contract["loss_weights"].items()
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
    scaler_enabled = device.type == "cuda" and args.amp == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    val_sequences = load_split_sequences(args.cache, "val")
    test_sequences = load_split_sequences(args.cache, "test")
    bounds = parent["cache_contract"]["condition_bounds"]
    mean_anchor_beta = float(
        parent["candidate_contract"]["mean_anchor_beta"]
    )
    dt = float(parent["dt"])
    f0_train_cpu = parent["cache_contract"]["f0_train"].float()
    f0_train = f0_train_cpu.to(device)
    delta_global_rms = float(
        parent["cache_contract"]["delta_global_rms"]
    )

    baseline_model = build_stepper(parent, device)
    baseline_val_prediction = rollout_candidate(
        baseline_model,
        val_sequences["field"][:, 0],
        val_sequences["k"],
        val_sequences["alpha"],
        float(val_sequences["phase_time"][0]),
        val_sequences["field"].shape[1] - 1,
        dt,
        bounds,
        device,
        snapshot_model=snapshot_model,
        mean_anchor_beta=mean_anchor_beta,
        amp=args.amp,
    )
    _, baseline_val_groups, _ = evaluate_sequence_predictions(
        baseline_val_prediction,
        val_sequences,
        model_name="stage8d2c_parent",
    )
    baseline_val_group = group_row(
        baseline_val_groups, "all_validation"
    )
    baseline_pos_cases = positivity_case_metrics(
        "stage8d2c_parent",
        baseline_val_prediction,
        val_sequences,
        f0_train_cpu.numpy(),
        delta_global_rms,
    )
    baseline_pos_groups = aggregate_positivity_metrics(
        baseline_pos_cases
    )
    baseline_pos_group = find_positivity_group(
        baseline_pos_groups,
        "stage8d2c_parent",
        "all_validation",
    )
    baseline_spectral_cases = spectral_case_metrics(
        "stage8d2c_parent",
        baseline_val_prediction,
        val_sequences,
    )
    baseline_spectral_group = find_spectral_group(
        aggregate_spectral_metrics(baseline_spectral_cases),
        "stage8d2c_parent",
        "all_validation",
    )
    baseline_val_macro = float(
        baseline_val_group["trajectory_case_macro_relative_l2"]
    )
    baseline_val_mode1 = float(
        baseline_val_group["mode1_relative_l2"]
    )
    trajectory_gate = baseline_val_macro * (
        1.0 + args.trajectory_gate_relative
    )
    mode1_gate = baseline_val_mode1 * (
        1.0 + args.mode1_gate_relative
    )

    datasets: dict[int, Stage8D2BWindowDataset] = {}
    best_state = cpu_state_dict(model)
    best_epoch = 0
    best_key = (math.inf, math.inf, math.inf, math.inf)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        horizon = curriculum_horizon(epoch, args.epochs)
        if horizon not in datasets:
            datasets[horizon] = Stage8D2BWindowDataset(
                args.cache, split="train", horizon=horizon
            )
        dataset = datasets[horizon]
        indices = sampled_window_indices(
            dataset,
            args.windows_per_case,
            args.seed,
            epoch,
        )
        loader = make_loader(
            dataset, indices, args.micro_batch_size, device
        )
        train_metrics = train_one_epoch(
            model,
            snapshot_model,
            loader,
            optimizer,
            scaler,
            device,
            args.amp,
            candidate_config,
            floors,
            weights,
            bounds,
            mean_anchor_beta,
            f0_train,
            delta_global_rms,
            accumulation_steps,
            args.tbptt_steps,
            args.gradient_clip,
        )

        model.eval()
        val_prediction = rollout_candidate(
            model,
            val_sequences["field"][:, 0],
            val_sequences["k"],
            val_sequences["alpha"],
            float(val_sequences["phase_time"][0]),
            val_sequences["field"].shape[1] - 1,
            dt,
            bounds,
            device,
            snapshot_model=snapshot_model,
            mean_anchor_beta=mean_anchor_beta,
            amp=args.amp,
        )
        _, val_groups, _ = evaluate_sequence_predictions(
            val_prediction,
            val_sequences,
            model_name=args.candidate,
        )
        val_group = group_row(val_groups, "all_validation")
        val_pos_cases = positivity_case_metrics(
            args.candidate,
            val_prediction,
            val_sequences,
            f0_train_cpu.numpy(),
            delta_global_rms,
        )
        val_pos = find_positivity_group(
            aggregate_positivity_metrics(val_pos_cases),
            args.candidate,
            "all_validation",
        )
        val_macro = float(
            val_group["trajectory_case_macro_relative_l2"]
        )
        val_mode1 = float(val_group["mode1_relative_l2"])
        val_excess = float(
            val_pos["negative_fraction_excess_mean_mean"]
        )
        val_mass_excess = float(
            val_pos["negative_mass_excess_normalized_mean_mean"]
        )
        gates_pass = (
            val_macro <= trajectory_gate and val_mode1 <= mode1_gate
        )
        key = (
            0.0 if gates_pass else 1.0,
            val_excess if gates_pass else val_macro,
            val_mass_excess if gates_pass else val_mode1,
            val_macro,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "candidate": args.candidate,
            "curriculum_horizon": horizon,
            "sampled_window_count": len(indices),
            "micro_batch_size": args.micro_batch_size,
            "effective_batch_size": args.effective_batch_size,
            "tbptt_steps": args.tbptt_steps,
            "val_trajectory_macro_l2": val_macro,
            "val_horizon30_macro_l2": float(
                val_group["horizon30_case_macro_relative_l2"]
            ),
            "val_mode1_l2": val_mode1,
            "val_negative_fraction_mean": float(
                val_pos["pred_negative_fraction_mean_mean"]
            ),
            "val_negative_fraction_excess_mean": val_excess,
            "val_negative_mass_excess_normalized_mean": val_mass_excess,
            "val_minimum_full_distribution": float(
                val_pos["pred_minimum_full_distribution_mean"]
            ),
            "trajectory_gate": trajectory_gate,
            "mode1_gate": mode1_gate,
            "selection_gate_pass": gates_pass,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            **train_metrics,
        }
        history.append(row)
        print(
            f"{args.candidate} epoch {epoch:03d}/{args.epochs} "
            f"H={horizon} total={row['train_total_loss']:.6e} "
            f"val={val_macro:.6e} neg_excess={val_excess:.6e} "
            f"gate={gates_pass} seconds={row['epoch_seconds']:.2f}"
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = cpu_state_dict(model)

    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    all_case_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_horizon_rows: list[dict[str, Any]] = []
    all_pos_cases: list[dict[str, Any]] = []
    all_spectral_cases: list[dict[str, Any]] = []
    for split_name, sequences in (
        ("val", val_sequences),
        ("test", test_sequences),
    ):
        prediction = rollout_candidate(
            model,
            sequences["field"][:, 0],
            sequences["k"],
            sequences["alpha"],
            float(sequences["phase_time"][0]),
            sequences["field"].shape[1] - 1,
            dt,
            bounds,
            device,
            snapshot_model=snapshot_model,
            mean_anchor_beta=mean_anchor_beta,
            amp=args.amp,
        )
        case_rows, group_rows, horizon_rows = (
            evaluate_sequence_predictions(
                prediction, sequences, args.candidate
            )
        )
        all_case_rows.extend(case_rows)
        all_group_rows.extend(group_rows)
        all_horizon_rows.extend(horizon_rows)
        all_pos_cases.extend(
            positivity_case_metrics(
                args.candidate,
                prediction,
                sequences,
                f0_train_cpu.numpy(),
                delta_global_rms,
            )
        )
        all_spectral_cases.extend(
            spectral_case_metrics(
                args.candidate, prediction, sequences
            )
        )
    all_pos_groups = aggregate_positivity_metrics(all_pos_cases)
    all_spectral_groups = aggregate_spectral_metrics(all_spectral_cases)
    val_group = group_row(all_group_rows, "all_validation")
    test_group = group_row(all_group_rows, "all_test")
    val_pos = find_positivity_group(
        all_pos_groups, args.candidate, "all_validation"
    )
    test_pos = find_positivity_group(
        all_pos_groups, args.candidate, "all_test"
    )
    val_spectral = find_spectral_group(
        all_spectral_groups, args.candidate, "all_validation"
    )
    test_spectral = find_spectral_group(
        all_spectral_groups, args.candidate, "all_test"
    )

    checkpoint_path = args.output_dir / "best.pt"
    checkpoint = {
        "stage": "stage9b_positivity_candidate",
        "candidate": args.candidate,
        "candidate_config": candidate_config,
        "model_config": parent["model_config"],
        "model_contract": parent["model_contract"],
        "model_state_dict": model.state_dict(),
        "mean_anchor_beta": mean_anchor_beta,
        "dt": dt,
        "cache_contract": parent["cache_contract"],
        "best_epoch": best_epoch,
        "selection_policy": {
            "trajectory_gate_relative": args.trajectory_gate_relative,
            "mode1_gate_relative": args.mode1_gate_relative,
            "primary_inside_gate": (
                "validation negative-fraction excess mean"
            ),
            "secondary_inside_gate": (
                "validation normalized negative-mass excess mean"
            ),
            "test_used_for_selection": False,
        },
        "validation_metrics": val_group,
        "validation_positivity_metrics": val_pos,
        "validation_spectral_metrics": val_spectral,
        "test_metrics": test_group,
        "test_positivity_metrics": test_pos,
        "test_spectral_metrics": test_spectral,
        "parent_checkpoint": {
            "path": str(args.parent_checkpoint),
            "sha256": sha256_file(args.parent_checkpoint),
        },
        "loss_contract_checkpoint": {
            "path": str(args.loss_contract_checkpoint),
            "sha256": sha256_file(args.loss_contract_checkpoint),
        },
    }
    torch.save(checkpoint, checkpoint_path)

    write_csv(args.output_dir / "training_history.csv", history)
    write_csv(args.output_dir / "rollout_case_metrics.csv", all_case_rows)
    write_csv(args.output_dir / "rollout_group_metrics.csv", all_group_rows)
    write_csv(args.output_dir / "rollout_horizon_metrics.csv", all_horizon_rows)
    write_csv(args.output_dir / "positivity_case_metrics.csv", all_pos_cases)
    write_csv(args.output_dir / "positivity_group_metrics.csv", all_pos_groups)
    write_csv(args.output_dir / "spectral_case_metrics.csv", all_spectral_cases)
    write_csv(args.output_dir / "spectral_group_metrics.csv", all_spectral_groups)

    peak = (
        int(torch.cuda.max_memory_allocated(cuda_index))
        if cuda_index is not None
        else 0
    )
    rollout_limit = 100.0 if args.mode == "smoke" else 20.0
    norm_limit = 1000.0 if args.mode == "smoke" else 100.0
    checks = {
        "parent_contract_valid": parent.get("stage")
        == "stage8d2c_final_rollout",
        "loss_contract_sha_valid": sha256_file(
            args.loss_contract_checkpoint
        )
        == expected_uniform_sha,
        "curriculum_complete": set(
            int(row["curriculum_horizon"]) for row in history
        )
        == set(CURRICULUM_HORIZONS),
        "training_finite": all(
            np.isfinite(float(row["train_total_loss"]))
            and np.isfinite(float(row["val_trajectory_macro_l2"]))
            and np.isfinite(
                float(row["val_negative_fraction_excess_mean"])
            )
            for row in history
        ),
        "final_metrics_finite": all(
            np.isfinite(float(value))
            for row in all_group_rows + all_pos_groups + all_spectral_groups
            for key, value in row.items()
            if key
            not in {
                "model",
                "split_group",
                "positivity_group",
                "spectral_group",
            }
        ),
        "test_rollout_bounded": float(
            test_group["trajectory_case_macro_relative_l2"]
        )
        < rollout_limit,
        "test_norm_bounded": float(
            test_group["max_predicted_to_true_norm_ratio"]
        )
        < norm_limit,
        "peak_gpu_memory_under_20_gib": peak < 20 * 1024**3,
        "checkpoint_written": checkpoint_path.is_file(),
    }
    passed = all(checks.values())
    acceptance = {
        "stage": "stage9b_positivity_candidate",
        "candidate": args.candidate,
        "mode": args.mode,
        "status": "PASS" if passed else "FAILED",
        "passed": passed,
        "checks": checks,
        "best_epoch": best_epoch,
        "baseline_validation_trajectory_macro_l2": baseline_val_macro,
        "baseline_validation_mode1_l2": baseline_val_mode1,
        "baseline_validation_positivity_metrics": baseline_pos_group,
        "baseline_validation_spectral_metrics": baseline_spectral_group,
        "validation_metrics": val_group,
        "validation_positivity_metrics": val_pos,
        "validation_spectral_metrics": val_spectral,
        "test_metrics": test_group,
        "test_positivity_metrics": test_pos,
        "test_spectral_metrics": test_spectral,
        "quality_flags": {
            "validation_trajectory_gate_pass": float(
                val_group["trajectory_case_macro_relative_l2"]
            )
            <= trajectory_gate,
            "validation_mode1_gate_pass": float(
                val_group["mode1_relative_l2"]
            )
            <= mode1_gate,
            "validation_negative_fraction_excess_improved": float(
                val_pos["negative_fraction_excess_mean_mean"]
            )
            < float(
                baseline_pos_group[
                    "negative_fraction_excess_mean_mean"
                ]
            ),
            "test_negative_fraction_below_0p10": float(
                test_pos["pred_negative_fraction_mean_mean"]
            )
            < 0.10,
            "test_negative_excess_below_0p08": float(
                test_pos["negative_fraction_excess_mean_mean"]
            )
            < 0.08,
        },
        "peak_gpu_memory_bytes": peak,
        "peak_gpu_memory_gib": peak / 1024**3,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    atomic_json(args.output_dir / "acceptance.json", acceptance)
    atomic_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "candidate_config": candidate_config,
            "accumulation_steps": accumulation_steps,
            "loss_floors": floors.as_dict(),
            "loss_weights": weights.as_dict(),
        },
    )
    atomic_json(
        args.output_dir / "environment.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
        },
    )

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(
        [row["epoch"] for row in history],
        [row["val_trajectory_macro_l2"] for row in history],
        label="validation trajectory",
    )
    plt.plot(
        [row["epoch"] for row in history],
        [
            row["val_negative_fraction_excess_mean"]
            for row in history
        ],
        label="validation negative-fraction excess",
    )
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "validation_metrics.png", dpi=180)
    plt.close()

    artifact_lines = []
    for artifact in sorted(args.output_dir.rglob("*")):
        if artifact.is_file() and artifact.name != "artifact_sha256.txt":
            artifact_lines.append(
                f"{sha256_file(artifact)}  "
                f"{artifact.relative_to(args.output_dir)}"
            )
    (args.output_dir / "artifact_sha256.txt").write_text(
        "\n".join(artifact_lines) + "\n",
        encoding="utf-8",
    )
    for dataset in datasets.values():
        dataset.close()
    print(json.dumps(json_safe(acceptance), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
