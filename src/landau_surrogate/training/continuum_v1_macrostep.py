"""Train conservative multi-field macro steppers on ``continuum_v1``."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from landau_surrogate.data.continuum_macrostep import (
    MacrostepTrajectory,
    MacrostepWindowDataset,
    compute_normalization,
    denormalize_state,
    load_macrostep_trajectory,
    normalize_state,
    normalized_condition,
)
from landau_surrogate.data.continuum_v1 import (
    REGIME_TO_INDEX,
    ContinuumCase,
    load_continuum_case_index,
)
from landau_surrogate.models.macrostep_1d import build_macrostep_model
from landau_surrogate.physics.macrostep import (
    positivity_penalty,
    project_conservative_state,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def select_cases(cases: list[ContinuumCase], limit: int | None) -> list[ContinuumCase]:
    """Select a deterministic parameter-spanning subset for smoke runs."""
    if limit is None or limit >= len(cases):
        return cases
    if limit < 1:
        raise ValueError("case limit must be positive")
    ordered = sorted(cases, key=lambda case: (case.K, case.alpha, case.case_id))
    indices = np.linspace(0, len(ordered) - 1, limit, dtype=int)
    return [ordered[int(index)] for index in indices]


def load_trajectories(
    name: str,
    cases: list[ContinuumCase],
    *,
    maximum_mode: int | None,
    temporal_stride: int,
) -> list[MacrostepTrajectory]:
    values = []
    for index, case in enumerate(cases):
        values.append(
            load_macrostep_trajectory(
                case,
                maximum_mode=maximum_mode,
                temporal_stride=temporal_stride,
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(
                json.dumps(
                    {"event": "load", "split": name, "loaded": index + 1, "total": len(cases)}
                ),
                flush=True,
            )
    return values


def make_loader(
    dataset: MacrostepWindowDataset,
    *,
    batch_size: int,
    training: bool,
    samples_per_epoch: int | None,
    regime_sampling_power: float,
    trajectories: list[MacrostepTrajectory],
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = training
    if training and samples_per_epoch is not None:
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        weights = np.ones(len(dataset), dtype=np.float64)
        if regime_sampling_power > 0.0:
            regime_index = np.asarray(
                [REGIME_TO_INDEX[item.case.regime] for item in trajectories],
                dtype=np.int64,
            )
            sample_regime = regime_index[dataset.case_indices]
            counts = np.bincount(sample_regime, minlength=len(REGIME_TO_INDEX))
            weights = np.power(counts[sample_regime], -regime_sampling_power)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(samples_per_epoch),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )


def curriculum_steps(curriculum: list[dict[str, Any]], epoch: int) -> int:
    remaining = int(epoch)
    for stage in curriculum:
        duration = int(stage["epochs"])
        if remaining <= duration:
            return int(stage["steps"])
        remaining -= duration
    return int(curriculum[-1]["steps"])


def total_curriculum_epochs(curriculum: list[dict[str, Any]]) -> int:
    return sum(int(stage["epochs"]) for stage in curriculum)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _total_energy_tensor(state: torch.Tensor, k_value: torch.Tensor) -> torch.Tensor:
    domain_length = 2.0 * torch.pi / k_value
    return 0.5 * domain_length * (
        state[:, 2].mean(dim=-1) + state[:, 3].square().mean(dim=-1)
    )


def predict_next(
    model: nn.Module,
    history_normalized: torch.Tensor,
    *,
    k_value: torch.Tensor,
    alpha: torch.Tensor,
    delta_t: torch.Tensor,
    normalization: dict[str, Any],
    projection_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    condition = normalized_condition(
        k_value, alpha, delta_t, normalization
    )
    unprojected_normalized = (
        history_normalized[:, -1] + model(history_normalized, condition)
    )
    unprojected = denormalize_state(unprojected_normalized, normalization)
    reference = denormalize_state(history_normalized[:, -1], normalization)
    projected = project_conservative_state(
        unprojected,
        reference,
        k_value,
        preserve_mass=bool(projection_config.get("preserve_mass", True)),
        preserve_momentum=bool(
            projection_config.get("preserve_momentum", True)
        ),
        poisson_project_electric=bool(
            projection_config.get("poisson_project_electric", True)
        ),
        maximum_mode=projection_config.get("maximum_mode"),
    )
    return normalize_state(projected, normalization), projected


def rollout_loss(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    active_steps: int,
    normalization: dict[str, Any],
    projection_config: dict[str, Any],
    loss_weights: dict[str, Any],
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    history, targets, k_value, alpha, delta_t, _case_index = batch
    history = history.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    k_value = k_value.to(device, non_blocking=True)
    alpha = alpha.to(device, non_blocking=True)
    delta_t = delta_t.to(device, non_blocking=True)
    if active_steps > targets.shape[1]:
        raise ValueError("active rollout exceeds dataset target count")
    history_normalized = normalize_state(history, normalization)
    state_loss = torch.zeros((), device=device)
    spectral_loss = torch.zeros((), device=device)
    mode_loss = torch.zeros((), device=device)
    energy_loss = torch.zeros((), device=device)
    positive_loss = torch.zeros((), device=device)
    with autocast_context(device, amp):
        for step in range(active_steps):
            target = targets[:, step]
            target_normalized = normalize_state(target, normalization)
            next_normalized, next_physical = predict_next(
                model,
                history_normalized,
                k_value=k_value,
                alpha=alpha,
                delta_t=delta_t[:, step],
                normalization=normalization,
                projection_config=projection_config,
            )
            state_loss = state_loss + torch.mean(
                (next_normalized - target_normalized) ** 2
            )
            prediction_hat = torch.fft.rfft(next_normalized.float(), dim=-1)
            target_hat = torch.fft.rfft(target_normalized.float(), dim=-1)
            spectral_loss = spectral_loss + torch.mean(
                torch.abs(prediction_hat - target_hat) ** 2
            ) / torch.clamp(torch.mean(torch.abs(target_hat) ** 2), min=1.0e-8)
            electric_mode_error = prediction_hat[:, 3, 1] - target_hat[:, 3, 1]
            mode_loss = mode_loss + torch.mean(torch.abs(electric_mode_error) ** 2) / torch.clamp(
                torch.mean(torch.abs(target_hat[:, 3, 1]) ** 2), min=1.0e-8
            )
            prediction_energy = _total_energy_tensor(next_physical, k_value)
            target_energy = _total_energy_tensor(target, k_value)
            energy_loss = energy_loss + torch.mean(
                ((prediction_energy - target_energy) / torch.clamp(target_energy.abs(), min=1.0e-8)) ** 2
            )
            positive_loss = positive_loss + positivity_penalty(
                next_physical,
                density_floor=float(loss_weights.get("density_floor", 1.0e-4)),
                pressure_floor=float(loss_weights.get("pressure_floor", 1.0e-5)),
            )
            history_normalized = torch.cat(
                (history_normalized[:, 1:], next_normalized[:, None]), dim=1
            )
        inverse_steps = 1.0 / active_steps
        state_loss = state_loss * inverse_steps
        spectral_loss = spectral_loss * inverse_steps
        mode_loss = mode_loss * inverse_steps
        energy_loss = energy_loss * inverse_steps
        positive_loss = positive_loss * inverse_steps
        loss = (
            float(loss_weights.get("state", 1.0)) * state_loss
            + float(loss_weights.get("spectral", 0.05)) * spectral_loss
            + float(loss_weights.get("electric_mode", 0.05)) * mode_loss
            + float(loss_weights.get("energy", 0.05)) * energy_loss
            + float(loss_weights.get("positivity", 1.0)) * positive_loss
        )
    components = {
        "loss": float(loss.detach()),
        "state": float(state_loss.detach()),
        "spectral": float(spectral_loss.detach()),
        "electric_mode": float(mode_loss.detach()),
        "energy": float(energy_loss.detach()),
        "positivity": float(positive_loss.detach()),
    }
    return loss, components


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    active_steps: int,
    normalization: dict[str, Any],
    projection_config: dict[str, Any],
    device: torch.device,
    amp: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    normalized_error = 0.0
    normalized_target = 0.0
    channel_error = np.zeros(4, dtype=np.float64)
    channel_target = np.zeros(4, dtype=np.float64)
    minimum_density = float("inf")
    minimum_pressure = float("inf")
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, targets, k_value, alpha, delta_t, _case_index = batch
        history = history.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        k_value = k_value.to(device, non_blocking=True)
        alpha = alpha.to(device, non_blocking=True)
        delta_t = delta_t.to(device, non_blocking=True)
        history_normalized = normalize_state(history, normalization)
        with autocast_context(device, amp):
            for step in range(active_steps):
                target = targets[:, step]
                target_normalized = normalize_state(target, normalization)
                next_normalized, next_physical = predict_next(
                    model,
                    history_normalized,
                    k_value=k_value,
                    alpha=alpha,
                    delta_t=delta_t[:, step],
                    normalization=normalization,
                    projection_config=projection_config,
                )
                error = (next_normalized - target_normalized).float()
                normalized_error += float(torch.sum(error.square()))
                normalized_target += float(torch.sum(target_normalized.float().square()))
                physical_error = (next_physical - target).float()
                channel_error += (
                    physical_error.square().sum(dim=(0, 2)).cpu().numpy()
                )
                channel_target += (
                    target.float().square().sum(dim=(0, 2)).cpu().numpy()
                )
                density = next_physical[:, 0]
                pressure = (
                    next_physical[:, 2]
                    - next_physical[:, 1].square() / density
                )
                minimum_density = min(minimum_density, float(density.min()))
                minimum_pressure = min(minimum_pressure, float(pressure.min()))
                history_normalized = torch.cat(
                    (history_normalized[:, 1:], next_normalized[:, None]), dim=1
                )
        sample_count += len(history)
    if sample_count == 0:
        raise RuntimeError("No validation batches were evaluated")
    channel_relative_l2 = np.sqrt(channel_error) / np.maximum(
        np.sqrt(channel_target), 1.0e-30
    )
    return {
        "rollout_steps": active_steps,
        "sample_count": sample_count,
        "normalized_state_relative_l2": float(
            np.sqrt(normalized_error) / max(np.sqrt(normalized_target), 1.0e-30)
        ),
        "channel_relative_l2": {
            name: float(value)
            for name, value in zip(
                ("M0", "M1", "M2", "E"), channel_relative_l2, strict=True
            )
        },
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
    }


def checkpoint_payload(
    *,
    model: nn.Module,
    model_kind: str,
    model_arguments: dict[str, Any],
    normalization: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    epoch: int,
    validation: dict[str, Any],
    split_case_ids: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "stage": "continuum_v1_conservative_macrostep",
        "protocol": "whole_trajectory_casewise_history_macrostep",
        "source_kind": "gkeyll_continuum_v1",
        "state_contract": ["M0", "M1", "M2", "E"],
        "target_kind": "future_state_residual",
        "model_kind": model_kind,
        "model_arguments": model_arguments,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "normalization": normalization,
        "config": config,
        "seed": seed,
        "epoch": epoch,
        "validation": validation,
        "split_case_ids": split_case_ids,
        "test_evaluated_during_selection": False,
    }


def training_state_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_epoch: int,
    best_value: float,
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an epoch-boundary state used for unattended restart."""
    return {
        "stage": "continuum_v1_conservative_macrostep_training_state",
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history_rows": history_rows,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    model_section = config["model"]
    training_config = config["training"]
    projection_config = config.get("projection", {})
    loss_weights = config.get("loss_weights", {})
    curriculum = list(training_config["rollout_curriculum"])
    configured_epochs = total_curriculum_epochs(curriculum)
    epochs = int(args.epochs or configured_epochs)
    seed = int(args.seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(training_config.get("torch_threads", 2)))
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    amp = bool(training_config.get("amp_bf16", True)) and device.type == "cuda"
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support bfloat16 AMP")

    all_cases = load_continuum_case_index(data_config["dataset_root"])
    cases_by_split = {
        split: select_cases(
            [case for case in all_cases if case.split == split],
            args.case_limit_per_split,
        )
        for split in ("train", "validation", "test")
    }
    trajectories = {
        split: load_trajectories(
            split,
            cases_by_split[split],
            maximum_mode=data_config.get("maximum_mode"),
            temporal_stride=int(data_config.get("temporal_stride", 1)),
        )
        for split in ("train", "validation")
    }
    normalization = compute_normalization(trajectories["train"])
    normalization["delta_t_scale"] = float(data_config.get("delta_t_scale", 1.0))
    model_arguments = dict(model_section["arguments"])
    history_steps = int(model_arguments["history_steps"])
    macro_stride = int(data_config["macro_stride"])
    maximum_rollout_steps = max(
        max(int(stage["steps"]) for stage in curriculum),
        int(training_config.get("validation_rollout_steps", 1)),
    )
    train_dataset = MacrostepWindowDataset(
        trajectories["train"],
        history_steps=history_steps,
        macro_stride=macro_stride,
        rollout_steps=maximum_rollout_steps,
        sample_stride=int(data_config.get("training_sample_stride", 1)),
    )
    validation_dataset = MacrostepWindowDataset(
        trajectories["validation"],
        history_steps=history_steps,
        macro_stride=macro_stride,
        rollout_steps=maximum_rollout_steps,
        sample_stride=int(data_config.get("validation_sample_stride", 10)),
    )
    batch_size = int(args.batch_size or training_config["batch_size"])
    validation_batch_size = int(
        training_config.get("validation_batch_size", batch_size)
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        training=True,
        samples_per_epoch=(
            int(training_config["samples_per_epoch"])
            if training_config.get("samples_per_epoch") is not None
            else None
        ),
        regime_sampling_power=float(
            training_config.get("regime_sampling_power", 0.0)
        ),
        trajectories=trajectories["train"],
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=validation_batch_size,
        training=False,
        samples_per_epoch=None,
        regime_sampling_power=0.0,
        trajectories=trajectories["validation"],
        seed=0,
        pin_memory=device.type == "cuda",
    )

    model_kind = str(model_section["kind"])
    model = build_macrostep_model(model_kind, model_arguments).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training_config.get("minimum_learning_rate", 1.0e-6)),
    )
    split_case_ids = {
        split: [case.case_id for case in cases]
        for split, cases in cases_by_split.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "config_path": str(args.config.resolve()),
        "seed": seed,
        "device": str(device),
        "parameter_count": parameter_count,
        "model_kind": model_kind,
        "model_arguments": model_arguments,
        "normalization": normalization,
        "split_case_ids": split_case_ids,
        "test_evaluated_during_selection": False,
        "macro_stride": macro_stride,
        "maximum_rollout_steps": maximum_rollout_steps,
        "automatic_resume_enabled": bool(args.resume),
    }
    if device.type == "cuda":
        run_contract["visible_gpu_name"] = torch.cuda.get_device_name(device)
    atomic_json(args.output_dir / "run_contract.json", run_contract)

    best_value = float("inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    start_epoch = 1
    resume_mode = "new_run"
    if args.resume:
        state_path = args.output_dir / "last.pt"
        history_path = args.output_dir / "history.json"
        best_path = args.output_dir / "best.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=False)
            if state.get("stage") != "continuum_v1_conservative_macrostep_training_state":
                raise ValueError(f"invalid training state: {state_path}")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            history_rows = list(state["history_rows"])
            best_epoch = int(state["best_epoch"])
            best_value = float(state["best_value"])
            start_epoch = int(state["epoch"]) + 1
            resume_mode = "full_training_state"
        elif history_path.exists() and best_path.exists():
            # Legacy interrupted runs predate last.pt. Continue from their best
            # weights with a fresh optimizer at the correct cosine-schedule LR.
            history_rows = json.loads(history_path.read_text(encoding="utf-8"))
            completed_epochs = max(
                (int(row["epoch"]) for row in history_rows), default=0
            )
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            best_epoch = int(checkpoint["epoch"])
            best_value = float(
                checkpoint["validation"]["normalized_state_relative_l2"]
            )
            start_epoch = completed_epochs + 1
            scheduler.last_epoch = completed_epochs
            resumed_lrs = scheduler._get_closed_form_lr()
            for group, learning_rate in zip(
                optimizer.param_groups, resumed_lrs, strict=True
            ):
                group["lr"] = learning_rate
            scheduler._last_lr = resumed_lrs
            resume_mode = "best_weights_optimizer_reset"
        print(
            json.dumps(
                {
                    "event": "resume",
                    "mode": resume_mode,
                    "start_epoch": start_epoch,
                    "target_epochs": epochs,
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )
    run_contract["resume_mode"] = resume_mode
    atomic_json(args.output_dir / "run_contract.json", run_contract)
    started = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        active_steps = min(
            curriculum_steps(curriculum, epoch), maximum_rollout_steps
        )
        model.train()
        sums = {
            key: 0.0
            for key in (
                "loss",
                "state",
                "spectral",
                "electric_mode",
                "energy",
                "positivity",
            )
        }
        sample_count = 0
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, components = rollout_loss(
                model,
                batch,
                active_steps=active_steps,
                normalization=normalization,
                projection_config=projection_config,
                loss_weights=loss_weights,
                device=device,
                amp=amp,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config.get("gradient_clip", 1.0))
            )
            optimizer.step()
            count = len(batch[0])
            sample_count += count
            for key, value in components.items():
                sums[key] += value * count
        if sample_count == 0:
            raise RuntimeError("No training batches were processed")
        scheduler.step()
        validation_steps = min(
            int(training_config.get("validation_rollout_steps", active_steps)),
            maximum_rollout_steps,
        )
        validation = evaluate(
            model,
            validation_loader,
            active_steps=validation_steps,
            normalization=normalization,
            projection_config=projection_config,
            device=device,
            amp=amp,
            max_batches=args.max_eval_batches,
        )
        row = {
            "epoch": epoch,
            "active_rollout_steps": active_steps,
            **{f"train_{key}": value / sample_count for key, value in sums.items()},
            "validation": validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "samples_seen": sample_count,
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history_rows.append(row)
        atomic_json(args.output_dir / "history.json", history_rows)
        print(json.dumps(row), flush=True)
        selection_value = float(validation["normalized_state_relative_l2"])
        if selection_value < best_value:
            best_value = selection_value
            best_epoch = epoch
            atomic_torch_save(
                args.output_dir / "best.pt",
                checkpoint_payload(
                    model=model,
                    model_kind=model_kind,
                    model_arguments=model_arguments,
                    normalization=normalization,
                    config=config,
                    seed=seed,
                    epoch=epoch,
                    validation=validation,
                    split_case_ids=split_case_ids,
                ),
            )
        atomic_torch_save(
            args.output_dir / "last.pt",
            training_state_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_value=best_value,
                history_rows=history_rows,
            ),
        )

    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation = evaluate(
        model,
        validation_loader,
        active_steps=int(training_config.get("validation_rollout_steps", 1)),
        normalization=normalization,
        projection_config=projection_config,
        device=device,
        amp=amp,
        max_batches=args.max_eval_batches,
    )
    summary = {
        "stage": "continuum_v1_conservative_macrostep",
        "protocol": "whole_trajectory_casewise_history_macrostep",
        "seed": seed,
        "model_kind": model_kind,
        "history_steps": history_steps,
        "macro_stride": macro_stride,
        "best_epoch": best_epoch,
        "epochs_completed": len(history_rows),
        "best_validation_relative_l2": best_value,
        "final_validation": final_validation,
        "parameter_count": parameter_count,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "split_case_ids": split_case_ids,
        "test_evaluated_during_selection": False,
        "resume_mode": resume_mode,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--case-limit-per-split", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume from output-dir/last.pt, or from legacy best.pt and "
            "history.json when no full training state exists"
        ),
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
