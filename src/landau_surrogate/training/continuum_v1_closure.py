"""Pure-supervised FNO closure training on the ``continuum_v1`` dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from landau_surrogate.data.continuum_v1 import (
    REGIME_TO_INDEX,
    ContinuumCase,
    load_continuum_case,
    load_continuum_case_index,
)
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


@dataclass
class FrameSplit:
    name: str
    cases: list[ContinuumCase]
    state: np.ndarray
    target: np.ndarray
    k: np.ndarray
    regime: np.ndarray
    case_index: np.ndarray


class StreamingMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.error_square = 0.0
        self.target_square = 0.0
        self.prediction_sum = 0.0
        self.target_sum = 0.0
        self.prediction_square = 0.0
        self.cross = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        error = prediction - target
        self.count += int(prediction.size)
        self.error_square += float(np.sum(error * error))
        self.target_square += float(np.sum(target * target))
        self.prediction_sum += float(np.sum(prediction))
        self.target_sum += float(np.sum(target))
        self.prediction_square += float(np.sum(prediction * prediction))
        self.cross += float(np.sum(prediction * target))

    def result(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("No samples were evaluated")
        prediction_variance = self.prediction_square - (
            self.prediction_sum * self.prediction_sum / self.count
        )
        target_variance = self.target_square - (
            self.target_sum * self.target_sum / self.count
        )
        covariance = self.cross - (
            self.prediction_sum * self.target_sum / self.count
        )
        denominator = math.sqrt(
            max(prediction_variance, 0.0) * max(target_variance, 0.0)
        )
        return {
            "relative_l2": math.sqrt(self.error_square)
            / max(math.sqrt(self.target_square), 1.0e-12),
            "rmse": math.sqrt(self.error_square / self.count),
            "correlation": covariance / denominator if denominator > 0.0 else 0.0,
            "scalar_count": self.count,
        }


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


def lowpass_torch(value: torch.Tensor, maximum_mode: int | None) -> torch.Tensor:
    if maximum_mode is None:
        return value
    transformed = torch.fft.rfft(value.float(), dim=-1)
    transformed[..., int(maximum_mode) + 1 :] = 0.0
    return torch.fft.irfft(transformed, n=value.shape[-1], dim=-1).to(value.dtype)


def select_cases(cases: list[ContinuumCase], limit: int | None) -> list[ContinuumCase]:
    """Select a deterministic parameter-spanning subset for smoke tests."""
    if limit is None or limit >= len(cases):
        return cases
    if limit < 1:
        raise ValueError("case limit must be positive")
    ordered = sorted(cases, key=lambda case: (case.K, case.alpha, case.case_id))
    indices = np.linspace(0, len(ordered) - 1, limit, dtype=int)
    return [ordered[int(index)] for index in indices]


def load_frame_split(
    name: str,
    cases: list[ContinuumCase],
    *,
    input_maximum_mode: int | None,
    target_maximum_mode: int | None,
    temporal_stride: int,
) -> FrameSplit:
    if not cases:
        raise ValueError(f"The {name} split is empty")
    frame_counts: list[int] = []
    nx: int | None = None
    for case in cases:
        with h5py.File(case.path, "r") as handle:
            shape = handle["diagnostics/central_moments"].shape
        if len(shape) != 3 or shape[-1] != 4:
            raise ValueError(f"Invalid central-moment shape {shape} in {case.path}")
        if nx is None:
            nx = int(shape[1])
        elif nx != int(shape[1]):
            raise ValueError("All cases in one training run must have the same Nx")
        frame_counts.append((int(shape[0]) + temporal_stride - 1) // temporal_stride)

    total = sum(frame_counts)
    assert nx is not None
    state = np.empty((total, 3, nx), dtype=np.float32)
    target = np.empty((total, nx), dtype=np.float32)
    k_value = np.empty(total, dtype=np.float32)
    regime = np.empty(total, dtype=np.int8)
    case_index = np.empty(total, dtype=np.int16)
    offset = 0
    for index, (case, count) in enumerate(zip(cases, frame_counts, strict=True)):
        trajectory = load_continuum_case(
            case,
            input_maximum_mode=input_maximum_mode,
            target_maximum_mode=target_maximum_mode,
            temporal_stride=temporal_stride,
        )
        if trajectory.state.shape[0] != count:
            raise ValueError(f"Unexpected frame count for {case.case_id}")
        section = slice(offset, offset + count)
        state[section] = trajectory.state
        target[section] = trajectory.heat_flux_gradient
        k_value[section] = case.K
        regime[section] = REGIME_TO_INDEX[case.regime]
        case_index[section] = index
        offset += count
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(
                f"loaded {name}: {index + 1}/{len(cases)} cases, "
                f"{offset} frames",
                flush=True,
            )
    return FrameSplit(name, cases, state, target, k_value, regime, case_index)


def compute_normalization(split: FrameSplit) -> dict[str, Any]:
    input_mean = split.state.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    input_std = split.state.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    gradient_mean = float(split.target.mean(dtype=np.float64))
    gradient_std = float(split.target.std(dtype=np.float64))
    k_mean = float(split.k.mean(dtype=np.float64))
    k_std = float(split.k.std(dtype=np.float64))
    if min(float(input_std.min()), gradient_std, k_std) <= 0.0:
        raise ValueError("Degenerate training normalization")
    return {
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "target_mean": gradient_mean,
        "target_std": gradient_std,
        "k_mean": k_mean,
        "k_std": k_std,
    }


def normalize_in_place(split: FrameSplit, normalization: dict[str, Any]) -> None:
    mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    std = np.asarray(normalization["input_std"], dtype=np.float32)
    split.state -= mean[None, :, None]
    split.state /= std[None, :, None]
    split.target -= float(normalization["gradient_mean"])
    split.target /= float(normalization["gradient_std"])
    split.k -= float(normalization["k_mean"])
    split.k /= float(normalization["k_std"])


def make_loader(
    split: FrameSplit,
    *,
    batch_size: int,
    training: bool,
    regime_sampling_power: float,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(split.state),
        torch.from_numpy(split.target),
        torch.from_numpy(split.k),
        torch.from_numpy(split.case_index.astype(np.int64, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = training
    if training and regime_sampling_power > 0.0:
        counts = np.bincount(split.regime, minlength=len(REGIME_TO_INDEX))
        if np.any(counts == 0):
            raise ValueError(f"Cannot balance absent regime; counts={counts.tolist()}")
        weights = np.power(counts[split.regime], -regime_sampling_power)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
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
        pin_memory=True,
        generator=generator,
    )


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    )


@torch.no_grad()
def evaluate(
    model: ClosureFNO1d,
    split: FrameSplit,
    *,
    normalization: dict[str, Any],
    batch_size: int,
    target_maximum_mode: int | None,
    device: torch.device,
    amp: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    loader = make_loader(
        split,
        batch_size=batch_size,
        training=False,
        regime_sampling_power=0.0,
        seed=0,
    )
    aggregate = StreamingMetrics()
    per_case = [StreamingMetrics() for _case in split.cases]
    target_mean = float(normalization["gradient_mean"])
    target_std = float(normalization["gradient_std"])
    model.eval()
    for batch_index, (state, target, k_value, case_index) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        state = state.to(device, non_blocking=True)
        k_value = k_value.to(device, non_blocking=True)
        with autocast_context(device, amp):
            prediction = lowpass_torch(
                model(state, k_value), target_maximum_mode
            )
        prediction_np = prediction.float().cpu().numpy() * target_std + target_mean
        target_np = target.numpy() * target_std + target_mean
        case_np = case_index.numpy()
        aggregate.update(prediction_np, target_np)
        for index in np.unique(case_np):
            mask = case_np == index
            per_case[int(index)].update(prediction_np[mask], target_np[mask])

    case_rows = []
    for case, accumulator in zip(split.cases, per_case, strict=True):
        if accumulator.count == 0:
            continue
        case_rows.append(
            {
                "case_id": case.case_id,
                "K": case.K,
                "alpha": case.alpha,
                "regime": case.regime,
                **accumulator.result(),
            }
        )
    return {**aggregate.result(), "cases": case_rows}


def checkpoint_payload(
    *,
    model: ClosureFNO1d,
    model_arguments: dict[str, Any],
    normalization: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    epoch: int,
    best_validation_relative_l2: float,
    split_case_ids: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "stage": "continuum_v1_pure_supervised_fno",
        "protocol": "whole_trajectory_casewise",
        "source_kind": "gkeyll_continuum_v1",
        "moment_definition": "central_n_u_p_with_p_equal_nT",
        "target_kind": "gradient",
        "model_arguments": model_arguments,
        "model_config": model_arguments,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "normalization": normalization,
        "config": config,
        "seed": seed,
        "epoch": epoch,
        "best_validation_relative_l2": best_validation_relative_l2,
        "split_case_ids": split_case_ids,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    model_config = dict(config["model"])
    training_config = config["training"]
    epochs = int(args.epochs or training_config["epochs"])
    batch_size = int(args.batch_size or training_config["batch_size"])
    evaluation_batch_size = int(training_config.get("evaluation_batch_size", batch_size))
    seed = int(args.seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("continuum_v1 formal closure training requires CUDA")
    if not torch.cuda.is_bf16_supported() and bool(training_config.get("amp_bf16", True)):
        raise RuntimeError("The selected GPU does not support configured bfloat16 AMP")
    amp = bool(training_config.get("amp_bf16", True))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    all_cases = load_continuum_case_index(data_config["dataset_root"])
    cases_by_split = {
        name: select_cases(
            [case for case in all_cases if case.split == name],
            args.case_limit_per_split,
        )
        for name in ("train", "validation", "test")
    }
    print(
        json.dumps(
            {
                "event": "case_split",
                "counts": {name: len(value) for name, value in cases_by_split.items()},
                "seed": seed,
                "device": str(device),
            }
        ),
        flush=True,
    )
    splits = {
        name: load_frame_split(
            name,
            cases,
            input_maximum_mode=data_config.get("input_maximum_mode"),
            target_maximum_mode=data_config.get("target_maximum_mode"),
            temporal_stride=int(data_config.get("temporal_stride", 1)),
        )
        for name, cases in cases_by_split.items()
    }
    normalization = compute_normalization(splits["train"])
    for split in splits.values():
        normalize_in_place(split, normalization)

    train_loader = make_loader(
        splits["train"],
        batch_size=batch_size,
        training=True,
        regime_sampling_power=float(training_config.get("regime_sampling_power", 0.0)),
        seed=seed,
    )
    model = ClosureFNO1d(**model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training_config.get("minimum_learning_rate", 1.0e-5)),
    )
    spectral_weight = float(training_config.get("spectral_loss_weight", 0.1))
    patience = int(training_config.get("patience", epochs))
    target_maximum_mode = data_config.get("target_maximum_mode")
    split_case_ids = {
        name: [case.case_id for case in split.cases]
        for name, split in splits.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "config_path": str(args.config.resolve()),
        "seed": seed,
        "device": str(device),
        "visible_gpu_name": torch.cuda.get_device_name(device),
        "parameter_count": parameter_count,
        "split_case_ids": split_case_ids,
        "normalization": normalization,
        "test_evaluated_during_selection": False,
    }
    atomic_json(args.output_dir / "run_contract.json", run_contract)

    best_value = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        pointwise_sum = 0.0
        spectral_sum = 0.0
        sample_count = 0
        epoch_started = time.monotonic()
        for batch_index, (state, target, k_value, _case_index) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            state = state.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            k_value = k_value.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                output = lowpass_torch(model(state, k_value), target_maximum_mode)
                pointwise = torch.mean((output - target) ** 2)
                output_hat = torch.fft.rfft(output.float(), dim=-1)
                target_hat = torch.fft.rfft(target.float(), dim=-1)
                spectral = torch.mean(torch.abs(output_hat - target_hat) ** 2) / torch.clamp(
                    torch.mean(torch.abs(target_hat) ** 2), min=1.0e-8
                )
                loss = pointwise + spectral_weight * spectral
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config.get("gradient_clip", 1.0))
            )
            optimizer.step()
            count = len(state)
            sample_count += count
            loss_sum += float(loss.detach()) * count
            pointwise_sum += float(pointwise.detach()) * count
            spectral_sum += float(spectral.detach()) * count
        if sample_count == 0:
            raise RuntimeError("No training batches were processed")
        scheduler.step()
        validation = evaluate(
            model,
            splits["validation"],
            normalization=normalization,
            batch_size=evaluation_batch_size,
            target_maximum_mode=target_maximum_mode,
            device=device,
            amp=amp,
            max_batches=args.max_eval_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "train_pointwise_mse": pointwise_sum / sample_count,
            "train_spectral_loss": spectral_sum / sample_count,
            "validation_relative_l2": validation["relative_l2"],
            "validation_rmse": validation["rmse"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.monotonic() - epoch_started,
            "samples_seen": sample_count,
        }
        history.append(row)
        atomic_json(args.output_dir / "history.json", history)
        print(json.dumps(row), flush=True)
        if validation["relative_l2"] < best_value:
            best_value = float(validation["relative_l2"])
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                args.output_dir / "best.pt",
                checkpoint_payload(
                    model=model,
                    model_arguments=model_config,
                    normalization=normalization,
                    config=config,
                    seed=seed,
                    epoch=epoch,
                    best_validation_relative_l2=best_value,
                    split_case_ids=split_case_ids,
                ),
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping after {stale} stale epochs", flush=True)
                break

    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {
        name: evaluate(
            model,
            split,
            normalization=normalization,
            batch_size=evaluation_batch_size,
            target_maximum_mode=target_maximum_mode,
            device=device,
            amp=amp,
            max_batches=args.max_eval_batches,
        )
        for name, split in splits.items()
    }
    summary = {
        "stage": "continuum_v1_pure_supervised_fno",
        "protocol": "whole_trajectory_casewise",
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_relative_l2": best_value,
        "metrics": metrics,
        "split_case_ids": split_case_ids,
        "parameter_count": parameter_count,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "test_evaluated_during_selection": False,
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
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
