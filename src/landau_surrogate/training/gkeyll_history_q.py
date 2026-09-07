"""Casewise history-FNO training that predicts heat flux before differentiation."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from landau_surrogate.models.closure_fno1d import ClosureFNO1d, spectral_derivative
from landau_surrogate.training.gkeyll_multicase import (
    load_cases,
    lowpass_numpy,
    lowpass_torch,
    metrics,
)


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(item) for item in value.split(","))
    if not offsets or offsets[-1] != 0 or any(item > 0 for item in offsets):
        raise ValueError("History offsets must be non-positive and end at zero")
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("History offsets must be unique and increasing")
    return offsets


def histories(state: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    start = -offsets[0]
    indices = np.arange(start, len(state), dtype=np.int64)
    return np.stack([state[indices + offset] for offset in offsets], axis=1)


def target_rows(value: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    return value[-offsets[0] :]


@torch.no_grad()
def predict(
    model: ClosureFNO1d,
    state: np.ndarray,
    k_normalized: float,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    heat_flux_std: float,
    maximum_mode: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    flat = state.reshape(len(state), -1, state.shape[-1])
    mean = np.tile(input_mean, state.shape[1]).reshape(1, -1, 1)
    std = np.tile(input_std, state.shape[1]).reshape(1, -1, 1)
    normalized = ((flat - mean) / std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(normalized)), batch_size=batch_size)
    values = []
    model.eval()
    for (batch,) in loader:
        batch = batch.to(device)
        condition = torch.full((len(batch),), k_normalized, device=device)
        output = lowpass_torch(model(batch, condition), maximum_mode)
        values.append((output * heat_flux_std).cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-offsets", default="-20,-10,-5,0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--maximum-mode", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--derivative-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    offsets = parse_offsets(args.history_offsets)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal history-FNO training requires CUDA")

    prepared = []
    for spec, trajectory in load_cases(args.manifest, args.dataset_root):
        state = lowpass_numpy(trajectory.state, args.maximum_mode)
        heat_flux = lowpass_numpy(trajectory.heat_flux, args.maximum_mode)
        heat_flux -= heat_flux.mean(axis=-1, keepdims=True)
        gradient = lowpass_numpy(trajectory.heat_flux_gradient, args.maximum_mode)
        prepared.append(
            (
                spec,
                trajectory,
                histories(state, offsets),
                target_rows(heat_flux, offsets),
                target_rows(gradient, offsets),
            )
        )
    train = [item for item in prepared if item[0]["split"] == "train"]
    validation = [item for item in prepared if item[0]["split"] == "validation"]
    test = [item for item in prepared if item[0]["split"] == "test"]

    train_history = np.concatenate([item[2] for item in train])
    train_flux = np.concatenate([item[3] for item in train])
    train_gradient = np.concatenate([item[4] for item in train])
    train_k = np.concatenate(
        [np.full(len(item[2]), item[1].k, dtype=np.float32) for item in train]
    )
    input_mean = train_history.mean(axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
    input_std = train_history.std(axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
    heat_flux_std = float(train_flux.std(dtype=np.float64))
    gradient_std = float(train_gradient.std(dtype=np.float64))
    k_mean = float(train_k.mean(dtype=np.float64))
    k_std = float(train_k.std(dtype=np.float64))
    if min(float(input_std.min()), heat_flux_std, gradient_std, k_std) <= 0.0:
        raise ValueError("Degenerate normalization")

    flat = train_history.reshape(len(train_history), -1, train_history.shape[-1])
    repeated_mean = np.tile(input_mean, len(offsets)).reshape(1, -1, 1)
    repeated_std = np.tile(input_std, len(offsets)).reshape(1, -1, 1)
    normalized_history = ((flat - repeated_mean) / repeated_std).astype(np.float32)
    normalized_flux = (train_flux / heat_flux_std).astype(np.float32)
    normalized_k = ((train_k - k_mean) / k_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_history),
            torch.from_numpy(normalized_flux),
            torch.from_numpy(normalized_k),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    model_arguments = {
        "state_channels": 3 * len(offsets),
        "include_k": True,
        "width": args.width,
        "modes": args.modes,
        "layers": args.layers,
        "enforce_zero_mean": True,
    }
    model = ClosureFNO1d(**model_arguments).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1.0e-5
    )
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history_rows = []
    stale = 0

    def evaluate(items: list[tuple]) -> tuple[float, list[dict]]:
        predicted_gradient, target_gradient, rows = [], [], []
        for spec, trajectory, state, flux, gradient in items:
            predicted_flux = predict(
                model,
                state,
                (trajectory.k - k_mean) / k_std,
                input_mean,
                input_std,
                heat_flux_std,
                args.maximum_mode,
                args.batch_size,
                device,
            )
            predicted = np.fft.irfft(
                1j
                * (np.arange(predicted_flux.shape[-1] // 2 + 1) * trajectory.k)
                * np.fft.rfft(predicted_flux, axis=-1),
                n=predicted_flux.shape[-1],
                axis=-1,
            ).astype(np.float32)
            flux_metrics = metrics(predicted_flux, flux)
            gradient_metrics = metrics(predicted, gradient)
            rows.append(
                {
                    "case_id": spec["case_id"],
                    "k": trajectory.k,
                    "alpha": trajectory.alpha,
                    "heat_flux_relative_l2": flux_metrics["relative_l2"],
                    "gradient_relative_l2": gradient_metrics["relative_l2"],
                    "gradient_correlation": gradient_metrics["correlation"],
                }
            )
            predicted_gradient.append(predicted)
            target_gradient.append(gradient)
        aggregate = metrics(
            np.concatenate(predicted_gradient), np.concatenate(target_gradient)
        )["relative_l2"]
        return aggregate, rows

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for state, flux, k_value in loader:
            state = state.to(device, non_blocking=True)
            flux = flux.to(device, non_blocking=True)
            k_value = k_value.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = lowpass_torch(model(state, k_value), args.maximum_mode)
            flux_loss = torch.mean((output - flux) ** 2)
            physical_output = output * heat_flux_std
            physical_target = flux * heat_flux_std
            physical_k = k_value * k_std + k_mean
            output_gradient = spectral_derivative(physical_output, physical_k)
            target_gradient = spectral_derivative(physical_target, physical_k)
            derivative_loss = torch.mean(((output_gradient - target_gradient) / gradient_std) ** 2)
            loss = flux_loss + args.derivative_weight * derivative_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_value, _ = evaluate(validation)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_gradient_relative_l2": validation_value,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if validation_value < best_value:
            best_value = validation_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    split_metrics = {}
    for name, items in (("train", train), ("validation", validation), ("test", test)):
        aggregate, rows = evaluate(items)
        split_metrics[name] = {"gradient_relative_l2": aggregate, "cases": rows}
    normalization = {
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "heat_flux_mean": 0.0,
        "heat_flux_std": heat_flux_std,
        "gradient_mean": 0.0,
        "gradient_std": gradient_std,
        "k_mean": k_mean,
        "k_std": k_std,
    }
    checkpoint = {
        "stage": "huang2025_gkeyll_history_q_v1",
        "protocol": "trajectory_casewise",
        "source_manifest": str(args.manifest),
        "history_frame_offsets": list(offsets),
        "history_time_offsets_nominal": [0.005 * value for value in offsets],
        "target_kind": "heat_flux",
        "maximum_mode": args.maximum_mode,
        "model_arguments": model_arguments,
        "normalization": normalization,
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "best_validation_gradient_relative_l2": best_value,
        "metrics": split_metrics,
    }
    summary = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    summary["epochs_completed"] = len(history_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "best.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output_dir / "best.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(history_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
