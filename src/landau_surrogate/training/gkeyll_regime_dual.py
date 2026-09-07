"""Regime-balanced dual-head history-residual FNO for enriched Gkeyll data."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from landau_surrogate.models.closure_fno1d import (
    DualHeadHistoryResidualFNO1d,
    spectral_derivative,
)
from landau_surrogate.training.gkeyll_history_q import histories, parse_offsets, target_rows
from landau_surrogate.training.gkeyll_multicase import load_cases, lowpass_numpy, lowpass_torch, metrics


def warm_start_base(model: DualHeadHistoryResidualFNO1d, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    if checkpoint.get("target_kind") == "dual_gradient_heat_flux":
        target = model.state_dict()
        copied = []
        for key in target:
            if key in source and source[key].shape == target[key].shape:
                target[key].copy_(source[key])
                copied.append(key)
        model.load_state_dict(target)
        return {
            "path": str(path),
            "kind": "full_dual_model",
            "copied_keys": copied,
            "copied_key_count": len(copied),
        }
    target = model.base.state_dict()
    copied = []
    for key in target:
        if key == "lift.weight":
            old = source[key]
            if old.shape[0] == target[key].shape[0] and old.shape[1] == 4:
                target[key].zero_()
                target[key][:, :4] = old
                copied.append(key)
        elif key in source and source[key].shape == target[key].shape:
            target[key].copy_(source[key])
            copied.append(key)
    model.base.load_state_dict(target)
    return {
        "path": str(path), "kind": "base_model",
        "copied_keys": copied, "copied_key_count": len(copied),
    }


def padded_histories(state: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    """Build histories for every time row, repeating the initial state before t=0."""
    indices = np.arange(len(state), dtype=np.int64)
    return np.stack(
        [state[np.maximum(indices + offset, 0)] for offset in offsets], axis=1
    )


@torch.no_grad()
def predict_case(
    model: DualHeadHistoryResidualFNO1d,
    history: np.ndarray,
    condition: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    gradient_std: float,
    heat_flux_std: float,
    maximum_mode: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = ((history - input_mean[None, None, :, None]) / input_std[None, None, :, None]).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(normalized)), batch_size=batch_size)
    gradients, fluxes = [], []
    model.eval()
    condition_tensor = torch.as_tensor(condition, dtype=torch.float32, device=device)
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        current_condition = condition_tensor.reshape(1, 2).expand(len(batch), -1)
        gradient, flux = model(batch, current_condition)
        gradients.append((lowpass_torch(gradient, maximum_mode) * gradient_std).cpu().numpy())
        fluxes.append((lowpass_torch(flux, maximum_mode) * heat_flux_std).cpu().numpy())
    return np.concatenate(gradients), np.concatenate(fluxes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--regime-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-offsets", default="-20,-10,-5,0")
    parser.add_argument(
        "--pad-history", action="store_true",
        help="Retain early frames by repeating the t=0 state for unavailable history.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=48000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--maximum-mode", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--heat-flux-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--spectral-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    offsets = parse_offsets(args.history_offsets)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal regime-balanced FNO training requires CUDA")

    labels_file = np.load(args.regime_labels)
    prepared = []
    select_target_rows = (
        (lambda value: value)
        if args.pad_history
        else (lambda value: target_rows(value, offsets))
    )
    for spec, trajectory in load_cases(args.manifest, args.dataset_root):
        state = lowpass_numpy(trajectory.state, args.maximum_mode)
        flux = lowpass_numpy(trajectory.heat_flux, args.maximum_mode)
        flux -= flux.mean(axis=-1, keepdims=True)
        gradient = lowpass_numpy(trajectory.heat_flux_gradient, args.maximum_mode)
        history_values = (
            padded_histories(state, offsets) if args.pad_history else histories(state, offsets)
        )
        prepared.append({
            "spec": spec,
            "trajectory": trajectory,
            "history": history_values,
            "flux": select_target_rows(flux),
            "gradient": select_target_rows(gradient),
            "regime": select_target_rows(labels_file[spec["case_id"]]),
        })
    labels_file.close()
    splits = {name: [item for item in prepared if item["spec"]["split"] == name] for name in ("train", "validation", "test")}
    if any(not value for value in splits.values()):
        raise ValueError("Manifest must contain train, validation, and test cases")

    train = splits["train"]
    train_history = np.concatenate([item["history"] for item in train])
    train_flux = np.concatenate([item["flux"] for item in train])
    train_gradient = np.concatenate([item["gradient"] for item in train])
    train_regime = np.concatenate([item["regime"] for item in train])
    train_case = np.concatenate([np.full(len(item["history"]), index) for index, item in enumerate(train)])
    train_k = np.concatenate([np.full(len(item["history"]), item["trajectory"].k) for item in train])
    train_alpha = np.concatenate([np.full(len(item["history"]), item["trajectory"].alpha) for item in train])

    input_mean = train_history.mean(axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
    input_std = train_history.std(axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
    gradient_std = float(train_gradient.std(dtype=np.float64))
    heat_flux_std = float(train_flux.std(dtype=np.float64))
    condition_mean = np.array([train_k.mean(), train_alpha.mean()], dtype=np.float32)
    condition_std = np.array([train_k.std(), train_alpha.std()], dtype=np.float32)
    if min(float(input_std.min()), gradient_std, heat_flux_std, float(condition_std.min())) <= 0.0:
        raise ValueError("Degenerate normalization")

    normalized_history = ((train_history - input_mean[None, None, :, None]) / input_std[None, None, :, None]).astype(np.float32)
    normalized_gradient = (train_gradient / gradient_std).astype(np.float32)
    normalized_flux = (train_flux / heat_flux_std).astype(np.float32)
    normalized_condition = ((np.stack((train_k, train_alpha), axis=1) - condition_mean) / condition_std).astype(np.float32)

    # Give every case equal probability and every regime present within a case equal probability.
    weights = np.empty(len(train_regime), dtype=np.float64)
    sampling_cells = []
    for case_index, item in enumerate(train):
        case_mask = train_case == case_index
        present = np.unique(train_regime[case_mask])
        for regime in present:
            cell = case_mask & (train_regime == regime)
            count = int(cell.sum())
            weights[cell] = 1.0 / (len(train) * len(present) * count)
            sampling_cells.append({"case_id": item["spec"]["case_id"], "regime": int(regime), "count": count})
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), args.samples_per_epoch, replacement=True, generator=generator)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_history), torch.from_numpy(normalized_gradient),
            torch.from_numpy(normalized_flux), torch.from_numpy(normalized_condition),
        ), batch_size=args.batch_size, sampler=sampler, pin_memory=True,
    )

    model_arguments = {"history_steps": len(offsets), "width": args.width, "modes": args.modes, "layers": args.layers}
    model = DualHeadHistoryResidualFNO1d(**model_arguments).to(device)
    warm_start = warm_start_base(model, args.warm_start) if args.warm_start else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-5)
    best_value, stale = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    history_rows = []

    def evaluate(items: list[dict], save: bool = False, split: str = "") -> dict:
        all_prediction, all_target, rows = [], [], []
        for item in items:
            trajectory = item["trajectory"]
            condition = (np.array([trajectory.k, trajectory.alpha]) - condition_mean) / condition_std
            prediction, flux_prediction = predict_case(
                model, item["history"], condition, input_mean, input_std, gradient_std,
                heat_flux_std, args.maximum_mode, args.batch_size, device,
            )
            row = {"case_id": item["spec"]["case_id"], "k": trajectory.k, "alpha": trajectory.alpha,
                   "gradient": metrics(prediction, item["gradient"]),
                   "heat_flux": metrics(flux_prediction, item["flux"]), "regimes": {}}
            for regime in np.unique(item["regime"]):
                selected = item["regime"] == regime
                row["regimes"][str(int(regime))] = metrics(prediction[selected], item["gradient"][selected])
            rows.append(row)
            all_prediction.append(prediction)
            all_target.append(item["gradient"])
            if save:
                directory = args.output_dir / "evaluation" / split
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / f"{item['spec']['case_id']}.npz",
                    time=select_target_rows(trajectory.time), x_over_l=trajectory.x_over_l,
                    target=item["gradient"], prediction=prediction, heat_flux_target=item["flux"],
                    heat_flux_prediction=flux_prediction, regime=item["regime"])
        return {"gradient_relative_l2": metrics(np.concatenate(all_prediction), np.concatenate(all_target))["relative_l2"], "cases": rows}

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for state, gradient_target, flux_target, condition in loader:
            state, gradient_target, flux_target, condition = [value.to(device, non_blocking=True) for value in (state, gradient_target, flux_target, condition)]
            optimizer.zero_grad(set_to_none=True)
            gradient_output, flux_output = model(state, condition)
            gradient_output = lowpass_torch(gradient_output, args.maximum_mode)
            flux_output = lowpass_torch(flux_output, args.maximum_mode)
            gradient_loss = torch.mean((gradient_output - gradient_target) ** 2)
            output_hat = torch.fft.rfft(gradient_output.float(), dim=-1)
            target_hat = torch.fft.rfft(gradient_target.float(), dim=-1)
            spectral_loss = torch.mean(torch.abs(output_hat - target_hat) ** 2) / torch.clamp(torch.mean(torch.abs(target_hat) ** 2), min=1.0e-8)
            flux_loss = torch.mean((flux_output - flux_target) ** 2)
            physical_k = condition[:, 0] * float(condition_std[0]) + float(condition_mean[0])
            flux_gradient = spectral_derivative(flux_output * heat_flux_std, physical_k) / gradient_std
            consistency_loss = torch.mean((flux_gradient - gradient_target) ** 2)
            loss = gradient_loss + args.spectral_weight * spectral_loss + args.heat_flux_weight * flux_loss + args.consistency_weight * consistency_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(splits["validation"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation_gradient_relative_l2": validation["gradient_relative_l2"],
               "history_gate": float(torch.sigmoid(model.history_gate_logit).detach()),
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_gradient_relative_l2"] < best_value:
            best_value = row["validation_gradient_relative_l2"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    split_metrics = {name: evaluate(items, save=True, split=name) for name, items in splits.items()}
    normalization = {"input_mean": input_mean.tolist(), "input_std": input_std.tolist(),
                     "gradient_mean": 0.0, "gradient_std": gradient_std,
                     "heat_flux_mean": 0.0, "heat_flux_std": heat_flux_std,
                     "condition_names": ["k", "alpha"], "condition_mean": condition_mean.tolist(),
                     "condition_std": condition_std.tolist()}
    checkpoint = {"stage": "gkeyll_regime_balanced_dual_history_v1", "protocol": "trajectory_casewise",
                  "source_manifest": str(args.manifest),
                  "regime_labels": str(args.regime_labels), "history_frame_offsets": list(offsets),
                  "history_time_offsets_nominal": [0.005 * value for value in offsets],
                  "history_padding": "repeat_initial" if args.pad_history else "drop_early",
                  "target_kind": "dual_gradient_heat_flux", "maximum_mode": args.maximum_mode,
                  "model_arguments": model_arguments, "normalization": normalization,
                  "warm_start": warm_start, "sampling_cells": sampling_cells, "metrics": split_metrics,
                  "best_validation_gradient_relative_l2": best_value,
                  "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()}}
    summary = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    summary["epochs_completed"] = len(history_rows)
    summary["best_history_gate"] = float(torch.sigmoid(model.history_gate_logit).detach().cpu())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "best.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output_dir / "best.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
