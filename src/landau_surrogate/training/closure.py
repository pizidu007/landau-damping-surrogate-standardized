"""Train a CUDA 1D FNO closure from PIC low-order moments."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from landau_surrogate.data.closure_dataset import (
    ClosureSnapshotDataset,
    ClosureStats,
    compute_closure_stats,
    load_pair_trajectories,
)
from landau_surrogate.data.paths import nonlinear_runs_root
from landau_surrogate.diagnostics.closure import closure_metrics, macro_average
from landau_surrogate.models.closure_fno1d import ClosureFNO1d, spectral_derivative


VARIANTS = {
    "nup": {"include_k": False, "history": 1, "target_kind": "gradient"},
    "nupk": {"include_k": True, "history": 1, "target_kind": "gradient"},
    "nupk_q": {"include_k": True, "history": 1, "target_kind": "heat_flux"},
    "nupk_history": {"include_k": True, "history": 3, "target_kind": "gradient"},
    "nupk_q_history3": {"include_k": True, "history": 3, "target_kind": "heat_flux"},
    "nupk_q_history5": {"include_k": True, "history": 5, "target_kind": "heat_flux"},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def predict_gradient_normalized(
    model: ClosureFNO1d,
    batch: dict[str, torch.Tensor],
    stats: ClosureStats,
    target_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(batch["state"], batch["k"])
    if target_kind == "gradient":
        physical = output * stats.gradient_std + stats.gradient_mean
        physical = physical - physical.mean(dim=-1, keepdim=True)
        return output, physical
    heat_flux = output * stats.heat_flux_std + stats.heat_flux_mean
    gradient = spectral_derivative(heat_flux, batch["k_physical"])
    gradient_normalized = (gradient - stats.gradient_mean) / stats.gradient_std
    return output, gradient_normalized * stats.gradient_std + stats.gradient_mean


def training_loss(
    model: ClosureFNO1d,
    batch: dict[str, torch.Tensor],
    stats: ClosureStats,
    target_kind: str,
    spectral_weight: float,
    gradient_weight: float,
) -> torch.Tensor:
    output, gradient_physical = predict_gradient_normalized(model, batch, stats, target_kind)
    pointwise = torch.mean((output - batch["target"]) ** 2)
    output_spectrum = torch.fft.rfft(output.float(), dim=-1)
    target_spectrum = torch.fft.rfft(batch["target"].float(), dim=-1)
    spectral = torch.mean(torch.abs(output_spectrum - target_spectrum) ** 2) / torch.clamp(
        torch.mean(torch.abs(target_spectrum) ** 2), min=1.0e-8
    )
    gradient = torch.mean(
        ((gradient_physical - batch["gradient_physical"]) / stats.gradient_std) ** 2
    )
    return pointwise + spectral_weight * spectral + gradient_weight * gradient


@torch.no_grad()
def evaluate(
    model: ClosureFNO1d,
    loader: DataLoader,
    dataset: ClosureSnapshotDataset,
    stats: ClosureStats,
    target_kind: str,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    predictions: dict[int, list[tuple[int, np.ndarray]]] = {}
    for batch in loader:
        moved = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        _raw, gradient = predict_gradient_normalized(model, moved, stats, target_kind)
        for i in range(len(gradient)):
            trajectory_index = int(batch["trajectory_index"][i])
            time_index = int(batch["time_index"][i])
            predictions.setdefault(trajectory_index, []).append(
                (time_index, gradient[i].float().cpu().numpy())
            )
    rows: list[dict[str, Any]] = []
    for trajectory_index, items in sorted(predictions.items()):
        trajectory = dataset.trajectories[trajectory_index]
        ordered = sorted(items)
        indices = np.asarray([item[0] for item in ordered], dtype=np.int64)
        prediction = np.stack([item[1] for item in ordered])
        target = trajectory.heat_flux_gradient[indices]
        if dataset.maximum_target_mode is not None:
            transformed = np.fft.rfft(target, axis=-1)
            transformed[..., int(dataset.maximum_target_mode) + 1 :] = 0.0
            target = np.fft.irfft(transformed, n=target.shape[-1], axis=-1)
        rows.append({
            "pair_id": trajectory.pair_id,
            "split": trajectory.split,
            "k": trajectory.k,
            "alpha": trajectory.alpha,
            **closure_metrics(prediction, target),
        })
    names = ["relative_l2", "rmse", "mae", "correlation", "spectral_relative_l2"]
    return macro_average(rows, names), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=nonlinear_runs_root() / "nonlinear_formal_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--spectral-weight", type=float, default=0.1)
    parser.add_argument("--gradient-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--maximum-target-mode", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Closure production training requires CUDA")
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    definition = VARIANTS[args.variant]
    train_trajectories = load_pair_trajectories(args.run_dir, ["train"])
    validation_trajectories = load_pair_trajectories(args.run_dir, ["validation"])
    stats = compute_closure_stats(train_trajectories)
    train_dataset = ClosureSnapshotDataset(
        train_trajectories, stats, definition["target_kind"], definition["history"],
        args.time_stride, args.maximum_target_mode,
    )
    validation_dataset = ClosureSnapshotDataset(
        validation_trajectories, stats, definition["target_kind"], definition["history"],
        args.time_stride, args.maximum_target_mode,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model_config = {
        "state_channels": 3 * int(definition["history"]),
        "include_k": bool(definition["include_k"]),
        "width": args.width,
        "modes": args.modes,
        "layers": args.layers,
        "enforce_zero_mean": True,
    }
    model = ClosureFNO1d(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05)
    use_amp = torch.cuda.is_bf16_supported()
    history_rows: list[dict[str, float]] = []
    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            moved = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = training_loss(
                    model, moved, stats, str(definition["target_kind"]), args.spectral_weight, args.gradient_weight
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite closure loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_metrics, _rows = evaluate(
            model, validation_loader, validation_dataset, stats, str(definition["target_kind"]), device
        )
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "validation_relative_l2": validation_metrics["relative_l2"],
            "validation_spectral_relative_l2": validation_metrics["spectral_relative_l2"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["relative_l2"] < best_metric:
            best_metric = validation_metrics["relative_l2"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    validation_metrics, validation_rows = evaluate(
        model, validation_loader, validation_dataset, stats, str(definition["target_kind"]), device
    )
    checkpoint = {
        "stage": "pic_heat_flux_closure_supervised_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "target_kind": definition["target_kind"],
        "history": definition["history"],
        "maximum_target_mode": args.maximum_target_mode,
        "model_config": model_config,
        "normalization": stats.as_dict(),
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "best_validation_relative_l2": best_metric,
        "train_pair_ids": [item.pair_id for item in train_trajectories],
        "validation_pair_ids": [item.pair_id for item in validation_trajectories],
        "run_dir": str(args.run_dir),
        "arguments": vars(args),
    }
    checkpoint["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in checkpoint["arguments"].items()}
    temporary = args.output_dir / "best.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output_dir / "best.pt")
    atomic_json(args.output_dir / "history.json", history_rows)
    atomic_json(args.output_dir / "validation_summary.json", {"metrics": validation_metrics, "pairs": validation_rows})

    plt.figure(figsize=(7, 4.5))
    plt.semilogy([row["epoch"] for row in history_rows], [row["train_loss"] for row in history_rows], label="train loss")
    plt.semilogy([row["epoch"] for row in history_rows], [row["validation_relative_l2"] for row in history_rows], label="validation pair-macro relative L2")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_curve.png", dpi=180)
    plt.close()
    print(json.dumps({"best_validation_relative_l2": best_metric, "checkpoint": str(args.output_dir / 'best.pt')}, indent=2))


if __name__ == "__main__":
    main()
