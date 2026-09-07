"""Train a paper-style, memoryless FNO for direct raw-M3-gradient closure."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from landau_surrogate.data.raw_moment_pic import interleaved_indices, load_raw_moment_pic_pair
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(target), 1.0e-12)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "correlation": float(np.corrcoef(prediction.ravel(), target.ravel())[0, 1]),
    }


@torch.no_grad()
def predict(model: torch.nn.Module, state: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(state)), batch_size=batch_size):
        values.append(model(batch.to(device), torch.zeros(len(batch), device=device)).cpu().numpy())
    return np.concatenate(values)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default="k0p350_a0p100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--maximum-target-mode", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Raw-moment FNO training requires CUDA")
    trajectory = load_raw_moment_pic_pair(
        args.case_dir, args.pair_id, maximum_target_mode=args.maximum_target_mode
    )
    train_index, validation_index, test_index = interleaved_indices(len(trajectory.time))
    input_mean = trajectory.public_state[train_index].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    input_std = trajectory.public_state[train_index].std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    target_mean = float(trajectory.third_moment_gradient[train_index].mean(dtype=np.float64))
    target_std = float(trajectory.third_moment_gradient[train_index].std(dtype=np.float64))
    state = ((trajectory.public_state - input_mean[None, :, None]) / input_std[None, :, None]).astype(np.float32)
    target = ((trajectory.third_moment_gradient - target_mean) / target_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(state[train_index]), torch.from_numpy(target[train_index])),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    model = ClosureFNO1d(3, False, args.width, args.modes, args.layers, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-5)
    best_value = float("inf"); best_state = copy.deepcopy(model.state_dict()); stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch_state, batch_target in loader:
            batch_state = batch_state.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_state, torch.zeros(len(batch_state), device=device))
            pointwise = torch.mean((output - batch_target) ** 2)
            output_hat = torch.fft.rfft(output.float(), dim=-1)
            target_hat = torch.fft.rfft(batch_target.float(), dim=-1)
            spectral = torch.mean(torch.abs(output_hat - target_hat) ** 2) / torch.clamp(
                torch.mean(torch.abs(target_hat) ** 2), min=1.0e-8
            )
            loss = pointwise + 0.1 * spectral
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        normalized = predict(model, state[validation_index], args.batch_size, device)
        validation = metrics(normalized * target_std + target_mean, trajectory.third_moment_gradient[validation_index])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation_relative_l2": validation["relative_l2"],
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(row); print(json.dumps(row), flush=True)
        if validation["relative_l2"] < best_value:
            best_value = validation["relative_l2"]; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    prediction = predict(model, state, args.batch_size, device) * target_std + target_mean
    summary = {
        "stage": "pic_raw_moment_single_trajectory", "pair_id": trajectory.pair_id,
        "k": trajectory.k, "alpha": trajectory.alpha, "replica_count": trajectory.replica_count,
        "protocol": "interleaved_same_trajectory_8_1_1",
        "split_counts": {"train": len(train_index), "validation": len(validation_index), "test": len(test_index)},
        "maximum_target_mode": args.maximum_target_mode,
        "normalization": {"input_mean": input_mean.tolist(), "input_std": input_std.tolist(),
                          "target_mean": target_mean, "target_std": target_std},
        "best_validation_relative_l2": best_value,
        "train": metrics(prediction[train_index], trajectory.third_moment_gradient[train_index]),
        "validation": metrics(prediction[validation_index], trajectory.third_moment_gradient[validation_index]),
        "test": metrics(prediction[test_index], trajectory.third_moment_gradient[test_index]),
        "epochs_completed": len(history), "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        **summary, "model_arguments": {"state_channels": 3, "include_k": False,
            "width": args.width, "modes": args.modes, "layers": args.layers,
            "enforce_zero_mean": True},
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "source_paths": trajectory.source_paths,
    }, args.output_dir / "best.pt")
    np.savez_compressed(
        args.output_dir / "evaluation.npz", time=trajectory.time, x_over_l=trajectory.x_over_l,
        target=trajectory.third_moment_gradient, prediction=prediction,
        train_index=train_index, validation_index=validation_index, test_index=test_index,
    )
    atomic_json(args.output_dir / "summary.json", summary); atomic_json(args.output_dir / "history.json", history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    extent = (float(trajectory.time[0]), float(trajectory.time[-1]), 0.0, 1.0)
    scale = float(np.quantile(np.abs(trajectory.third_moment_gradient), 0.995))
    for axis, value, title in zip(axes[:2], (trajectory.third_moment_gradient, prediction), ("PIC raw-M3 gradient", "memoryless FNO")):
        image = axis.imshow(value.T, origin="lower", aspect="auto", extent=extent,
                            cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set_title(title); axis.set_xlabel("time")
    error = np.abs(prediction - trajectory.third_moment_gradient)
    error_image = axes[2].imshow(error.T, origin="lower", aspect="auto", extent=extent,
                                 cmap="magma", vmin=0.0, vmax=float(np.quantile(error, 0.995)))
    axes[2].set_title("absolute error"); axes[2].set_xlabel("time"); axes[0].set_ylabel("x/L")
    fig.colorbar(image, ax=axes[:2], shrink=0.8, label=r"$\partial_x M_3$")
    fig.colorbar(error_image, ax=axes[2], shrink=0.8, label="absolute error")
    fig.subplots_adjust(wspace=0.18, right=0.96)
    fig.savefig(args.output_dir / "closure_xt.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
