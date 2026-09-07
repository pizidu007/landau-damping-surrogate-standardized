"""Evaluate one frozen closure checkpoint without modifying model selection."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.closure_dataset import load_pair_trajectories
from landau_surrogate.data.paths import nonlinear_runs_root
from landau_surrogate.diagnostics.closure import closure_metrics, macro_average
from landau_surrogate.inference.closure import load_closure_checkpoint, predict_physical_gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=nonlinear_runs_root() / "nonlinear_formal_v1")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hp-blend", type=float, default=0.0)
    parser.add_argument("--baseline-summary", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Closure evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_closure_checkpoint(args.checkpoint, device)
    if not 0.0 <= args.hp_blend <= 1.0:
        parser.error("--hp-blend must be between 0 and 1")
    hp_scale = 0.0
    if args.hp_blend > 0.0:
        if args.baseline_summary is None:
            parser.error("--baseline-summary is required with --hp-blend")
        hp_scale = float(
            json.loads(args.baseline_summary.read_text())["train_fit"]["hp_scale"]
        )
    history = int(checkpoint["history"])
    rows = []
    with torch.no_grad():
        for trajectory in load_pair_trajectories(args.run_dir, [args.split]):
            states = []
            for time_index in range(history - 1, len(trajectory.time)):
                states.append(trajectory.state[time_index - history + 1 : time_index + 1])
            state = torch.from_numpy(np.asarray(states, dtype=np.float32))
            predictions = []
            for start in range(0, len(state), args.batch_size):
                batch = state[start : start + args.batch_size].to(device)
                k_value = torch.full((len(batch),), trajectory.k, device=device)
                predictions.append(
                    predict_physical_gradient(model, checkpoint, batch, k_value).cpu().numpy()
                )
            prediction = np.concatenate(predictions)
            target = trajectory.heat_flux_gradient[history - 1 :]
            maximum_target_mode = checkpoint.get("maximum_target_mode")
            if maximum_target_mode is not None:
                maximum_target_mode = int(maximum_target_mode)
                target_hat = np.fft.rfft(target, axis=-1)
                target_hat[..., maximum_target_mode + 1 :] = 0.0
                target = np.fft.irfft(target_hat, n=target.shape[-1], axis=-1)
                prediction_hat = np.fft.rfft(prediction, axis=-1)
                prediction_hat[..., maximum_target_mode + 1 :] = 0.0
                prediction = np.fft.irfft(
                    prediction_hat, n=prediction.shape[-1], axis=-1
                )
            if args.hp_blend > 0.0:
                state_values = trajectory.state[history - 1 :]
                temperature_perturbation = state_values[:, 2] - state_values[:, 0]
                modes = (
                    np.arange(state_values.shape[-1] // 2 + 1, dtype=np.float64)
                    * trajectory.k
                )
                hp_prediction = hp_scale * np.fft.irfft(
                    modes[None]
                    * np.fft.rfft(temperature_perturbation, axis=-1),
                    n=state_values.shape[-1],
                    axis=-1,
                )
                prediction = (
                    (1.0 - args.hp_blend) * prediction
                    + args.hp_blend * hp_prediction
                )
            metrics = closure_metrics(prediction, target)
            rows.append({
                "pair_id": trajectory.pair_id,
                "split": trajectory.split,
                "k": trajectory.k,
                "alpha": trajectory.alpha,
                **metrics,
            })
            np.savez_compressed(
                args.output_dir / f"{trajectory.pair_id}.npz",
                time=trajectory.time[history - 1 :],
                prediction=prediction,
                target=target,
                density=trajectory.state[history - 1 :, 0],
                velocity=trajectory.state[history - 1 :, 1],
                pressure=trajectory.state[history - 1 :, 2],
                field_energy=trajectory.field_energy[history - 1 :],
            )
    fields = list(rows[0])
    temporary = args.output_dir / "pair_metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output_dir / "pair_metrics.csv")
    names = ["relative_l2", "rmse", "mae", "correlation", "spectral_relative_l2"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "pair_count": len(rows),
        "hp_blend": args.hp_blend,
        "macro": macro_average(rows, names),
    }
    temporary_json = args.output_dir / "summary.json.tmp"
    temporary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary_json, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
