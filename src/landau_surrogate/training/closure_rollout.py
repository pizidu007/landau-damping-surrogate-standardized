"""Short-horizon differentiable fluid-rollout fine tuning for a closure FNO."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.closure_dataset import PairTrajectory, load_pair_trajectories
from landau_surrogate.data.paths import nonlinear_runs_root
from landau_surrogate.fluid.multimoment_1d import fluid_energies, rk4_step
from landau_surrogate.inference.closure import load_closure_checkpoint, predict_physical_gradient


def sample_windows(
    trajectories: list[PairTrajectory],
    count: int,
    length: int,
    history: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    k_values = []
    for _ in range(count):
        trajectory = trajectories[rng.randrange(len(trajectories))]
        start = rng.randrange(history - 1, len(trajectory.time) - length)
        states.append(trajectory.state[start - history + 1 : start + length + 1])
        k_values.append(trajectory.k)
    return np.asarray(states, dtype=np.float32), np.asarray(k_values, dtype=np.float32)


def advance(
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    sequence: torch.Tensor,
    k_value: torch.Tensor,
    interval: float,
    substep_dt: float,
    maximum_mode: int,
) -> torch.Tensor:
    substeps = max(1, int(round(interval / substep_dt)))
    dt = interval / substeps
    current = sequence[:, -1]
    prefix = sequence[:, :-1]
    for _ in range(substeps):
        def factory(_stage: torch.Tensor):
            def closure(stage: torch.Tensor) -> torch.Tensor:
                model_input = torch.cat((prefix, stage[:, None]), dim=1)
                return predict_physical_gradient(model, checkpoint, model_input, k_value)
            return closure
        current = rk4_step(
            current, k_value, dt, factory,
            density_floor=1.0e-4, pressure_floor=1.0e-5, clamp_output=False,
            maximum_mode=maximum_mode,
        )
        current = torch.stack(
            (
                torch.clamp(current[:, 0], min=1.0e-4),
                current[:, 1],
                torch.clamp(current[:, 2], min=1.0e-5),
            ), dim=1,
        )
    return current


def window_loss(
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    windows: torch.Tensor,
    k_value: torch.Tensor,
    substep_dt: float,
    closure_weight: float,
    field_weight: float,
    maximum_mode: int,
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    state_std = torch.as_tensor(
        normalization["input_std"], device=windows.device, dtype=windows.dtype
    ).reshape(1, 3, 1)
    gradient_std = float(normalization["gradient_std"])
    history = int(checkpoint["history"])
    sequence = windows[:, :history]
    current = sequence[:, -1]
    losses = []
    field_losses = []
    interval = 0.1
    rollout_steps = windows.shape[1] - history
    for step in range(1, rollout_steps + 1):
        current = advance(
            model, checkpoint, sequence, k_value, interval, substep_dt, maximum_mode
        )
        target_state = windows[:, history - 1 + step]
        losses.append(torch.mean(((current - target_state) / state_std) ** 2))
        predicted_field = fluid_energies(current, k_value)[0]
        target_field = fluid_energies(target_state, k_value)[0]
        field_floor = torch.clamp(target_field.detach() * 1.0e-8, min=1.0e-14)
        field_losses.append(torch.mean(
            (torch.log(torch.maximum(predicted_field, field_floor)) -
             torch.log(torch.maximum(target_field, field_floor))) ** 2
        ))
        sequence = torch.cat((sequence[:, 1:], current[:, None]), dim=1)
    predicted_gradient = predict_physical_gradient(model, checkpoint, windows[:, :history], k_value)
    # The exact closure label is not present in this window tensor; constrain
    # the update magnitude to remain close to the frozen starting model via a
    # teacher prediction injected by the caller.
    teacher = checkpoint.get("_teacher_gradient")
    if isinstance(teacher, torch.Tensor):
        losses.append(closure_weight * torch.mean(((predicted_gradient - teacher) / gradient_std) ** 2))
    field, _kinetic, total = fluid_energies(current, k_value)
    finite_penalty = torch.nan_to_num(field.mean() + 1.0e-4 * total.mean(), nan=1.0e6, posinf=1.0e6)
    field_loss = torch.stack(field_losses).mean() if field_losses else torch.zeros((), device=windows.device)
    return torch.stack(losses).mean() + field_weight * field_loss + 0.0 * finite_penalty


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    trajectories: list[PairTrajectory],
    device: torch.device,
    window: int,
    substep_dt: float,
    maximum_mode: int,
) -> float:
    values = []
    normalization = checkpoint["normalization"]
    history = int(checkpoint["history"])
    state_std = torch.as_tensor(normalization["input_std"], device=device).reshape(1, 3, 1)
    for trajectory in trajectories:
        starts = np.linspace(history - 1, len(trajectory.time) - window - 1, 4, dtype=int)
        for start in starts:
            truth = torch.from_numpy(
                trajectory.state[start - history + 1 : start + window + 1]
            ).to(device)
            sequence = truth[:history][None]
            current = sequence[:, -1]
            k_value = torch.tensor([trajectory.k], device=device)
            for step in range(1, window + 1):
                current = advance(
                    model, checkpoint, sequence, k_value, 0.1, substep_dt, maximum_mode
                )
                values.append(float(torch.mean(
                    ((current - truth[history - 1 + step : history + step]) / state_std) ** 2
                )))
                sequence = torch.cat((sequence[:, 1:], current[:, None]), dim=1)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=nonlinear_runs_root() / "nonlinear_formal_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--windows-per-epoch", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--substep-dt", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--closure-weight", type=float, default=0.2)
    parser.add_argument("--field-weight", type=float, default=0.0)
    parser.add_argument("--maximum-mode", type=int, default=24)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Rollout fine tuning requires CUDA")
    model, checkpoint = load_closure_checkpoint(args.checkpoint, device)
    history = int(checkpoint["history"])
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    train_trajectories = load_pair_trajectories(args.run_dir, ["train"])
    validation_trajectories = load_pair_trajectories(args.run_dir, ["validation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    rng = random.Random(args.seed)
    history_rows = []
    best_value = validation_loss(
        model, checkpoint, validation_trajectories, device,
        args.window, args.substep_dt, args.maximum_mode,
    )
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        states, k_values = sample_windows(
            train_trajectories, args.windows_per_epoch, args.window, history, rng
        )
        order = np.random.default_rng(args.seed + epoch).permutation(len(states))
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            windows = torch.from_numpy(states[indices]).to(device)
            k_value = torch.from_numpy(k_values[indices]).to(device)
            with torch.no_grad():
                checkpoint["_teacher_gradient"] = predict_physical_gradient(
                    teacher, checkpoint, windows[:, :history], k_value
                )
            optimizer.zero_grad(set_to_none=True)
            loss = window_loss(
                model, checkpoint, windows, k_value, args.substep_dt,
                args.closure_weight, args.field_weight, args.maximum_mode,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite rollout fine-tuning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        checkpoint.pop("_teacher_gradient", None)
        model.eval()
        value = validation_loss(
            model, checkpoint, validation_trajectories, device,
            args.window, args.substep_dt, args.maximum_mode,
        )
        row = {"epoch": epoch, "train_rollout_loss": float(np.mean(losses)), "validation_rollout_loss": value}
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    output = dict(checkpoint)
    output.pop("_teacher_gradient", None)
    output.update({
        "stage": "pic_heat_flux_closure_rollout_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(args.checkpoint),
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "best_validation_rollout_loss": best_value,
        "rollout_finetune_arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "best_rollout.pt.tmp"
    torch.save(output, temporary)
    os.replace(temporary, args.output_dir / "best_rollout.pt")
    temporary_json = args.output_dir / "history.json.tmp"
    temporary_json.write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
    os.replace(temporary_json, args.output_dir / "history.json")
    print(json.dumps({"best_validation_rollout_loss": best_value, "checkpoint": str(args.output_dir / 'best_rollout.pt')}, indent=2))


if __name__ == "__main__":
    main()
