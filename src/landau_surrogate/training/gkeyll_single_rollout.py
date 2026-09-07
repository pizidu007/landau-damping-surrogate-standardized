"""Rollout-aware fine tuning for the strict Huang single-case closure."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.huang2025 import load_huang_mat
from landau_surrogate.fluid.multimoment_1d import (
    poisson_electric,
    rk4_step_ampere,
    spectral_filter,
)
from landau_surrogate.models.closure_fno1d import ClosureFNO1d
from landau_surrogate.training.gkeyll_multicase import lowpass_numpy


ALLOWED_STAGES = {
    "huang2025_gkeyll_reproduction",
    "huang2025_gkeyll_single_rollout_v1",
}


def interpolate(time: np.ndarray, value: np.ndarray, query: float) -> np.ndarray:
    upper = int(np.clip(np.searchsorted(time, query, side="right"), 1, len(time) - 1))
    lower = upper - 1
    weight = np.float32((query - time[lower]) / max(time[upper] - time[lower], 1.0e-12))
    return value[lower] * (1.0 - weight) + value[upper] * weight


def load_model(path: Path, device: torch.device) -> tuple[ClosureFNO1d, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") not in ALLOWED_STAGES:
        raise ValueError("Expected a strict single-case supervised/rollout checkpoint")
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def initial_state(trajectory, start_time: float, maximum_mode: int, device: torch.device):
    moments = lowpass_numpy(
        interpolate(trajectory.time, trajectory.state, start_time)[None], maximum_mode
    )
    moments_tensor = torch.from_numpy(moments).to(device)
    k_value = torch.tensor([trajectory.k], device=device)
    electric = poisson_electric(moments_tensor[:, 0], k_value)
    return torch.cat((moments_tensor, electric[:, None]), dim=1), k_value


def closure_gradient(
    model: ClosureFNO1d,
    checkpoint: dict,
    state: torch.Tensor,
    maximum_mode: int,
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    mean = torch.as_tensor(
        normalization["input_mean"], device=state.device
    ).reshape(1, 3, 1)
    std = torch.as_tensor(
        normalization["input_std"], device=state.device
    ).reshape(1, 3, 1)
    normalized = (state[:, :3] - mean) / std
    prediction = model(normalized, torch.zeros(len(state), device=state.device))
    prediction = prediction * float(normalization["target_std"]) + float(
        normalization["target_mean"]
    )
    return spectral_filter(prediction[:, None], maximum_mode)[:, 0]


def advance(
    model: ClosureFNO1d,
    checkpoint: dict,
    state: torch.Tensor,
    k_value: torch.Tensor,
    steps: int,
    dt: float,
    maximum_mode: int,
) -> torch.Tensor:
    for _ in range(steps):
        def factory(_stage: torch.Tensor):
            def closure(current: torch.Tensor) -> torch.Tensor:
                return closure_gradient(model, checkpoint, current, maximum_mode)
            return closure

        state = rk4_step_ampere(
            state, k_value, dt, factory, clamp_output=False,
            maximum_mode=maximum_mode,
        )
    return state


def endpoint_loss(
    model: ClosureFNO1d,
    teacher: ClosureFNO1d,
    checkpoint: dict,
    state: torch.Tensor,
    target: torch.Tensor,
    k_value: torch.Tensor,
    maximum_mode: int,
    field_weight: float,
    phase_weight: float,
    teacher_weight: float,
    spectral_weight: float,
    high_mode_cutoff: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    normalization = checkpoint["normalization"]
    state_std = torch.as_tensor(
        normalization["input_std"], device=state.device
    ).reshape(1, 3, 1)
    state_value = torch.mean(((state[:, :3] - target) / state_std) ** 2)
    target_electric = poisson_electric(target[:, 0], k_value)
    predicted_energy = 0.5 * state[:, 3].square().mean(dim=-1)
    target_energy = 0.5 * target_electric.square().mean(dim=-1)
    floor = torch.clamp(target_energy.detach() * 1.0e-8, min=1.0e-14)
    field_value = torch.mean(
        (torch.log(torch.maximum(predicted_energy, floor))
         - torch.log(torch.maximum(target_energy, floor))) ** 2
    )
    predicted_mode = torch.fft.rfft(state[:, 3].float(), dim=-1)[:, 1]
    target_mode = torch.fft.rfft(target_electric.float(), dim=-1)[:, 1]
    phase_value = torch.mean(
        torch.abs(predicted_mode - target_mode).square()
        / torch.clamp(torch.abs(target_mode).square().detach(), min=1.0e-10)
    )
    current_gradient = closure_gradient(model, checkpoint, state, maximum_mode)
    with torch.no_grad():
        teacher_gradient = closure_gradient(teacher, checkpoint, state, maximum_mode)
    teacher_value = torch.mean(
        ((current_gradient - teacher_gradient) / float(normalization["target_std"])) ** 2
    )
    transformed = torch.fft.rfft(current_gradient.float(), dim=-1)
    if high_mode_cutoff + 1 < transformed.shape[-1]:
        high = transformed[:, high_mode_cutoff + 1 :]
        spectral_value = torch.mean(torch.abs(high).square()) / torch.clamp(
            torch.mean(torch.abs(transformed[:, 1:]).square()).detach(), min=1.0e-10
        )
    else:
        spectral_value = torch.zeros((), device=state.device)
    total = (
        state_value + field_weight * field_value + phase_weight * phase_value
        + teacher_weight * teacher_value + spectral_weight * spectral_value
    )
    return total, {
        "state": float(state_value.detach()),
        "field": float(field_value.detach()),
        "phase": float(phase_value.detach()),
        "teacher": float(teacher_value.detach()),
        "spectral": float(spectral_value.detach()),
    }


@torch.no_grad()
def validation_loss(
    model: ClosureFNO1d,
    checkpoint: dict,
    trajectory,
    horizon_steps: int,
    dt: float,
    maximum_mode: int,
    device: torch.device,
) -> float:
    model.eval()
    horizon = horizon_steps * dt
    upper = float(trajectory.time[-1]) - horizon
    starts = np.linspace(float(trajectory.time[0]), upper, 4)
    state_std = torch.as_tensor(
        checkpoint["normalization"]["input_std"], device=device
    ).reshape(1, 3, 1)
    values = []
    for start_time in starts:
        state, k_value = initial_state(trajectory, float(start_time), maximum_mode, device)
        state = advance(
            model, checkpoint, state, k_value, horizon_steps, dt, maximum_mode
        )
        target_np = lowpass_numpy(
            interpolate(trajectory.time, trajectory.state, float(start_time) + horizon)[None],
            maximum_mode,
        )
        target = torch.from_numpy(target_np).to(device)
        values.append(float(torch.mean(((state[:, :3] - target) / state_std) ** 2)))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--tbptt-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--windows-per-epoch", type=int, default=4)
    parser.add_argument("--maximum-mode", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--field-weight", type=float, default=0.05)
    parser.add_argument("--phase-weight", type=float, default=0.05)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--high-mode-cutoff", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Strict single-case rollout training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trajectory = load_huang_mat(args.trajectory)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
    )
    rng = random.Random(args.seed)
    horizon = args.horizon_steps * args.dt
    best_value = validation_loss(
        model, checkpoint, trajectory, args.horizon_steps, args.dt,
        args.maximum_mode, device,
    )
    best_state = copy.deepcopy(model.state_dict())
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, components = [], []
        for _ in range(args.windows_per_epoch):
            start_time = rng.uniform(
                float(trajectory.time[0]), float(trajectory.time[-1]) - horizon
            )
            state, k_value = initial_state(
                trajectory, start_time, args.maximum_mode, device
            )
            completed = 0
            while completed < args.horizon_steps:
                steps = min(args.tbptt_steps, args.horizon_steps - completed)
                state = advance(
                    model, checkpoint, state, k_value, steps, args.dt,
                    args.maximum_mode,
                )
                completed += steps
                target_np = lowpass_numpy(
                    interpolate(
                        trajectory.time, trajectory.state,
                        start_time + completed * args.dt,
                    )[None],
                    args.maximum_mode,
                )
                target = torch.from_numpy(target_np).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, detail = endpoint_loss(
                    model, teacher, checkpoint, state, target, k_value,
                    args.maximum_mode, args.field_weight, args.phase_weight,
                    args.teacher_weight, args.spectral_weight,
                    args.high_mode_cutoff,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite single-case rollout loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                components.append(detail)
                state = state.detach()
        value = validation_loss(
            model, checkpoint, trajectory, args.horizon_steps, args.dt,
            args.maximum_mode, device,
        )
        row = {
            "epoch": epoch,
            "horizon_steps": args.horizon_steps,
            "horizon_time": horizon,
            "train_loss": float(np.mean(losses)),
            "validation_state_loss": value,
            "mean_components": {
                key: float(np.mean([item[key] for item in components]))
                for key in ("state", "field", "phase", "teacher", "spectral")
            },
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    curriculum = list(checkpoint.get("single_rollout_curriculum", []))
    curriculum.append({
        "parent_checkpoint": str(args.checkpoint),
        "horizon_steps": args.horizon_steps,
        "horizon_time": horizon,
        "tbptt_steps": args.tbptt_steps,
        "best_validation_state_loss": best_value,
        "spectral_weight": args.spectral_weight,
    })
    output = dict(checkpoint)
    output.update({
        "stage": "huang2025_gkeyll_single_rollout_v1",
        "parent_checkpoint": str(args.checkpoint),
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "single_rollout_curriculum": curriculum,
        "best_validation_rollout_loss": best_value,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "best_rollout.pt.tmp"
    torch.save(output, temporary)
    os.replace(temporary, args.output_dir / "best_rollout.pt")
    (args.output_dir / "history.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_validation_rollout_loss": best_value}, indent=2))


if __name__ == "__main__":
    main()
