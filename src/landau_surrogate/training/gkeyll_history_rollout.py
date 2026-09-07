"""Truncated-BPTT rollout curriculum for history-conditioned Gkeyll q-FNOs."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.fluid.multimoment_1d import (
    poisson_electric,
    rk4_step_ampere,
    spectral_filter,
)
from landau_surrogate.models.closure_fno1d import ClosureFNO1d, closure_from_physical
from landau_surrogate.training.gkeyll_multicase import load_cases, lowpass_numpy


STAGES = {"huang2025_gkeyll_history_q_v1", "huang2025_gkeyll_history_rollout_v1"}


def interpolate(time: np.ndarray, value: np.ndarray, query: float | np.ndarray) -> np.ndarray:
    points = np.atleast_1d(np.asarray(query, dtype=np.float64))
    upper = np.searchsorted(time, points, side="right")
    upper = np.clip(upper, 1, len(time) - 1)
    lower = upper - 1
    denominator = np.maximum(time[upper] - time[lower], 1.0e-12)
    weight = ((points - time[lower]) / denominator).astype(np.float32)
    result = value[lower] * (1.0 - weight[:, None, None]) + value[upper] * weight[:, None, None]
    return result[0] if np.ndim(query) == 0 else result


def load_model(path: Path, device: torch.device) -> tuple[ClosureFNO1d, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") not in STAGES:
        raise ValueError("Expected a supervised or rollout history-q checkpoint")
    if checkpoint.get("target_kind") != "heat_flux":
        raise ValueError("History rollout requires a heat-flux checkpoint")
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def lag_steps(checkpoint: dict, dt: float) -> tuple[int, ...]:
    times = checkpoint["history_time_offsets_nominal"]
    return tuple(int(np.floor(abs(float(value)) / dt + 0.5)) for value in times)


def history_tensor(
    current: torch.Tensor,
    buffer: list[torch.Tensor],
    lags: tuple[int, ...],
) -> torch.Tensor:
    current_moments = current[:, :3]
    values = []
    for lag in lags:
        if lag == 0:
            values.append(current_moments)
        else:
            index = max(0, len(buffer) - 1 - lag)
            values.append(buffer[index])
    return torch.stack(values, dim=1)


def closure_gradient(
    model: ClosureFNO1d,
    checkpoint: dict,
    current: torch.Tensor,
    buffer: list[torch.Tensor],
    lags: tuple[int, ...],
    k_value: torch.Tensor,
    maximum_mode: int,
) -> torch.Tensor:
    history = history_tensor(current, buffer, lags)
    gradient = closure_from_physical(
        model, history, k_value, checkpoint["normalization"], checkpoint["target_kind"]
    )
    return spectral_filter(gradient[:, None], maximum_mode)[:, 0]


def initial_state_and_buffer(
    trajectory,
    start_time: float,
    dt: float,
    maximum_lag: int,
    maximum_mode: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    times = start_time - np.arange(maximum_lag, -1, -1, dtype=np.float64) * dt
    moments = interpolate(trajectory.time, trajectory.state, times)
    moments = lowpass_numpy(moments, maximum_mode)
    tensor = torch.from_numpy(moments).to(device)
    buffer = [tensor[index : index + 1] for index in range(len(tensor))]
    k_value = torch.tensor([trajectory.k], device=device)
    electric = poisson_electric(buffer[-1][:, 0], k_value)
    state = torch.cat((buffer[-1], electric[:, None]), dim=1)
    return state, buffer


def advance_segment(
    model: ClosureFNO1d,
    checkpoint: dict,
    state: torch.Tensor,
    buffer: list[torch.Tensor],
    lags: tuple[int, ...],
    k_value: torch.Tensor,
    steps: int,
    dt: float,
    maximum_mode: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    maximum_lag = max(lags)
    for _ in range(steps):
        def factory(_stage: torch.Tensor):
            def closure(current: torch.Tensor) -> torch.Tensor:
                return closure_gradient(
                    model, checkpoint, current, buffer, lags, k_value, maximum_mode
                )
            return closure

        state = rk4_step_ampere(
            state,
            k_value,
            dt,
            factory,
            clamp_output=False,
            maximum_mode=maximum_mode,
        )
        buffer.append(state[:, :3])
        if len(buffer) > maximum_lag + 1:
            buffer.pop(0)
    return state, buffer


def segment_loss(
    model: ClosureFNO1d,
    teacher: ClosureFNO1d,
    checkpoint: dict,
    state: torch.Tensor,
    buffer: list[torch.Tensor],
    lags: tuple[int, ...],
    target: torch.Tensor,
    k_value: torch.Tensor,
    maximum_mode: int,
    field_weight: float,
    phase_weight: float,
    teacher_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    normalization = checkpoint["normalization"]
    state_std = torch.as_tensor(
        normalization["input_std"], device=state.device
    ).reshape(1, 3, 1)
    prediction = state[:, :3]
    state_value = torch.mean(((prediction - target) / state_std) ** 2)
    target_electric = poisson_electric(target[:, 0], k_value)
    predicted_electric = state[:, 3]
    predicted_energy = 0.5 * predicted_electric.square().mean(dim=-1)
    target_energy = 0.5 * target_electric.square().mean(dim=-1)
    floor = torch.clamp(target_energy.detach() * 1.0e-8, min=1.0e-14)
    field_value = torch.mean(
        (torch.log(torch.maximum(predicted_energy, floor))
         - torch.log(torch.maximum(target_energy, floor))) ** 2
    )
    predicted_mode = torch.fft.rfft(predicted_electric.float(), dim=-1)[:, 1]
    target_mode = torch.fft.rfft(target_electric.float(), dim=-1)[:, 1]
    phase_value = torch.mean(
        torch.abs(predicted_mode - target_mode).square()
        / torch.clamp(torch.abs(target_mode).square().detach(), min=1.0e-10)
    )
    current_gradient = closure_gradient(
        model, checkpoint, state, buffer, lags, k_value, maximum_mode
    )
    with torch.no_grad():
        teacher_gradient = closure_gradient(
            teacher, checkpoint, state, buffer, lags, k_value, maximum_mode
        )
    teacher_value = torch.mean(
        ((current_gradient - teacher_gradient) / float(normalization["gradient_std"])) ** 2
    )
    total = (
        state_value
        + field_weight * field_value
        + phase_weight * phase_value
        + teacher_weight * teacher_value
    )
    return total, {
        "state": float(state_value.detach()),
        "field": float(field_value.detach()),
        "phase": float(phase_value.detach()),
        "teacher": float(teacher_value.detach()),
    }


@torch.no_grad()
def validation_loss(
    model: ClosureFNO1d,
    checkpoint: dict,
    cases: list[tuple],
    horizon_steps: int,
    dt: float,
    tbptt_steps: int,
    maximum_mode: int,
    device: torch.device,
) -> float:
    model.eval()
    lags = lag_steps(checkpoint, dt)
    values = []
    state_std = torch.as_tensor(
        checkpoint["normalization"]["input_std"], device=device
    ).reshape(1, 3, 1)
    horizon = horizon_steps * dt
    for _spec, trajectory in cases:
        lower = float(trajectory.time[0]) + max(lags) * dt
        upper = float(trajectory.time[-1]) - horizon
        for start_time in np.linspace(lower, upper, 2):
            state, buffer = initial_state_and_buffer(
                trajectory, float(start_time), dt, max(lags), maximum_mode, device
            )
            remaining = horizon_steps
            while remaining:
                steps = min(tbptt_steps, remaining)
                state, buffer = advance_segment(
                    model, checkpoint, state, buffer, lags,
                    torch.tensor([trajectory.k], device=device), steps, dt, maximum_mode,
                )
                remaining -= steps
            target_np = lowpass_numpy(
                interpolate(trajectory.time, trajectory.state, float(start_time) + horizon),
                maximum_mode,
            )
            target = torch.from_numpy(target_np[None]).to(device)
            values.append(float(torch.mean(((state[:, :3] - target) / state_std) ** 2)))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--tbptt-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--windows-per-epoch", type=int, default=4)
    parser.add_argument("--maximum-mode", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--field-weight", type=float, default=0.1)
    parser.add_argument("--phase-weight", type=float, default=0.2)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("History rollout curriculum requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    all_cases = load_cases(args.manifest, args.dataset_root)
    train = [item for item in all_cases if item[0]["split"] == "train"]
    validation = [item for item in all_cases if item[0]["split"] == "validation"]
    lags = lag_steps(checkpoint, args.dt)
    maximum_lag = max(lags)
    horizon = args.horizon_steps * args.dt
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
    )
    rng = random.Random(args.seed)
    best_value = validation_loss(
        model, checkpoint, validation, args.horizon_steps, args.dt,
        args.tbptt_steps, args.maximum_mode, device,
    )
    best_state = copy.deepcopy(model.state_dict())
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, components = [], []
        for _ in range(args.windows_per_epoch):
            _spec, trajectory = train[rng.randrange(len(train))]
            lower = float(trajectory.time[0]) + maximum_lag * args.dt
            upper = float(trajectory.time[-1]) - horizon
            start_time = rng.uniform(lower, upper)
            state, buffer = initial_state_and_buffer(
                trajectory, start_time, args.dt, maximum_lag,
                args.maximum_mode, device,
            )
            k_value = torch.tensor([trajectory.k], device=device)
            completed = 0
            while completed < args.horizon_steps:
                steps = min(args.tbptt_steps, args.horizon_steps - completed)
                state, buffer = advance_segment(
                    model, checkpoint, state, buffer, lags, k_value,
                    steps, args.dt, args.maximum_mode,
                )
                completed += steps
                target_np = lowpass_numpy(
                    interpolate(
                        trajectory.time, trajectory.state,
                        start_time + completed * args.dt,
                    ),
                    args.maximum_mode,
                )
                target = torch.from_numpy(target_np[None]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, detail = segment_loss(
                    model, teacher, checkpoint, state, buffer, lags, target,
                    k_value, args.maximum_mode, args.field_weight,
                    args.phase_weight, args.teacher_weight,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite history rollout loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                components.append(detail)
                state = state.detach()
                buffer = [value.detach() for value in buffer]
        value = validation_loss(
            model, checkpoint, validation, args.horizon_steps, args.dt,
            args.tbptt_steps, args.maximum_mode, device,
        )
        row = {
            "epoch": epoch,
            "horizon_time": horizon,
            "train_loss": float(np.mean(losses)),
            "validation_state_loss": value,
            "mean_components": {
                key: float(np.mean([item[key] for item in components]))
                for key in ("state", "field", "phase", "teacher")
            },
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    curriculum = list(checkpoint.get("curriculum_history", []))
    curriculum.append(
        {
            "parent_checkpoint": str(args.checkpoint),
            "horizon_steps": args.horizon_steps,
            "horizon_time": horizon,
            "tbptt_steps": args.tbptt_steps,
            "best_validation_state_loss": best_value,
        }
    )
    output = dict(checkpoint)
    output.update(
        {
            "stage": "huang2025_gkeyll_history_rollout_v1",
            "parent_checkpoint": str(args.checkpoint),
            "model_state_dict": {
                key: value.cpu() for key, value in model.state_dict().items()
            },
            "curriculum_history": curriculum,
            "best_validation_rollout_loss": best_value,
        }
    )
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
