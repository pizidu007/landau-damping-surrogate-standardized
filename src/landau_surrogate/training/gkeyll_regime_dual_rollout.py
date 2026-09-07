"""Regime-balanced truncated-BPTT rollout tuning for the dual-head Gkeyll FNO."""
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
    rk4_step,
    rk4_step_ampere,
    spectral_filter,
)
from landau_surrogate.models.closure_fno1d import DualHeadHistoryResidualFNO1d
from landau_surrogate.training.gkeyll_history_rollout import interpolate
from landau_surrogate.training.gkeyll_multicase import load_cases, lowpass_numpy


STAGES = {"gkeyll_regime_balanced_dual_history_v1", "gkeyll_regime_balanced_dual_rollout_v1"}


def load_model(path: Path, device: torch.device) -> tuple[DualHeadHistoryResidualFNO1d, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") not in STAGES:
        raise ValueError("Expected a regime-balanced dual-head checkpoint")
    model = DualHeadHistoryResidualFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def lag_steps(checkpoint: dict, dt: float) -> tuple[int, ...]:
    times = checkpoint.get("history_time_offsets_nominal")
    if times is None:
        times = [0.005 * value for value in checkpoint["history_frame_offsets"]]
    return tuple(int(np.floor(abs(float(value)) / dt + 0.5)) for value in times)


def history_tensor(current: torch.Tensor, buffer: list[torch.Tensor], lags: tuple[int, ...]) -> torch.Tensor:
    values = []
    for lag in lags:
        values.append(current[:, :3] if lag == 0 else buffer[max(0, len(buffer) - 1 - lag)])
    return torch.stack(values, dim=1)


def closure_gradient(
    model: DualHeadHistoryResidualFNO1d,
    checkpoint: dict,
    current: torch.Tensor,
    buffer: list[torch.Tensor],
    lags: tuple[int, ...],
    k_value: torch.Tensor,
    alpha_value: torch.Tensor,
    maximum_mode: int,
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    history = history_tensor(current, buffer, lags)
    mean = torch.as_tensor(normalization["input_mean"], device=current.device).reshape(1, 1, 3, 1)
    std = torch.as_tensor(normalization["input_std"], device=current.device).reshape(1, 1, 3, 1)
    condition_mean = torch.as_tensor(normalization["condition_mean"], device=current.device)
    condition_std = torch.as_tensor(normalization["condition_std"], device=current.device)
    condition = (torch.stack((k_value, alpha_value), dim=1) - condition_mean) / condition_std
    gradient, _flux = model((history - mean) / std, condition)
    gradient = gradient * float(normalization["gradient_std"])
    return spectral_filter(gradient[:, None], maximum_mode)[:, 0]


def closure_gradient_from_history(
    model: DualHeadHistoryResidualFNO1d,
    checkpoint: dict,
    history: torch.Tensor,
    k_value: torch.Tensor,
    alpha_value: torch.Tensor,
    maximum_mode: int,
) -> torch.Tensor:
    """Evaluate the learned closure on an explicitly supplied physical history."""
    normalization = checkpoint["normalization"]
    mean = torch.as_tensor(normalization["input_mean"], device=history.device).reshape(1, 1, 3, 1)
    std = torch.as_tensor(normalization["input_std"], device=history.device).reshape(1, 1, 3, 1)
    condition_mean = torch.as_tensor(normalization["condition_mean"], device=history.device)
    condition_std = torch.as_tensor(normalization["condition_std"], device=history.device)
    condition = (torch.stack((k_value, alpha_value), dim=1) - condition_mean) / condition_std
    gradient, _flux = model((history - mean) / std, condition)
    gradient = gradient * float(normalization["gradient_std"])
    return spectral_filter(gradient[:, None], maximum_mode)[:, 0]


def initial_state_and_buffer(trajectory, start_time: float, dt: float, maximum_lag: int,
                             maximum_mode: int, device: torch.device,
                             field_solver: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    times = start_time - np.arange(maximum_lag, -1, -1, dtype=np.float64) * dt
    moments = lowpass_numpy(interpolate(trajectory.time, trajectory.state, times), maximum_mode)
    tensor = torch.from_numpy(moments).to(device)
    buffer = [tensor[index:index + 1] for index in range(len(tensor))]
    k_value = torch.tensor([trajectory.k], device=device)
    if field_solver == "ampere":
        electric = poisson_electric(buffer[-1][:, 0], k_value)
        state = torch.cat((buffer[-1], electric[:, None]), dim=1)
    else:
        state = buffer[-1]
    return state, buffer


def advance_segment(model, checkpoint, state, buffer, lags, k_value, alpha_value,
                    steps: int, dt: float, maximum_mode: int, field_solver: str):
    maximum_lag = max(lags)
    for _ in range(steps):
        def factory(_stage):
            def closure(current):
                return closure_gradient(model, checkpoint, current, buffer, lags,
                                        k_value, alpha_value, maximum_mode)
            return closure
        stepper = rk4_step_ampere if field_solver == "ampere" else rk4_step
        state = stepper(state, k_value, dt, factory, clamp_output=False,
                        maximum_mode=maximum_mode)
        buffer.append(state[:, :3])
        if len(buffer) > maximum_lag + 1:
            buffer.pop(0)
    return state, buffer


def balanced_cells(cases: list[tuple], labels: dict[str, np.ndarray], minimum_time: float,
                   horizon: float) -> list[tuple]:
    cells = []
    for spec, trajectory in cases:
        regime = labels[spec["case_id"]]
        allowed_time = (trajectory.time >= minimum_time + horizon) & (trajectory.time <= trajectory.time[-1])
        for value in np.unique(regime):
            indices = np.flatnonzero((regime == value) & allowed_time)
            if len(indices):
                cells.append((spec, trajectory, int(value), indices))
    return cells


@torch.no_grad()
def validation_loss(model, checkpoint, cases, labels, horizon_steps, dt, tbptt_steps,
                    maximum_mode, device, field_solver, field_weight,
                    complex_field_weight, maximum_cells) -> float:
    model.eval()
    lags = lag_steps(checkpoint, dt)
    horizon = horizon_steps * dt
    cells = balanced_cells(cases, labels, max(lags) * dt, horizon)
    if maximum_cells > 0 and len(cells) > maximum_cells:
        selected = np.linspace(0, len(cells) - 1, maximum_cells, dtype=int)
        cells = [cells[int(index)] for index in selected]
    state_std = torch.as_tensor(checkpoint["normalization"]["input_std"], device=device).reshape(1, 3, 1)
    values = []
    for _spec, trajectory, _regime, indices in cells:
        target_index = int(indices[len(indices) // 2])
        start_time = float(trajectory.time[target_index] - horizon)
        state, buffer = initial_state_and_buffer(
            trajectory, start_time, dt, max(lags), maximum_mode, device, field_solver
        )
        k_value = torch.tensor([trajectory.k], device=device)
        alpha_value = torch.tensor([trajectory.alpha], device=device)
        completed = 0
        while completed < horizon_steps:
            steps = min(tbptt_steps, horizon_steps - completed)
            state, buffer = advance_segment(model, checkpoint, state, buffer, lags, k_value,
                                            alpha_value, steps, dt, maximum_mode, field_solver)
            completed += steps
        target_np = lowpass_numpy(interpolate(trajectory.time, trajectory.state,
                                              start_time + horizon), maximum_mode)
        target = torch.from_numpy(target_np[None]).to(device)
        state_loss = torch.mean(((state[:, :3] - target) / state_std) ** 2)
        predicted_electric = (
            state[:, 3]
            if field_solver == "ampere"
            else poisson_electric(state[:, 0], k_value)
        )
        target_electric = poisson_electric(target[:, 0], k_value)
        predicted_energy = 0.5 * predicted_electric.square().mean(dim=-1)
        target_energy = 0.5 * target_electric.square().mean(dim=-1)
        floor = torch.clamp(target_energy.detach() * 1.0e-8, min=1.0e-14)
        field_loss = torch.mean(
            (torch.log(torch.maximum(predicted_energy, floor))
             - torch.log(torch.maximum(target_energy, floor))) ** 2
        )
        nx = predicted_electric.shape[-1]
        predicted_mode = torch.fft.rfft(predicted_electric.float(), dim=-1)[:, 1] / nx
        target_mode = torch.fft.rfft(target_electric.float(), dim=-1)[:, 1] / nx
        initial_mode_scale = max(float(trajectory.alpha / (2.0 * trajectory.k)), 1.0e-6)
        complex_field_loss = torch.mean(
            torch.abs((predicted_mode - target_mode) / initial_mode_scale) ** 2
        )
        values.append(float(
            state_loss + field_weight * field_loss
            + complex_field_weight * complex_field_loss
        ))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--regime-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--tbptt-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--windows-per-epoch", type=int, default=8)
    parser.add_argument("--maximum-mode", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--field-weight", type=float, default=0.1)
    parser.add_argument("--complex-field-weight", type=float, default=0.0)
    parser.add_argument("--teacher-weight", type=float, default=0.2)
    parser.add_argument("--truth-closure-weight", type=float, default=0.0)
    parser.add_argument("--field-solver", choices=("poisson", "ampere"), default="poisson")
    parser.add_argument(
        "--validation-max-cells", type=int, default=0,
        help="Evenly subsample validation case-regime cells; zero uses all cells.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Regime-balanced rollout tuning requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    all_cases = load_cases(args.manifest, args.dataset_root)
    train = [item for item in all_cases if item[0]["split"] == "train"]
    validation = [item for item in all_cases if item[0]["split"] == "validation"]
    label_file = np.load(args.regime_labels)
    labels = {key: np.asarray(label_file[key]) for key in label_file.files}
    label_file.close()
    lags = lag_steps(checkpoint, args.dt)
    horizon = args.horizon_steps * args.dt
    cells = balanced_cells(train, labels, max(lags) * args.dt, horizon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    rng = random.Random(args.seed)
    best_value = validation_loss(model, checkpoint, validation, labels, args.horizon_steps,
                                 args.dt, args.tbptt_steps, args.maximum_mode, device,
                                 args.field_solver, args.field_weight,
                                 args.complex_field_weight, args.validation_max_cells)
    best_state = copy.deepcopy(model.state_dict())
    state_std = torch.as_tensor(checkpoint["normalization"]["input_std"], device=device).reshape(1, 3, 1)
    gradient_std = float(checkpoint["normalization"]["gradient_std"])
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        order = list(range(len(cells))); rng.shuffle(order)
        for window in range(args.windows_per_epoch):
            _spec, trajectory, regime, indices = cells[order[window % len(order)]]
            target_index = int(indices[rng.randrange(len(indices))])
            start_time = float(trajectory.time[target_index] - horizon)
            state, buffer = initial_state_and_buffer(
                trajectory, start_time, args.dt, max(lags), args.maximum_mode,
                device, args.field_solver,
            )
            k_value = torch.tensor([trajectory.k], device=device)
            alpha_value = torch.tensor([trajectory.alpha], device=device)
            completed = 0
            while completed < args.horizon_steps:
                steps = min(args.tbptt_steps, args.horizon_steps - completed)
                state, buffer = advance_segment(model, checkpoint, state, buffer, lags, k_value,
                                                alpha_value, steps, args.dt, args.maximum_mode,
                                                args.field_solver)
                completed += steps
                target_time = start_time + completed * args.dt
                target_np = lowpass_numpy(interpolate(trajectory.time, trajectory.state,
                                                       target_time), args.maximum_mode)
                target = torch.from_numpy(target_np[None]).to(device)
                target_electric = poisson_electric(target[:, 0], k_value)
                predicted_electric = (
                    state[:, 3]
                    if args.field_solver == "ampere"
                    else poisson_electric(state[:, 0], k_value)
                )
                predicted_energy = 0.5 * predicted_electric.square().mean(dim=-1)
                target_energy = 0.5 * target_electric.square().mean(dim=-1)
                floor = torch.clamp(target_energy.detach() * 1.0e-8, min=1.0e-14)
                state_loss = torch.mean(((state[:, :3] - target) / state_std) ** 2)
                field_loss = torch.mean((torch.log(torch.maximum(predicted_energy, floor)) -
                                         torch.log(torch.maximum(target_energy, floor))) ** 2)
                nx = predicted_electric.shape[-1]
                predicted_mode = torch.fft.rfft(predicted_electric.float(), dim=-1)[:, 1] / nx
                target_mode = torch.fft.rfft(target_electric.float(), dim=-1)[:, 1] / nx
                initial_mode_scale = torch.clamp(alpha_value / (2.0 * k_value), min=1.0e-6)
                complex_field_loss = torch.mean(
                    torch.abs((predicted_mode - target_mode) / initial_mode_scale) ** 2
                )
                current_gradient = closure_gradient(model, checkpoint, state, buffer, lags,
                                                    k_value, alpha_value, args.maximum_mode)
                with torch.no_grad():
                    teacher_gradient = closure_gradient(teacher, checkpoint, state, buffer, lags,
                                                        k_value, alpha_value, args.maximum_mode)
                teacher_loss = torch.mean(((current_gradient - teacher_gradient) / gradient_std) ** 2)
                truth_times = target_time - np.asarray(lags, dtype=np.float64) * args.dt
                truth_history_np = lowpass_numpy(
                    interpolate(trajectory.time, trajectory.state, truth_times),
                    args.maximum_mode,
                )
                truth_history = torch.from_numpy(truth_history_np[None]).to(device)
                supervised_gradient = closure_gradient_from_history(
                    model, checkpoint, truth_history, k_value, alpha_value,
                    args.maximum_mode,
                )
                truth_gradient_np = lowpass_numpy(
                    interpolate(
                        trajectory.time, trajectory.heat_flux_gradient,
                        np.asarray([target_time]),
                    ),
                    args.maximum_mode,
                )[0]
                truth_gradient = torch.from_numpy(truth_gradient_np[None]).to(device)
                truth_closure_loss = torch.mean(
                    ((supervised_gradient - truth_gradient) / gradient_std) ** 2
                )
                loss = (
                    state_loss
                    + args.field_weight * field_loss
                    + args.complex_field_weight * complex_field_loss
                    + args.teacher_weight * teacher_loss
                    + args.truth_closure_weight * truth_closure_loss
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite rollout loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(float(loss.detach()))
                state = state.detach(); buffer = [value.detach() for value in buffer]
        value = validation_loss(model, checkpoint, validation, labels, args.horizon_steps,
                                args.dt, args.tbptt_steps, args.maximum_mode, device,
                                args.field_solver, args.field_weight,
                                args.complex_field_weight, args.validation_max_cells)
        row = {"epoch": epoch, "horizon_time": horizon, "train_loss": float(np.mean(losses)),
               "validation_rollout_loss": value, "balanced_cell_count": len(cells)}
        rows.append(row); print(json.dumps(row), flush=True)
        if value < best_value:
            best_value = value; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    curriculum = list(checkpoint.get("curriculum_history", []))
    curriculum.append({"parent_checkpoint": str(args.checkpoint), "horizon_steps": args.horizon_steps,
                       "horizon_time": horizon, "best_validation_state_loss": best_value,
                       "regime_balanced": True})
    output = dict(checkpoint)
    output.update({"stage": "gkeyll_regime_balanced_dual_rollout_v1",
                   "parent_checkpoint": str(args.checkpoint),
                   "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                   "curriculum_history": curriculum, "best_validation_rollout_loss": best_value})
    output["rollout_finetune_arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "best_rollout.pt.tmp"; torch.save(output, temporary)
    os.replace(temporary, args.output_dir / "best_rollout.pt")
    (args.output_dir / "history.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation_rollout_loss": best_value}, indent=2))


if __name__ == "__main__":
    main()
