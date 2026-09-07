"""Fifty-solver-step differentiable rollout fine tuning for Gkeyll cases."""
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
from landau_surrogate.models.closure_fno1d import ClosureFNO1d
from landau_surrogate.training.gkeyll_multicase import load_cases, lowpass_numpy


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ClosureFNO1d, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "huang2025_gkeyll_multicase_v1":
        raise ValueError("Expected the supervised multi-case Gkeyll checkpoint")
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def closure_gradient(
    model: ClosureFNO1d,
    checkpoint: dict,
    state: torch.Tensor,
    k_value: torch.Tensor,
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    mean = torch.as_tensor(normalization["input_mean"], device=state.device).reshape(1, 3, 1)
    std = torch.as_tensor(normalization["input_std"], device=state.device).reshape(1, 3, 1)
    k_normalized = (k_value - float(normalization["k_mean"])) / float(normalization["k_std"])
    output = model((state - mean) / std, k_normalized)
    return output * float(normalization["target_std"]) + float(normalization["target_mean"])


def advance(
    model: ClosureFNO1d,
    checkpoint: dict,
    initial: torch.Tensor,
    k_value: torch.Tensor,
    solver_steps: int,
    dt: float,
    maximum_mode: int,
) -> torch.Tensor:
    state = spectral_filter(initial, maximum_mode)

    def factory(_stage: torch.Tensor):
        def closure(current: torch.Tensor) -> torch.Tensor:
            return closure_gradient(model, checkpoint, current, k_value)
        return closure

    for _ in range(solver_steps):
        state = rk4_step_ampere(
            state, k_value, dt, factory, clamp_output=False, maximum_mode=maximum_mode
        )
    return state


def sample_batch(
    cases: list[tuple],
    batch_size: int,
    frame_stride: int,
    maximum_mode: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial, target, k_values = [], [], []
    for _ in range(batch_size):
        _spec, trajectory = cases[rng.randrange(len(cases))]
        start = rng.randrange(0, len(trajectory.time) - frame_stride)
        state = lowpass_numpy(trajectory.state[[start, start + frame_stride]], maximum_mode)
        initial.append(state[0])
        target.append(state[1])
        k_values.append(trajectory.k)
    return np.asarray(initial), np.asarray(target), np.asarray(k_values, dtype=np.float32)


@torch.no_grad()
def validation_loss(
    model: ClosureFNO1d,
    checkpoint: dict,
    cases: list[tuple],
    solver_steps: int,
    dt: float,
    frame_stride: int,
    maximum_mode: int,
    device: torch.device,
) -> float:
    values = []
    std = torch.as_tensor(checkpoint["normalization"]["input_std"], device=device).reshape(1, 3, 1)
    for _spec, trajectory in cases:
        starts = np.linspace(0, len(trajectory.time) - frame_stride - 1, 4, dtype=int)
        for start in starts:
            moments = lowpass_numpy(
                trajectory.state[[start, start + frame_stride]], maximum_mode
            )
            moment0 = torch.from_numpy(moments[0:1]).to(device)
            k_value = torch.tensor([trajectory.k], device=device)
            electric0 = poisson_electric(moment0[:, 0], k_value)
            initial = torch.cat((moment0, electric0[:, None]), dim=1)
            prediction = advance(
                model, checkpoint, initial, k_value, solver_steps, dt, maximum_mode
            )[:, :3]
            target = torch.from_numpy(moments[1:2]).to(device)
            values.append(float(torch.mean(((prediction - target) / std) ** 2)))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--solver-steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--maximum-mode", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--teacher-weight", type=float, default=0.2)
    parser.add_argument("--field-weight", type=float, default=0.1)
    args = parser.parse_args()
    if not np.isclose(args.solver_steps * args.dt, args.frame_stride * 0.005):
        raise ValueError("Solver horizon must equal the Gkeyll frame horizon")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Gkeyll rollout fine tuning requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    all_cases = load_cases(args.manifest, args.dataset_root)
    train = [(spec, trajectory) for spec, trajectory in all_cases if spec["split"] == "train"]
    validation = [(spec, trajectory) for spec, trajectory in all_cases if spec["split"] == "validation"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    state_std = torch.as_tensor(
        checkpoint["normalization"]["input_std"], device=device
    ).reshape(1, 3, 1)
    target_std = float(checkpoint["normalization"]["target_std"])
    rng = random.Random(args.seed)
    best_value = validation_loss(
        model, checkpoint, validation, args.solver_steps, args.dt,
        args.frame_stride, args.maximum_mode, device,
    )
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.batches_per_epoch):
            initial_np, target_np, k_np = sample_batch(
                train, args.batch_size, args.frame_stride, args.maximum_mode, rng
            )
            initial_moments = torch.from_numpy(initial_np).to(device)
            target = torch.from_numpy(target_np).to(device)
            k_value = torch.from_numpy(k_np).to(device)
            electric0 = poisson_electric(initial_moments[:, 0], k_value)
            initial = torch.cat((initial_moments, electric0[:, None]), dim=1)
            with torch.no_grad():
                teacher_gradient = closure_gradient(teacher, checkpoint, initial_moments, k_value)
            optimizer.zero_grad(set_to_none=True)
            prediction4 = advance(
                model, checkpoint, initial, k_value, args.solver_steps, args.dt,
                args.maximum_mode,
            )
            prediction = prediction4[:, :3]
            state_loss = torch.mean(((prediction - target) / state_std) ** 2)
            target_electric = poisson_electric(target[:, 0], k_value)
            field_floor = torch.clamp(
                0.5 * target_electric.square().mean(dim=-1).detach() * 1.0e-8,
                min=1.0e-14,
            )
            predicted_energy = 0.5 * prediction4[:, 3].square().mean(dim=-1)
            target_energy = 0.5 * target_electric.square().mean(dim=-1)
            field_loss = torch.mean(
                (torch.log(torch.maximum(predicted_energy, field_floor))
                 - torch.log(torch.maximum(target_energy, field_floor))) ** 2
            )
            gradient = closure_gradient(model, checkpoint, initial_moments, k_value)
            teacher_loss = torch.mean(((gradient - teacher_gradient) / target_std) ** 2)
            loss = state_loss + args.field_weight * field_loss + args.teacher_weight * teacher_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite rollout loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        value = validation_loss(
            model, checkpoint, validation, args.solver_steps, args.dt,
            args.frame_stride, args.maximum_mode, device,
        )
        row = {
            "epoch": epoch, "train_rollout_loss": float(np.mean(losses)),
            "validation_rollout_loss": value,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    output = dict(checkpoint)
    output.update(
        {
            "stage": "huang2025_gkeyll_multicase_rollout_v1",
            "parent_checkpoint": str(args.checkpoint),
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "best_validation_rollout_loss": best_value,
            "rollout_finetune_arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "best_rollout.pt.tmp"
    torch.save(output, temporary)
    os.replace(temporary, args.output_dir / "best_rollout.pt")
    history_path = args.output_dir / "history.json.tmp"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(history_path, args.output_dir / "history.json")
    print(json.dumps({"best_validation_rollout_loss": best_value}, indent=2))


if __name__ == "__main__":
    main()
