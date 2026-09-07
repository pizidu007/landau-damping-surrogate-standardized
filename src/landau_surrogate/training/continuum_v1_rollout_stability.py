"""Closed-loop stability fine tuning for the continuum_v1 closure FNO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from landau_surrogate.data.continuum_v1 import (
    ContinuumCase,
    load_continuum_case,
    load_continuum_case_index,
)
from landau_surrogate.fluid.multimoment_1d import (
    poisson_electric,
    rk4_step_ampere,
    spectral_filter,
)
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


@dataclass(frozen=True)
class CachedTrajectory:
    case: ContinuumCase
    state: np.ndarray
    gradient: np.ndarray
    time: np.ndarray


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_cache(
    cases: list[ContinuumCase], maximum_mode: int, name: str
) -> list[CachedTrajectory]:
    result = []
    for index, case in enumerate(cases):
        trajectory = load_continuum_case(
            case,
            input_maximum_mode=maximum_mode,
            target_maximum_mode=maximum_mode,
        )
        result.append(
            CachedTrajectory(
                case=case,
                state=trajectory.state,
                gradient=trajectory.heat_flux_gradient,
                time=trajectory.time,
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(f"cached {name}: {index + 1}/{len(cases)}", flush=True)
    return result


def interpolate(value: np.ndarray, time_values: np.ndarray, query: float) -> np.ndarray:
    upper = int(np.clip(np.searchsorted(time_values, query, side="right"), 1, len(time_values) - 1))
    lower = upper - 1
    weight = np.float32(
        (query - time_values[lower])
        / max(float(time_values[upper] - time_values[lower]), 1.0e-12)
    )
    return value[lower] * (1.0 - weight) + value[upper] * weight


def physical_batch(
    trajectories: list[CachedTrajectory],
    times: list[float],
    attribute: str,
) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            [
                interpolate(getattr(trajectory, attribute), trajectory.time, query)
                for trajectory, query in zip(trajectories, times, strict=True)
            ]
        ),
        dtype=np.float32,
    )


def closure_gradient(
    model: ClosureFNO1d,
    checkpoint: dict[str, Any],
    state: torch.Tensor,
    k_value: torch.Tensor,
    maximum_mode: int,
) -> torch.Tensor:
    normalization = checkpoint["normalization"]
    mean = torch.as_tensor(
        normalization["input_mean"], device=state.device, dtype=state.dtype
    ).reshape(1, 3, 1)
    std = torch.as_tensor(
        normalization["input_std"], device=state.device, dtype=state.dtype
    ).reshape(1, 3, 1)
    normalized_k = (k_value - float(normalization["k_mean"])) / float(
        normalization["k_std"]
    )
    output = model((state - mean) / std, normalized_k)
    output = (
        output * float(normalization["gradient_std"])
        + float(normalization["gradient_mean"])
    )
    return spectral_filter(output[:, None], maximum_mode)[:, 0]


def advance_segment(
    model: ClosureFNO1d,
    checkpoint: dict[str, Any],
    state: torch.Tensor,
    k_value: torch.Tensor,
    *,
    steps: int,
    dt: float,
    maximum_mode: int,
) -> torch.Tensor:
    def factory(_stage: torch.Tensor):
        def closure(current: torch.Tensor) -> torch.Tensor:
            return closure_gradient(
                model, checkpoint, current, k_value, maximum_mode
            )

        return closure

    for _ in range(steps):
        state = rk4_step_ampere(
            state,
            k_value,
            dt,
            factory,
            clamp_output=False,
            maximum_mode=maximum_mode,
        )
    return state


def loss_components(
    model: ClosureFNO1d,
    checkpoint: dict[str, Any],
    state: torch.Tensor,
    target_state: torch.Tensor,
    target_gradient: torch.Tensor,
    k_value: torch.Tensor,
    alpha_value: torch.Tensor,
    maximum_mode: int,
    stability_limits: dict[str, float],
) -> dict[str, torch.Tensor]:
    normalization = checkpoint["normalization"]
    state_std = torch.as_tensor(
        normalization["input_std"], device=state.device, dtype=state.dtype
    ).reshape(1, 3, 1)
    normalized_error = (state[:, :3] - target_state) / state_std
    state_loss = torch.mean(normalized_error.square())

    target_electric = poisson_electric(target_state[:, 0], k_value)
    field_scale = torch.clamp(alpha_value / k_value, min=1.0e-4)
    electric_loss = torch.mean(
        ((state[:, 3] - target_electric) / field_scale[:, None]).square()
    )
    predicted_energy = 0.5 * state[:, 3].square().mean(dim=-1)
    target_energy = 0.5 * target_electric.square().mean(dim=-1)
    initial_energy_scale = 0.25 * field_scale.square()
    energy_floor = torch.clamp(initial_energy_scale * 1.0e-8, min=1.0e-14)
    energy_loss = torch.mean(
        (
            torch.log(torch.maximum(predicted_energy, energy_floor))
            - torch.log(torch.maximum(target_energy, energy_floor))
        ).square()
    )

    supervised_gradient = closure_gradient(
        model, checkpoint, target_state, k_value, maximum_mode
    )
    off_manifold_gradient = closure_gradient(
        model, checkpoint, state[:, :3], k_value, maximum_mode
    )
    gradient_std = float(normalization["gradient_std"])
    supervised_loss = torch.mean(
        ((supervised_gradient - target_gradient) / gradient_std).square()
    )
    recovery_loss = torch.mean(
        ((off_manifold_gradient - target_gradient) / gradient_std).square()
    )

    density_floor = float(stability_limits["density_soft_floor"])
    pressure_floor = float(stability_limits["pressure_soft_floor"])
    density_violation = torch.relu(density_floor - state[:, 0]) / density_floor
    pressure_violation = torch.relu(pressure_floor - state[:, 2]) / pressure_floor
    positivity_loss = torch.mean(density_violation.square()) + torch.mean(
        pressure_violation.square()
    )
    input_mean = torch.as_tensor(
        normalization["input_mean"], device=state.device, dtype=state.dtype
    ).reshape(1, 3, 1)
    normalized_state = (state[:, :3] - input_mean) / state_std
    excursion_limit = float(stability_limits["state_sigma_limit"])
    excursion_loss = torch.mean(
        torch.relu(torch.abs(normalized_state) - excursion_limit).square()
    )

    transformed = torch.fft.rfft(off_manifold_gradient.float(), dim=-1)
    tail_energy = torch.mean(torch.abs(transformed[:, 7:9]).square())
    retained_energy = torch.mean(torch.abs(transformed[:, 1:9]).square()).detach()
    spectral_tail_loss = tail_energy / torch.clamp(retained_energy, min=1.0e-10)
    gradient_rms = torch.sqrt(torch.mean(off_manifold_gradient.square(), dim=-1) + 1.0e-12)
    closure_sigma_limit = float(stability_limits["closure_rms_sigma_limit"])
    closure_limit_loss = torch.mean(
        torch.relu(
            gradient_rms / (closure_sigma_limit * gradient_std) - 1.0
        ).square()
    )
    return {
        "state": state_loss,
        "electric": electric_loss,
        "energy": energy_loss,
        "supervised": supervised_loss,
        "recovery": recovery_loss,
        "positivity": positivity_loss,
        "excursion": excursion_loss,
        "spectral_tail": spectral_tail_loss,
        "closure_limit": closure_limit_loss,
    }


def weighted_loss(
    components: dict[str, torch.Tensor], weights: dict[str, float]
) -> torch.Tensor:
    return sum(
        (components[name] * float(weights.get(name, 0.0)) for name in components),
        torch.zeros((), device=next(iter(components.values())).device),
    )


def sample_windows(
    cache: list[CachedTrajectory],
    *,
    batch_size: int,
    horizon: float,
    batch_index: int,
    rng: random.Random,
) -> tuple[list[CachedTrajectory], list[float]]:
    by_regime: dict[str, list[CachedTrajectory]] = {
        regime: [item for item in cache if item.case.regime == regime]
        for regime in ("weak", "transition", "strong_nonlinear")
    }
    if any(not values for values in by_regime.values()):
        raise ValueError("Every training regime must be represented")
    selected = []
    starts = []
    regimes = tuple(by_regime)
    time_buckets = ((0.0, 20.0), (20.0, 45.0), (45.0, 80.0 - horizon))
    for index in range(batch_size):
        regime = regimes[(batch_index * batch_size + index) % len(regimes)]
        trajectory = rng.choice(by_regime[regime])
        bucket = time_buckets[(batch_index + index) % len(time_buckets)]
        lower = max(bucket[0], float(trajectory.time[0]))
        upper = min(bucket[1], float(trajectory.time[-1]) - horizon)
        if upper <= lower:
            lower, upper = float(trajectory.time[0]), float(trajectory.time[-1]) - horizon
        selected.append(trajectory)
        starts.append(rng.uniform(lower, upper))
    return selected, starts


def regime_representatives(
    cases: list[ContinuumCase],
) -> list[ContinuumCase]:
    """Return one deterministic case per regime for a representative smoke run."""
    result = []
    for regime in ("weak", "transition", "strong_nonlinear"):
        try:
            result.append(next(case for case in cases if case.regime == regime))
        except StopIteration as error:
            raise ValueError(f"Missing {regime} case") from error
    return result


def initial_fluid_state(
    trajectories: list[CachedTrajectory],
    times: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    moments = torch.from_numpy(physical_batch(trajectories, times, "state")).to(device)
    k_value = torch.tensor(
        [item.case.K for item in trajectories], device=device, dtype=moments.dtype
    )
    alpha_value = torch.tensor(
        [item.case.alpha for item in trajectories], device=device, dtype=moments.dtype
    )
    electric = poisson_electric(moments[:, 0], k_value)
    return torch.cat((moments, electric[:, None]), dim=1), k_value, alpha_value


@torch.no_grad()
def validation_metrics(
    model: ClosureFNO1d,
    checkpoint: dict[str, Any],
    cache: list[CachedTrajectory],
    *,
    starts_per_case: list[float],
    horizon_steps: int,
    dt: float,
    maximum_mode: int,
    weights: dict[str, float],
    stability_limits: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    trajectories = []
    starts = []
    horizon = horizon_steps * dt
    for trajectory in cache:
        for requested in starts_per_case:
            trajectories.append(trajectory)
            starts.append(
                min(float(requested), float(trajectory.time[-1]) - horizon)
            )
    state, k_value, alpha_value = initial_fluid_state(trajectories, starts, device)
    state = advance_segment(
        model,
        checkpoint,
        state,
        k_value,
        steps=horizon_steps,
        dt=dt,
        maximum_mode=maximum_mode,
    )
    target_times = [value + horizon for value in starts]
    target_state = torch.from_numpy(
        physical_batch(trajectories, target_times, "state")
    ).to(device)
    target_gradient = torch.from_numpy(
        physical_batch(trajectories, target_times, "gradient")
    ).to(device)
    components = loss_components(
        model,
        checkpoint,
        state,
        target_state,
        target_gradient,
        k_value,
        alpha_value,
        maximum_mode,
        stability_limits,
    )
    finite = torch.isfinite(state).all(dim=(1, 2))
    values = {
        name: float(torch.nan_to_num(value, nan=1.0e6, posinf=1.0e6))
        for name, value in components.items()
    }
    score = weighted_loss(components, weights)
    values["score"] = float(torch.nan_to_num(score, nan=1.0e6, posinf=1.0e6))
    values["finite_fraction"] = float(finite.float().mean())
    return values


def checkpoint_payload(
    parent: dict[str, Any],
    model: ClosureFNO1d,
    *,
    parent_path: Path,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    best_validation: dict[str, float],
) -> dict[str, Any]:
    result = dict(parent)
    result.update(
        {
            "stage": "continuum_v1_rollout_stability_fno",
            "parent_checkpoint": str(parent_path.resolve()),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "rollout_stability_config": config,
            "rollout_stability_history": history,
            "best_validation_rollout": best_validation,
            "test_evaluated_during_selection": False,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("rollout stability fine tuning requires CUDA")
    checkpoint_path = Path(config["parent_checkpoint"])
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if parent.get("stage") not in {
        "continuum_v1_pure_supervised_fno",
        "continuum_v1_rollout_stability_fno",
    }:
        raise ValueError("Expected a continuum_v1 supervised or rollout checkpoint")
    model = ClosureFNO1d(**parent["model_arguments"])
    model.load_state_dict(parent["model_state_dict"])
    model.to(device)

    dt = float(config["solver"]["dt"])
    maximum_mode = int(config["solver"]["maximum_mode"])
    tbptt_steps = int(config["training"]["tbptt_steps"])
    batch_size = int(config["training"]["batch_size"])
    curriculum = list(config["training"]["curriculum"])
    validation_starts = list(config["validation"]["start_times"])
    validation_horizon_steps = int(config["validation"]["horizon_steps"])
    if args.smoke:
        batch_size = 2
        curriculum = [{"horizon_steps": 20, "epochs": 1, "batches_per_epoch": 1}]
        validation_starts = [5.0]
        validation_horizon_steps = 20

    all_cases = load_continuum_case_index(config["dataset_root"])
    train_cases = [case for case in all_cases if case.split == "train"]
    validation_cases = [case for case in all_cases if case.split == "validation"]
    if args.smoke:
        train_cases = regime_representatives(train_cases)
        validation_cases = regime_representatives(validation_cases)
    train_cache = load_cache(train_cases, maximum_mode, "train")
    validation_cache = load_cache(validation_cases, maximum_mode, "validation")
    weights = {key: float(value) for key, value in config["loss_weights"].items()}
    stability_limits = {
        "density_soft_floor": 0.50,
        "pressure_soft_floor": 0.25,
        "state_sigma_limit": 8.0,
        "closure_rms_sigma_limit": 8.0,
        **{
            key: float(value)
            for key, value in config.get("stability_limits", {}).items()
        },
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    rng = random.Random(args.seed)
    baseline = validation_metrics(
        model,
        parent,
        validation_cache,
        starts_per_case=validation_starts,
        horizon_steps=validation_horizon_steps,
        dt=dt,
        maximum_mode=maximum_mode,
        weights=weights,
        stability_limits=stability_limits,
        device=device,
    )
    print(json.dumps({"event": "baseline_validation", **baseline}), flush=True)
    history: list[dict[str, Any]] = []
    best_validation = baseline
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        args.output_dir / "best_rollout.pt",
        checkpoint_payload(
            parent,
            model,
            parent_path=checkpoint_path,
            config=config,
            history=history,
            best_validation=best_validation,
        ),
    )
    started = time.monotonic()
    global_batch = 0
    for stage_index, stage in enumerate(curriculum, start=1):
        horizon_steps = int(stage["horizon_steps"])
        if horizon_steps % tbptt_steps != 0:
            raise ValueError("Every curriculum horizon must be divisible by tbptt_steps")
        horizon = horizon_steps * dt
        for epoch in range(1, int(stage["epochs"]) + 1):
            model.train()
            component_rows = []
            loss_values = []
            epoch_started = time.monotonic()
            for _batch in range(int(stage["batches_per_epoch"])):
                trajectories, starts = sample_windows(
                    train_cache,
                    batch_size=batch_size,
                    horizon=horizon,
                    batch_index=global_batch,
                    rng=rng,
                )
                global_batch += 1
                state, k_value, alpha_value = initial_fluid_state(
                    trajectories, starts, device
                )
                completed = 0
                while completed < horizon_steps:
                    state = advance_segment(
                        model,
                        parent,
                        state,
                        k_value,
                        steps=tbptt_steps,
                        dt=dt,
                        maximum_mode=maximum_mode,
                    )
                    completed += tbptt_steps
                    target_times = [
                        value + completed * dt for value in starts
                    ]
                    target_state = torch.from_numpy(
                        physical_batch(trajectories, target_times, "state")
                    ).to(device)
                    target_gradient = torch.from_numpy(
                        physical_batch(trajectories, target_times, "gradient")
                    ).to(device)
                    components = loss_components(
                        model,
                        parent,
                        state,
                        target_state,
                        target_gradient,
                        k_value,
                        alpha_value,
                        maximum_mode,
                        stability_limits,
                    )
                    loss = weighted_loss(components, weights)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite rollout stability loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["training"]["gradient_clip"]),
                    )
                    optimizer.step()
                    loss_values.append(float(loss.detach()))
                    component_rows.append(
                        {name: float(value.detach()) for name, value in components.items()}
                    )
                    state = state.detach()
            validation = validation_metrics(
                model,
                parent,
                validation_cache,
                starts_per_case=validation_starts,
                horizon_steps=validation_horizon_steps,
                dt=dt,
                maximum_mode=maximum_mode,
                weights=weights,
                stability_limits=stability_limits,
                device=device,
            )
            row = {
                "stage": stage_index,
                "epoch": epoch,
                "horizon_steps": horizon_steps,
                "horizon_time": horizon,
                "train_loss": float(np.mean(loss_values)),
                "train_components": {
                    name: float(np.mean([item[name] for item in component_rows]))
                    for name in component_rows[0]
                },
                "validation": validation,
                "epoch_seconds": time.monotonic() - epoch_started,
            }
            history.append(row)
            atomic_json(args.output_dir / "history.json", history)
            print(json.dumps(row), flush=True)
            if (
                validation["finite_fraction"] == 1.0
                and validation["score"] < best_validation["score"]
            ):
                best_validation = validation
                atomic_torch_save(
                    args.output_dir / "best_rollout.pt",
                    checkpoint_payload(
                        parent,
                        model,
                        parent_path=checkpoint_path,
                        config=config,
                        history=history,
                        best_validation=best_validation,
                    ),
                )

    summary = {
        "stage": "continuum_v1_rollout_stability_fno",
        "parent_checkpoint": str(checkpoint_path.resolve()),
        "seed": args.seed,
        "train_case_count": len(train_cases),
        "validation_case_count": len(validation_cases),
        "test_case_count_used": 0,
        "baseline_validation": baseline,
        "best_validation": best_validation,
        "improvement_fraction": 1.0
        - best_validation["score"] / max(baseline["score"], 1.0e-30),
        "epochs_completed": len(history),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
