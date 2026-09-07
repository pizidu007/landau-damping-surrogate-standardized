"""Closed-loop fluid reproduction on the public Huang Gkeyll trajectory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.huang2025 import load_huang_mat
from landau_surrogate.fluid.multimoment_1d import (
    fluid_energies, fluid_energies_ampere, poisson_electric, rk4_step, rk4_step_ampere,
    rk4_step_raw_moments, rk4_step_raw_moments_ampere, spectral_filter,
)
from landau_surrogate.models.closure_fno1d import (
    ClosureFNO1d, DualHeadHistoryResidualFNO1d, closure_from_physical,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--closure", choices=("fno", "hp"), default="fno")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--hp-scale", type=float, default=float(np.sqrt(8.0 / np.pi)),
        help="Coefficient multiplying |k|(p-n) in the HP heat-flux-gradient closure.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--max-time", type=float, default=40.0)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--maximum-mode", type=int, default=24)
    parser.add_argument("--field-solver", choices=("poisson", "ampere"), default="poisson")
    parser.add_argument("--moment-formulation", choices=("raw", "primitive"), default="raw")
    parser.add_argument("--closure-scale", type=float, default=1.0)
    parser.add_argument("--time-grid", choices=("source", "fixed"), default="source")
    parser.add_argument("--initial-closure", choices=("fno", "truth"), default="fno")
    parser.add_argument(
        "--closure-evaluation", choices=("stage", "step"), default="stage",
        help="Evaluate FNO at every RK stage or once and freeze it for the full step.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Huang closed-loop reproduction requires CUDA")
    checkpoint = None
    model = None
    normalization = None
    dual_model = False
    history_model = False
    if args.closure == "fno":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for FNO closure")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") not in (
            "huang2025_gkeyll_reproduction", "huang2025_gkeyll_multicase_v1",
            "huang2025_gkeyll_multicase_rollout_v1",
            "huang2025_gkeyll_history_q_v1", "huang2025_gkeyll_history_rollout_v1",
            "gkeyll_regime_balanced_dual_history_v1",
            "gkeyll_regime_balanced_dual_rollout_v1",
            "huang2025_gkeyll_single_rollout_v1",
        ):
            raise ValueError("Expected a Huang/Gkeyll reproduction checkpoint")
        dual_model = checkpoint.get("target_kind") == "dual_gradient_heat_flux"
        model = (
            DualHeadHistoryResidualFNO1d(**checkpoint["model_arguments"])
            if dual_model else ClosureFNO1d(**checkpoint["model_arguments"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        normalization = checkpoint["normalization"]
        history_model = checkpoint.get("target_kind") in ("heat_flux", "dual_gradient_heat_flux")
        if not history_model:
            mean = torch.tensor(normalization["input_mean"], device=device).reshape(1, 3, 1)
            std = torch.tensor(normalization["input_std"], device=device).reshape(1, 3, 1)
            target_mean = float(normalization["target_mean"])
            target_std = float(normalization["target_std"])
    trajectory = load_huang_mat(args.mat)
    maximum_time = min(float(args.max_time), float(trajectory.time[-1]))
    start_time = max(float(trajectory.time[0]), float(args.start_time))
    if start_time >= maximum_time:
        raise ValueError("start-time must be smaller than max-time")
    if args.time_grid == "fixed":
        output_time = start_time + np.arange(
            int(np.floor((maximum_time - start_time) / args.dt)) + 1,
            dtype=np.float64,
        ) * args.dt
    else:
        selected = (trajectory.time >= start_time) & (trajectory.time <= maximum_time)
        output_time = trajectory.time[selected]
        if len(output_time) == 0 or output_time[0] > start_time + 1.0e-12:
            output_time = np.concatenate(([start_time], output_time))

    def interpolate_initial(value: np.ndarray) -> np.ndarray:
        result = np.empty((1, *value.shape[1:]), dtype=np.float32)
        for index in np.ndindex(value.shape[1:]):
            result[(0, *index)] = np.interp(
                start_time, trajectory.time, value[(slice(None), *index)]
            )
        return result

    public_state = torch.from_numpy(interpolate_initial(trajectory.state)).to(device)
    k_value = torch.tensor([trajectory.k], device=device)
    if checkpoint is not None and checkpoint["model_arguments"].get("include_k", False):
        model_k = (k_value - float(normalization["k_mean"])) / float(normalization["k_std"])
    else:
        model_k = torch.zeros_like(k_value)
    if args.moment_formulation == "raw":
        density, velocity, second_moment = public_state[:, 0], public_state[:, 1], public_state[:, 2]
        raw_state = torch.stack((density, density * velocity, second_moment), dim=1)
        if args.field_solver == "ampere":
            electric = poisson_electric(density, k_value)
            state = torch.cat((raw_state, electric[:, None]), dim=1)
            stepper = rk4_step_raw_moments_ampere
        else:
            state = raw_state
            stepper = rk4_step_raw_moments
    else:
        if args.field_solver == "ampere":
            electric = poisson_electric(public_state[:, 0], k_value)
            state = torch.cat((public_state, electric[:, None]), dim=1)
            stepper = rk4_step_ampere
        else:
            state = public_state
            stepper = rk4_step

    def as_public(current: torch.Tensor) -> torch.Tensor:
        if args.moment_formulation == "primitive":
            return current[:, :3]
        safe_density = torch.clamp(current[:, 0], min=1.0e-6)
        return torch.stack((current[:, 0], current[:, 1] / safe_density, current[:, 2]), dim=1)

    if history_model:
        nominal_offsets = checkpoint.get(
            "history_time_offsets_nominal",
            [0.005 * value for value in checkpoint["history_frame_offsets"]],
        )
        history_lags = tuple(
            int(np.floor(abs(float(value)) / args.dt + 0.5))
            for value in nominal_offsets
        )
        maximum_history_lag = max(history_lags)
        history_buffer = [as_public(state).detach()] * (maximum_history_lag + 1)
    else:
        history_lags = ()
        maximum_history_lag = 0
        history_buffer = []

    def factory(_stage: torch.Tensor):
        def closure(current: torch.Tensor) -> torch.Tensor:
            if args.closure == "hp":
                current_public = as_public(current)
                temperature_perturbation = current_public[:, 2] - current_public[:, 0]
                mode = torch.arange(
                    temperature_perturbation.shape[-1] // 2 + 1,
                    device=device, dtype=temperature_perturbation.dtype,
                )
                wave_number = k_value.reshape(-1, 1) * mode.reshape(1, -1)
                return args.hp_scale * torch.fft.irfft(
                    wave_number * torch.fft.rfft(
                        temperature_perturbation.float(), dim=-1
                    ),
                    n=temperature_perturbation.shape[-1], dim=-1,
                ).to(temperature_perturbation.dtype)
            if history_model:
                current_public = as_public(current)
                values = []
                for lag in history_lags:
                    if lag == 0:
                        values.append(current_public)
                    else:
                        index = max(0, len(history_buffer) - 1 - lag)
                        values.append(history_buffer[index])
                physical_history = torch.stack(values, dim=1)
                if dual_model:
                    assert normalization is not None and model is not None
                    history_mean = torch.as_tensor(
                        normalization["input_mean"], device=device
                    ).reshape(1, 1, 3, 1)
                    history_std = torch.as_tensor(
                        normalization["input_std"], device=device
                    ).reshape(1, 1, 3, 1)
                    condition_mean = torch.as_tensor(
                        normalization["condition_mean"], device=device
                    )
                    condition_std = torch.as_tensor(
                        normalization["condition_std"], device=device
                    )
                    alpha_value = torch.full_like(k_value, trajectory.alpha)
                    condition = (
                        torch.stack((k_value, alpha_value), dim=1) - condition_mean
                    ) / condition_std
                    gradient, _heat_flux = model(
                        (physical_history - history_mean) / history_std, condition
                    )
                    gradient = gradient * float(normalization["gradient_std"])
                else:
                    assert model is not None and normalization is not None
                    gradient = closure_from_physical(
                        model,
                        physical_history,
                        k_value.expand(len(current)),
                        normalization,
                        checkpoint["target_kind"],
                    )
                return args.closure_scale * spectral_filter(
                    gradient[:, None], args.maximum_mode
                )[:, 0]
            assert model is not None
            normalized = (as_public(current) - mean) / std
            raw = model(normalized, model_k.expand(len(current))) * target_std + target_mean
            return args.closure_scale * raw
        return closure

    initial_gradient = torch.from_numpy(
        interpolate_initial(trajectory.heat_flux_gradient[:, None, :])[:, 0]
    ).to(device)

    def initial_factory(_stage: torch.Tensor):
        def closure(current: torch.Tensor) -> torch.Tensor:
            return initial_gradient.expand(len(current), -1)
        return closure

    recorded = [as_public(state).cpu()]
    electric = state[:, 3] if args.field_solver == "ampere" else poisson_electric(state[:, 0], k_value)
    energies = [(0.5 * (2.0 * torch.pi / k_value) * electric.square().mean(dim=-1)).cpu()]
    clamp_count = 0
    integration_steps = 0
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(1, len(output_time)):
            interval = float(output_time[index] - output_time[index - 1])
            substeps = 1 if args.time_grid == "fixed" else max(1, int(np.ceil(interval / args.dt)))
            step_dt = interval / substeps
            for _ in range(substeps):
                integration_steps += 1
                closure_factory = (
                    initial_factory
                    if args.initial_closure == "truth" and integration_steps == 1
                    else factory
                )
                if (
                    closure_factory is factory
                    and args.closure_evaluation == "step"
                ):
                    frozen_gradient = factory(state)(state).detach()

                    def frozen_factory(_stage: torch.Tensor, value=frozen_gradient):
                        def closure(_current: torch.Tensor) -> torch.Tensor:
                            return value
                        return closure

                    closure_factory = frozen_factory
                if args.moment_formulation == "raw":
                    raw = stepper(state, k_value, step_dt, closure_factory, maximum_mode=args.maximum_mode)
                    safe_density = torch.clamp(raw[:, 0], min=1.0e-4)
                    central_pressure = raw[:, 2] - raw[:, 1].square() / safe_density
                    clamp_count += int(torch.count_nonzero((raw[:, 0] < 1.0e-4) | (central_pressure < 1.0e-5)))
                    second_moment = torch.maximum(
                        raw[:, 2], raw[:, 1].square() / safe_density + 1.0e-5
                    )
                    components = [safe_density, raw[:, 1], second_moment]
                    if args.field_solver == "ampere":
                        components.append(raw[:, 3])
                    state = torch.stack(components, dim=1)
                else:
                    raw = stepper(
                        state, k_value, step_dt, closure_factory, clamp_output=False,
                        maximum_mode=args.maximum_mode,
                    )
                    clamp_count += int(torch.count_nonzero((raw[:, 0] < 1.0e-4) | (raw[:, 2] < 1.0e-5)))
                    components = [torch.clamp(raw[:, 0], min=1.0e-4), raw[:, 1], torch.clamp(raw[:, 2], min=1.0e-5)]
                    if args.field_solver == "ampere":
                        components.append(raw[:, 3])
                    state = torch.stack(components, dim=1)
                if history_model:
                    history_buffer.append(as_public(state).detach())
                    if len(history_buffer) > maximum_history_lag + 1:
                        history_buffer.pop(0)
            if not torch.isfinite(state).all():
                break
            recorded.append(as_public(state).cpu())
            electric = state[:, 3] if args.field_solver == "ampere" else poisson_electric(state[:, 0], k_value)
            energies.append((0.5 * (2.0 * torch.pi / k_value) * electric.square().mean(dim=-1)).cpu())
    runtime = time.perf_counter() - started
    prediction = torch.cat(recorded).numpy()
    field_energy = torch.cat(energies).numpy()
    model_time = output_time[: len(prediction)]
    truth = np.empty((len(model_time), trajectory.state.shape[1], trajectory.state.shape[2]), dtype=np.float32)
    for channel in range(trajectory.state.shape[1]):
        for cell in range(trajectory.state.shape[2]):
            truth[:, channel, cell] = np.interp(
                model_time, trajectory.time, trajectory.state[:, channel, cell]
            )
    density_tensor = torch.from_numpy(truth[:, 0]).to(device)
    truth_electric = poisson_electric(density_tensor, torch.full((len(truth),), trajectory.k, device=device))
    domain_length = 2.0 * np.pi / trajectory.k
    truth_field_energy = (0.5 * domain_length * torch.mean(truth_electric.square(), dim=-1)).cpu().numpy()
    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-12))
    floor = max(float(truth_field_energy[0]) * 1.0e-10, 1.0e-14)
    summary = {
        "protocol": (
            checkpoint.get("protocol", "trajectory_casewise")
            if checkpoint is not None else "hammett_perkins"
        ),
        "closure": args.closure,
        "hp_scale": args.hp_scale if args.closure == "hp" else None,
        "field_solver": args.field_solver,
        "moment_formulation": args.moment_formulation,
        "dt": args.dt, "time_grid": args.time_grid,
        "start_time": start_time,
        "initial_closure": args.initial_closure,
        "closure_evaluation": args.closure_evaluation,
        "integration_steps": integration_steps,
        "maximum_mode": args.maximum_mode,
        "closure_scale": args.closure_scale,
        "completed_time": float(model_time[-1]),
        "runtime_seconds": runtime, "clamp_count": clamp_count,
        "density_relative_l2": rel(prediction[:, 0], truth[:, 0]),
        "velocity_relative_l2": rel(prediction[:, 1], truth[:, 1]),
        "pressure_relative_l2": rel(prediction[:, 2], truth[:, 2]),
        "field_energy_log10_rmse": float(np.sqrt(np.mean((
            np.log10(np.maximum(field_energy, floor)) - np.log10(np.maximum(truth_field_energy, floor))
        ) ** 2))),
        "minimum_density": float(prediction[:, 0].min()),
        "minimum_pressure": float(prediction[:, 2].min()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "rollout.npz", time=model_time,
        prediction=prediction, truth=truth, field_energy=field_energy,
        truth_field_energy=truth_field_energy,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
