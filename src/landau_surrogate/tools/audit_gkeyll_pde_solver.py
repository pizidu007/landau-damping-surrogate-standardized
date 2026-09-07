"""Separate PDE discretization, spectral truncation, and learned-closure errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.huang2025 import load_huang_mat
from landau_surrogate.fluid.multimoment_1d import (
    fluid_rhs,
    fluid_rhs_ampere,
    poisson_electric,
    spectral_filter,
)
from landau_surrogate.training.gkeyll_multicase import lowpass_numpy


def interpolate(time: np.ndarray, value: np.ndarray, query: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(time, query, side="right")
    upper = np.clip(upper, 1, len(time) - 1)
    lower = upper - 1
    weight = ((query - time[lower]) / np.maximum(time[upper] - time[lower], 1.0e-14)).astype(np.float32)
    return value[lower] * (1.0 - weight[:, None, None]) + value[upper] * weight[:, None, None]


def rel(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-12))


def field_energy(state: torch.Tensor, k: torch.Tensor, ampere: bool) -> torch.Tensor:
    electric = state[:, 3] if ampere else poisson_electric(state[:, 0], k)
    return 0.5 * (2.0 * torch.pi / k) * electric.square().mean(dim=-1)


def rhs_residual(trajectory, maximum_mode: int, device: torch.device) -> dict:
    state = lowpass_numpy(trajectory.state, maximum_mode)
    gradient = lowpass_numpy(trajectory.heat_flux_gradient, maximum_mode)
    time_derivative = np.gradient(state.astype(np.float64), trajectory.time, axis=0).astype(np.float32)
    rows = []
    batch_size = 512
    for start in range(0, len(state), batch_size):
        stop = min(start + batch_size, len(state))
        current = torch.from_numpy(state[start:stop]).to(device)
        closure = torch.from_numpy(gradient[start:stop]).to(device)
        k_value = torch.full((stop - start,), trajectory.k, device=device)
        rows.append(fluid_rhs(current, k_value, lambda _state, value=closure: value,
                              maximum_mode=maximum_mode).cpu().numpy())
    prediction = np.concatenate(rows)[2:-2]
    target = time_derivative[2:-2]
    names = ("density", "velocity", "pressure")
    return {
        "aggregate_relative_l2": rel(prediction, target),
        "channels": {name: rel(prediction[:, index], target[:, index]) for index, name in enumerate(names)},
    }


def rk4_non_autonomous(state, k, dt: float, q0, qhalf, q1, maximum_mode: int, ampere: bool):
    rhs_function = fluid_rhs_ampere if ampere else fluid_rhs

    def rhs(value, closure):
        filtered = spectral_filter(value, maximum_mode)
        return rhs_function(filtered, k, lambda _state: closure,
                            maximum_mode=maximum_mode)

    k1 = rhs(state, q0)
    k2 = rhs(state + 0.5 * dt * k1, qhalf)
    k3 = rhs(state + 0.5 * dt * k2, qhalf)
    k4 = rhs(state + dt * k3, q1)
    return spectral_filter(state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), maximum_mode)


@torch.no_grad()
def oracle_rollout(trajectory, maximum_mode: int, dt: float, ampere: bool,
                   device: torch.device, record_interval: float = 0.05) -> dict:
    stop_time = min(40.0, float(trajectory.time[-1]))
    step_count = int(np.floor(stop_time / dt + 1.0e-10))
    grid = np.arange(step_count + 1, dtype=np.float64) * dt
    half_grid = grid[:-1] + 0.5 * dt
    gradient = lowpass_numpy(trajectory.heat_flux_gradient, maximum_mode)
    q0 = interpolate(trajectory.time, gradient[:, None, :], grid[:-1])[:, 0]
    qhalf = interpolate(trajectory.time, gradient[:, None, :], half_grid)[:, 0]
    q1 = interpolate(trajectory.time, gradient[:, None, :], grid[1:])[:, 0]

    initial = torch.from_numpy(lowpass_numpy(trajectory.state[0:1], maximum_mode)).to(device)
    k_value = torch.tensor([trajectory.k], device=device)
    if ampere:
        initial_electric = poisson_electric(initial[:, 0], k_value)
        state = torch.cat((initial, initial_electric[:, None]), dim=1)
    else:
        state = initial
    stride = max(1, int(round(record_interval / dt)))
    indices, states, energies, constraint = [0], [state[:, :3].cpu()], [field_energy(state, k_value, ampere).cpu()], []
    for index in range(step_count):
        state = rk4_non_autonomous(
            state, k_value, dt,
            torch.from_numpy(q0[index:index+1]).to(device),
            torch.from_numpy(qhalf[index:index+1]).to(device),
            torch.from_numpy(q1[index:index+1]).to(device),
            maximum_mode, ampere,
        )
        if (index + 1) % stride == 0 or index + 1 == step_count:
            indices.append(index + 1)
            states.append(state[:, :3].cpu())
            energies.append(field_energy(state, k_value, ampere).cpu())
            if ampere:
                poisson = poisson_electric(state[:, 0], k_value)
                constraint.append(float(torch.linalg.vector_norm(state[:, 3] - poisson) /
                                        torch.clamp(torch.linalg.vector_norm(poisson), min=1.0e-12)))
    output_time = grid[np.asarray(indices)]
    prediction = torch.cat(states).numpy()
    predicted_energy = torch.cat(energies).numpy()
    truth = interpolate(trajectory.time, trajectory.state, output_time)
    truth = lowpass_numpy(truth, maximum_mode)
    truth_tensor = torch.from_numpy(truth).to(device)
    truth_energy = field_energy(truth_tensor, torch.full((len(truth),), trajectory.k, device=device), False).cpu().numpy()
    floor = max(float(truth_energy[0]) * 1.0e-10, 1.0e-14)
    return {
        "dt": dt,
        "maximum_mode": maximum_mode,
        "field_solver": "ampere" if ampere else "poisson",
        "completed_time": float(output_time[-1]),
        "state_relative_l2": rel(prediction, truth),
        "density_relative_l2": rel(prediction[:, 0], truth[:, 0]),
        "velocity_relative_l2": rel(prediction[:, 1], truth[:, 1]),
        "pressure_relative_l2": rel(prediction[:, 2], truth[:, 2]),
        "field_energy_log10_rmse": float(np.sqrt(np.mean((
            np.log10(np.maximum(predicted_energy, floor)) - np.log10(np.maximum(truth_energy, floor))
        )**2))),
        "maximum_poisson_constraint_relative_l2": max(constraint) if constraint else 0.0,
        "minimum_density": float(prediction[:, 0].min()),
        "minimum_pressure": float(prediction[:, 2].min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dts", default="0.01,0.005,0.002")
    parser.add_argument("--modes", default="8,31")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal PDE solver audit requires CUDA")
    trajectory = load_huang_mat(args.trajectory)
    modes = [int(value) for value in args.modes.split(",")]
    dts = [float(value) for value in args.dts.split(",")]
    residuals = {str(mode): rhs_residual(trajectory, mode, device) for mode in modes}
    rollouts = []
    for mode in modes:
        for ampere in (False, True):
            for dt in dts:
                row = oracle_rollout(trajectory, mode, dt, ampere, device)
                rollouts.append(row)
                print(json.dumps(row), flush=True)
    result = {
        "case_id": args.trajectory.parent.parent.name,
        "k": trajectory.k,
        "alpha": trajectory.alpha,
        "closure": "time-dependent Gkeyll truth heat-flux gradient",
        "rhs_residuals": residuals,
        "oracle_rollouts": rollouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
