"""Free-run a paper-style raw-moment FNO closure against one PIC trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from landau_surrogate.data.raw_moment_pic import load_raw_moment_pic_pair, spectral_derivative
from landau_surrogate.fluid.multimoment_1d import (
    poisson_electric, rk4_step_raw_moments, rk4_step_raw_moments_ampere,
)
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default="k0p350_a0p100")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--maximum-mode", type=int, default=24)
    parser.add_argument("--field-solver", choices=("ampere", "poisson"), default="ampere")
    parser.add_argument("--closure-mode", choices=("learned", "truth-time"), default="learned")
    parser.add_argument("--no-prescribed-first-step", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Raw-moment closure rollout requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "pic_raw_moment_single_trajectory":
        raise ValueError("Expected a pic_raw_moment_single_trajectory checkpoint")
    trajectory = load_raw_moment_pic_pair(
        args.case_dir, args.pair_id,
        maximum_target_mode=int(checkpoint["maximum_target_mode"]),
    )
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.to(device).eval()
    normalization = checkpoint["normalization"]
    input_mean = torch.tensor(normalization["input_mean"], device=device).reshape(1, 3, 1)
    input_std = torch.tensor(normalization["input_std"], device=device).reshape(1, 3, 1)
    target_mean = float(normalization["target_mean"]); target_std = float(normalization["target_std"])
    k_value = torch.tensor([trajectory.k], device=device)
    raw = torch.from_numpy(trajectory.raw_state[0:1]).to(device)
    if args.field_solver == "ampere":
        electric = torch.from_numpy(trajectory.electric[0:1]).to(device)
        state = torch.cat((raw, electric[:, None]), dim=1)
        stepper = rk4_step_raw_moments_ampere
    else:
        state = raw
        stepper = rk4_step_raw_moments
    initial_gradient = torch.from_numpy(trajectory.third_moment_gradient[0:1]).to(device)
    # The learned target follows the checkpoint cutoff.  The truth-time mode is
    # an oracle consistency audit and therefore uses the complete resolved M3
    # derivative; otherwise it would conflate closure error with label filtering.
    oracle_gradient = spectral_derivative(trajectory.third_moment, trajectory.k).astype(np.float32)
    truth_gradient = torch.from_numpy(oracle_gradient).to(device)

    def public_state(current: torch.Tensor) -> torch.Tensor:
        safe_density = torch.clamp(current[:, 0], min=1.0e-6)
        return torch.stack((current[:, 0], current[:, 1] / safe_density, current[:, 2]), dim=1)

    def learned_closure(current: torch.Tensor) -> torch.Tensor:
        normalized = (public_state(current) - input_mean) / input_std
        return model(normalized, torch.zeros(len(current), device=device)) * target_std + target_mean

    recorded_raw = [state[:, :3].cpu()]
    recorded_public = [public_state(state[:, :3]).cpu()]
    recorded_electric = []
    recorded_field = []
    recorded_total = []
    def record_energy() -> None:
        electric_now = state[:, 3] if args.field_solver == "ampere" else poisson_electric(state[:, 0], k_value)
        domain_length = 2.0 * torch.pi / k_value
        field = 0.5 * domain_length * electric_now.square().mean(dim=-1)
        kinetic = 0.5 * domain_length * state[:, 2].mean(dim=-1)
        recorded_electric.append(electric_now.cpu()); recorded_field.append(field.cpu())
        recorded_total.append((field + kinetic).cpu())
    record_energy()
    stop_index = min(
        len(trajectory.time) - 1,
        int(np.searchsorted(trajectory.time, args.max_time, side="right") - 1),
    )
    clamp_count = 0; started = time.perf_counter(); completed_index = 0
    with torch.no_grad():
        for output_index in range(1, stop_index + 1):
            interval = float(trajectory.time[output_index] - trajectory.time[output_index - 1])
            substeps = max(1, int(round(interval / args.dt))); step_dt = interval / substeps
            for substep in range(substeps):
                # Huang et al. prescribe the closure over the first physical
                # time step.  Here one saved PIC interval is split into many
                # RK4 substeps, so the prescription must cover the complete
                # first interval rather than only its first numerical substep.
                prescribed = not args.no_prescribed_first_step and output_index == 1
                midpoint_fraction = (substep + 0.5) / substeps
                midpoint_gradient = (
                    (1.0 - midpoint_fraction) * truth_gradient[output_index - 1 : output_index]
                    + midpoint_fraction * truth_gradient[output_index : output_index + 1]
                )
                def factory(_stage: torch.Tensor):
                    if args.closure_mode == "truth-time":
                        return lambda current: midpoint_gradient.expand(len(current), -1)
                    if prescribed:
                        return lambda current: initial_gradient.expand(len(current), -1)
                    return learned_closure
                candidate = stepper(
                    state, k_value, step_dt, factory, maximum_mode=args.maximum_mode
                )
                safe_density = torch.clamp(candidate[:, 0], min=1.0e-4)
                central_pressure = candidate[:, 2] - candidate[:, 1].square() / safe_density
                clamp_count += int(torch.count_nonzero(
                    (candidate[:, 0] < 1.0e-4) | (central_pressure < 1.0e-5)
                ))
                second = torch.maximum(
                    candidate[:, 2], candidate[:, 1].square() / safe_density + 1.0e-5
                )
                components = [safe_density, candidate[:, 1], second]
                if args.field_solver == "ampere":
                    components.append(candidate[:, 3])
                state = torch.stack(components, dim=1)
                if not torch.isfinite(state).all():
                    break
            if not torch.isfinite(state).all():
                break
            completed_index = output_index
            recorded_raw.append(state[:, :3].cpu()); recorded_public.append(public_state(state[:, :3]).cpu())
            record_energy()
    runtime = time.perf_counter() - started
    raw_prediction = torch.cat(recorded_raw).numpy()
    public_prediction = torch.cat(recorded_public).numpy()
    electric_prediction = torch.cat(recorded_electric).numpy()
    field_prediction = torch.cat(recorded_field).numpy()
    total_prediction = torch.cat(recorded_total).numpy()
    output_time = trajectory.time[: len(raw_prediction)]
    truth_raw = trajectory.raw_state[: len(raw_prediction)]
    truth_public = trajectory.public_state[: len(raw_prediction)]
    truth_field = trajectory.field_energy[: len(raw_prediction)]
    floor = max(float(truth_field[0]) * 1.0e-10, 1.0e-14)
    field_error = float(np.sqrt(np.mean((
        np.log10(np.maximum(field_prediction, floor)) - np.log10(np.maximum(truth_field, floor))
    ) ** 2)))
    central_pressure = raw_prediction[:, 2] - raw_prediction[:, 1] ** 2 / np.maximum(raw_prediction[:, 0], 1.0e-8)
    summary = {
        "stage": "pic_raw_moment_single_trajectory_rollout", "pair_id": args.pair_id,
        "checkpoint": str(args.checkpoint), "checkpoint_seed": int(checkpoint["seed"]),
        "field_solver": args.field_solver, "dt": args.dt, "maximum_mode": args.maximum_mode,
        "closure_mode": args.closure_mode,
        "prescribed_first_saved_interval": not args.no_prescribed_first_step,
        "completed_time": float(output_time[-1]), "target_time": float(args.max_time),
        "runtime_seconds": runtime, "clamp_count": clamp_count,
        "finite": bool(np.isfinite(raw_prediction).all()),
        "density_relative_l2": relative_l2(raw_prediction[:, 0], truth_raw[:, 0]),
        "momentum_relative_l2": relative_l2(raw_prediction[:, 1], truth_raw[:, 1]),
        "second_moment_relative_l2": relative_l2(raw_prediction[:, 2], truth_raw[:, 2]),
        "field_energy_log10_rmse": field_error,
        "minimum_density": float(raw_prediction[:, 0].min()),
        "minimum_central_pressure": float(central_pressure.min()),
        "total_energy_relative_span": float(np.ptp(total_prediction / total_prediction[0] - 1.0)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / f"{args.pair_id}.npz", time=output_time,
        raw_prediction=raw_prediction, raw_truth=truth_raw,
        public_prediction=public_prediction, public_truth=truth_public,
        electric=electric_prediction, truth_electric=trajectory.electric[:len(raw_prediction)],
        field_energy=field_prediction, truth_field_energy=truth_field,
        total_energy=total_prediction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
