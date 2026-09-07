"""Run multi-moment fluid rollouts with ML, HP, or zero heat-flux closure."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from landau_surrogate.data.closure_dataset import PairTrajectory, load_pair_trajectories
from landau_surrogate.data.paths import nonlinear_runs_root
from landau_surrogate.diagnostics.closure import relative_l2
from landau_surrogate.fluid.multimoment_1d import (
    fluid_energies, fluid_energies_ampere, rk4_step, rk4_step_ampere,
)
from landau_surrogate.inference.closure import load_closure_checkpoint, predict_physical_gradient


def hp_gradient(state: torch.Tensor, k_value: torch.Tensor, scale: float) -> torch.Tensor:
    temperature_perturbation = state[:, 2] - state[:, 0]
    nx = state.shape[-1]
    mode = torch.arange(nx // 2 + 1, device=state.device, dtype=state.dtype)
    wave_number = k_value.reshape(-1, 1) * mode.reshape(1, -1)
    return scale * torch.fft.irfft(
        wave_number * torch.fft.rfft(temperature_perturbation.float(), dim=-1),
        n=nx, dim=-1,
    ).to(state.dtype)


def fit_rate(time_values: np.ndarray, energy: np.ndarray, start: float, stop: float) -> float:
    mask = (time_values >= start) & (time_values <= stop) & (energy > 0.0) & np.isfinite(energy)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.polyfit(time_values[mask], 0.5 * np.log(energy[mask]), 1)[0])


def bounce_frequency(time_values: np.ndarray, energy: np.ndarray, start: float = 25.0) -> float:
    mask = (time_values >= start) & (energy > 0.0) & np.isfinite(energy)
    if np.count_nonzero(mask) < 8:
        return float("nan")
    time_selected = time_values[mask]
    signal = np.log(energy[mask])
    signal = signal - np.polyval(np.polyfit(time_selected, signal, 2), time_selected)
    spectrum = np.abs(np.fft.rfft(signal))
    frequency = np.fft.rfftfreq(len(signal), d=float(np.median(np.diff(time_selected))))
    allowed = (frequency > 0.0) & (2.0 * np.pi * frequency <= 1.0)
    if np.count_nonzero(allowed) == 0:
        return float("nan")
    selected = np.flatnonzero(allowed)
    return float(2.0 * np.pi * frequency[selected[np.argmax(spectrum[selected])]])


def rollout_metrics(
    trajectory: PairTrajectory,
    time_values: np.ndarray,
    prediction: np.ndarray,
    field_energy: np.ndarray,
    total_energy: np.ndarray,
    start_index: int,
) -> dict[str, float]:
    truth_state = trajectory.state[start_index : start_index + len(prediction)]
    truth_field = trajectory.field_energy[start_index : start_index + len(prediction)]
    field_floor = max(float(truth_field[0]) * 1.0e-8, 1.0e-14)
    log_error = np.sqrt(
        np.mean(
            (
                np.log10(np.maximum(field_energy, field_floor))
                - np.log10(np.maximum(truth_field, field_floor))
            ) ** 2
        )
    )
    metrics = {
        "density_relative_l2": relative_l2(prediction[:, 0], truth_state[:, 0]),
        "velocity_relative_l2": relative_l2(prediction[:, 1], truth_state[:, 1]),
        "pressure_relative_l2": relative_l2(prediction[:, 2], truth_state[:, 2]),
        "field_energy_log10_rmse": float(log_error),
        "predicted_damping_rate": fit_rate(time_values, field_energy, 0.0, 15.0),
        "truth_damping_rate": fit_rate(time_values, truth_field, 0.0, 15.0),
        "predicted_nonlinear_rate": fit_rate(time_values, field_energy, 30.0, 60.0),
        "truth_nonlinear_rate": fit_rate(time_values, truth_field, 30.0, 60.0),
        "predicted_bounce_frequency": bounce_frequency(time_values, field_energy),
        "truth_bounce_frequency": bounce_frequency(time_values, truth_field),
        "minimum_density": float(np.min(prediction[:, 0])),
        "minimum_pressure": float(np.min(prediction[:, 2])),
        "total_energy_relative_span": float(np.ptp(total_energy / total_energy[0] - 1.0)),
        "finite": float(np.isfinite(prediction).all() and np.isfinite(field_energy).all()),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=nonlinear_runs_root() / "nonlinear_formal_v1")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--closure", choices=("model", "hp", "zero"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--maximum-mode", type=int, default=24)
    parser.add_argument("--hp-blend", type=float, default=0.0)
    parser.add_argument("--field-solver", choices=("poisson", "ampere"), default="poisson")
    parser.add_argument(
        "--pair-id", action="append", default=[],
        help="Optionally restrict the selected split to one or more parameter-pair IDs.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production closure rollout requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    checkpoint: dict[str, Any] | None = None
    history = 1
    if args.closure == "model":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for model closure")
        model, checkpoint = load_closure_checkpoint(args.checkpoint, device)
        history = int(checkpoint["history"])
    if not 0.0 <= args.hp_blend <= 1.0:
        parser.error("--hp-blend must be between 0 and 1")
    hp_scale = 0.0
    if args.closure == "hp" or args.hp_blend > 0.0:
        if args.baseline_summary is None:
            parser.error("--baseline-summary is required for HP or blended closure")
        hp_scale = float(json.loads(args.baseline_summary.read_text())["train_fit"]["hp_scale"])

    trajectories = load_pair_trajectories(args.run_dir, [args.split])
    if args.pair_id:
        requested = set(args.pair_id)
        trajectories = [item for item in trajectories if item.pair_id in requested]
        missing = requested - {item.pair_id for item in trajectories}
        if missing:
            raise ValueError(
                f"Requested pairs are absent from split {args.split!r}: {sorted(missing)}"
            )
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        start_index = history - 1
        stop_index = min(
            len(trajectory.time) - 1,
            int(np.searchsorted(trajectory.time, args.max_time, side="right") - 1),
        )
        primitive_state = torch.from_numpy(trajectory.state[start_index : start_index + 1]).to(device)
        if args.field_solver == "ampere":
            electric = torch.from_numpy(trajectory.electric[start_index : start_index + 1]).to(device)
            state = torch.cat((primitive_state, electric[:, None]), dim=1)
        else:
            state = primitive_state
        from landau_surrogate.fluid.multimoment_1d import spectral_filter
        state = spectral_filter(state, args.maximum_mode)
        k_value = torch.tensor([trajectory.k], device=device, dtype=state.dtype)
        recorded = [state[:, :3]]
        field_values = []
        kinetic_values = []
        total_values = []
        prefix = [
            torch.from_numpy(trajectory.state[index : index + 1]).to(device)
            for index in range(start_index)
        ]
        clamp_count = 0
        started = time.perf_counter()
        with torch.no_grad():
            energy_function = fluid_energies_ampere if args.field_solver == "ampere" else fluid_energies
            initial_energies = energy_function(state, k_value)
            field_values.append(initial_energies[0])
            kinetic_values.append(initial_energies[1])
            total_values.append(initial_energies[2])
            for output_index in range(start_index + 1, stop_index + 1):
                interval = float(trajectory.time[output_index] - trajectory.time[output_index - 1])
                substeps = max(1, int(round(interval / args.dt)))
                dt = interval / substeps
                for _ in range(substeps):
                    def closure_factory(stage: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
                        def closure(current: torch.Tensor) -> torch.Tensor:
                            if args.closure == "zero":
                                return torch.zeros_like(current[:, 0])
                            if args.closure == "hp":
                                return hp_gradient(current, k_value, hp_scale)
                            assert model is not None and checkpoint is not None
                            required = history - 1
                            history_states = prefix[-required:] if required else []
                            sequence = torch.stack(history_states + [current], dim=1) if required else current[:, None]
                            learned = predict_physical_gradient(
                                model, checkpoint, sequence, k_value
                            )
                            if args.hp_blend > 0.0:
                                hp_value = hp_gradient(current, k_value, hp_scale)
                                return (1.0 - args.hp_blend) * learned + args.hp_blend * hp_value
                            return learned
                        return closure
                    stepper = rk4_step_ampere if args.field_solver == "ampere" else rk4_step
                    raw_next = stepper(
                        state, k_value, dt, closure_factory,
                        density_floor=1.0e-4, pressure_floor=1.0e-5, clamp_output=False,
                        maximum_mode=args.maximum_mode,
                    )
                    clamp_count += int(torch.count_nonzero((raw_next[:, 0] < 1.0e-4) | (raw_next[:, 2] < 1.0e-5)))
                    components = [
                        torch.clamp(raw_next[:, 0], min=1.0e-4), raw_next[:, 1],
                        torch.clamp(raw_next[:, 2], min=1.0e-5),
                    ]
                    if args.field_solver == "ampere":
                        components.append(raw_next[:, 3])
                    state = torch.stack(components, dim=1)
                    if not torch.isfinite(state).all():
                        break
                if not torch.isfinite(state).all():
                    break
                recorded.append(state[:, :3])
                prefix.append(state[:, :3])
                energies = energy_function(state, k_value)
                field_values.append(energies[0])
                kinetic_values.append(energies[1])
                total_values.append(energies[2])
        runtime = time.perf_counter() - started
        prediction = torch.cat(recorded).cpu().numpy()
        field_energy = torch.cat(field_values).cpu().numpy()
        kinetic_energy = torch.cat(kinetic_values).cpu().numpy()
        total_energy = torch.cat(total_values).cpu().numpy()
        time_values = trajectory.time[start_index : start_index + len(prediction)]
        metrics = rollout_metrics(
            trajectory, time_values, prediction, field_energy, total_energy, start_index
        )
        row = {
            "closure": args.closure if args.hp_blend == 0.0 else f"model_hp_{args.hp_blend:.2f}",
            "field_solver": args.field_solver,
            "pair_id": trajectory.pair_id,
            "split": trajectory.split,
            "k": trajectory.k,
            "alpha": trajectory.alpha,
            "runtime_seconds": runtime,
            "clamp_count": clamp_count,
            "completed_time": float(time_values[-1]),
            **metrics,
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"{trajectory.pair_id}.npz",
            time=time_values,
            prediction=prediction,
            truth=trajectory.state[start_index : start_index + len(prediction)],
            field_energy=field_energy,
            truth_field_energy=trajectory.field_energy[start_index : start_index + len(prediction)],
            kinetic_energy=kinetic_energy,
            total_energy=total_energy,
            truth_total_energy=trajectory.total_energy[start_index : start_index + len(prediction)],
        )
        print(json.dumps(row), flush=True)

    temporary = args.output_dir / "pair_metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output_dir / "pair_metrics.csv")
    numeric = [
        "density_relative_l2", "velocity_relative_l2", "pressure_relative_l2",
        "field_energy_log10_rmse", "minimum_density", "minimum_pressure",
        "total_energy_relative_span", "runtime_seconds", "completed_time",
    ]
    summary = {
        "closure": args.closure if args.hp_blend == 0.0 else f"model_hp_{args.hp_blend:.2f}",
        "field_solver": args.field_solver,
        "split": args.split,
        "pair_count": len(rows),
        "macro": {name: float(np.mean([float(row[name]) for row in rows])) for name in numeric},
    }
    temporary_json = args.output_dir / "summary.json.tmp"
    temporary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary_json, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
