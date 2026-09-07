"""Run and plot HP/FNO fluid closures for all held-out continuum_v1 cases."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from landau_surrogate.data.continuum_v1 import (
    ContinuumCase,
    load_continuum_case_index,
    spectral_derivative,
)
from landau_surrogate.fluid.multimoment_1d import (
    rk4_step_ampere,
    spectral_filter,
)
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


ClosureFactory = Callable[[torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]
COLORS = {"truth": "black", "hp": "#0072B2", "fno": "#D55E00"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_initial_conditions(
    cases: list[ContinuumCase], output_time: np.ndarray, maximum_mode: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = []
    electrics = []
    initial_gradients = []
    truth_energy = np.empty((len(output_time), len(cases)), dtype=np.float64)
    for index, case in enumerate(cases):
        with h5py.File(case.path, "r") as handle:
            central = np.asarray(
                handle["diagnostics/central_moments"][0], dtype=np.float32
            )
            electric = np.asarray(
                handle["diagnostics/electric_field"][0], dtype=np.float32
            )
            energy_time = np.asarray(
                handle["diagnostics/field_energy_time"], dtype=np.float64
            )
            stored_energy = np.asarray(
                handle["diagnostics/field_energy"], dtype=np.float64
            )
        density = central[:, 0]
        velocity = central[:, 1]
        pressure = density * central[:, 2]
        states.append(np.stack((density, velocity, pressure), axis=0))
        electrics.append(electric)
        initial_gradients.append(
            spectral_derivative(
                central[:, 3][None], case.K, maximum_mode=maximum_mode
            )[0]
        )
        if stored_energy.ndim == 2:
            stored_energy = stored_energy[:, 0]
        # Gkeyll stores integral(E^2); the fluid diagnostic uses 0.5*integral(E^2).
        # This factor cancels after normalization but is retained for unit consistency.
        truth_energy[:, index] = np.interp(
            output_time, energy_time, 0.5 * stored_energy
        )
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(electrics, dtype=np.float32),
        np.asarray(initial_gradients, dtype=np.float32),
        truth_energy,
    )


def normalized_energy(energy: np.ndarray) -> np.ndarray:
    denominator = np.maximum(energy[0:1], 1.0e-30)
    return energy / denominator


def log_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not np.any(valid):
        return float("nan")
    floor = 1.0e-10
    difference = np.log10(np.maximum(prediction[valid], floor)) - np.log10(
        np.maximum(truth[valid], floor)
    )
    return float(np.sqrt(np.mean(difference**2)))


@torch.no_grad()
def run_rollout(
    closure_name: str,
    *,
    cases: list[ContinuumCase],
    initial_state: np.ndarray,
    initial_electric: np.ndarray,
    initial_gradient: np.ndarray,
    output_time: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
    dt: float,
    maximum_mode: int,
    hp_scale: float,
    divergence_limit: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    state = torch.cat(
        (
            torch.from_numpy(initial_state).to(device),
            torch.from_numpy(initial_electric[:, None]).to(device),
        ),
        dim=1,
    )
    state = spectral_filter(state, maximum_mode)
    k_value = torch.tensor([case.K for case in cases], device=device)
    active = torch.ones(len(cases), dtype=torch.bool, device=device)
    clamp_count = torch.zeros(len(cases), dtype=torch.int64, device=device)
    completed_time = np.full(len(cases), float(output_time[0]), dtype=np.float64)
    energies = np.full((len(output_time), len(cases)), np.nan, dtype=np.float64)
    domain_length = 2.0 * torch.pi / k_value
    energies[0] = (
        0.5 * domain_length * state[:, 3].square().mean(dim=-1)
    ).cpu().numpy()
    initial_gradient_tensor = torch.from_numpy(initial_gradient).to(device)

    model = None
    mean = std = model_k = None
    gradient_mean = gradient_std = 0.0
    if closure_name == "fno":
        model = ClosureFNO1d(**checkpoint["model_arguments"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        normalization = checkpoint["normalization"]
        mean = torch.tensor(
            normalization["input_mean"], device=device
        ).reshape(1, 3, 1)
        std = torch.tensor(
            normalization["input_std"], device=device
        ).reshape(1, 3, 1)
        model_k = (k_value - float(normalization["k_mean"])) / float(
            normalization["k_std"]
        )
        gradient_mean = float(normalization["gradient_mean"])
        gradient_std = float(normalization["gradient_std"])

    def learned_factory(_stage: torch.Tensor):
        def closure(current: torch.Tensor) -> torch.Tensor:
            if closure_name == "hp":
                pressure_minus_density = current[:, 2] - current[:, 0]
                mode = torch.arange(
                    pressure_minus_density.shape[-1] // 2 + 1,
                    device=device,
                    dtype=pressure_minus_density.dtype,
                )
                absolute_wave_number = k_value[:, None] * mode[None]
                return hp_scale * torch.fft.irfft(
                    absolute_wave_number
                    * torch.fft.rfft(pressure_minus_density.float(), dim=-1),
                    n=pressure_minus_density.shape[-1],
                    dim=-1,
                )
            assert model is not None
            assert mean is not None and std is not None and model_k is not None
            return model((current - mean) / std, model_k) * gradient_std + gradient_mean

        return closure

    def truth_factory(_stage: torch.Tensor):
        def closure(_current: torch.Tensor) -> torch.Tensor:
            return initial_gradient_tensor

        return closure

    equilibrium = torch.zeros_like(state)
    equilibrium[:, 0] = 1.0
    equilibrium[:, 2] = 1.0
    output_index = 1
    integration_step = 0
    started = time.perf_counter()
    while output_index < len(output_time):
        interval = float(output_time[output_index] - output_time[output_index - 1])
        substeps = max(1, int(math.ceil(interval / dt - 1.0e-10)))
        step_dt = interval / substeps
        for _substep in range(substeps):
            integration_step += 1
            factory: ClosureFactory = (
                truth_factory if integration_step == 1 else learned_factory
            )
            raw = rk4_step_ampere(
                state,
                k_value,
                step_dt,
                factory,
                density_floor=1.0e-4,
                pressure_floor=1.0e-5,
                clamp_output=False,
                maximum_mode=maximum_mode,
            )
            finite = torch.isfinite(raw).all(dim=(1, 2))
            bounded = torch.amax(torch.abs(raw), dim=(1, 2)) < divergence_limit
            newly_active = active & finite & bounded
            density_bad = raw[:, 0] < 1.0e-4
            pressure_bad = raw[:, 2] < 1.0e-5
            clamp_count += torch.sum(density_bad | pressure_bad, dim=-1)
            safe = torch.stack(
                (
                    torch.clamp(raw[:, 0], min=1.0e-4),
                    raw[:, 1],
                    torch.clamp(raw[:, 2], min=1.0e-5),
                    raw[:, 3],
                ),
                dim=1,
            )
            state = torch.where(newly_active[:, None, None], safe, equilibrium)
            active = newly_active
        current_energy = (
            0.5 * domain_length * state[:, 3].square().mean(dim=-1)
        ).cpu().numpy()
        active_np = active.cpu().numpy()
        energies[output_index, active_np] = current_energy[active_np]
        completed_time[active_np] = output_time[output_index]
        if output_index % 200 == 0 or output_index + 1 == len(output_time):
            print(
                json.dumps(
                    {
                        "closure": closure_name,
                        "time": float(output_time[output_index]),
                        "active_cases": int(active.sum()),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        output_index += 1
        if not bool(active.any()):
            print(
                json.dumps(
                    {
                        "closure": closure_name,
                        "time": float(output_time[output_index - 1]),
                        "active_cases": 0,
                        "event": "all_cases_stopped",
                    }
                ),
                flush=True,
            )
            break

    rows = []
    clamp_np = clamp_count.cpu().numpy()
    for index, case in enumerate(cases):
        rows.append(
            {
                "case_id": case.case_id,
                "K": case.K,
                "alpha": case.alpha,
                "regime": case.regime,
                "closure": closure_name,
                "completed_time": float(completed_time[index]),
                "clamp_count": int(clamp_np[index]),
                "finite_to_final_time": bool(completed_time[index] >= output_time[-1]),
            }
        )
    return energies, rows


def save_rollout(
    path: Path,
    *,
    output_time: np.ndarray,
    cases: list[ContinuumCase],
    truth_energy: np.ndarray,
    predicted_energy: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        time=output_time,
        case_id=np.asarray([case.case_id for case in cases]),
        K=np.asarray([case.K for case in cases]),
        alpha=np.asarray([case.alpha for case in cases]),
        truth_field_energy=truth_energy,
        field_energy=predicted_energy,
    )


def plot_one_case(
    axis: plt.Axes,
    *,
    time_values: np.ndarray,
    truth: np.ndarray,
    hp: np.ndarray,
    fno: np.ndarray,
    case: ContinuumCase,
    show_labels: bool,
) -> tuple[float, float, float, float]:
    truth_normalized = normalized_energy(truth[:, None])[:, 0]
    hp_normalized = hp / max(hp[0], 1.0e-30)
    fno_normalized = fno / max(fno[0], 1.0e-30)
    hp_valid = np.flatnonzero(np.isfinite(hp_normalized))
    fno_valid = np.flatnonzero(np.isfinite(fno_normalized))
    common_last = min(int(hp_valid[-1]), int(fno_valid[-1]))
    common_slice = slice(0, common_last + 1)
    hp_error = log_rmse(hp_normalized[common_slice], truth_normalized[common_slice])
    fno_error = log_rmse(fno_normalized[common_slice], truth_normalized[common_slice])
    hp_full_error = log_rmse(hp_normalized, truth_normalized)
    fno_stop_time = float(time_values[fno_valid[-1]])
    axis.semilogy(time_values, truth_normalized, color=COLORS["truth"], lw=1.35)
    axis.semilogy(time_values, hp_normalized, color=COLORS["hp"], lw=1.0)
    axis.semilogy(time_values, fno_normalized, color=COLORS["fno"], lw=1.0)
    abbreviation = {"weak": "W", "transition": "T", "strong_nonlinear": "S"}[
        case.regime
    ]
    stop_label = (
        "FNO reached final time"
        if len(fno_valid) == len(time_values)
        else f"FNO stopped at t={fno_stop_time:.1f}"
    )
    axis.set_title(
        f"K={case.K:.3f}, A={case.alpha:.3f} [{abbreviation}]\n"
        f"pre-stop log-RMSE: HP {hp_error:.2f}, FNO {fno_error:.2f}\n"
        f"{stop_label}",
        fontsize=7.4,
    )
    axis.set_xlim(float(time_values[0]), float(time_values[-1]))
    axis.set_ylim(1.0e-8, 1.0e3)
    axis.grid(True, which="both", alpha=0.18)
    if show_labels:
        axis.set_xlabel("time")
        axis.set_ylabel("normalized field energy")
    return hp_error, fno_error, hp_full_error, fno_stop_time


def plot_results(
    output_dir: Path,
    *,
    cases: list[ContinuumCase],
    time_values: np.ndarray,
    truth_energy: np.ndarray,
    hp_energy: np.ndarray,
    fno_energy: np.ndarray,
) -> list[dict[str, Any]]:
    figure, axes = plt.subplots(6, 6, figsize=(18.0, 16.0), sharex=True, sharey=True)
    metric_rows = []
    individual_dir = output_dir / "individual_cases"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for index, (axis, case) in enumerate(zip(axes.ravel(), cases, strict=True)):
        hp_error, fno_error, hp_full_error, fno_stop_time = plot_one_case(
            axis,
            time_values=time_values,
            truth=truth_energy[:, index],
            hp=hp_energy[:, index],
            fno=fno_energy[:, index],
            case=case,
            show_labels=False,
        )
        metric_rows.append(
            {
                "case_id": case.case_id,
                "K": case.K,
                "alpha": case.alpha,
                "regime": case.regime,
                "comparison_time": fno_stop_time,
                "hp_prestop_field_energy_log10_rmse": hp_error,
                "fno_prestop_field_energy_log10_rmse": fno_error,
                "hp_full_t80_field_energy_log10_rmse": hp_full_error,
                "hp_completed_time": float(
                    time_values[np.flatnonzero(np.isfinite(hp_energy[:, index]))[-1]]
                ),
                "fno_completed_time": float(
                    time_values[np.flatnonzero(np.isfinite(fno_energy[:, index]))[-1]]
                ),
            }
        )
        row, column = divmod(index, 6)
        if column == 0:
            axis.set_ylabel("normalized energy", fontsize=8)
        if row == 5:
            axis.set_xlabel("time", fontsize=8)
        axis.tick_params(labelsize=7)

        single_figure, single_axis = plt.subplots(figsize=(7.2, 4.6))
        plot_one_case(
            single_axis,
            time_values=time_values,
            truth=truth_energy[:, index],
            hp=hp_energy[:, index],
            fno=fno_energy[:, index],
            case=case,
            show_labels=True,
        )
        single_axis.plot([], [], color=COLORS["truth"], label="Gkeyll truth")
        single_axis.plot([], [], color=COLORS["hp"], label="Fluid + HP")
        single_axis.plot([], [], color=COLORS["fno"], label="Fluid + FNO")
        single_axis.legend(frameon=False, ncol=3, loc="lower left")
        single_figure.tight_layout()
        single_figure.savefig(
            individual_dir / f"{case.case_id}_field_energy.png", dpi=185
        )
        plt.close(single_figure)

    handles = [
        plt.Line2D([], [], color=COLORS[name], lw=1.8, label=label)
        for name, label in (
            ("truth", "Gkeyll truth"),
            ("hp", "Fluid + HP"),
            ("fno", "Fluid + FNO"),
        )
    ]
    figure.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        frameon=False,
    )
    figure.suptitle(
        "All 36 held-out trajectories: normalized electric-field energy",
        y=0.997,
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    figure.savefig(output_dir / "all_test_field_energy_truth_hp_fno.png", dpi=190)
    plt.close(figure)
    return metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--output-dt", type=float, default=0.02)
    parser.add_argument("--max-time", type=float, default=80.0)
    parser.add_argument("--maximum-mode", type=int, default=32)
    parser.add_argument("--hp-scale", type=float, default=float(np.sqrt(8.0 / np.pi)))
    parser.add_argument("--divergence-limit", type=float, default=1.0e5)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0 or args.output_dt <= 0.0 or args.max_time <= 0.0:
        raise ValueError("time arguments must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("continuum_v1 closed-loop evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") not in {
        "continuum_v1_pure_supervised_fno",
        "continuum_v1_rollout_stability_fno",
    }:
        raise ValueError("Expected a continuum_v1 supervised closure checkpoint")
    supervised_maximum_mode = int(
        checkpoint["config"]["data"]["target_maximum_mode"]
    )
    if args.maximum_mode > supervised_maximum_mode:
        raise ValueError(
            "Rollout maximum mode cannot exceed the supervised target band"
        )

    cases = sorted(
        [
            case
            for case in load_continuum_case_index(args.dataset_root)
            if case.split == "test"
        ],
        key=lambda case: (case.K, case.alpha, case.case_id),
    )
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    output_time = np.arange(
        int(np.floor(args.max_time / args.output_dt + 1.0e-9)) + 1,
        dtype=np.float64,
    ) * args.output_dt
    initial_state, initial_electric, initial_gradient, truth_energy = (
        load_initial_conditions(cases, output_time, args.maximum_mode)
    )

    closure_results: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    for closure_name in ("hp", "fno"):
        rollout_path = args.output_dir / f"{closure_name}_rollout.npz"
        status_path = args.output_dir / f"{closure_name}_status.json"
        if args.reuse_existing and rollout_path.exists() and status_path.exists():
            with np.load(rollout_path) as stored:
                stored_time = np.asarray(stored["time"])
                stored_cases = [str(value) for value in stored["case_id"]]
                energy = np.asarray(stored["field_energy"])
            expected_cases = [case.case_id for case in cases]
            if not np.array_equal(stored_time, output_time) or stored_cases != expected_cases:
                raise ValueError(f"Stored {closure_name} rollout contract mismatch")
            rows = json.loads(status_path.read_text(encoding="utf-8"))
            print(f"reusing completed {closure_name} rollout", flush=True)
        else:
            energy, rows = run_rollout(
                closure_name,
                cases=cases,
                initial_state=initial_state,
                initial_electric=initial_electric,
                initial_gradient=initial_gradient,
                output_time=output_time,
                checkpoint=checkpoint,
                device=device,
                dt=args.dt,
                maximum_mode=args.maximum_mode,
                hp_scale=args.hp_scale,
                divergence_limit=args.divergence_limit,
            )
            save_rollout(
                rollout_path,
                output_time=output_time,
                cases=cases,
                truth_energy=truth_energy,
                predicted_energy=energy,
            )
            atomic_json(status_path, rows)
        closure_results[closure_name] = (energy, rows)

    metric_rows = plot_results(
        args.output_dir,
        cases=cases,
        time_values=output_time,
        truth_energy=truth_energy,
        hp_energy=closure_results["hp"][0],
        fno_energy=closure_results["fno"][0],
    )
    summary = {
        "protocol": {
            "split": "whole-trajectory held-out test",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_seed": checkpoint["seed"],
            "checkpoint_selection": (
                "minimum finite validation closed-loop composite score"
                if checkpoint.get("stage") == "continuum_v1_rollout_stability_fno"
                else "lowest validation relative_l2 across seeds"
            ),
            "field_solver": "ampere",
            "moment_formulation": "primitive_central_n_u_p",
            "dt": args.dt,
            "output_dt": args.output_dt,
            "maximum_mode": args.maximum_mode,
            "supervised_target_maximum_mode": supervised_maximum_mode,
            "hp_scale": args.hp_scale,
            "initial_closure": "truth_first_RK4_step",
            "energy_normalization": "each trajectory divided by its t=0 value",
        },
        "case_count": len(cases),
        "case_metrics": metric_rows,
        "hp_status": closure_results["hp"][1],
        "fno_status": closure_results["fno"][1],
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"case_count": len(cases), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
