"""Batched, unclipped, time-dependent heat-flux oracle for Round 11."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from landau_surrogate.data.continuum_v1 import (
    load_continuum_case, load_continuum_case_index, spectral_lowpass,
)
from landau_surrogate.fluid.multimoment_1d import poisson_electric
from landau_surrogate.tools.audit_gkeyll_pde_solver import rk4_non_autonomous


ROOT = Path("/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def interpolate(values: np.ndarray, times: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Linear interpolation using the actual output-trigger times."""
    queries = np.asarray(queries, dtype=np.float64)
    if queries.min() < times[0] - 1e-8 or queries.max() > times[-1] + 1e-8:
        raise ValueError("Oracle interpolation must stay within kinetic data")
    upper = np.clip(np.searchsorted(times, queries, side="right"), 1, len(times) - 1)
    lower = upper - 1
    weight = ((queries - times[lower]) / (times[upper] - times[lower])).astype(np.float32)
    weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))
    return np.ascontiguousarray(values[lower] + weight * (values[upper] - values[lower]))


def select_cases(cases):
    """Deterministic train/validation coverage; no diagnostic test access."""
    selected = {}
    for regime in ("weak", "transition", "strong_nonlinear"):
        pool = sorted((c for c in cases if c.split == "validation" and c.regime == regime),
                      key=lambda c: (c.K, -c.alpha))
        for case in (pool[0], pool[-1]):
            selected[case.case_id] = case
    train = [c for c in cases if c.split == "train"]
    for key in (lambda c: (c.K, -c.alpha), lambda c: (-c.K, -c.alpha),
                lambda c: abs(c.K - 0.39) + abs(c.alpha - 0.12)):
        case = min(train, key=key)
        selected[case.case_id] = case
    return list(selected.values())


def electric_numpy(state: np.ndarray, k: np.ndarray) -> np.ndarray:
    modes = k[:, None] * np.arange(state.shape[-1] // 2 + 1)[None, :]
    charge = np.fft.rfft(1.0 - state[:, :, 0], axis=-1)
    result = np.zeros_like(charge)
    result[..., 1:] = charge[..., 1:] / (1j * modes[None, :, 1:])
    return np.fft.irfft(result, n=state.shape[-1], axis=-1)


def metrics(pred: np.ndarray, truth: np.ndarray, case, output_time: np.ndarray,
            failed_time: float | None) -> dict:
    valid = np.isfinite(pred).all(axis=(1, 2))
    count = int(valid.sum())
    row = {"case_id": case.case_id, "split": case.split, "regime": case.regime,
           "K": case.K, "alpha": case.alpha, "complete": bool(valid.all()),
           "failure_time": failed_time, "valid_output_count": count}
    if not count:
        return row
    p, t = pred[valid], truth[valid]
    energy = 0.5 * np.mean(p[:, 3] ** 2, axis=-1)
    target_energy = 0.5 * np.mean(t[:, 3] ** 2, axis=-1)
    initial_energy = max(float(np.mean(truth[0, 3] ** 2) / 2), 1e-30)
    floor = initial_energy * 1e-8
    field_error = float(np.sqrt(np.mean((np.log10(np.maximum(energy, floor))
                       - np.log10(np.maximum(target_energy, floor))) ** 2)))
    equilibrium = np.array([1., 0., 1., 0.])[None, :, None]
    perturbation_l2 = float(np.linalg.norm(p - t) / max(np.linalg.norm(t - equilibrium), 1e-12))
    total = 0.5 * np.mean(p[:, 2] + p[:, 0] * p[:, 1] ** 2 + p[:, 3] ** 2, axis=-1)
    initial_total = 0.5 * np.mean(pred[0, 2] + pred[0, 0] * pred[0, 1] ** 2 + pred[0, 3] ** 2)
    phat = np.fft.rfft(p[:, 3], axis=-1)[:, 1]
    that = np.fft.rfft(t[:, 3], axis=-1)[:, 1]
    signal = np.abs(that) > max(float(np.max(np.abs(that))) * 1e-4, 1e-12)
    predicted_signal = np.abs(phat) > max(float(np.max(np.abs(that))) * 1e-6, 1e-14)
    phase_error = np.abs(np.angle(phat * np.conj(that)))
    # No predicted signal means undefined phase, not a zero phase error.
    phase_error = np.where(predicted_signal,phase_error,np.pi)
    phase = float(np.mean(phase_error[signal])) if signal.any() else None
    row.update({"completed_time": float(output_time[valid][-1]),
                "full_t80_field_energy_log10_rmse": field_error if valid.all() else None,
                "prefix_field_energy_log10_rmse": field_error,
                "perturbation_relative_l2": perturbation_l2,
                "state_relative_l2": float(np.linalg.norm(p - t) / np.linalg.norm(t)),
                "electric_mode1_phase_mae": phase,
                "predicted_phase_missing_frames":int((signal & ~predicted_signal).sum()),
                "total_energy_max_relative_drift": float(np.max(np.abs(total - initial_total)) / abs(initial_total)),
                "mass_max_absolute_drift": float(np.max(np.abs(p[:, 0].mean(-1) - pred[0, 0].mean()))),
                "momentum_max_absolute_drift": float(np.max(np.abs((p[:, 0] * p[:, 1]).mean(-1)
                                                                           - (pred[0, 0] * pred[0, 1]).mean()))),
                "minimum_density": float(p[:, 0].min()), "minimum_pressure": float(p[:, 2].min())})
    return row


@torch.no_grad()
def run(cases, trajectories, mode, dt, t_end, output_dt, device, output, dtype=torch.float32):
    steps = round(t_end / dt)
    stride = round(output_dt / dt)
    if not np.isclose(steps * dt, t_end) or not np.isclose(stride * dt, output_dt):
        raise ValueError("dt must divide t_end and output_dt")
    grid = np.arange(2 * steps + 1, dtype=np.float64) * (dt / 2)
    output_time = np.arange(steps // stride + 1, dtype=np.float64) * output_dt
    gradients = np.stack([interpolate(spectral_lowpass(t.heat_flux_gradient, mode), t.time, grid)
                          for t in trajectories], axis=1)
    truth_moments = np.stack([interpolate(spectral_lowpass(t.state, mode), t.time, output_time)
                             for t in trajectories], axis=1)
    k_np = np.array([c.K for c in cases])
    truth_electric = electric_numpy(truth_moments, k_np)
    truth = np.concatenate((truth_moments, truth_electric[:, :, None]), axis=2).astype(np.float32)
    full_moments = np.stack([interpolate(t.state, t.time, output_time) for t in trajectories], axis=1)
    full_electric = electric_numpy(full_moments, k_np)
    full_truth = np.concatenate((full_moments, full_electric[:, :, None]), axis=2).astype(np.float32)
    state = torch.as_tensor(truth[0], device=device, dtype=dtype).clone()
    k = torch.as_tensor(k_np, dtype=state.dtype, device=device)
    g = torch.as_tensor(gradients, device=device, dtype=dtype)
    del gradients
    output_states = torch.full((len(output_time),) + state.shape, float("nan"), device=device, dtype=dtype)
    output_states[0] = state
    active = torch.ones(len(cases), dtype=torch.bool, device=device)
    failed_step = torch.full((len(cases),), steps + 1, dtype=torch.int64, device=device)
    equilibrium = torch.zeros_like(state)
    equilibrium[:, 0] = equilibrium[:, 2] = 1.
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps):
        value = rk4_non_autonomous(state, k, dt, g[2*step], g[2*step+1], g[2*step+2], mode, True)
        good = torch.isfinite(value).all(dim=(1, 2)) & (value[:, 0].amin(-1) > 0) & (value[:, 2].amin(-1) > 0)
        good &= value.abs().amax(dim=(1, 2)) < 100
        failed_step = torch.where(active & ~good, step + 1, failed_step)
        active &= good
        state = torch.where(active[:, None, None], value, equilibrium)
        if (step + 1) % stride == 0:
            output_states[(step + 1) // stride] = torch.where(active[:, None, None], state, float("nan"))
        if (step + 1) % max(1, round(10 / dt)) == 0:
            print(json.dumps({"mode": mode, "dt": dt, "time": (step+1)*dt,
                              "active": int(active.sum()), "seconds": time.perf_counter()-started}), flush=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    prediction = output_states.cpu().numpy()
    failures = failed_step.cpu().numpy()
    rows = [metrics(prediction[:, i], truth[:, i], c, output_time,
                    float(failures[i]*dt) if failures[i] <= steps else None) for i, c in enumerate(cases)]
    for i, row in enumerate(rows):
        row["filtered_truth_state_error_vs_full"] = float(np.linalg.norm(truth[:, i] - full_truth[:, i])
                                                          / np.linalg.norm(full_truth[:, i]))
        row["filtered_truth_gradient_tail_fraction"] = float(
            np.sum((trajectories[i].heat_flux_gradient - spectral_lowpass(trajectories[i].heat_flux_gradient, mode))**2)
            / max(np.sum(trajectories[i].heat_flux_gradient**2), 1e-30))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "rollout.npz", time=output_time, prediction=prediction,
                        truth=truth, full_truth=full_truth, case_ids=np.array([c.case_id for c in cases]))
    summary = {"mode": mode, "dt": dt, "t_end": t_end, "oracle": True, "dtype": str(dtype),
               "wall_seconds": seconds, "case_count": len(cases), "complete": sum(r["complete"] for r in rows),
               "clipping": False, "field_solver": "ampere", "case_metrics": rows}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modes", default="8,16,24")
    parser.add_argument("--dts", default="0.002")
    parser.add_argument("--t-end", type=float, default=80.)
    parser.add_argument("--output-dt", type=float, default=0.1)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--case-selection",choices=["representative","validation"],default="representative")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Round 11 production oracle requires CUDA")
    indexed=load_continuum_case_index(args.dataset_root)
    cases = select_cases(indexed) if args.case_selection=="representative" else [c for c in indexed if c.split=="validation"]
    trajectories = [load_continuum_case(c) for c in cases]
    manifest = {"protocol": "round11_time_dependent_oracle", "test_used": False,"selection":args.case_selection,
                "cases": [{**asdict(c), "path": str(c.path)} for c in cases]}
    atomic_json(args.output_dir / "cases.json", manifest)
    print(json.dumps({"selected": [c.case_id for c in cases]}), flush=True)
    runs = []
    for mode in map(int, args.modes.split(",")):
        for dt in map(float, args.dts.split(",")):
            target = args.output_dir / f"mode{mode}_dt{dt:g}"
            marker = target / "summary.json"
            if marker.exists():
                previous = json.loads(marker.read_text())
                if previous["t_end"] != args.t_end:
                    raise ValueError("Existing run has a different end time")
                runs.append(previous)
            else:
                runs.append(run(cases, trajectories, mode, dt, args.t_end, args.output_dt, device, target,
                                getattr(torch, args.dtype)))
            atomic_json(args.output_dir / "summary.json", {**manifest, "runs": runs})


if __name__ == "__main__":
    main()
