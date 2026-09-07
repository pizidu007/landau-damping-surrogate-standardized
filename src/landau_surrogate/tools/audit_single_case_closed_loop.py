"""Diagnose teacher-forced versus free-running closure error for one trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.huang2025 import load_huang_mat
from landau_surrogate.fluid.multimoment_1d import derivative, spectral_filter
from landau_surrogate.models.closure_fno1d import ClosureFNO1d
from landau_surrogate.training.gkeyll_multicase import lowpass_numpy


def interpolate(time: np.ndarray, value: np.ndarray, query: np.ndarray) -> np.ndarray:
    upper = np.clip(np.searchsorted(time, query, side="right"), 1, len(time) - 1)
    lower = upper - 1
    weight = ((query - time[lower]) / np.maximum(time[upper] - time[lower], 1.0e-14)).astype(np.float32)
    return value[lower] * (1.0 - weight[:, None, None]) + value[upper] * weight[:, None, None]


def lowpass(value: np.ndarray, maximum_mode: int) -> np.ndarray:
    return lowpass_numpy(value, maximum_mode).astype(np.float32)


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-12))


def first_crossing(time: np.ndarray, value: np.ndarray, threshold: float) -> float | None:
    selected = np.flatnonzero(value >= threshold)
    return None if len(selected) == 0 else float(time[selected[0]])


@torch.no_grad()
def predict(
    model: ClosureFNO1d,
    state: np.ndarray,
    checkpoint: dict,
    maximum_mode: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    normalization = checkpoint["normalization"]
    mean = np.asarray(normalization["input_mean"], dtype=np.float32)[None, :, None]
    std = np.asarray(normalization["input_std"], dtype=np.float32)[None, :, None]
    normalized = ((state - mean) / std).astype(np.float32)
    target_mean = float(normalization["target_mean"])
    target_std = float(normalization["target_std"])
    output = []
    for start in range(0, len(state), batch_size):
        batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
        prediction = model(batch, torch.zeros(len(batch), device=device))
        prediction = prediction * target_std + target_mean
        prediction = spectral_filter(prediction[:, None], maximum_mode)[:, 0]
        output.append(prediction.cpu().numpy())
    return np.concatenate(output)


def band(value: np.ndarray, lower: int, upper: int) -> np.ndarray:
    transformed = np.fft.rfft(value, axis=-1)
    mode = np.arange(transformed.shape[-1])
    transformed[..., (mode < lower) | (mode > upper)] = 0.0
    return np.fft.irfft(transformed, n=value.shape[-1], axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-mode", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    trajectory = load_huang_mat(args.trajectory)
    with np.load(args.rollout) as values:
        time = np.asarray(values["time"], dtype=np.float64)
        rollout_state = np.asarray(values["prediction"], dtype=np.float32)
        truth_state = np.asarray(values["truth"], dtype=np.float32)
        field_energy = np.asarray(values["field_energy"], dtype=np.float32)
        truth_field_energy = np.asarray(values["truth_field_energy"], dtype=np.float32)

    truth_state = lowpass(truth_state, args.maximum_mode)
    rollout_state = lowpass(rollout_state, args.maximum_mode)
    target = interpolate(
        trajectory.time,
        trajectory.heat_flux_gradient[:, None, :],
        time,
    )[:, 0]
    target = lowpass(target, args.maximum_mode)
    truth_prediction = predict(
        model, truth_state, checkpoint, args.maximum_mode, args.batch_size, device
    )
    rollout_prediction = predict(
        model, rollout_state, checkpoint, args.maximum_mode, args.batch_size, device
    )

    state_std = np.asarray(checkpoint["normalization"]["input_std"], dtype=np.float32)[None, :, None]
    normalized_state_distance = np.sqrt(
        np.mean(((rollout_state - truth_state) / state_std) ** 2, axis=(1, 2))
    )
    target_rms = max(float(np.sqrt(np.mean(target**2))), 1.0e-12)
    truth_closure_nrmse = np.sqrt(np.mean((truth_prediction - target) ** 2, axis=1)) / target_rms
    rollout_closure_nrmse = np.sqrt(np.mean((rollout_prediction - target) ** 2, axis=1)) / target_rms
    feedback_nrmse = np.sqrt(np.mean((rollout_prediction - truth_prediction) ** 2, axis=1)) / target_rms
    field_floor = max(float(truth_field_energy[0]) * 1.0e-10, 1.0e-14)
    field_log_error = np.abs(
        np.log10(np.maximum(field_energy, field_floor))
        - np.log10(np.maximum(truth_field_energy, field_floor))
    )

    k = torch.full((len(time),), trajectory.k, device=device)
    truth_tensor = torch.from_numpy(truth_state).to(device)
    density, velocity, pressure = truth_tensor[:, 0], truth_tensor[:, 1], truth_tensor[:, 2]
    pressure_advection = (-velocity * derivative(pressure, k, args.maximum_mode)).cpu().numpy()
    pressure_compression = (-3.0 * pressure * derivative(velocity, k, args.maximum_mode)).cpu().numpy()
    pressure_total = pressure_advection + pressure_compression - target

    result: dict[str, object] = {
        "case": {"k": trajectory.k, "alpha": trajectory.alpha},
        "protocol": checkpoint.get("protocol"),
        "maximum_mode": args.maximum_mode,
        "offline_truth_manifold_relative_l2": relative_l2(truth_prediction, target),
        "free_path_closure_relative_l2": relative_l2(rollout_prediction, target),
        "feedback_shift_relative_l2": relative_l2(rollout_prediction, truth_prediction),
        "spectral_bands": {},
        "first_crossings": {
            "normalized_state_distance": {
                str(level): first_crossing(time, normalized_state_distance, level)
                for level in (0.001, 0.01, 0.05, 0.1)
            },
            "closure_nrmse_truth_path": {
                str(level): first_crossing(time, truth_closure_nrmse, level)
                for level in (0.01, 0.05, 0.1)
            },
            "closure_nrmse_free_path": {
                str(level): first_crossing(time, rollout_closure_nrmse, level)
                for level in (0.01, 0.05, 0.1, 0.5)
            },
            "feedback_nrmse": {
                str(level): first_crossing(time, feedback_nrmse, level)
                for level in (0.01, 0.05, 0.1, 0.5)
            },
            "field_log10_absolute_error": {
                str(level): first_crossing(time, field_log_error, level)
                for level in (0.01, 0.05, 0.1, 0.3)
            },
        },
        "windows": {},
    }
    spectral_bands = [(1, min(8, args.maximum_mode))]
    if args.maximum_mode >= 9:
        spectral_bands.append((9, args.maximum_mode))
    for lower, upper in spectral_bands:
        key = f"modes_{lower}_{upper}"
        result["spectral_bands"][key] = {
            "truth_path_relative_l2": relative_l2(
                band(truth_prediction, lower, upper), band(target, lower, upper)
            ),
            "free_path_relative_l2": relative_l2(
                band(rollout_prediction, lower, upper), band(target, lower, upper)
            ),
        }
    for lower, upper in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40.0001)):
        selected = (time >= lower) & (time < upper)
        closure_error = truth_prediction[selected] - target[selected]
        result["windows"][f"{lower:g}-{min(upper, 40):g}"] = {
            "truth_path_closure_relative_l2": relative_l2(
                truth_prediction[selected], target[selected]
            ),
            "free_path_closure_relative_l2": relative_l2(
                rollout_prediction[selected], target[selected]
            ),
            "feedback_shift_relative_l2": relative_l2(
                rollout_prediction[selected], truth_prediction[selected]
            ),
            "normalized_state_distance_rms": float(
                np.sqrt(np.mean(normalized_state_distance[selected] ** 2))
            ),
            "closure_error_rms_over_pressure_rhs_rms": float(
                np.sqrt(np.mean(closure_error**2))
                / max(float(np.sqrt(np.mean(pressure_total[selected] ** 2))), 1.0e-12)
            ),
            "pressure_term_rms": {
                "advection": float(np.sqrt(np.mean(pressure_advection[selected] ** 2))),
                "compression": float(np.sqrt(np.mean(pressure_compression[selected] ** 2))),
                "heat_flux_gradient": float(np.sqrt(np.mean(target[selected] ** 2))),
                "total_rhs": float(np.sqrt(np.mean(pressure_total[selected] ** 2))),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
