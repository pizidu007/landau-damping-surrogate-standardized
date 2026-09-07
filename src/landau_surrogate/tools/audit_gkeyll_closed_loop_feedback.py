"""Measure how a closure changes after its fluid rollout leaves the truth manifold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.huang2025 import load_huang_mat
from landau_surrogate.models.closure_fno1d import DualHeadHistoryResidualFNO1d
from landau_surrogate.training.gkeyll_multicase import lowpass_numpy


def interpolate(time: np.ndarray, value: np.ndarray, query: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(time, query, side="right")
    upper = np.clip(upper, 1, len(time) - 1)
    lower = upper - 1
    weight = ((query - time[lower]) / np.maximum(time[upper] - time[lower], 1.0e-14)).astype(np.float32)
    return value[lower] * (1.0 - weight[:, None]) + value[upper] * weight[:, None]


def make_history(state: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    index = np.arange(len(state))
    return np.stack([state[np.maximum(index - lag, 0)] for lag in lags], axis=1)


def rel(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-12))


@torch.no_grad()
def predict(model, history, condition, normalization, batch_size, device):
    mean = np.asarray(normalization["input_mean"], dtype=np.float32)
    std = np.asarray(normalization["input_std"], dtype=np.float32)
    normalized = ((history - mean[None, None, :, None]) / std[None, None, :, None]).astype(np.float32)
    output = []
    condition_tensor = torch.from_numpy(condition.astype(np.float32)).to(device)[None]
    for start in range(0, len(normalized), batch_size):
        batch = torch.from_numpy(normalized[start:start + batch_size]).to(device)
        gradient, _flux = model(batch, condition_tensor.expand(len(batch), -1))
        output.append((gradient * float(normalization["gradient_std"])).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Closed-loop feedback audit requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DualHeadHistoryResidualFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    normalization = checkpoint["normalization"]
    trajectory = load_huang_mat(args.trajectory)
    with np.load(args.rollout) as values:
        time = np.asarray(values["time"])
        rollout_state = np.asarray(values["prediction"], dtype=np.float32)
        truth_state = np.asarray(values["truth"], dtype=np.float32)
    maximum_mode = int(checkpoint["maximum_mode"])
    truth_state = lowpass_numpy(truth_state, maximum_mode)
    rollout_state = lowpass_numpy(rollout_state, maximum_mode)
    nominal = checkpoint.get(
        "history_time_offsets_nominal",
        [0.005 * value for value in checkpoint["history_frame_offsets"]],
    )
    dt = float(np.median(np.diff(time)))
    lags = tuple(int(np.floor(abs(float(value)) / dt + 0.5)) for value in nominal)
    condition_mean = np.asarray(normalization["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(normalization["condition_std"], dtype=np.float32)
    condition = (np.asarray([trajectory.k, trajectory.alpha], dtype=np.float32) - condition_mean) / condition_std
    truth_prediction = predict(model, make_history(truth_state, lags), condition,
                               normalization, args.batch_size, device)
    rollout_prediction = predict(model, make_history(rollout_state, lags), condition,
                                 normalization, args.batch_size, device)
    target = interpolate(trajectory.time, trajectory.heat_flux_gradient, time)
    target = lowpass_numpy(target, maximum_mode)
    state_std = np.asarray(normalization["input_std"], dtype=np.float32)[None, :, None]
    normalized_state_distance = np.sqrt(np.mean(((rollout_state - truth_state) / state_std) ** 2, axis=(1, 2)))

    result = {"case_id": args.trajectory.parent.parent.name, "dt": dt, "lags": list(lags), "windows": {}}
    for lower, upper in ((0, 10), (10, 20), (20, 30), (30, 40.0001)):
        selected = (time >= lower) & (time < upper)
        result["windows"][f"{lower}-{min(upper,40):g}"] = {
            "truth_manifold_closure_relative_l2": rel(truth_prediction[selected], target[selected]),
            "rollout_path_vs_truth_time_closure_relative_l2": rel(rollout_prediction[selected], target[selected]),
            "feedback_shift_relative_to_truth_target": rel(rollout_prediction[selected], truth_prediction[selected]),
            "normalized_state_distance_rms": float(np.sqrt(np.mean(normalized_state_distance[selected] ** 2))),
        }
    result["aggregate"] = {
        "truth_manifold_closure_relative_l2": rel(truth_prediction, target),
        "rollout_path_vs_truth_time_closure_relative_l2": rel(rollout_prediction, target),
        "feedback_shift_relative_to_truth_target": rel(rollout_prediction, truth_prediction),
        "normalized_state_distance_rms": float(np.sqrt(np.mean(normalized_state_distance ** 2))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
