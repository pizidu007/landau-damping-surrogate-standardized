"""Fit train-only closure baselines and evaluate one sealed split."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from landau_surrogate.data.closure_dataset import load_pair_trajectories
from landau_surrogate.data.paths import nonlinear_runs_root
from landau_surrogate.diagnostics.closure import closure_metrics, macro_average


def hp_unit(state: np.ndarray, k_value: float) -> np.ndarray:
    temperature_perturbation = state[:, 2] - state[:, 0]
    modes = np.arange(state.shape[-1] // 2 + 1, dtype=np.float64) * k_value
    return np.fft.irfft(
        np.abs(modes)[None] * np.fft.rfft(temperature_perturbation, axis=-1),
        n=state.shape[-1], axis=-1,
    ).astype(np.float32)


def fit_parameters(run_dir: Path) -> dict[str, object]:
    trajectories = load_pair_trajectories(run_dir, ["train"])
    hp_x = np.concatenate([hp_unit(item.state, item.k).reshape(-1) for item in trajectories])
    y = np.concatenate([item.heat_flux_gradient.reshape(-1) for item in trajectories]).astype(np.float64)
    hp_scale = float(np.dot(hp_x, y) / max(float(np.dot(hp_x, hp_x)), 1.0e-30))
    design = []
    targets = []
    for item in trajectories:
        state = item.state.transpose(0, 2, 1).reshape(-1, 3).astype(np.float64)
        state[:, 0] -= 1.0
        state[:, 2] -= 1.0
        design.append(np.column_stack((state, np.ones(len(state)))))
        targets.append(item.heat_flux_gradient.reshape(-1).astype(np.float64))
    x = np.concatenate(design)
    y_local = np.concatenate(targets)
    ridge = 1.0e-6 * np.eye(x.shape[1])
    local_coefficients = np.linalg.solve(x.T @ x + ridge, x.T @ y_local)
    return {"hp_scale": hp_scale, "local_coefficients": local_coefficients.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=nonlinear_runs_root() / "nonlinear_formal_v1")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = fit_parameters(args.run_dir)
    coefficients = np.asarray(parameters["local_coefficients"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for item in load_pair_trajectories(args.run_dir, [args.split]):
        local_state = item.state.transpose(0, 2, 1).astype(np.float64)
        local_state[..., 0] -= 1.0
        local_state[..., 2] -= 1.0
        local_design = np.concatenate((local_state, np.ones((*local_state.shape[:2], 1))), axis=-1)
        predictions = {
            "zero": np.zeros_like(item.heat_flux_gradient),
            "hp_calibrated": float(parameters["hp_scale"]) * hp_unit(item.state, item.k),
            "local_ridge": np.einsum("txc,c->tx", local_design, coefficients),
        }
        for model_name, prediction in predictions.items():
            rows.append({
                "model": model_name,
                "pair_id": item.pair_id,
                "split": item.split,
                "k": item.k,
                "alpha": item.alpha,
                **closure_metrics(prediction, item.heat_flux_gradient),
            })
    temporary = args.output_dir / "pair_metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output_dir / "pair_metrics.csv")
    metric_names = ["relative_l2", "rmse", "mae", "correlation", "spectral_relative_l2"]
    macro = {
        model: macro_average([row for row in rows if row["model"] == model], metric_names)
        for model in ("zero", "hp_calibrated", "local_ridge")
    }
    summary = {"split": args.split, "train_fit": parameters, "macro": macro}
    temporary_json = args.output_dir / "summary.json.tmp"
    temporary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary_json, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
