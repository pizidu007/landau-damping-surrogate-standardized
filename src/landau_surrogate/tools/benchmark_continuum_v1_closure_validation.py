"""Benchmark the frozen small-step closure baseline on validation trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from landau_surrogate.data.continuum_v1 import load_continuum_case_index
from landau_surrogate.tools.run_continuum_v1_test_rollouts import (
    load_initial_conditions,
    log_rmse,
    normalized_energy,
    run_rollout,
    save_rollout,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--output-dt", type=float, default=0.02)
    parser.add_argument("--max-time", type=float, default=80.0)
    parser.add_argument("--maximum-mode", type=int, default=8)
    parser.add_argument("--divergence-limit", type=float, default=1.0e5)
    parser.add_argument("--case-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0 or args.output_dt <= 0.0 or args.max_time <= 0.0:
        raise ValueError("time arguments must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("closure timing benchmark requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cases = sorted(
        (
            case
            for case in load_continuum_case_index(args.dataset_root)
            if case.split == "validation"
        ),
        key=lambda case: (case.K, case.alpha, case.case_id),
    )
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    output_time = (
        np.arange(
            int(np.floor(args.max_time / args.output_dt + 1.0e-9)) + 1,
            dtype=np.float64,
        )
        * args.output_dt
    )
    initial_state, initial_electric, initial_gradient, truth_energy = (
        load_initial_conditions(cases, output_time, args.maximum_mode)
    )
    started = time.perf_counter()
    predicted_energy, status = run_rollout(
        "fno",
        cases=cases,
        initial_state=initial_state,
        initial_electric=initial_electric,
        initial_gradient=initial_gradient,
        output_time=output_time,
        checkpoint=checkpoint,
        device=device,
        dt=args.dt,
        maximum_mode=args.maximum_mode,
        hp_scale=float(np.sqrt(8.0 / np.pi)),
        divergence_limit=args.divergence_limit,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started

    rows = []
    for index, case in enumerate(cases):
        valid = np.isfinite(predicted_energy[:, index])
        prediction = normalized_energy(predicted_energy[:, index, None])[:, 0]
        truth = normalized_energy(truth_energy[:, index, None])[:, 0]
        rows.append(
            {
                **status[index],
                "field_energy_log10_rmse": log_rmse(prediction[valid], truth[valid]),
            }
        )
    errors = np.asarray(
        [row["field_energy_log10_rmse"] for row in rows], dtype=np.float64
    )
    integration_steps = sum(
        max(
            1,
            int(
                math.ceil(
                    float(output_time[index] - output_time[index - 1]) / args.dt
                    - 1.0e-10
                )
            ),
        )
        for index in range(1, len(output_time))
    )
    model_calls = max(integration_steps - 1, 0) * 4
    summary = {
        "stage": "continuum_v1_frozen_closure_validation_benchmark",
        "split": "validation",
        "checkpoint": str(args.checkpoint.resolve()),
        "case_count": len(cases),
        "dt": args.dt,
        "output_dt": args.output_dt,
        "maximum_mode": args.maximum_mode,
        "integration_steps": integration_steps,
        "model_calls": model_calls,
        "wall_seconds": wall_seconds,
        "complete_case_count": int(sum(row["finite_to_final_time"] for row in rows)),
        "violation_free_case_count": int(sum(row["clamp_count"] == 0 for row in rows)),
        "field_energy_log10_rmse_median": float(np.nanmedian(errors)),
        "field_energy_log10_rmse_mean": float(np.nanmean(errors)),
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rollout(
        args.output_dir / "fno_rollout.npz",
        output_time=output_time,
        cases=cases,
        truth_energy=truth_energy,
        predicted_energy=predicted_energy,
    )
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
