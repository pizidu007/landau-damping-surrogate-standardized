"""Run truth-seeded, then fully free conservative macro-step rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from landau_surrogate.data.continuum_macrostep import (
    denormalize_state,
    load_macrostep_trajectory,
    normalize_state,
    normalized_condition,
)
from landau_surrogate.data.continuum_v1 import load_continuum_case_index
from landau_surrogate.diagnostics.macrostep import (
    aggregate_case_metrics,
    trajectory_metrics,
)
from landau_surrogate.models.macrostep_1d import build_macrostep_model
from landau_surrogate.physics.macrostep import (
    pressure_from_raw_moments,
    project_conservative_state,
)
from landau_surrogate.training.continuum_v1_macrostep import select_cases


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@torch.inference_mode()
def run_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to open the test split without --allow-test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "continuum_v1_conservative_macrostep":
        raise ValueError("checkpoint is not a continuum_v1 macrostep model")
    config = checkpoint["config"]
    data_config = config["data"]
    projection_config = config.get("projection", {})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_macrostep_model(
        checkpoint["model_kind"], checkpoint["model_arguments"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalization = checkpoint["normalization"]
    history_steps = int(checkpoint["model_arguments"]["history_steps"])
    macro_stride = int(data_config["macro_stride"])

    all_cases = load_continuum_case_index(data_config["dataset_root"])
    cases = select_cases(
        [case for case in all_cases if case.split == args.split], args.case_limit
    )
    expected_ids = set(checkpoint["split_case_ids"][args.split])
    if not {case.case_id for case in cases}.issubset(expected_ids):
        raise ValueError("rollout split differs from checkpoint contract")
    trajectories = [
        load_macrostep_trajectory(
            case,
            maximum_mode=data_config.get("maximum_mode"),
            temporal_stride=int(data_config.get("temporal_stride", 1)),
        )
        for case in cases
    ]
    frame_counts = {len(item.time) for item in trajectories}
    grid_sizes = {item.state.shape[-1] for item in trajectories}
    if len(frame_counts) != 1 or len(grid_sizes) != 1:
        raise ValueError("batched rollouts require a common frame count and Nx")
    stored_indices = np.arange(0, len(trajectories[0].time), macro_stride)
    if args.maximum_time is not None:
        reference_time = trajectories[0].time[stored_indices]
        stored_indices = stored_indices[reference_time <= args.maximum_time + 1.0e-9]
    if len(stored_indices) <= history_steps:
        raise ValueError("rollout horizon is shorter than the history seed")

    truth = np.stack(
        [item.state[stored_indices] for item in trajectories], axis=0
    )
    times = np.stack(
        [item.time[stored_indices] for item in trajectories], axis=0
    )
    prediction = np.full_like(truth, np.nan)
    prediction[:, :history_steps] = truth[:, :history_steps]
    history_physical = torch.as_tensor(
        truth[:, :history_steps], dtype=torch.float32, device=device
    )
    history_normalized = normalize_state(history_physical, normalization)
    k_value = torch.tensor(
        [case.K for case in cases], dtype=torch.float32, device=device
    )
    alpha = torch.tensor(
        [case.alpha for case in cases], dtype=torch.float32, device=device
    )
    active = torch.ones(len(cases), dtype=torch.bool, device=device)
    equilibrium = torch.zeros(
        (len(cases), 4, truth.shape[-1]), dtype=torch.float32, device=device
    )
    equilibrium[:, 0] = 1.0
    equilibrium[:, 2] = 1.0
    equilibrium_normalized = normalize_state(equilibrium, normalization)
    density_floor = float(args.density_floor)
    pressure_floor = float(args.pressure_floor)
    divergence_limit = float(args.divergence_limit)
    started = time.perf_counter()
    for output_index in range(history_steps, len(stored_indices)):
        delta_t = torch.as_tensor(
            times[:, output_index] - times[:, output_index - 1],
            dtype=torch.float32,
            device=device,
        )
        condition = normalized_condition(
            k_value, alpha, delta_t, normalization
        )
        raw_normalized = history_normalized[:, -1] + model(
            history_normalized, condition
        )
        raw = denormalize_state(raw_normalized, normalization)
        reference = denormalize_state(history_normalized[:, -1], normalization)
        next_state = project_conservative_state(
            raw,
            reference,
            k_value,
            preserve_mass=bool(projection_config.get("preserve_mass", True)),
            preserve_momentum=bool(
                projection_config.get("preserve_momentum", True)
            ),
            poisson_project_electric=bool(
                projection_config.get("poisson_project_electric", True)
            ),
            maximum_mode=projection_config.get("maximum_mode"),
        )
        density = next_state[:, 0]
        pressure = pressure_from_raw_moments(next_state)
        valid = (
            torch.isfinite(next_state).all(dim=(1, 2))
            & (torch.amax(torch.abs(next_state), dim=(1, 2)) < divergence_limit)
            & (torch.amin(density, dim=1) >= density_floor)
            & (torch.amin(pressure, dim=1) >= pressure_floor)
        )
        new_active = active & valid
        next_np = next_state.float().cpu().numpy()
        active_np = new_active.cpu().numpy()
        prediction[active_np, output_index] = next_np[active_np]
        safe_normalized = normalize_state(next_state, normalization)
        safe_normalized = torch.where(
            new_active[:, None, None], safe_normalized, equilibrium_normalized
        )
        history_normalized = torch.cat(
            (history_normalized[:, 1:], safe_normalized[:, None]), dim=1
        )
        active = new_active
        if output_index % 100 == 0 or output_index + 1 == len(stored_indices):
            print(
                json.dumps(
                    {
                        "event": "rollout",
                        "output_index": output_index,
                        "outputs": len(stored_indices),
                        "active_cases": int(active.sum()),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        if not bool(active.any()):
            break
    elapsed = time.perf_counter() - started

    rows = []
    for index, case in enumerate(cases):
        metrics = trajectory_metrics(
            prediction[index],
            truth[index],
            time=times[index],
            k_value=case.K,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
            divergence_limit=divergence_limit,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "K": case.K,
                "alpha": case.alpha,
                "regime": case.regime,
                **metrics,
            }
        )
    aggregate = aggregate_case_metrics(rows)
    summary = {
        "stage": "continuum_v1_macrostep_free_rollout",
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "case_count": len(cases),
        "history_steps": history_steps,
        "macro_stride": macro_stride,
        "truth_seed_frames": history_steps,
        "truth_feedback_after_seed": False,
        "output_count": int(len(stored_indices)),
        "model_calls": int(max(len(stored_indices) - history_steps, 0)),
        "wall_seconds": elapsed,
        "aggregate": aggregate,
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", summary)
    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        case_id=np.asarray([case.case_id for case in cases]),
        K=np.asarray([case.K for case in cases]),
        alpha=np.asarray([case.alpha for case in cases]),
        time=times,
        truth=truth,
        prediction=prediction,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--maximum-time", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--density-floor", type=float, default=1.0e-4)
    parser.add_argument("--pressure-floor", type=float, default=1.0e-5)
    parser.add_argument("--divergence-limit", type=float, default=1.0e6)
    return parser.parse_args()


def main() -> None:
    run_rollouts(parse_args())


if __name__ == "__main__":
    main()
