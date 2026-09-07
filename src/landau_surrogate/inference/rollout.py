#!/usr/bin/env python3
"""Production inference for the frozen Stage 9B positivity-constrained rollout model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.rollout_cache import (
    load_case_sequence,
    split_case_indices,
)
from landau_surrogate.diagnostics.rollout import (
    build_snapshot_model,
    rollout_candidate,
)
from landau_surrogate.diagnostics.rollout import build_stepper
from landau_surrogate.physics.closure import sha256_file


def snap_to_closed_interval(
    value: float,
    lower: float,
    upper: float,
) -> tuple[float, bool]:
    scale = max(1.0, abs(lower), abs(upper))
    tolerance = max(
        1.0e-7,
        32.0 * float(np.finfo(np.float32).eps) * scale,
    )
    numeric = float(value)
    if lower - tolerance <= numeric < lower:
        return float(lower), True
    if upper < numeric <= upper + tolerance:
        return float(upper), True
    return numeric, False


def resolve_cache_initial_state(
    cache_path: Path,
    split: str,
    split_position: int,
    time_index: int,
) -> tuple[np.ndarray, float, float, float, str, int]:
    indices = split_case_indices(cache_path, split)
    if split_position < 0 or split_position >= len(indices):
        raise IndexError(split_position)
    case_index = int(indices[split_position])
    sequence = load_case_sequence(cache_path, case_index)
    if time_index < 0 or time_index >= len(sequence["phase_time"]):
        raise IndexError(time_index)
    return (
        np.asarray(sequence["field"][time_index], dtype=np.float32),
        float(sequence["k"]),
        float(sequence["alpha"]),
        float(sequence["phase_time"][time_index]),
        str(sequence["case_id"]),
        case_index,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--amp", choices=("none", "bf16", "fp16"), default="bf16"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-npz", type=Path)
    source.add_argument("--cache", type=Path)
    parser.add_argument(
        "--initial-key", default="normalized_delta_f0"
    )
    parser.add_argument("--k", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--start-time", type=float)
    parser.add_argument(
        "--cache-split",
        choices=("train", "val", "test"),
        default="val",
    )
    parser.add_argument("--cache-split-position", type=int, default=0)
    parser.add_argument("--cache-time-index", type=int, default=0)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument(
        "--allow-cache-mismatch",
        action="store_true",
        help=(
            "Allow a cache whose SHA256 differs from the checkpoint contract. "
            "Use only for an intentional compatibility experiment."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("stage") != "stage9b_final_positivity_rollout":
        raise RuntimeError("Not a frozen Stage 9B checkpoint.")
    contract = checkpoint["cache_contract"]
    bounds = contract["condition_bounds"]
    dt = float(checkpoint["dt"])

    if args.cache is not None:
        expected_cache_sha256 = str(contract.get("cache_sha256", ""))
        actual_cache_sha256 = sha256_file(args.cache)
        if (
            expected_cache_sha256
            and actual_cache_sha256 != expected_cache_sha256
            and not args.allow_cache_mismatch
        ):
            raise RuntimeError(
                "Cache SHA256 does not match the checkpoint contract: "
                f"expected={expected_cache_sha256}, "
                f"actual={actual_cache_sha256}. "
                "Use the historical_exact cache, or pass "
                "--allow-cache-mismatch for an intentional experiment."
            )
        (
            initial,
            k_value,
            alpha_value,
            start_time,
            case_id,
            case_index,
        ) = resolve_cache_initial_state(
            args.cache,
            args.cache_split,
            args.cache_split_position,
            args.cache_time_index,
        )
        source = {
            "type": "cache_case",
            "cache": str(args.cache),
            "cache_sha256": actual_cache_sha256,
            "split": args.cache_split,
            "split_position": args.cache_split_position,
            "case_id": case_id,
            "case_index": case_index,
            "time_index": args.cache_time_index,
        }
    else:
        if args.k is None or args.alpha is None or args.start_time is None:
            raise ValueError(
                "--k, --alpha and --start-time are required with NPZ."
            )
        with np.load(args.initial_npz, allow_pickle=False) as archive:
            if args.initial_key not in archive:
                raise KeyError(args.initial_key)
            initial = np.asarray(
                archive[args.initial_key], dtype=np.float32
            )
        k_value = float(args.k)
        alpha_value = float(args.alpha)
        start_time = float(args.start_time)
        source = {
            "type": "npz",
            "path": str(args.initial_npz),
            "key": args.initial_key,
        }

    expected_shape = tuple(
        checkpoint["model_contract"]["field_shape"]
    )
    if initial.shape != expected_shape:
        raise ValueError(
            f"Initial shape {initial.shape} != {expected_shape}"
        )
    if not np.isfinite(initial).all():
        raise ValueError("Initial state is non-finite.")

    original = {
        "k": k_value,
        "alpha": alpha_value,
        "start_time": start_time,
    }
    k_value, k_snap = snap_to_closed_interval(
        k_value, bounds["k_min"], bounds["k_max"]
    )
    alpha_value, alpha_snap = snap_to_closed_interval(
        alpha_value,
        bounds["alpha_min"],
        bounds["alpha_max"],
    )
    start_time, time_snap = snap_to_closed_interval(
        start_time,
        bounds["time_min"],
        bounds["time_max"],
    )
    final_time = start_time + args.steps * dt
    outside = {}
    if not bounds["k_min"] <= k_value <= bounds["k_max"]:
        outside["k"] = k_value
    if not bounds["alpha_min"] <= alpha_value <= bounds["alpha_max"]:
        outside["alpha"] = alpha_value
    if not bounds["time_min"] <= start_time <= bounds["time_max"]:
        outside["start_time"] = start_time
    if final_time > bounds["time_max"]:
        outside["final_time"] = final_time
    if outside and not args.allow_extrapolation:
        raise ValueError(
            f"Rollout leaves frozen domain: {outside}. "
            "Use --allow-extrapolation explicitly."
        )

    device = (
        torch.device(args.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index or 0))
    model = build_stepper(checkpoint, device)
    snapshot = build_snapshot_model(
        {
            "model_config": checkpoint["snapshot_model_config"],
            "model_state_dict": checkpoint[
                "snapshot_model_state_dict"
            ],
            "cache_contract": contract,
        },
        device,
    )
    rollout = rollout_candidate(
        model,
        initial,
        np.asarray([k_value], dtype=np.float32),
        np.asarray([alpha_value], dtype=np.float32),
        start_time,
        args.steps,
        dt,
        bounds,
        device,
        snapshot_model=snapshot,
        mean_anchor_beta=float(checkpoint["mean_anchor_beta"]),
        amp=args.amp,
    )[0]

    global_rms = float(contract["delta_global_rms"])
    delta_f0 = rollout * np.float32(global_rms)
    f0_train = contract["f0_train"].cpu().numpy().astype(np.float32)
    f_phase = delta_f0 + f0_train[None, None, :]
    mean_delta = np.mean(delta_f0, axis=1)
    nonzero = delta_f0 - mean_delta[:, None, :]
    phase_time = start_time + np.arange(
        args.steps + 1, dtype=np.float32
    ) * dt
    decomposition_error = float(
        np.max(
            np.abs(delta_f0 - (nonzero + mean_delta[:, None, :]))
        )
    )
    if not all(
        np.isfinite(array).all()
        for array in (rollout, delta_f0, f_phase, mean_delta, nonzero)
    ):
        raise RuntimeError("Non-finite inference output.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        normalized_delta_f0=rollout,
        delta_f0=delta_f0,
        f_phase=f_phase,
        mean_delta=mean_delta,
        nonzero=nonzero,
        phase_time=phase_time,
        normalized_x=contract["normalized_x"].cpu().numpy(),
        velocity=contract["velocity"].cpu().numpy(),
        source_x_index=contract["source_x_index"].cpu().numpy(),
        source_v_index=contract["source_v_index"].cpu().numpy(),
        k=np.asarray([k_value], dtype=np.float32),
        alpha=np.asarray([alpha_value], dtype=np.float32),
        dt=np.asarray([dt], dtype=np.float32),
    )
    snapped = {}
    for name, did_snap, value in (
        ("k", k_snap, k_value),
        ("alpha", alpha_snap, alpha_value),
        ("start_time", time_snap, start_time),
    ):
        if did_snap:
            snapped[name] = {
                "original": original[name],
                "snapped": value,
            }
    metadata = {
        "stage": checkpoint["stage"],
        "version": checkpoint["version"],
        "selected_candidate": checkpoint["selected_candidate"],
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "source_initial_state": source,
        "k": k_value,
        "alpha": alpha_value,
        "start_time": start_time,
        "steps": args.steps,
        "dt": dt,
        "final_time": final_time,
        "outside_training_domain": outside,
        "snapped_boundary_values": snapped,
        "rollout_shape": list(rollout.shape),
        "finite": True,
        "decomposition_max_abs": decomposition_error,
        "source_dataset_sha256": contract[
            "source_dataset_sha256"
        ],
        "cache_sha256": contract["cache_sha256"],
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
