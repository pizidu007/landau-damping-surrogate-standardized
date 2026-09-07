#!/usr/bin/env python3
"""Production rollout plus density/Poisson/electric/energy reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from landau_surrogate.data.rollout_cache import load_case_sequence, split_case_indices
from landau_surrogate.diagnostics.rollout import build_snapshot_model, build_stepper, rollout_candidate
from landau_surrogate.physics.closure import closure_from_normalized_delta, resolve_mean_anchor_beta


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def snap(value: float, lower: float, upper: float) -> float:
    tolerance = max(1.0e-7, 32.0 * np.finfo(np.float32).eps * max(1.0, abs(lower), abs(upper)))
    if lower - tolerance <= value < lower:
        return float(lower)
    if upper < value <= upper + tolerance:
        return float(upper)
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path)
    source.add_argument("--initial-npz", type=Path)
    parser.add_argument("--cache-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--cache-split-position", type=int, default=0)
    parser.add_argument("--cache-time-index", type=int, default=0)
    parser.add_argument("--initial-key", default="normalized_delta_f0")
    parser.add_argument("--k", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--allow-extrapolation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "model_config", "cache_contract", "dt",
        "snapshot_model_state_dict", "snapshot_model_config",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing}")
    mean_anchor_beta, mean_anchor_beta_source = resolve_mean_anchor_beta(checkpoint)
    contract = checkpoint["cache_contract"]
    bounds = contract["condition_bounds"]
    dt = float(checkpoint["dt"])

    if args.cache is not None:
        indices = split_case_indices(args.cache, args.cache_split)
        if args.cache_split_position < 0 or args.cache_split_position >= len(indices):
            raise IndexError(args.cache_split_position)
        case_index = int(indices[args.cache_split_position])
        sequence = load_case_sequence(args.cache, case_index)
        if args.cache_time_index < 0 or args.cache_time_index >= len(sequence["phase_time"]):
            raise IndexError(args.cache_time_index)
        initial = np.asarray(sequence["field"][args.cache_time_index], dtype=np.float32)
        k_value = float(sequence["k"])
        alpha_value = float(sequence["alpha"])
        start_time = float(sequence["phase_time"][args.cache_time_index])
        source = {
            "type": "cache", "path": str(args.cache), "split": args.cache_split,
            "split_position": args.cache_split_position, "case_index": case_index,
            "case_id": sequence["case_id"], "time_index": args.cache_time_index,
        }
    else:
        if args.k is None or args.alpha is None or args.start_time is None:
            raise ValueError("--k, --alpha and --start-time are required with --initial-npz")
        with np.load(args.initial_npz, allow_pickle=False) as archive:
            if args.initial_key not in archive:
                raise KeyError(args.initial_key)
            initial = np.asarray(archive[args.initial_key], dtype=np.float32)
        k_value = float(args.k)
        alpha_value = float(args.alpha)
        start_time = float(args.start_time)
        source = {"type": "npz", "path": str(args.initial_npz), "key": args.initial_key}

    expected_shape = tuple(int(item) for item in checkpoint["model_contract"]["field_shape"])
    if initial.shape != expected_shape:
        raise ValueError(f"Initial shape {initial.shape} != {expected_shape}")
    k_value = snap(k_value, float(bounds["k_min"]), float(bounds["k_max"]))
    alpha_value = snap(alpha_value, float(bounds["alpha_min"]), float(bounds["alpha_max"]))
    start_time = snap(start_time, float(bounds["time_min"]), float(bounds["time_max"]))
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
        raise ValueError(f"Rollout leaves frozen domain: {outside}. Use --allow-extrapolation explicitly.")

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index or 0))
    stepper = build_stepper(checkpoint, device)
    snapshot = build_snapshot_model(
        {
            "model_config": checkpoint["snapshot_model_config"],
            "model_state_dict": checkpoint["snapshot_model_state_dict"],
            "cache_contract": contract,
        },
        device,
    )
    rollout = rollout_candidate(
        stepper, initial, np.asarray([k_value], dtype=np.float32),
        np.asarray([alpha_value], dtype=np.float32), start_time, args.steps, dt,
        bounds, device, snapshot_model=snapshot,
        mean_anchor_beta=mean_anchor_beta, amp=args.amp,
    )[0]
    f0_train = contract["f0_train"].cpu().numpy().astype(np.float32)
    velocity = contract["velocity"].cpu().numpy().astype(np.float32)
    closure = closure_from_normalized_delta(
        rollout, f0_train, float(contract["delta_global_rms"]), velocity, k_value
    )
    phase_time = start_time + np.arange(args.steps + 1, dtype=np.float64) * dt
    arrays = {
        "normalized_delta_f0": rollout.astype(np.float32),
        "delta_f0": np.asarray(closure["delta_f0"], dtype=np.float32),
        "f_phase": np.asarray(closure["f_phase"], dtype=np.float32),
        "electron_density": np.asarray(closure["electron_density"], dtype=np.float32),
        "charge_density": np.asarray(closure["charge_density"], dtype=np.float32),
        "electric_field": np.asarray(closure["electric_field"], dtype=np.float32),
        "electric_mode_real": np.real(closure["electric_mode_complex"]).astype(np.float32),
        "electric_mode_imag": np.imag(closure["electric_mode_complex"]).astype(np.float32),
        "density_mode_real": np.real(closure["density_mode_complex"]).astype(np.float32),
        "density_mode_imag": np.imag(closure["density_mode_complex"]).astype(np.float32),
        "field_energy": np.asarray(closure["field_energy"], dtype=np.float64),
        "kinetic_energy": np.asarray(closure["kinetic_energy"], dtype=np.float64),
        "total_energy": np.asarray(closure["total_energy"], dtype=np.float64),
        "phase_time": phase_time,
        "normalized_x": contract["normalized_x"].cpu().numpy().astype(np.float32),
        "velocity": velocity,
        "k": np.asarray([k_value], dtype=np.float32),
        "alpha": np.asarray([alpha_value], dtype=np.float32),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Non-finite closure inference output.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "selected_candidate": checkpoint.get("selected_candidate"),
        "mean_anchor_beta": mean_anchor_beta,
        "mean_anchor_beta_source": mean_anchor_beta_source,
        "source": source,
        "shape": list(rollout.shape),
        "steps": args.steps,
        "dt": dt,
        "start_time": start_time,
        "final_time": final_time,
        "outside_training_domain": outside,
        "allow_extrapolation": args.allow_extrapolation,
        "finite": True,
        "poisson_residual_absolute_max": float(np.max(np.abs(closure["poisson_residual"]))),
        "mean_charge_absolute_max": float(np.max(np.abs(np.mean(closure["charge_density"], axis=-1)))),
        "total_energy_relative_span": float((np.max(closure["total_energy"]) - np.min(closure["total_energy"])) / max(abs(float(closure["total_energy"][0])), 1.0e-30)),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_safe(metadata), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
