#!/usr/bin/env python3
"""
Inference interface for the frozen Stage 8D-1D final FNO baseline.

The checkpoint is self-contained for the derived 128x193 grid. It stores:
- model state/config;
- normalized x and velocity grids;
- train-only global RMS;
- coarse f0_train(v);
- source-grid index mapping and source hashes.

Outputs
-------
normalized_delta_f0
    Model output in the Stage 8D-0 normalized representation.
delta_f0
    Physical-scale perturbation on the 128x193 derived grid.
f_phase
    f0_train + delta_f0 on the same derived grid.
mean_delta
    x-mean perturbation.
nonzero
    zero-x-mean component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from landau_surrogate.models.snapshot_fno import ConditionalFNO2d


def normalize(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return 2.0 * (value - lower) / (upper - lower) - 1.0


def build_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> ConditionalFNO2d:
    contract = checkpoint["cache_contract"]
    model = ConditionalFNO2d(
        normalized_x=contract["normalized_x"].cpu().numpy(),
        velocity=contract["velocity"].cpu().numpy(),
        **checkpoint["model_config"],
    )
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    return model.to(device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-extrapolation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("stage") != "stage8d1d_final_baseline":
        raise RuntimeError(
            "Checkpoint is not a frozen Stage 8D-1D baseline."
        )

    bounds = checkpoint["cache_contract"]["condition_bounds"]
    values = {
        "k": args.k,
        "alpha": args.alpha,
        "time": args.time,
    }
    ranges = {
        "k": (bounds["k_min"], bounds["k_max"]),
        "alpha": (
            bounds["alpha_min"],
            bounds["alpha_max"],
        ),
        "time": (bounds["time_min"], bounds["time_max"]),
    }
    outside = {
        name: value
        for name, value in values.items()
        if not (
            ranges[name][0]
            <= value
            <= ranges[name][1]
        )
    }
    if outside and not args.allow_extrapolation:
        raise ValueError(
            "Requested condition is outside the frozen training domain: "
            f"{outside}. Use --allow-extrapolation explicitly."
        )

    device = (
        torch.device(args.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(
            device.index if device.index is not None else 0
        )

    normalized_condition = torch.tensor(
        [
            [
                normalize(
                    args.k,
                    bounds["k_min"],
                    bounds["k_max"],
                ),
                normalize(
                    args.alpha,
                    bounds["alpha_min"],
                    bounds["alpha_max"],
                ),
                normalize(
                    args.time,
                    bounds["time_min"],
                    bounds["time_max"],
                ),
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    physical_condition = torch.tensor(
        [[args.k, args.alpha, args.time]],
        dtype=torch.float32,
        device=device,
    )

    model = build_model(checkpoint, device)
    with torch.no_grad():
        prediction = model(
            normalized_condition,
            physical_condition,
        )

    normalized_delta = (
        prediction["field"][0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    normalized_mean = (
        prediction["mean_delta"][0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    normalized_nonzero = (
        prediction["nonzero"][0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    cache_contract = checkpoint["cache_contract"]
    global_rms = float(cache_contract["delta_global_rms"])
    delta_f0 = normalized_delta * np.float32(global_rms)
    mean_delta = normalized_mean * np.float32(global_rms)
    nonzero = normalized_nonzero * np.float32(global_rms)
    f0_train = (
        cache_contract["f0_train"]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    f_phase = delta_f0 + f0_train[None, :]

    if not all(
        np.isfinite(array).all()
        for array in (
            normalized_delta,
            delta_f0,
            f_phase,
            mean_delta,
            nonzero,
        )
    ):
        raise RuntimeError("Inference produced non-finite values.")

    reconstruction_error = float(
        np.max(
            np.abs(
                normalized_delta
                - (
                    normalized_nonzero
                    + normalized_mean[None, :]
                )
            )
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        normalized_delta_f0=normalized_delta,
        delta_f0=delta_f0,
        f_phase=f_phase,
        normalized_mean_delta=normalized_mean,
        normalized_nonzero=normalized_nonzero,
        mean_delta=mean_delta,
        nonzero=nonzero,
        normalized_x=(
            cache_contract["normalized_x"]
            .cpu()
            .numpy()
            .astype(np.float32)
        ),
        velocity=(
            cache_contract["velocity"]
            .cpu()
            .numpy()
            .astype(np.float32)
        ),
        source_x_index=(
            cache_contract["source_x_index"]
            .cpu()
            .numpy()
            .astype(np.int32)
        ),
        source_v_index=(
            cache_contract["source_v_index"]
            .cpu()
            .numpy()
            .astype(np.int32)
        ),
        condition=np.asarray(
            [args.k, args.alpha, args.time],
            dtype=np.float32,
        ),
        normalized_condition=(
            normalized_condition[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        ),
        delta_global_rms=np.asarray(
            [global_rms], dtype=np.float32
        ),
    )
    metadata = {
        "stage": checkpoint["stage"],
        "version": checkpoint["version"],
        "selected_candidate": checkpoint[
            "selected_candidate"
        ],
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "condition": values,
        "normalized_condition": (
            normalized_condition[0].cpu().tolist()
        ),
        "outside_training_domain": outside,
        "field_shape": list(normalized_delta.shape),
        "decomposition_max_abs": reconstruction_error,
        "source_dataset_sha256": cache_contract[
            "source_dataset_sha256"
        ],
        "cache_sha256": cache_contract["cache_sha256"],
    }
    metadata_path = args.output.with_suffix(
        args.output.suffix + ".json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
