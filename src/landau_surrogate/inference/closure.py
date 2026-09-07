"""Checkpoint loading and batched evaluation for PIC heat-flux closures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from landau_surrogate.models.closure_fno1d import ClosureFNO1d, closure_from_physical


def load_closure_checkpoint(
    path: Path, device: torch.device
) -> tuple[ClosureFNO1d, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") not in {
        "pic_heat_flux_closure_supervised_v1",
        "pic_heat_flux_closure_rollout_v1",
    }:
        raise ValueError(f"Unsupported closure checkpoint stage: {checkpoint.get('stage')}")
    model = ClosureFNO1d(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_physical_gradient(
    model: ClosureFNO1d,
    checkpoint: dict[str, Any],
    state: torch.Tensor,
    k_value: torch.Tensor,
) -> torch.Tensor:
    return closure_from_physical(
        model,
        state,
        k_value,
        checkpoint["normalization"],
        str(checkpoint["target_kind"]),
    )


__all__ = ["load_closure_checkpoint", "predict_physical_gradient"]
