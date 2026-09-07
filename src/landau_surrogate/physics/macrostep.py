"""Differentiable physical constraints for conservative macro-step states."""

from __future__ import annotations

import torch

from landau_surrogate.fluid.multimoment_1d import poisson_electric, spectral_filter


def pressure_from_raw_moments(state: torch.Tensor) -> torch.Tensor:
    """Return central pressure from ``[M0,M1,M2,E]`` states."""
    density = state[..., 0, :]
    momentum = state[..., 1, :]
    second_moment = state[..., 2, :]
    return second_moment - momentum.square() / density


def project_conservative_state(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    k_value: torch.Tensor,
    *,
    preserve_mass: bool = True,
    preserve_momentum: bool = True,
    poisson_project_electric: bool = True,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    """Project a physical next state without applying positivity clamps.

    Mass and momentum means are inherited from the previous state.  For the
    periodic electrostatic system, Poisson projection makes the electric field
    exactly consistent with the predicted density and fixes its zero mode.
    """
    if prediction.ndim != 3 or prediction.shape[1] != 4:
        raise ValueError(f"Expected prediction [B,4,Nx], got {prediction.shape}")
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes differ")
    if k_value.shape != (prediction.shape[0],):
        raise ValueError("k_value must have one value per batch item")
    density, momentum, second_moment, electric = prediction.unbind(dim=1)
    if preserve_mass:
        density = density - density.mean(dim=-1, keepdim=True)
        density = density + reference[:, 0].mean(dim=-1, keepdim=True)
    if preserve_momentum:
        momentum = momentum - momentum.mean(dim=-1, keepdim=True)
        momentum = momentum + reference[:, 1].mean(dim=-1, keepdim=True)
    if poisson_project_electric:
        electric = poisson_electric(density, k_value)
    else:
        electric = electric - electric.mean(dim=-1, keepdim=True)
    result = torch.stack((density, momentum, second_moment, electric), dim=1)
    return spectral_filter(result, maximum_mode)


def positivity_penalty(
    state: torch.Tensor,
    *,
    density_floor: float = 1.0e-4,
    pressure_floor: float = 1.0e-5,
) -> torch.Tensor:
    density = state[:, 0]
    pressure = pressure_from_raw_moments(state)
    return (
        torch.relu(float(density_floor) - density).square().mean()
        + torch.relu(float(pressure_floor) - pressure).square().mean()
    )


def poisson_residual(
    state: torch.Tensor, k_value: torch.Tensor
) -> torch.Tensor:
    expected = poisson_electric(state[:, 0], k_value)
    scale = torch.clamp(expected.square().mean(), min=1.0e-10)
    return (state[:, 3] - expected).square().mean() / scale


__all__ = [
    "poisson_residual",
    "positivity_penalty",
    "pressure_from_raw_moments",
    "project_conservative_state",
]
