"""Periodic one-dimensional Fourier neural operator for heat-flux closure."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(max(in_channels * out_channels, 1))
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        size = value.shape[-1]
        transformed = torch.fft.rfft(value.float(), dim=-1)
        count = min(self.modes, transformed.shape[-1])
        output = torch.zeros(
            value.shape[0], self.out_channels, transformed.shape[-1],
            device=value.device, dtype=torch.cfloat,
        )
        output[..., :count] = torch.einsum(
            "bim,iom->bom", transformed[..., :count], self.weight[..., :count]
        )
        return torch.fft.irfft(output, n=size, dim=-1).to(value.dtype)


class FNOBlock1d(nn.Module):
    def __init__(
        self,
        width: int,
        modes: int,
        normalization: str = "group",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, 1)
        if normalization not in ("group", "none"):
            raise ValueError(f"Unsupported FNO normalization {normalization!r}")
        if activation not in ("gelu", "relu"):
            raise ValueError(f"Unsupported FNO activation {activation!r}")
        self.normalization = normalization
        self.activation = activation
        self.norm = nn.GroupNorm(1, width) if normalization == "group" else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.norm(self.spectral(value) + self.local(value))
        return F.gelu(value) if self.activation == "gelu" else F.relu(value)


class ClosureFNO1d(nn.Module):
    def __init__(
        self,
        state_channels: int = 3,
        include_k: bool = True,
        width: int = 48,
        modes: int = 24,
        layers: int = 4,
        enforce_zero_mean: bool = True,
        normalization: str = "group",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.state_channels = int(state_channels)
        self.include_k = bool(include_k)
        self.width = int(width)
        self.modes = int(modes)
        self.layers = int(layers)
        self.enforce_zero_mean = bool(enforce_zero_mean)
        self.normalization = str(normalization)
        self.activation = str(activation)
        input_channels = state_channels + int(include_k)
        self.lift = nn.Conv1d(input_channels, width, 1)
        self.blocks = nn.ModuleList([
            FNOBlock1d(width, modes, self.normalization, self.activation)
            for _ in range(layers)
        ])
        project_activation: nn.Module = nn.GELU() if self.activation == "gelu" else nn.ReLU()
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1),
            project_activation,
            nn.Conv1d(width, 1, 1),
        )

    def forward(self, state: torch.Tensor, k_condition: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1] != self.state_channels:
            raise ValueError(
                f"Expected state [B,{self.state_channels},Nx], got {tuple(state.shape)}"
            )
        value = state
        if self.include_k:
            k_channel = k_condition.reshape(-1, 1, 1).expand(-1, 1, state.shape[-1])
            value = torch.cat((value, k_channel), dim=1)
        value = self.lift(value)
        for block in self.blocks:
            value = block(value)
        output = self.project(value)[:, 0]
        if self.enforce_zero_mean:
            output = output - output.mean(dim=-1, keepdim=True)
        return output


class FNOMap1d(nn.Module):
    """Small reusable periodic FNO map with an arbitrary output channel count."""

    def __init__(self, input_channels: int, output_channels: int, width: int, modes: int, layers: int) -> None:
        super().__init__()
        self.lift = nn.Conv1d(input_channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, output_channels, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.lift(value)
        for block in self.blocks:
            value = block(value)
        return self.project(value)


class DualHeadHistoryResidualFNO1d(nn.Module):
    """Current-state gradient FNO plus a gated history residual and heat-flux head."""

    def __init__(
        self,
        history_steps: int = 4,
        width: int = 64,
        modes: int = 8,
        layers: int = 4,
        history_gate_logit: float = -3.0,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.width = int(width)
        self.modes = int(modes)
        self.layers = int(layers)
        # Both branches receive normalized k and alpha as constant spatial channels.
        self.base = FNOMap1d(3 + 2, 1, width, modes, layers)
        self.history = FNOMap1d(3 * history_steps + 2, 2, width, modes, layers)
        self.history_gate_logit = nn.Parameter(torch.tensor(float(history_gate_logit)))

    @staticmethod
    def _conditions(condition: torch.Tensor, nx: int) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != 2:
            raise ValueError(f"Expected conditions [B,2], got {tuple(condition.shape)}")
        return condition.unsqueeze(-1).expand(-1, -1, nx)

    def forward(self, history: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 4 or history.shape[1:3] != (self.history_steps, 3):
            raise ValueError(
                f"Expected history [B,{self.history_steps},3,Nx], got {tuple(history.shape)}"
            )
        batch, _steps, _channels, nx = history.shape
        condition_field = self._conditions(condition, nx)
        current = history[:, -1]
        base_gradient = self.base(torch.cat((current, condition_field), dim=1))[:, 0]
        flattened = history.reshape(batch, self.history_steps * 3, nx)
        historical = self.history(torch.cat((flattened, condition_field), dim=1))
        residual_gradient = historical[:, 0]
        heat_flux = historical[:, 1]
        gradient = base_gradient + torch.sigmoid(self.history_gate_logit) * residual_gradient
        gradient = gradient - gradient.mean(dim=-1, keepdim=True)
        heat_flux = heat_flux - heat_flux.mean(dim=-1, keepdim=True)
        return gradient, heat_flux


def spectral_derivative(value: torch.Tensor, k_fundamental: torch.Tensor) -> torch.Tensor:
    """Differentiate periodic values on one physical period for each batch item."""
    nx = value.shape[-1]
    mode = torch.arange(nx // 2 + 1, device=value.device, dtype=value.dtype)
    wave_number = k_fundamental.reshape(-1, 1) * mode.reshape(1, -1)
    return torch.fft.irfft(
        1j * wave_number * torch.fft.rfft(value.float(), dim=-1), n=nx, dim=-1
    ).to(value.dtype)


def closure_from_physical(
    model: ClosureFNO1d,
    physical_state: torch.Tensor,
    k_physical: torch.Tensor,
    normalization: dict[str, object],
    target_kind: str,
) -> torch.Tensor:
    """Apply a trained closure to physical states with differentiable scaling."""
    if physical_state.ndim == 4:
        batch, history, channels, nx = physical_state.shape
        physical_state = physical_state.reshape(batch, history * channels, nx)
    history = physical_state.shape[1] // 3
    mean = torch.as_tensor(
        normalization["input_mean"], device=physical_state.device, dtype=physical_state.dtype
    ).repeat(history).reshape(1, -1, 1)
    std = torch.as_tensor(
        normalization["input_std"], device=physical_state.device, dtype=physical_state.dtype
    ).repeat(history).reshape(1, -1, 1)
    normalized_state = (physical_state - mean) / std
    k_mean = float(normalization["k_mean"])
    k_std = float(normalization["k_std"])
    normalized_k = (k_physical - k_mean) / k_std
    output = model(normalized_state, normalized_k)
    if target_kind == "gradient":
        gradient = output * float(normalization["gradient_std"]) + float(
            normalization["gradient_mean"]
        )
        return gradient - gradient.mean(dim=-1, keepdim=True)
    heat_flux = output * float(normalization["heat_flux_std"]) + float(
        normalization["heat_flux_mean"]
    )
    return spectral_derivative(heat_flux, k_physical)


__all__ = [
    "ClosureFNO1d",
    "DualHeadHistoryResidualFNO1d",
    "FNOMap1d",
    "SpectralConv1d",
    "closure_from_physical",
    "spectral_derivative",
]
