#!/usr/bin/env python3
"""
Stage 8D-1C conditional 2D FNO model and physics-aware losses.

Task
----
(k, alpha, t) -> normalized delta_f0(x, v, t)

The x direction is periodic and handled directly by Fourier modes. The velocity
direction is non-periodic; zero padding is applied only along v before every
spectral block and removed before the output heads.

Variants
--------
single_head
    One full-field output head.

dual_head
    Separate mean_delta(v,t) and zero-x-mean nonzero(x,v,t) heads. The final
    field is their exact sum:
        delta_f0 = mean_delta + nonzero
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


VALID_VARIANTS = {"single_head", "dual_head"}
VALID_CONDITION_MODES = {"basic", "phase_aware"}


@dataclass(frozen=True)
class LossFloors:
    relative_mse: float
    mode1_energy: float
    resonance_mse: float
    mean_delta_mse: float

    def as_dict(self) -> dict[str, float]:
        return {
            "relative_mse": float(self.relative_mse),
            "mode1_energy": float(self.mode1_energy),
            "resonance_mse": float(self.resonance_mse),
            "mean_delta_mse": float(self.mean_delta_mse),
        }


@dataclass(frozen=True)
class LossWeights:
    relative: float
    mode1: float
    resonance: float
    mean_delta: float

    def as_dict(self) -> dict[str, float]:
        return {
            "global_mse": 1.0,
            "relative": float(self.relative),
            "mode1": float(self.mode1),
            "resonance": float(self.resonance),
            "mean_delta": float(self.mean_delta),
        }


def condition_feature_dimension(
    condition_mode: str,
    time_harmonics: int,
    phase_harmonics: int,
) -> int:
    if condition_mode not in VALID_CONDITION_MODES:
        raise ValueError(condition_mode)
    if condition_mode == "basic":
        return 3
    return 10 + 2 * time_harmonics + 2 * phase_harmonics


def build_condition_features(
    normalized_condition: torch.Tensor,
    physical_condition: torch.Tensor,
    condition_mode: str,
    time_harmonics: int,
    phase_harmonics: int,
) -> torch.Tensor:
    """Build basic or phase-aware features using frozen physical ranges."""
    if normalized_condition.ndim != 2 or normalized_condition.shape[1] != 3:
        raise ValueError(
            f"normalized_condition must be [B,3], got "
            f"{tuple(normalized_condition.shape)}"
        )
    if physical_condition.shape != normalized_condition.shape:
        raise ValueError(
            "physical_condition must match normalized_condition."
        )
    if condition_mode not in VALID_CONDITION_MODES:
        raise ValueError(condition_mode)

    normalized = normalized_condition.float()
    if condition_mode == "basic":
        return normalized

    physical = physical_condition.float()
    k_norm = normalized[:, 0]
    alpha_norm = normalized[:, 1]
    time_norm = normalized[:, 2]

    k_value = physical[:, 0]
    alpha_value = physical[:, 1]
    time_value = physical[:, 2]

    time_unit = time_value / 15.0
    phase_proxy = k_value * time_value
    log_alpha = torch.log(
        torch.clamp(alpha_value, min=1.0e-6)
    )
    log_alpha = (log_alpha - math.log(0.005)) / (
        math.log(0.05) - math.log(0.005)
    )
    log_alpha = 2.0 * log_alpha - 1.0

    columns = [
        k_norm,
        alpha_norm,
        time_norm,
        k_norm * alpha_norm,
        k_norm * time_norm,
        alpha_norm * time_norm,
        time_norm * time_norm,
        alpha_norm * alpha_norm,
        phase_proxy,
        log_alpha,
    ]

    for harmonic in range(1, time_harmonics + 1):
        angle = 2.0 * math.pi * harmonic * time_unit
        columns.extend([torch.sin(angle), torch.cos(angle)])

    for harmonic in range(1, phase_harmonics + 1):
        angle = harmonic * phase_proxy
        columns.extend([torch.sin(angle), torch.cos(angle)])

    return torch.stack(columns, dim=1)


def build_coordinate_channels(
    normalized_x: torch.Tensor,
    velocity: torch.Tensor,
    x_coordinate_harmonics: int,
) -> torch.Tensor:
    """
    Return [C,Nx,Nv] periodic-x and non-periodic-v coordinate channels.
    """
    x_grid, v_grid = torch.meshgrid(
        normalized_x.float(),
        velocity.float(),
        indexing="ij",
    )
    v_norm = v_grid / 6.0
    channels = [
        v_norm,
        v_norm * v_norm,
    ]
    for harmonic in range(1, x_coordinate_harmonics + 1):
        angle = 2.0 * math.pi * harmonic * x_grid
        channels.extend([torch.sin(angle), torch.cos(angle)])
    return torch.stack(channels, dim=0)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        condition_mode: str,
        time_harmonics: int,
        phase_harmonics: int,
    ):
        super().__init__()
        if condition_mode not in VALID_CONDITION_MODES:
            raise ValueError(condition_mode)
        self.condition_mode = condition_mode
        self.time_harmonics = int(time_harmonics)
        self.phase_harmonics = int(phase_harmonics)
        input_dim = condition_feature_dimension(
            self.condition_mode,
            self.time_harmonics,
            self.phase_harmonics,
        )
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        normalized_condition: torch.Tensor,
        physical_condition: torch.Tensor,
    ) -> torch.Tensor:
        features = build_condition_features(
            normalized_condition,
            physical_condition,
            self.condition_mode,
            self.time_harmonics,
            self.phase_harmonics,
        )
        return self.network(features)


class SpectralConv2d(nn.Module):
    """
    2D spectral convolution with separate positive/negative x-mode weights.

    FFTs are always executed in float32. This avoids CUDA half/bfloat16 FFT
    restrictions for the non-power-of-two velocity grid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_v: int,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_v = int(modes_v)

        scale = 1.0 / math.sqrt(
            max(in_channels * out_channels, 1)
        )
        self.weight_positive = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_v,
                2,
            )
        )
        self.weight_negative = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_v,
                2,
            )
        )

    @staticmethod
    def _complex_weight(parameter: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(parameter.contiguous())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values_float = values.float()
        batch, _, nx, nv = values_float.shape
        transformed = torch.fft.rfft2(
            values_float,
            dim=(-2, -1),
            norm="ortho",
        )
        output = torch.zeros(
            batch,
            self.out_channels,
            nx,
            transformed.shape[-1],
            device=values.device,
            dtype=torch.cfloat,
        )

        modes_x = min(self.modes_x, nx // 2)
        modes_v = min(self.modes_v, transformed.shape[-1])
        positive_weight = self._complex_weight(
            self.weight_positive[:, :, :modes_x, :modes_v]
        )
        negative_weight = self._complex_weight(
            self.weight_negative[:, :, :modes_x, :modes_v]
        )

        output[:, :, :modes_x, :modes_v] = torch.einsum(
            "bixv,ioxv->boxv",
            transformed[:, :, :modes_x, :modes_v],
            positive_weight,
        )
        output[:, :, -modes_x:, :modes_v] = torch.einsum(
            "bixv,ioxv->boxv",
            transformed[:, :, -modes_x:, :modes_v],
            negative_weight,
        )
        return torch.fft.irfft2(
            output,
            s=(nx, nv),
            dim=(-2, -1),
            norm="ortho",
        )


class FNOBlock(nn.Module):
    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_v: int,
        norm_groups: int,
    ):
        super().__init__()
        self.spectral = SpectralConv2d(
            width, width, modes_x, modes_v
        )
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        groups = min(norm_groups, width)
        while width % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectral = self.spectral(values)
        pointwise = self.pointwise(values).float()
        update = F.gelu(spectral + pointwise)
        return self.norm(values.float() + update)


class ConditionalFNO2d(nn.Module):
    def __init__(
        self,
        normalized_x: np.ndarray | torch.Tensor,
        velocity: np.ndarray | torch.Tensor,
        variant: str,
        width: int = 32,
        layers: int = 4,
        modes_x: int = 16,
        modes_v: int = 24,
        v_padding: int = 16,
        condition_channels: int = 16,
        condition_hidden_dim: int = 128,
        condition_hidden_layers: int = 2,
        condition_mode: str = "phase_aware",
        time_harmonics: int = 8,
        phase_harmonics: int = 8,
        x_coordinate_harmonics: int = 2,
        norm_groups: int = 4,
    ):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(variant)
        if condition_mode not in VALID_CONDITION_MODES:
            raise ValueError(condition_mode)
        self.variant = variant
        self.condition_mode = condition_mode
        self.width = int(width)
        self.layers_count = int(layers)
        self.modes_x = int(modes_x)
        self.modes_v = int(modes_v)
        self.v_padding = int(v_padding)
        self.condition_channels = int(condition_channels)
        self.time_harmonics = int(time_harmonics)
        self.phase_harmonics = int(phase_harmonics)
        self.x_coordinate_harmonics = int(
            x_coordinate_harmonics
        )

        x_tensor = torch.as_tensor(
            normalized_x, dtype=torch.float32
        )
        v_tensor = torch.as_tensor(
            velocity, dtype=torch.float32
        )
        self.register_buffer(
            "normalized_x", x_tensor, persistent=True
        )
        self.register_buffer("velocity", v_tensor, persistent=True)
        coordinate_channels = build_coordinate_channels(
            x_tensor,
            v_tensor,
            self.x_coordinate_harmonics,
        )
        self.register_buffer(
            "coordinate_channels",
            coordinate_channels,
            persistent=True,
        )

        self.condition_encoder = ConditionEncoder(
            output_dim=self.condition_channels,
            hidden_dim=condition_hidden_dim,
            hidden_layers=condition_hidden_layers,
            condition_mode=self.condition_mode,
            time_harmonics=self.time_harmonics,
            phase_harmonics=self.phase_harmonics,
        )
        input_channels = (
            self.condition_channels
            + int(coordinate_channels.shape[0])
        )
        self.lift = nn.Conv2d(
            input_channels, self.width, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            [
                FNOBlock(
                    self.width,
                    self.modes_x,
                    self.modes_v,
                    norm_groups,
                )
                for _ in range(self.layers_count)
            ]
        )

        if self.variant == "single_head":
            self.full_head = nn.Sequential(
                nn.Conv2d(
                    self.width,
                    self.width,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Conv2d(self.width, 1, kernel_size=1),
            )
        else:
            self.nonzero_head = nn.Sequential(
                nn.Conv2d(
                    self.width,
                    self.width,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Conv2d(self.width, 1, kernel_size=1),
            )
            self.mean_head = nn.Sequential(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Conv1d(self.width, 1, kernel_size=1),
            )

    @property
    def field_shape(self) -> tuple[int, int]:
        return (
            int(self.normalized_x.numel()),
            int(self.velocity.numel()),
        )

    def _input_tensor(
        self,
        normalized_condition: torch.Tensor,
        physical_condition: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition_encoder(
            normalized_condition,
            physical_condition,
        )
        batch = condition.shape[0]
        nx, nv = self.field_shape
        condition_grid = condition[:, :, None, None].expand(
            batch, self.condition_channels, nx, nv
        )
        coordinates = self.coordinate_channels[None].expand(
            batch, -1, -1, -1
        )
        return torch.cat(
            [condition_grid, coordinates], dim=1
        )

    def forward(
        self,
        normalized_condition: torch.Tensor,
        physical_condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        values = self.lift(
            self._input_tensor(
                normalized_condition,
                physical_condition,
            )
        )
        if self.v_padding > 0:
            values = F.pad(
                values,
                (self.v_padding, self.v_padding, 0, 0),
                mode="constant",
                value=0.0,
            )
        for block in self.blocks:
            values = block(values)
        if self.v_padding > 0:
            values = values[
                :, :, :, self.v_padding : -self.v_padding
            ]

        if self.variant == "single_head":
            field = self.full_head(values).squeeze(1).float()
            mean_delta = torch.mean(field, dim=1)
            nonzero = field - mean_delta[:, None, :]
        else:
            raw_nonzero = self.nonzero_head(values).squeeze(1).float()
            nonzero = raw_nonzero - torch.mean(
                raw_nonzero, dim=1, keepdim=True
            )
            mean_latent = torch.mean(values.float(), dim=2)
            mean_delta = self.mean_head(
                mean_latent
            ).squeeze(1).float()
            field = nonzero + mean_delta[:, None, :]

        return {
            "field": field,
            "mean_delta": mean_delta,
            "nonzero": nonzero,
        }

    def contract(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "field_shape": list(self.field_shape),
            "width": self.width,
            "layers": self.layers_count,
            "modes_x": self.modes_x,
            "modes_v": self.modes_v,
            "v_padding": self.v_padding,
            "condition_channels": self.condition_channels,
            "condition_mode": self.condition_mode,
            "time_harmonics": self.time_harmonics,
            "phase_harmonics": self.phase_harmonics,
            "x_coordinate_harmonics": (
                self.x_coordinate_harmonics
            ),
            "coordinate_channels": int(
                self.coordinate_channels.shape[0]
            ),
        }


def compute_training_loss(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    phase_velocity: torch.Tensor,
    velocity: torch.Tensor,
    floors: LossFloors,
    weights: LossWeights,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    field = prediction["field"].float()
    target = target.float()
    error = field - target

    if sample_weights is None:
        normalized_weights = torch.ones(
            target.shape[0],
            device=target.device,
            dtype=torch.float32,
        )
    else:
        normalized_weights = sample_weights.float().reshape(-1)
        if normalized_weights.shape[0] != target.shape[0]:
            raise ValueError(
                "sample_weights must have one value per batch sample."
            )
        if not torch.isfinite(normalized_weights).all():
            raise ValueError("sample_weights contain non-finite values.")
        normalized_weights = torch.clamp(
            normalized_weights, min=1.0e-6
        )
    normalized_weights = normalized_weights / torch.clamp(
        torch.mean(normalized_weights), min=1.0e-6
    )

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * normalized_weights) / torch.sum(
            normalized_weights
        )

    per_sample_error_mse = torch.mean(
        error * error, dim=(1, 2)
    )
    per_sample_target_mse = torch.mean(
        target * target, dim=(1, 2)
    )
    global_mse = weighted_mean(per_sample_error_mse)
    relative_loss = weighted_mean(
        per_sample_error_mse
        / (
            per_sample_target_mse
            + float(floors.relative_mse)
        )
    )

    prediction_fft = torch.fft.rfft(
        field, dim=1, norm="forward"
    )
    target_fft = torch.fft.rfft(
        target, dim=1, norm="forward"
    )
    mode1_error = prediction_fft[:, 1] - target_fft[:, 1]
    mode1_error_energy = torch.mean(
        torch.abs(mode1_error) ** 2, dim=1
    )
    mode1_target_energy = torch.mean(
        torch.abs(target_fft[:, 1]) ** 2, dim=1
    )
    mode1_loss = weighted_mean(
        mode1_error_energy
        / (
            mode1_target_energy
            + float(floors.mode1_energy)
        )
    )

    resonance_mask = (
        torch.abs(
            velocity[None, :] - phase_velocity[:, None]
        )
        <= 0.5
    ).float()
    resonance_count = torch.clamp(
        torch.sum(resonance_mask, dim=1) * target.shape[1],
        min=1.0,
    )
    resonance_error = torch.sum(
        error * error * resonance_mask[:, None, :],
        dim=(1, 2),
    ) / resonance_count
    resonance_target = torch.sum(
        target * target * resonance_mask[:, None, :],
        dim=(1, 2),
    ) / resonance_count
    resonance_loss = weighted_mean(
        resonance_error
        / (
            resonance_target
            + float(floors.resonance_mse)
        )
    )

    target_mean = torch.mean(target, dim=1)
    predicted_mean = prediction["mean_delta"].float()
    mean_error_mse = torch.mean(
        (predicted_mean - target_mean) ** 2,
        dim=1,
    )
    mean_target_mse = torch.mean(
        target_mean * target_mean, dim=1
    )
    mean_delta_loss = weighted_mean(
        mean_error_mse
        / (
            mean_target_mse
            + float(floors.mean_delta_mse)
        )
    )

    total = (
        global_mse
        + float(weights.relative) * relative_loss
        + float(weights.mode1) * mode1_loss
        + float(weights.resonance) * resonance_loss
        + float(weights.mean_delta) * mean_delta_loss
    )
    components = {
        "total": total,
        "global_mse": global_mse,
        "relative": relative_loss,
        "mode1": mode1_loss,
        "resonance": resonance_loss,
        "mean_delta": mean_delta_loss,
    }
    return total, components


__all__ = [
    "ConditionalFNO2d",
    "LossFloors",
    "LossWeights",
    "VALID_CONDITION_MODES",
    "VALID_VARIANTS",
    "build_condition_features",
    "build_coordinate_channels",
    "compute_training_loss",
]


# Public, stage-neutral alias. The original class name is retained for
# checkpoint compatibility and traceability.
SnapshotFNO = ConditionalFNO2d
