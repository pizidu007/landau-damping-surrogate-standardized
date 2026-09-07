"""History-conditioned residual models for conservative 1D macro stepping."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from landau_surrogate.models.closure_fno1d import FNOBlock1d


Projection = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _validate_inputs(
    history: torch.Tensor,
    condition: torch.Tensor,
    history_steps: int,
    state_channels: int,
) -> None:
    if history.ndim != 4 or history.shape[1:3] != (
        history_steps,
        state_channels,
    ):
        raise ValueError(
            f"Expected history [B,{history_steps},{state_channels},Nx], got "
            f"{tuple(history.shape)}"
        )
    if condition.ndim != 2 or condition.shape != (history.shape[0], 3):
        raise ValueError(
            f"Expected condition [B,3]=(K,alpha,delta_t), got "
            f"{tuple(condition.shape)}"
        )


class HistoryResidualFNO1d(nn.Module):
    """Periodic FNO mapping a field history to the next-state increment."""

    def __init__(
        self,
        *,
        history_steps: int = 4,
        state_channels: int = 4,
        width: int = 96,
        modes: int = 24,
        layers: int = 4,
        normalization: str = "group",
        activation: str = "gelu",
        zero_initialize_output: bool = True,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.state_channels = int(state_channels)
        self.width = int(width)
        self.modes = int(modes)
        self.layers = int(layers)
        input_channels = self.history_steps * self.state_channels + 3
        self.lift = nn.Conv1d(input_channels, self.width, 1)
        self.blocks = nn.ModuleList(
            [
                FNOBlock1d(
                    self.width,
                    self.modes,
                    normalization=normalization,
                    activation=activation,
                )
                for _ in range(self.layers)
            ]
        )
        nonlinearity: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.project_hidden = nn.Conv1d(self.width, self.width, 1)
        self.project_activation = nonlinearity
        self.project_output = nn.Conv1d(self.width, self.state_channels, 1)
        if zero_initialize_output:
            nn.init.zeros_(self.project_output.weight)
            nn.init.zeros_(self.project_output.bias)

    def forward(
        self, history: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        _validate_inputs(
            history, condition, self.history_steps, self.state_channels
        )
        batch, _steps, _channels, nx = history.shape
        flattened = history.reshape(batch, -1, nx)
        condition_field = condition.unsqueeze(-1).expand(-1, -1, nx)
        value = self.lift(torch.cat((flattened, condition_field), dim=1))
        for block in self.blocks:
            value = block(value)
        value = self.project_activation(self.project_hidden(value))
        return self.project_output(value)

    def predict_next(
        self,
        history: torch.Tensor,
        condition: torch.Tensor,
        projection: Projection | None = None,
    ) -> torch.Tensor:
        next_state = history[:, -1] + self(history, condition)
        return projection(next_state, history[:, -1]) if projection else next_state


class PeriodicResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 3, padding=1, padding_mode="circular"
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 3, padding=1, padding_mode="circular"
        )
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = F.gelu(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return F.gelu(value + residual)


class PeriodicUNet1d(nn.Module):
    """Parameter-matched convolutional alternative to the FNO macro stepper."""

    def __init__(
        self,
        *,
        history_steps: int = 4,
        state_channels: int = 4,
        base_width: int = 48,
        depth: int = 3,
        zero_initialize_output: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.history_steps = int(history_steps)
        self.state_channels = int(state_channels)
        self.base_width = int(base_width)
        self.depth = int(depth)
        input_channels = self.history_steps * self.state_channels + 3
        widths = [self.base_width * 2**index for index in range(self.depth + 1)]
        self.input_block = PeriodicResidualBlock1d(input_channels, widths[0])
        self.down_blocks = nn.ModuleList(
            [
                PeriodicResidualBlock1d(widths[index], widths[index + 1])
                for index in range(self.depth)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                PeriodicResidualBlock1d(
                    widths[index + 1] + widths[index], widths[index]
                )
                for index in range(self.depth - 1, -1, -1)
            ]
        )
        self.output = nn.Conv1d(widths[0], self.state_channels, 1)
        if zero_initialize_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self, history: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        _validate_inputs(
            history, condition, self.history_steps, self.state_channels
        )
        batch, _steps, _channels, nx = history.shape
        divisor = 2**self.depth
        if nx % divisor:
            raise ValueError(f"Nx={nx} must be divisible by 2**depth={divisor}")
        flattened = history.reshape(batch, -1, nx)
        condition_field = condition.unsqueeze(-1).expand(-1, -1, nx)
        value = self.input_block(torch.cat((flattened, condition_field), dim=1))
        skips = [value]
        for block in self.down_blocks:
            value = F.avg_pool1d(value, kernel_size=2, stride=2)
            value = block(value)
            skips.append(value)
        for block, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            value = F.interpolate(value, size=skip.shape[-1], mode="nearest")
            value = block(torch.cat((value, skip), dim=1))
        return self.output(value)

    def predict_next(
        self,
        history: torch.Tensor,
        condition: torch.Tensor,
        projection: Projection | None = None,
    ) -> torch.Tensor:
        next_state = history[:, -1] + self(history, condition)
        return projection(next_state, history[:, -1]) if projection else next_state


def build_macrostep_model(kind: str, arguments: dict) -> nn.Module:
    if kind == "fno":
        return HistoryResidualFNO1d(**arguments)
    if kind == "unet":
        return PeriodicUNet1d(**arguments)
    raise ValueError(f"Unknown macrostep model kind: {kind}")


__all__ = [
    "HistoryResidualFNO1d",
    "PeriodicUNet1d",
    "build_macrostep_model",
]
