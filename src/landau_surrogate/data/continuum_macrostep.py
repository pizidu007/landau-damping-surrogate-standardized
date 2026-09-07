"""Conservative multi-field windows for ``continuum_v1`` macro stepping.

The state contract is ``[M0, M1, M2, E]``.  History frames and rollout
targets are separated by the same macro stride so that training windows and
free rollouts have identical temporal semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from landau_surrogate.data.continuum_v1 import (
    ContinuumCase,
    spectral_lowpass,
)


STATE_CHANNELS = ("M0", "M1", "M2", "E")


@dataclass(frozen=True)
class MacrostepTrajectory:
    """One physical trajectory retained in memory for window sampling."""

    case: ContinuumCase
    state: np.ndarray
    time: np.ndarray
    x_over_l: np.ndarray

    def __post_init__(self) -> None:
        if self.state.ndim != 3 or self.state.shape[1] != len(STATE_CHANNELS):
            raise ValueError(
                f"state must be [time,{len(STATE_CHANNELS)},x], got "
                f"{self.state.shape}"
            )
        if self.time.shape != (self.state.shape[0],):
            raise ValueError("time and state frame counts differ")
        if self.x_over_l.shape != (self.state.shape[-1],):
            raise ValueError("x coordinate and state grid sizes differ")


def load_macrostep_trajectory(
    case: ContinuumCase,
    *,
    maximum_mode: int | None = None,
    temporal_stride: int = 1,
) -> MacrostepTrajectory:
    """Load raw conservative moments and electric field for one case."""
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")
    with h5py.File(case.path, "r") as handle:
        if str(handle.attrs.get("dataset_version", "")) != "continuum_v1":
            raise ValueError(f"Not a continuum_v1 trajectory: {case.path}")
        if str(handle.attrs.get("case_id", "")) != case.case_id:
            raise ValueError(f"case_id metadata mismatch in {case.path}")
        if int(handle.attrs.get("source_solver_use_gpu", 0)) != 1:
            raise ValueError(f"Non-CUDA trajectory is not eligible: {case.path}")
        k_value = float(handle.attrs["K"])
        alpha = float(handle.attrs["alpha"])
        if not np.isclose(k_value, case.K) or not np.isclose(alpha, case.alpha):
            raise ValueError(f"parameter metadata mismatch in {case.path}")
        raw = np.asarray(
            handle["diagnostics/raw_moments"][::temporal_stride],
            dtype=np.float32,
        )
        electric = np.asarray(
            handle["diagnostics/electric_field"][::temporal_stride],
            dtype=np.float32,
        )
        time = np.asarray(
            handle["diagnostics/time"][::temporal_stride], dtype=np.float64
        )
        x = np.asarray(handle["coordinates/x_cell"][:], dtype=np.float64)

    if raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError(f"Expected raw moments [time,x,4], got {raw.shape}")
    if electric.shape != raw.shape[:2] or time.shape != (raw.shape[0],):
        raise ValueError(f"state coordinate shape mismatch in {case.path}")
    if x.shape != (raw.shape[1],):
        raise ValueError(f"space coordinate shape mismatch in {case.path}")
    if not np.isfinite(raw).all() or not np.isfinite(electric).all():
        raise ValueError(f"non-finite state in {case.path}")
    if not np.isfinite(time).all() or not np.all(np.diff(time) > 0.0):
        raise ValueError(f"invalid time coordinate in {case.path}")

    state = np.concatenate(
        (np.moveaxis(raw[..., :3], -1, 1), electric[:, None, :]), axis=1
    )
    state = spectral_lowpass(state, maximum_mode)
    density = state[:, 0]
    pressure = state[:, 2] - state[:, 1] ** 2 / density
    if float(density.min()) <= 0.0 or float(pressure.min()) <= 0.0:
        raise ValueError(f"non-positive density or pressure in {case.path}")
    domain_length = 2.0 * np.pi / case.K
    return MacrostepTrajectory(
        case=case,
        state=np.ascontiguousarray(state, dtype=np.float32),
        time=time,
        x_over_l=x / domain_length,
    )


def compute_normalization(
    trajectories: Sequence[MacrostepTrajectory],
) -> dict[str, Any]:
    """Compute state and parameter statistics using training cases only."""
    if not trajectories:
        raise ValueError("training trajectories are empty")
    channel_sum = np.zeros(len(STATE_CHANNELS), dtype=np.float64)
    channel_square = np.zeros(len(STATE_CHANNELS), dtype=np.float64)
    scalar_count = 0
    for trajectory in trajectories:
        value = trajectory.state.astype(np.float64, copy=False)
        channel_sum += value.sum(axis=(0, 2))
        channel_square += np.square(value).sum(axis=(0, 2))
        scalar_count += value.shape[0] * value.shape[2]
    mean = channel_sum / scalar_count
    variance = np.maximum(channel_square / scalar_count - mean**2, 0.0)
    std = np.sqrt(variance)
    if float(std.min()) <= 1.0e-12:
        raise ValueError(f"degenerate state normalization: {std.tolist()}")

    k_values = np.asarray([item.case.K for item in trajectories], dtype=np.float64)
    alpha_values = np.asarray(
        [item.case.alpha for item in trajectories], dtype=np.float64
    )
    k_std = float(k_values.std())
    alpha_std = float(alpha_values.std())
    if min(k_std, alpha_std) <= 1.0e-12:
        raise ValueError("degenerate parameter normalization")
    return {
        "state_channels": list(STATE_CHANNELS),
        "state_mean": mean.astype(np.float32).tolist(),
        "state_std": std.astype(np.float32).tolist(),
        "K_mean": float(k_values.mean()),
        "K_std": k_std,
        "alpha_mean": float(alpha_values.mean()),
        "alpha_std": alpha_std,
        "delta_t_scale": 1.0,
    }


def normalize_state(value: torch.Tensor, normalization: dict[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(
        normalization["state_mean"], dtype=value.dtype, device=value.device
    )
    std = torch.as_tensor(
        normalization["state_std"], dtype=value.dtype, device=value.device
    )
    shape = (1,) * (value.ndim - 2) + (len(mean), 1)
    return (value - mean.reshape(shape)) / std.reshape(shape)


def denormalize_state(
    value: torch.Tensor, normalization: dict[str, Any]
) -> torch.Tensor:
    mean = torch.as_tensor(
        normalization["state_mean"], dtype=value.dtype, device=value.device
    )
    std = torch.as_tensor(
        normalization["state_std"], dtype=value.dtype, device=value.device
    )
    shape = (1,) * (value.ndim - 2) + (len(mean), 1)
    return value * std.reshape(shape) + mean.reshape(shape)


def normalized_condition(
    k_value: torch.Tensor,
    alpha: torch.Tensor,
    delta_t: torch.Tensor,
    normalization: dict[str, Any],
) -> torch.Tensor:
    """Return the condition contract ``[normalized K, normalized alpha, dt]``."""
    return torch.stack(
        (
            (k_value - float(normalization["K_mean"]))
            / float(normalization["K_std"]),
            (alpha - float(normalization["alpha_mean"]))
            / float(normalization["alpha_std"]),
            delta_t / float(normalization.get("delta_t_scale", 1.0)),
        ),
        dim=-1,
    )


class MacrostepWindowDataset(Dataset):
    """Lazy windows over in-memory physical trajectories.

    The latest history frame is followed by ``rollout_steps`` targets.  Every
    adjacent history/target frame is separated by ``macro_stride`` stored
    frames, which makes recurrent training identical to free rollout.
    """

    def __init__(
        self,
        trajectories: Sequence[MacrostepTrajectory],
        *,
        history_steps: int,
        macro_stride: int,
        rollout_steps: int = 1,
        sample_stride: int = 1,
    ) -> None:
        if history_steps < 1 or macro_stride < 1 or rollout_steps < 1:
            raise ValueError("history_steps, macro_stride, and rollout_steps must be positive")
        if sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        self.trajectories = list(trajectories)
        self.history_steps = int(history_steps)
        self.macro_stride = int(macro_stride)
        self.rollout_steps = int(rollout_steps)
        case_indices: list[np.ndarray] = []
        anchor_indices: list[np.ndarray] = []
        first_anchor = (self.history_steps - 1) * self.macro_stride
        for case_index, trajectory in enumerate(self.trajectories):
            last_anchor = (
                len(trajectory.time) - 1 - self.rollout_steps * self.macro_stride
            )
            if last_anchor < first_anchor:
                continue
            anchors = np.arange(
                first_anchor, last_anchor + 1, sample_stride, dtype=np.int32
            )
            anchor_indices.append(anchors)
            case_indices.append(
                np.full(len(anchors), case_index, dtype=np.int16)
            )
        if not anchor_indices:
            raise ValueError("no valid macrostep windows")
        self.case_indices = np.concatenate(case_indices)
        self.anchor_indices = np.concatenate(anchor_indices)

    def __len__(self) -> int:
        return int(len(self.anchor_indices))

    def __getitem__(self, index: int):
        case_index = int(self.case_indices[index])
        anchor = int(self.anchor_indices[index])
        trajectory = self.trajectories[case_index]
        history_indices = anchor - self.macro_stride * np.arange(
            self.history_steps - 1, -1, -1, dtype=np.int64
        )
        target_indices = anchor + self.macro_stride * np.arange(
            1, self.rollout_steps + 1, dtype=np.int64
        )
        history = np.ascontiguousarray(trajectory.state[history_indices])
        targets = np.ascontiguousarray(trajectory.state[target_indices])
        chain_indices = np.concatenate(([anchor], target_indices))
        delta_t = np.diff(trajectory.time[chain_indices]).astype(np.float32)
        return (
            torch.from_numpy(history),
            torch.from_numpy(targets),
            torch.tensor(trajectory.case.K, dtype=torch.float32),
            torch.tensor(trajectory.case.alpha, dtype=torch.float32),
            torch.from_numpy(delta_t),
            torch.tensor(case_index, dtype=torch.int64),
        )


__all__ = [
    "MacrostepTrajectory",
    "MacrostepWindowDataset",
    "STATE_CHANNELS",
    "compute_normalization",
    "denormalize_state",
    "load_macrostep_trajectory",
    "normalize_state",
    "normalized_condition",
]
