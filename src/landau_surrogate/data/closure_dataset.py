"""Datasets and normalization for PIC-derived heat-flux closure learning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STATE_NAMES = ("density", "velocity", "pressure")


def spectral_filter_numpy(value: np.ndarray, maximum_mode: int | None) -> np.ndarray:
    """Return a real periodic field with modes above the cutoff removed."""
    if maximum_mode is None:
        return value
    transformed = np.fft.rfft(value, axis=-1)
    transformed[..., int(maximum_mode) + 1 :] = 0.0
    return np.fft.irfft(transformed, n=value.shape[-1], axis=-1).astype(value.dtype)


@dataclass(frozen=True)
class ClosureStats:
    input_mean: tuple[float, ...]
    input_std: tuple[float, ...]
    gradient_mean: float
    gradient_std: float
    heat_flux_mean: float
    heat_flux_std: float
    k_mean: float
    k_std: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_mean": list(self.input_mean),
            "input_std": list(self.input_std),
            "gradient_mean": self.gradient_mean,
            "gradient_std": self.gradient_std,
            "heat_flux_mean": self.heat_flux_mean,
            "heat_flux_std": self.heat_flux_std,
            "k_mean": self.k_mean,
            "k_std": self.k_std,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClosureStats":
        return cls(
            input_mean=tuple(float(x) for x in value["input_mean"]),
            input_std=tuple(float(x) for x in value["input_std"]),
            gradient_mean=float(value["gradient_mean"]),
            gradient_std=float(value["gradient_std"]),
            heat_flux_mean=float(value["heat_flux_mean"]),
            heat_flux_std=float(value["heat_flux_std"]),
            k_mean=float(value["k_mean"]),
            k_std=float(value["k_std"]),
        )


@dataclass(frozen=True)
class PairTrajectory:
    pair_id: str
    split: str
    k: float
    alpha: float
    time: np.ndarray
    state: np.ndarray
    heat_flux: np.ndarray
    heat_flux_gradient: np.ndarray
    electric: np.ndarray
    field_energy: np.ndarray
    total_energy: np.ndarray
    seed_state_std: np.ndarray
    seed_gradient_std: np.ndarray
    source_paths: tuple[str, ...]


def _case_metadata(path: Path) -> tuple[float, float, int, str]:
    with h5py.File(path, "r") as handle:
        case = json.loads(str(handle.attrs["case_json"]))
        return (
            float(case["k"]),
            float(case["alpha"]),
            int(case["seed"]),
            str(handle.attrs["split"]),
        )


def discover_case_groups(run_dir: Path) -> dict[tuple[float, float], list[Path]]:
    groups: dict[tuple[float, float], list[Path]] = {}
    for path in sorted((run_dir / "cases").glob("*.h5")):
        k_value, alpha, _seed, _split = _case_metadata(path)
        groups.setdefault((k_value, alpha), []).append(path)
    if not groups:
        raise FileNotFoundError(f"No case HDF5 files under {run_dir / 'cases'}")
    return groups


def load_pair_trajectories(
    run_dir: Path,
    splits: Iterable[str] | None = None,
    average_seeds: bool = True,
) -> list[PairTrajectory]:
    selected = set(splits) if splits is not None else None
    trajectories: list[PairTrajectory] = []
    for (k_value, alpha), paths in sorted(discover_case_groups(run_dir).items()):
        states: list[np.ndarray] = []
        gradients: list[np.ndarray] = []
        heat_fluxes: list[np.ndarray] = []
        electrics: list[np.ndarray] = []
        field_energies: list[np.ndarray] = []
        total_energies: list[np.ndarray] = []
        split_names: set[str] = set()
        time: np.ndarray | None = None
        ordered_paths = sorted(paths, key=lambda p: _case_metadata(p)[2])
        for path in ordered_paths:
            with h5py.File(path, "r") as handle:
                split_names.add(str(handle.attrs["split"]))
                current_time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
                if time is None:
                    time = current_time
                elif not np.array_equal(time, current_time):
                    raise ValueError(f"Time-grid mismatch in {path}")
                states.append(
                    np.stack(
                        [np.asarray(handle[f"fluid/{name}"], dtype=np.float32) for name in STATE_NAMES],
                        axis=1,
                    )
                )
                gradients.append(np.asarray(handle["fluid/heat_flux_gradient"], dtype=np.float32))
                heat_fluxes.append(np.asarray(handle["fluid/heat_flux"], dtype=np.float32))
                electrics.append(np.asarray(handle["fields/electric"], dtype=np.float32))
                field_energies.append(np.asarray(handle["energies/field_energy"], dtype=np.float64))
                total_energies.append(np.asarray(handle["energies/total_energy"], dtype=np.float64))
        if len(split_names) != 1:
            raise ValueError(f"Pair {(k_value, alpha)} crosses splits: {split_names}")
        split = next(iter(split_names))
        if selected is not None and split not in selected:
            continue
        state_stack = np.stack(states)
        gradient_stack = np.stack(gradients)
        reducer = np.mean if average_seeds else lambda x, axis=0: x[0]
        pair_id = f"k{k_value:.3f}_a{alpha:.3f}".replace(".", "p")
        trajectories.append(
            PairTrajectory(
                pair_id=pair_id,
                split=split,
                k=k_value,
                alpha=alpha,
                time=np.asarray(time),
                state=np.asarray(reducer(state_stack, axis=0), dtype=np.float32),
                heat_flux=np.asarray(reducer(np.stack(heat_fluxes), axis=0), dtype=np.float32),
                heat_flux_gradient=np.asarray(reducer(gradient_stack, axis=0), dtype=np.float32),
                electric=np.asarray(reducer(np.stack(electrics), axis=0), dtype=np.float32),
                field_energy=np.asarray(reducer(np.stack(field_energies), axis=0), dtype=np.float64),
                total_energy=np.asarray(reducer(np.stack(total_energies), axis=0), dtype=np.float64),
                seed_state_std=np.asarray(np.std(state_stack, axis=0), dtype=np.float32),
                seed_gradient_std=np.asarray(np.std(gradient_stack, axis=0), dtype=np.float32),
                source_paths=tuple(str(path) for path in ordered_paths),
            )
        )
    return trajectories


def compute_closure_stats(trajectories: Iterable[PairTrajectory]) -> ClosureStats:
    items = list(trajectories)
    if not items:
        raise ValueError("Cannot compute normalization from an empty trajectory set")
    state = np.concatenate([item.state for item in items], axis=0).astype(np.float64)
    gradient = np.concatenate([item.heat_flux_gradient for item in items], axis=0).astype(np.float64)
    heat_flux = np.concatenate([item.heat_flux for item in items], axis=0).astype(np.float64)
    k_values = np.asarray([item.k for item in items], dtype=np.float64)
    input_mean = state.mean(axis=(0, 2))
    input_std = state.std(axis=(0, 2))
    input_std = np.maximum(input_std, 1.0e-8)
    return ClosureStats(
        input_mean=tuple(float(x) for x in input_mean),
        input_std=tuple(float(x) for x in input_std),
        gradient_mean=float(gradient.mean()),
        gradient_std=max(float(gradient.std()), 1.0e-8),
        heat_flux_mean=float(heat_flux.mean()),
        heat_flux_std=max(float(heat_flux.std()), 1.0e-8),
        k_mean=float(k_values.mean()),
        k_std=max(float(k_values.std()), 1.0e-8),
    )


class ClosureSnapshotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        trajectories: Iterable[PairTrajectory],
        stats: ClosureStats,
        target_kind: str = "gradient",
        history: int = 1,
        time_stride: int = 1,
        maximum_target_mode: int | None = None,
    ) -> None:
        self.trajectories = list(trajectories)
        self.stats = stats
        self.target_kind = target_kind
        self.history = int(history)
        self.time_stride = int(time_stride)
        self.maximum_target_mode = maximum_target_mode
        if target_kind not in {"gradient", "heat_flux"}:
            raise ValueError(target_kind)
        if self.history < 1 or self.time_stride < 1:
            raise ValueError("history and time_stride must be positive")
        self.index: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(self.trajectories):
            for time_index in range(self.history - 1, len(trajectory.time), self.time_stride):
                self.index.append((trajectory_index, time_index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory_index, time_index = self.index[index]
        trajectory = self.trajectories[trajectory_index]
        state = trajectory.state[time_index - self.history + 1 : time_index + 1]
        mean = np.asarray(self.stats.input_mean, dtype=np.float32)[None, :, None]
        std = np.asarray(self.stats.input_std, dtype=np.float32)[None, :, None]
        state_normalized = ((state - mean) / std).reshape(-1, state.shape[-1])
        filtered_heat_flux = spectral_filter_numpy(
            trajectory.heat_flux[time_index], self.maximum_target_mode
        )
        filtered_gradient = spectral_filter_numpy(
            trajectory.heat_flux_gradient[time_index], self.maximum_target_mode
        )
        target = (
            filtered_gradient
            if self.target_kind == "gradient"
            else filtered_heat_flux - np.mean(filtered_heat_flux)
        )
        target_mean = self.stats.gradient_mean if self.target_kind == "gradient" else self.stats.heat_flux_mean
        target_std = self.stats.gradient_std if self.target_kind == "gradient" else self.stats.heat_flux_std
        return {
            "state": torch.from_numpy(np.asarray(state_normalized, dtype=np.float32)),
            "k": torch.tensor((trajectory.k - self.stats.k_mean) / self.stats.k_std, dtype=torch.float32),
            "k_physical": torch.tensor(trajectory.k, dtype=torch.float32),
            "target": torch.from_numpy(np.asarray((target - target_mean) / target_std, dtype=np.float32)),
            "target_physical": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "gradient_physical": torch.from_numpy(
                np.asarray(filtered_gradient, dtype=np.float32)
            ),
            "trajectory_index": torch.tensor(trajectory_index, dtype=torch.int64),
            "time_index": torch.tensor(time_index, dtype=torch.int64),
        }


__all__ = [
    "ClosureSnapshotDataset",
    "ClosureStats",
    "PairTrajectory",
    "compute_closure_stats",
    "discover_case_groups",
    "load_pair_trajectories",
    "spectral_filter_numpy",
]
