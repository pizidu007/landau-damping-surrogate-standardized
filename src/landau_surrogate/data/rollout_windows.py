#!/usr/bin/env python3
"""
Lazy multi-step window reader for Stage 8D-2B.

For a selected horizon H, each sample is

    field[t_0 : t_0 + H + 1]

from one case only. Case-level train/validation/test splits remain frozen.
Windows never cross case boundaries and all conditions correspond to the input
times used by the autoregressive stepper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_NAME_TO_CODE = {"train": 0, "val": 1, "test": 2}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class Stage8D2BWindowDataset(Dataset):
    """Process-safe lazy HDF5 reader for fixed-length rollout windows."""

    def __init__(
        self,
        cache_path: str | Path,
        split: str,
        horizon: int,
        case_ids: Sequence[str] | None = None,
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(self.cache_path)
        if split not in SPLIT_NAME_TO_CODE:
            raise ValueError(f"Unsupported split: {split}")
        if horizon < 1:
            raise ValueError("horizon must be positive.")

        self.split = split
        self.horizon = int(horizon)
        self._handle: h5py.File | None = None
        self._handle_pid: int | None = None

        with h5py.File(self.cache_path, "r") as handle:
            if str(handle.attrs.get("status", "")) != "COMPLETE":
                raise RuntimeError("Stage 8D-1A cache is not COMPLETE.")

            self.normalized_x = np.asarray(
                handle["grids"]["normalized_x"][...],
                dtype=np.float32,
            )
            self.velocity = np.asarray(
                handle["grids"]["velocity"][...],
                dtype=np.float32,
            )
            self.phase_time = np.asarray(
                handle["grids"]["phase_time"][...],
                dtype=np.float32,
            )
            self.case_ids = tuple(
                _decode(item)
                for item in handle["cases"]["case_id"][...]
            )
            self.case_split_code = np.asarray(
                handle["cases"]["split_code"][...],
                dtype=np.uint8,
            )
            self.case_group_code = np.asarray(
                handle["cases"]["group_code"][...],
                dtype=np.uint8,
            )
            self.case_phase_velocity = np.asarray(
                handle["cases"]["phase_velocity"][...],
                dtype=np.float32,
            )

            phase_count = int(self.phase_time.size)
            if self.horizon >= phase_count:
                raise ValueError(
                    f"horizon={self.horizon} must be < "
                    f"phase_count={phase_count}"
                )
            expected_samples = len(self.case_ids) * phase_count
            expected_shape = (
                expected_samples,
                self.normalized_x.size,
                self.velocity.size,
            )
            actual_shape = tuple(
                int(value)
                for value in handle["samples"]["field"].shape
            )
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected field shape: {actual_shape}"
                )

            case_index = np.asarray(
                handle["samples"]["case_index"][...],
                dtype=np.int32,
            )
            time_index = np.asarray(
                handle["samples"]["time_index"][...],
                dtype=np.int32,
            )
            if not np.array_equal(
                case_index,
                np.repeat(
                    np.arange(
                        len(self.case_ids), dtype=np.int32
                    ),
                    phase_count,
                ),
            ):
                raise RuntimeError(
                    "Cache sample ordering is not case-major."
                )
            if not np.array_equal(
                time_index,
                np.tile(
                    np.arange(phase_count, dtype=np.int32),
                    len(self.case_ids),
                ),
            ):
                raise RuntimeError(
                    "Cache time ordering is not contiguous."
                )

        selected = np.nonzero(
            self.case_split_code == SPLIT_NAME_TO_CODE[split]
        )[0].astype(np.int64)
        if case_ids is not None:
            requested = set(case_ids)
            selected = np.asarray(
                [
                    index
                    for index in selected
                    if self.case_ids[int(index)] in requested
                ],
                dtype=np.int64,
            )
        if selected.size == 0:
            raise RuntimeError("No cases selected.")

        self.selected_case_indices = selected
        start_count = self.phase_count - self.horizon
        window_case: list[int] = []
        window_start: list[int] = []
        for case_index_value in selected:
            for start_index in range(start_count):
                window_case.append(int(case_index_value))
                window_start.append(start_index)

        self.window_case_indices = np.asarray(
            window_case, dtype=np.int32
        )
        self.window_start_indices = np.asarray(
            window_start, dtype=np.int16
        )

    @property
    def phase_count(self) -> int:
        return int(self.phase_time.size)

    @property
    def field_shape(self) -> tuple[int, int]:
        return (
            int(self.normalized_x.size),
            int(self.velocity.size),
        )

    @property
    def dt(self) -> float:
        differences = np.diff(self.phase_time.astype(np.float64))
        if not np.allclose(differences, differences[0]):
            raise RuntimeError("Phase-time grid is not uniform.")
        return float(differences[0])

    def __len__(self) -> int:
        return int(self.window_case_indices.size)

    def _sample_index(
        self,
        case_index: int,
        time_index: int,
    ) -> int:
        return case_index * self.phase_count + time_index

    def _get_handle(self) -> h5py.File:
        pid = os.getpid()
        if (
            self._handle is None
            or self._handle_pid != pid
            or not self._handle.id.valid
        ):
            self.close()
            self._handle = h5py.File(self.cache_path, "r")
            self._handle_pid = pid
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None
                self._handle_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __getitem__(self, item: int) -> dict[str, Any]:
        case_index = int(self.window_case_indices[item])
        start_index = int(self.window_start_indices[item])
        stop_index = start_index + self.horizon + 1
        sample_start = self._sample_index(
            case_index, start_index
        )
        sample_stop = self._sample_index(
            case_index, stop_index
        )

        handle = self._get_handle()
        frames = np.asarray(
            handle["samples"]["field"][
                sample_start:sample_stop
            ],
            dtype=np.float32,
        )
        condition = np.asarray(
            handle["samples"]["condition"][
                sample_start : sample_stop - 1
            ],
            dtype=np.float32,
        )
        physical = np.asarray(
            handle["samples"]["physical_condition"][
                sample_start : sample_stop - 1
            ],
            dtype=np.float32,
        )

        expected_frames = (
            self.horizon + 1,
            *self.field_shape,
        )
        if frames.shape != expected_frames:
            raise RuntimeError(
                f"Window shape {frames.shape} != {expected_frames}"
            )
        if condition.shape != (self.horizon, 3):
            raise RuntimeError(
                f"Condition shape {condition.shape} is invalid."
            )

        return {
            "window_index": torch.tensor(
                item, dtype=torch.int64
            ),
            "case_index": torch.tensor(
                case_index, dtype=torch.int64
            ),
            "case_id": self.case_ids[case_index],
            "start_time_index": torch.tensor(
                start_index, dtype=torch.int64
            ),
            "frames": torch.from_numpy(
                np.ascontiguousarray(frames)
            ),
            "condition": torch.from_numpy(
                np.ascontiguousarray(condition)
            ),
            "physical_condition": torch.from_numpy(
                np.ascontiguousarray(physical)
            ),
            "phase_velocity": torch.tensor(
                float(self.case_phase_velocity[case_index]),
                dtype=torch.float32,
            ),
            "dt": torch.tensor(self.dt, dtype=torch.float32),
        }


def expected_window_count(
    case_count: int,
    phase_count: int,
    horizon: int,
) -> int:
    if horizon < 1 or horizon >= phase_count:
        raise ValueError(horizon)
    return int(case_count * (phase_count - horizon))


__all__ = [
    "SPLIT_NAME_TO_CODE",
    "Stage8D2BWindowDataset",
    "expected_window_count",
]


# Public, stage-neutral alias.
RolloutWindowDataset = Stage8D2BWindowDataset
