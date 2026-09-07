#!/usr/bin/env python3
"""
Lazy pair/sequence readers for Stage 8D-2A time stepping.

The reader uses the frozen Stage 8D-1A compact cache:

    110 cases x 31 snapshots x 128 x 193

Case-level train/validation/test splits are preserved. Pair samples never cross
case boundaries:

    current = field(case, t_i)
    target  = field(case, t_{i+1})
    dt      = 0.5 phase-time units
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
GROUP_CODE_TO_NAME = {
    0: "none",
    1: "validation",
    2: "legacy",
    3: "interstitial",
    4: "boundary",
}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class Stage8D2PairDataset(Dataset):
    """Lazy one-step pair reader with process-local HDF5 handles."""

    def __init__(
        self,
        cache_path: str | Path,
        split: str,
        case_ids: Sequence[str] | None = None,
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(self.cache_path)
        if split not in SPLIT_NAME_TO_CODE:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
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
            self.case_k = np.asarray(
                handle["cases"]["k"][...], dtype=np.float32
            )
            self.case_alpha = np.asarray(
                handle["cases"]["alpha"][...],
                dtype=np.float32,
            )
            self.case_phase_velocity = np.asarray(
                handle["cases"]["phase_velocity"][...],
                dtype=np.float32,
            )
            self.case_split_code = np.asarray(
                handle["cases"]["split_code"][...],
                dtype=np.uint8,
            )
            self.case_group_code = np.asarray(
                handle["cases"]["group_code"][...],
                dtype=np.uint8,
            )

            field_shape = tuple(
                int(item)
                for item in handle["samples"]["field"].shape
            )
            expected_samples = (
                len(self.case_ids) * len(self.phase_time)
            )
            if field_shape != (
                expected_samples,
                self.normalized_x.size,
                self.velocity.size,
            ):
                raise RuntimeError(
                    f"Unexpected field shape: {field_shape}"
                )

            case_index = np.asarray(
                handle["samples"]["case_index"][...],
                dtype=np.int32,
            )
            time_index = np.asarray(
                handle["samples"]["time_index"][...],
                dtype=np.int32,
            )
            expected_case_index = np.repeat(
                np.arange(len(self.case_ids), dtype=np.int32),
                len(self.phase_time),
            )
            expected_time_index = np.tile(
                np.arange(len(self.phase_time), dtype=np.int32),
                len(self.case_ids),
            )
            if not np.array_equal(case_index, expected_case_index):
                raise RuntimeError(
                    "Cache sample ordering is not case-major."
                )
            if not np.array_equal(time_index, expected_time_index):
                raise RuntimeError(
                    "Cache sample time ordering is not contiguous."
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
        pair_case: list[int] = []
        pair_time: list[int] = []
        for case_index_value in selected:
            for time_index_value in range(
                len(self.phase_time) - 1
            ):
                pair_case.append(int(case_index_value))
                pair_time.append(time_index_value)
        self.pair_case_indices = np.asarray(
            pair_case, dtype=np.int32
        )
        self.pair_time_indices = np.asarray(
            pair_time, dtype=np.int16
        )

    @property
    def field_shape(self) -> tuple[int, int]:
        return (
            int(self.normalized_x.size),
            int(self.velocity.size),
        )

    @property
    def phase_count(self) -> int:
        return int(self.phase_time.size)

    @property
    def dt(self) -> float:
        differences = np.diff(self.phase_time.astype(np.float64))
        if differences.size == 0:
            return 0.0
        if not np.allclose(differences, differences[0]):
            raise RuntimeError("Phase-time grid is not uniform.")
        return float(differences[0])

    def __len__(self) -> int:
        return int(self.pair_case_indices.size)

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
        case_index = int(self.pair_case_indices[item])
        time_index = int(self.pair_time_indices[item])
        handle = self._get_handle()
        fields = handle["samples"]["field"]

        current = np.asarray(
            fields[
                self._sample_index(case_index, time_index)
            ],
            dtype=np.float32,
        )
        target = np.asarray(
            fields[
                self._sample_index(case_index, time_index + 1)
            ],
            dtype=np.float32,
        )
        current_condition = np.asarray(
            handle["samples"]["condition"][
                self._sample_index(case_index, time_index)
            ],
            dtype=np.float32,
        )
        next_condition = np.asarray(
            handle["samples"]["condition"][
                self._sample_index(case_index, time_index + 1)
            ],
            dtype=np.float32,
        )
        current_physical = np.asarray(
            handle["samples"]["physical_condition"][
                self._sample_index(case_index, time_index)
            ],
            dtype=np.float32,
        )
        next_physical = np.asarray(
            handle["samples"]["physical_condition"][
                self._sample_index(case_index, time_index + 1)
            ],
            dtype=np.float32,
        )

        return {
            "pair_index": torch.tensor(item, dtype=torch.int64),
            "case_index": torch.tensor(
                case_index, dtype=torch.int64
            ),
            "case_id": self.case_ids[case_index],
            "time_index": torch.tensor(
                time_index, dtype=torch.int64
            ),
            "current": torch.from_numpy(
                np.ascontiguousarray(current)
            ),
            "target": torch.from_numpy(
                np.ascontiguousarray(target)
            ),
            "increment": torch.from_numpy(
                np.ascontiguousarray(target - current)
            ),
            "condition": torch.from_numpy(current_condition),
            "next_condition": torch.from_numpy(next_condition),
            "physical_condition": torch.from_numpy(
                current_physical
            ),
            "next_physical_condition": torch.from_numpy(
                next_physical
            ),
            "phase_velocity": torch.tensor(
                float(self.case_phase_velocity[case_index]),
                dtype=torch.float32,
            ),
            "dt": torch.tensor(self.dt, dtype=torch.float32),
        }


def split_case_indices(
    cache_path: str | Path,
    split: str,
) -> np.ndarray:
    if split not in SPLIT_NAME_TO_CODE:
        raise ValueError(split)
    with h5py.File(cache_path, "r") as handle:
        codes = np.asarray(
            handle["cases"]["split_code"][...],
            dtype=np.uint8,
        )
    return np.nonzero(
        codes == SPLIT_NAME_TO_CODE[split]
    )[0].astype(np.int64)


def load_case_sequence(
    cache_path: str | Path,
    case_index: int,
) -> dict[str, Any]:
    """Load one complete 31-frame case sequence into memory."""
    path = Path(cache_path)
    with h5py.File(path, "r") as handle:
        phase_time = np.asarray(
            handle["grids"]["phase_time"][...],
            dtype=np.float32,
        )
        phase_count = int(phase_time.size)
        case_count = int(handle["cases"]["case_id"].shape[0])
        if case_index < 0 or case_index >= case_count:
            raise IndexError(case_index)

        start = case_index * phase_count
        stop = start + phase_count
        case_id_raw = handle["cases"]["case_id"][case_index]
        return {
            "case_index": int(case_index),
            "case_id": _decode(case_id_raw),
            "k": float(handle["cases"]["k"][case_index]),
            "alpha": float(
                handle["cases"]["alpha"][case_index]
            ),
            "phase_velocity": float(
                handle["cases"]["phase_velocity"][case_index]
            ),
            "split_code": int(
                handle["cases"]["split_code"][case_index]
            ),
            "group_code": int(
                handle["cases"]["group_code"][case_index]
            ),
            "phase_time": phase_time,
            "field": np.asarray(
                handle["samples"]["field"][start:stop],
                dtype=np.float32,
            ),
            "condition": np.asarray(
                handle["samples"]["condition"][start:stop],
                dtype=np.float32,
            ),
            "physical_condition": np.asarray(
                handle["samples"]["physical_condition"][
                    start:stop
                ],
                dtype=np.float32,
            ),
            "normalized_x": np.asarray(
                handle["grids"]["normalized_x"][...],
                dtype=np.float32,
            ),
            "velocity": np.asarray(
                handle["grids"]["velocity"][...],
                dtype=np.float32,
            ),
        }


def cache_rollout_contract(
    cache_path: str | Path,
) -> dict[str, Any]:
    path = Path(cache_path)
    with h5py.File(path, "r") as handle:
        phase_time = np.asarray(
            handle["grids"]["phase_time"][...],
            dtype=np.float64,
        )
        return {
            "status": str(handle.attrs.get("status", "")),
            "case_count": int(
                handle["cases"]["case_id"].shape[0]
            ),
            "phase_count": int(phase_time.size),
            "pair_count": int(
                handle["cases"]["case_id"].shape[0]
                * (phase_time.size - 1)
            ),
            "field_shape": list(
                handle["samples"]["field"].shape[1:]
            ),
            "phase_time_start": float(phase_time[0]),
            "phase_time_end": float(phase_time[-1]),
            "dt": float(phase_time[1] - phase_time[0]),
            "source_dataset_sha256": str(
                handle.attrs.get("source_dataset_sha256", "")
            ),
            "normalization_sha256": str(
                handle.attrs.get("normalization_sha256", "")
            ),
        }


__all__ = [
    "GROUP_CODE_TO_NAME",
    "SPLIT_NAME_TO_CODE",
    "Stage8D2PairDataset",
    "cache_rollout_contract",
    "load_case_sequence",
    "split_case_indices",
]


# Public, stage-neutral alias.
OneStepPairDataset = Stage8D2PairDataset
