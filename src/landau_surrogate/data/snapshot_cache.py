#!/usr/bin/env python3
"""
Lazy reader for the Stage 8D-1A compact normalized delta-f cache.

The cache is a derived, read-only training product. It never replaces or
modifies the frozen Stage 8C-4 mother HDF5.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_NAME_TO_CODE = {"train": 0, "val": 1, "test": 2}
GROUP_NAME_TO_CODE = {
    "none": 0,
    "validation": 1,
    "legacy": 2,
    "interstitial": 3,
    "boundary": 4,
}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class Stage8D1CacheDataset(Dataset):
    """Lazy sample reader for normalized delta_f0 snapshots."""

    def __init__(
        self,
        cache_path: str | Path,
        split: str = "train",
        evaluation_group: str | None = None,
        case_ids: Sequence[str] | None = None,
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(self.cache_path)
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if (
            evaluation_group is not None
            and evaluation_group not in GROUP_NAME_TO_CODE
        ):
            raise ValueError(
                f"Unsupported evaluation group: {evaluation_group}"
            )

        self.split = split
        self.evaluation_group = evaluation_group
        self._requested_case_ids = (
            set(case_ids) if case_ids is not None else None
        )
        self._handle: h5py.File | None = None
        self._handle_pid: int | None = None

        with h5py.File(self.cache_path, "r") as handle:
            if str(handle.attrs.get("status", "")) != "COMPLETE":
                raise RuntimeError("Stage 8D-1A cache is not COMPLETE.")
            self.normalized_x = np.asarray(
                handle["grids"]["normalized_x"][...], dtype=np.float32
            )
            self.velocity = np.asarray(
                handle["grids"]["velocity"][...], dtype=np.float32
            )
            self.phase_time = np.asarray(
                handle["grids"]["phase_time"][...], dtype=np.float32
            )
            self.case_ids = tuple(
                _decode(item) for item in handle["cases"]["case_id"][...]
            )
            self.case_k = np.asarray(
                handle["cases"]["k"][...], dtype=np.float32
            )
            self.case_alpha = np.asarray(
                handle["cases"]["alpha"][...], dtype=np.float32
            )
            self.case_phase_velocity = np.asarray(
                handle["cases"]["phase_velocity"][...], dtype=np.float32
            )
            self.case_split_code = np.asarray(
                handle["cases"]["split_code"][...], dtype=np.uint8
            )
            self.case_group_code = np.asarray(
                handle["cases"]["group_code"][...], dtype=np.uint8
            )

            sample_case_index = np.asarray(
                handle["samples"]["case_index"][...], dtype=np.int32
            )
            sample_split_code = np.asarray(
                handle["samples"]["split_code"][...], dtype=np.uint8
            )
            sample_group_code = np.asarray(
                handle["samples"]["group_code"][...], dtype=np.uint8
            )

            mask = np.ones(sample_case_index.shape[0], dtype=bool)
            if split != "all":
                mask &= sample_split_code == SPLIT_NAME_TO_CODE[split]
            if evaluation_group is not None:
                mask &= (
                    sample_group_code
                    == GROUP_NAME_TO_CODE[evaluation_group]
                )
            if self._requested_case_ids is not None:
                allowed_case_indices = {
                    index
                    for index, identifier in enumerate(self.case_ids)
                    if identifier in self._requested_case_ids
                }
                mask &= np.asarray(
                    [
                        int(case_index) in allowed_case_indices
                        for case_index in sample_case_index
                    ],
                    dtype=bool,
                )
            self.sample_indices = np.nonzero(mask)[0].astype(np.int64)

        if self.sample_indices.size == 0:
            raise RuntimeError("No samples selected from cache.")

    def __len__(self) -> int:
        return int(self.sample_indices.size)

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
        sample_index = int(self.sample_indices[item])
        handle = self._get_handle()
        samples = handle["samples"]
        case_index = int(samples["case_index"][sample_index])
        time_index = int(samples["time_index"][sample_index])
        field = np.asarray(
            samples["field"][sample_index], dtype=np.float32
        )
        condition = np.asarray(
            samples["condition"][sample_index], dtype=np.float32
        )
        physical_condition = np.asarray(
            samples["physical_condition"][sample_index],
            dtype=np.float32,
        )

        return {
            "sample_index": torch.tensor(
                sample_index, dtype=torch.int64
            ),
            "case_index": torch.tensor(case_index, dtype=torch.int64),
            "case_id": self.case_ids[case_index],
            "time_index": torch.tensor(time_index, dtype=torch.int64),
            "condition": torch.from_numpy(condition),
            "physical_condition": torch.from_numpy(
                physical_condition
            ),
            "field": torch.from_numpy(
                np.ascontiguousarray(field)
            ),
            "phase_velocity": torch.tensor(
                float(self.case_phase_velocity[case_index]),
                dtype=torch.float32,
            ),
        }


def cache_metadata(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path)
    with h5py.File(path, "r") as handle:
        return {
            "status": str(handle.attrs.get("status", "")),
            "version": str(handle.attrs.get("dataset_version", "")),
            "source_dataset_sha256": str(
                handle.attrs.get("source_dataset_sha256", "")
            ),
            "normalization_sha256": str(
                handle.attrs.get("normalization_sha256", "")
            ),
            "sample_count": int(handle["samples"]["field"].shape[0]),
            "case_count": int(handle["cases"]["case_id"].shape[0]),
            "field_shape": list(
                handle["samples"]["field"].shape[1:]
            ),
            "x_stride": int(handle.attrs["x_stride"]),
            "v_stride": int(handle.attrs["v_stride"]),
        }


def split_sample_indices(
    cache_path: str | Path,
    split: str,
) -> np.ndarray:
    if split not in SPLIT_NAME_TO_CODE:
        raise ValueError(split)
    with h5py.File(cache_path, "r") as handle:
        codes = np.asarray(
            handle["samples"]["split_code"][...], dtype=np.uint8
        )
    return np.nonzero(codes == SPLIT_NAME_TO_CODE[split])[0].astype(
        np.int64
    )


def load_split_arrays(
    cache_path: str | Path,
    split: str,
) -> dict[str, np.ndarray]:
    """Load one complete split into memory for POD construction/evaluation."""
    indices = split_sample_indices(cache_path, split)
    with h5py.File(cache_path, "r") as handle:
        samples = handle["samples"]
        return {
            "indices": indices,
            "field": np.asarray(
                samples["field"][indices], dtype=np.float32
            ),
            "condition": np.asarray(
                samples["condition"][indices], dtype=np.float32
            ),
            "physical_condition": np.asarray(
                samples["physical_condition"][indices],
                dtype=np.float32,
            ),
            "case_index": np.asarray(
                samples["case_index"][indices], dtype=np.int32
            ),
            "time_index": np.asarray(
                samples["time_index"][indices], dtype=np.int16
            ),
            "group_code": np.asarray(
                samples["group_code"][indices], dtype=np.uint8
            ),
        }


__all__ = [
    "GROUP_NAME_TO_CODE",
    "SPLIT_NAME_TO_CODE",
    "Stage8D1CacheDataset",
    "cache_metadata",
    "load_split_arrays",
    "split_sample_indices",
]


# Public, stage-neutral aliases.
SnapshotCacheDataset = Stage8D1CacheDataset
