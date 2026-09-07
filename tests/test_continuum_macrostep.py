from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from landau_surrogate.data.continuum_macrostep import (
    MacrostepWindowDataset,
    compute_normalization,
    denormalize_state,
    load_macrostep_trajectory,
    normalize_state,
    normalized_condition,
)
from landau_surrogate.data.continuum_v1 import ContinuumCase


def _write_case(path: Path, case_id: str = "macro_train") -> ContinuumCase:
    path.parent.mkdir(parents=True, exist_ok=True)
    nt, nx = 13, 16
    k_value, alpha = 0.4, 0.1
    phase = 2.0 * np.pi * np.arange(nx) / nx
    raw = np.zeros((nt, nx, 4), dtype=np.float32)
    electric = np.zeros((nt, nx), dtype=np.float32)
    for frame in range(nt):
        raw[frame, :, 0] = 1.0 + 0.05 * np.cos(phase + frame * 0.1)
        raw[frame, :, 1] = 0.02 * np.sin(phase + frame * 0.1)
        raw[frame, :, 2] = 1.0 + 0.03 * np.cos(phase + frame * 0.1)
        electric[frame] = -0.05 / k_value * np.sin(phase + frame * 0.1)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            dataset_version="continuum_v1",
            case_id=case_id,
            K=k_value,
            alpha=alpha,
            source_solver_use_gpu=1,
        )
        diagnostics = handle.create_group("diagnostics")
        diagnostics.create_dataset("raw_moments", data=raw)
        diagnostics.create_dataset("electric_field", data=electric)
        diagnostics.create_dataset("time", data=np.arange(nt) * 0.02)
        coordinates = handle.create_group("coordinates")
        length = 2.0 * np.pi / k_value
        coordinates.create_dataset(
            "x_cell", data=(np.arange(nx) + 0.5) * length / nx
        )
    return ContinuumCase(
        case_id=case_id,
        K=k_value,
        alpha=alpha,
        split="train",
        regime="strong_nonlinear",
        path=path,
    )


def test_macrostep_loader_and_window_indices(tmp_path: Path) -> None:
    case = _write_case(tmp_path / "trajectory.h5")
    trajectory = load_macrostep_trajectory(case)
    assert trajectory.state.shape == (13, 4, 16)
    dataset = MacrostepWindowDataset(
        [trajectory], history_steps=3, macro_stride=2, rollout_steps=2
    )
    assert len(dataset) == 5
    history, targets, k_value, alpha, delta_t, case_index = dataset[0]
    np.testing.assert_allclose(history.numpy(), trajectory.state[[0, 2, 4]])
    np.testing.assert_allclose(targets.numpy(), trajectory.state[[6, 8]])
    np.testing.assert_allclose(delta_t.numpy(), [0.04, 0.04])
    assert float(k_value) == pytest.approx(case.K)
    assert float(alpha) == pytest.approx(case.alpha)
    assert int(case_index) == 0


def test_training_normalization_round_trip_and_condition(tmp_path: Path) -> None:
    first = load_macrostep_trajectory(
        _write_case(tmp_path / "first.h5", "first")
    )
    second_case = _write_case(tmp_path / "second.h5", "second")
    second_case = ContinuumCase(
        case_id=second_case.case_id,
        K=0.5,
        alpha=0.15,
        split=second_case.split,
        regime=second_case.regime,
        path=second_case.path,
    )
    with h5py.File(second_case.path, "r+") as handle:
        handle.attrs["K"] = second_case.K
        handle.attrs["alpha"] = second_case.alpha
    second = load_macrostep_trajectory(second_case)
    normalization = compute_normalization([first, second])
    value = torch.from_numpy(first.state[:2])
    restored = denormalize_state(normalize_state(value, normalization), normalization)
    torch.testing.assert_close(restored, value)
    condition = normalized_condition(
        torch.tensor([first.case.K]),
        torch.tensor([first.case.alpha]),
        torch.tensor([0.1]),
        normalization,
    )
    assert condition.shape == (1, 3)
    assert float(condition[0, 2]) == pytest.approx(0.1)
