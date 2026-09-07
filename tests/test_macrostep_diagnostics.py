from __future__ import annotations

import numpy as np
import torch

from landau_surrogate.diagnostics.macrostep import (
    aggregate_case_metrics,
    trajectory_metrics,
)
from landau_surrogate.physics.macrostep import (
    pressure_from_raw_moments,
    project_conservative_state,
)


def _trajectory() -> np.ndarray:
    nt, nx = 8, 32
    phase = 2.0 * np.pi * np.arange(nx) / nx
    state = np.zeros((nt, 4, nx), dtype=np.float64)
    for frame in range(nt):
        state[frame, 0] = 1.0 + 0.05 * np.cos(phase + 0.1 * frame)
        state[frame, 1] = 0.02 * np.sin(phase + 0.1 * frame)
        state[frame, 2] = 1.0
        state[frame, 3] = -0.125 * np.sin(phase + 0.1 * frame)
    return state


def test_identical_trajectory_has_zero_errors() -> None:
    truth = _trajectory()
    metrics = trajectory_metrics(
        truth.copy(), truth, time=np.arange(len(truth)) * 0.1, k_value=0.4
    )
    assert metrics["finite_to_final_time"]
    assert metrics["state_relative_l2"] == 0.0
    assert metrics["field_energy_log10_rmse"] == 0.0
    assert metrics["electric_mode_one"]["phase_mae_radians"] == 0.0
    aggregate = aggregate_case_metrics([metrics])
    assert aggregate["complete_case_count"] == 1


def test_projection_preserves_means_and_enforces_poisson() -> None:
    truth = torch.tensor(_trajectory()[:2], dtype=torch.float32)
    prediction = truth + 0.01 * torch.randn_like(truth)
    k_value = torch.tensor([0.4, 0.4])
    projected = project_conservative_state(prediction, truth, k_value)
    torch.testing.assert_close(
        projected[:, :2].mean(dim=-1), truth[:, :2].mean(dim=-1), atol=1e-6, rtol=0
    )
    assert float(projected[:, 3].mean(dim=-1).abs().max()) < 1.0e-6
    assert torch.all(pressure_from_raw_moments(truth) > 0.0)
