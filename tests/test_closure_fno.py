from __future__ import annotations

import math

import numpy as np
import torch

from landau_surrogate.data.closure_dataset import spectral_filter_numpy
from landau_surrogate.models.closure_fno1d import (
    ClosureFNO1d, DualHeadHistoryResidualFNO1d, spectral_derivative,
)
from landau_surrogate.training.gkeyll_regime_dual import padded_histories


def test_closure_fno_shape_and_zero_mean() -> None:
    model = ClosureFNO1d(state_channels=3, width=8, modes=4, layers=2)
    output = model(torch.randn(2, 3, 32), torch.tensor([0.0, 1.0]))
    assert output.shape == (2, 32)
    assert torch.max(torch.abs(output.mean(dim=-1))).item() < 1.0e-6


def test_closure_fno_paper_like_relu_without_normalization() -> None:
    model = ClosureFNO1d(
        state_channels=3, width=8, modes=4, layers=2,
        normalization="none", activation="relu",
    )
    output = model(torch.randn(2, 3, 32), torch.zeros(2))
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()
    assert torch.max(torch.abs(output.mean(dim=-1))).item() < 1.0e-6


def test_dual_history_fno_shapes_zero_means_and_gradient_flow() -> None:
    model = DualHeadHistoryResidualFNO1d(history_steps=4, width=8, modes=4, layers=2)
    gradient, heat_flux = model(torch.randn(2, 4, 3, 32), torch.randn(2, 2))
    assert gradient.shape == heat_flux.shape == (2, 32)
    assert torch.max(torch.abs(gradient.mean(dim=-1))).item() < 1.0e-6
    assert torch.max(torch.abs(heat_flux.mean(dim=-1))).item() < 1.0e-6
    (gradient.square().mean() + heat_flux.square().mean()).backward()
    assert model.history_gate_logit.grad is not None


def test_spectral_derivative_uses_physical_fundamental() -> None:
    nx = 64
    coordinate = torch.arange(nx, dtype=torch.float32) / nx
    value = torch.sin(2.0 * math.pi * coordinate)[None]
    result = spectral_derivative(value, torch.tensor([0.35]))
    expected = 0.35 * torch.cos(2.0 * math.pi * coordinate)[None]
    assert torch.max(torch.abs(result - expected)).item() < 2.0e-5


def test_numpy_spectral_filter_removes_unresolved_mode() -> None:
    phase = 2.0 * np.pi * np.arange(64) / 64
    value = np.sin(3.0 * phase) + 0.5 * np.sin(17.0 * phase)
    result = spectral_filter_numpy(value.astype(np.float32), 8)
    np.testing.assert_allclose(result, np.sin(3.0 * phase), atol=2.0e-6)


def test_padded_histories_repeat_initial_state_before_time_zero() -> None:
    state = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
    result = padded_histories(state, (-3, -1, 0))[:, :, 0, 0]
    np.testing.assert_array_equal(
        result,
        np.array(
            [[0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 2, 3], [1, 3, 4]],
            dtype=np.float32,
        ),
    )
