from __future__ import annotations

import pytest
import torch

from landau_surrogate.models.macrostep_1d import (
    HistoryResidualFNO1d,
    PeriodicUNet1d,
)


@pytest.mark.parametrize(
    "model",
    [
        HistoryResidualFNO1d(
            history_steps=4, state_channels=4, width=16, modes=6, layers=2
        ),
        PeriodicUNet1d(
            history_steps=4, state_channels=4, base_width=8, depth=2
        ),
    ],
)
def test_macrostep_models_are_identity_at_initialization(model) -> None:
    history = torch.randn(3, 4, 4, 32)
    condition = torch.randn(3, 3)
    increment = model(history, condition)
    torch.testing.assert_close(increment, torch.zeros_like(increment))
    torch.testing.assert_close(model.predict_next(history, condition), history[:, -1])
    model.predict_next(history, condition).square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_fno_condition_changes_nonzero_initialized_output() -> None:
    torch.manual_seed(3)
    model = HistoryResidualFNO1d(
        history_steps=1,
        state_channels=4,
        width=12,
        modes=5,
        layers=1,
        zero_initialize_output=False,
    )
    history = torch.randn(2, 1, 4, 24)
    first = model(history, torch.zeros(2, 3))
    second = model(history, torch.ones(2, 3))
    assert not torch.allclose(first, second)


def test_unet_rejects_incompatible_grid() -> None:
    model = PeriodicUNet1d(history_steps=1, base_width=8, depth=3)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 1, 4, 30), torch.zeros(1, 3))
