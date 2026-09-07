
import numpy as np
import torch

from landau_surrogate.models.snapshot_fno import SnapshotFNO
from landau_surrogate.models.stepper_fno import StepperFNO


def grids():
    return (
        np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float32),
        np.linspace(-4.0, 4.0, 9, dtype=np.float32),
    )


def test_snapshot_forward_and_decomposition():
    x, v = grids()
    model = SnapshotFNO(
        normalized_x=x,
        velocity=v,
        variant="dual_head",
        width=8,
        layers=2,
        modes_x=3,
        modes_v=3,
        v_padding=2,
        condition_channels=4,
        condition_hidden_dim=16,
        condition_hidden_layers=1,
        time_harmonics=1,
        phase_harmonics=1,
        x_coordinate_harmonics=1,
    )
    normalized = torch.zeros(2, 3)
    physical = torch.tensor([[0.45, 0.005, 0.0], [0.55, 0.025, 1.0]])
    out = model(normalized, physical)
    assert out["field"].shape == (2, 8, 9)
    reconstructed = out["nonzero"] + out["mean_delta"][:, None, :]
    torch.testing.assert_close(out["field"], reconstructed)
    torch.testing.assert_close(out["nonzero"].mean(dim=1), torch.zeros_like(out["mean_delta"]), atol=1e-5, rtol=1e-5)


def test_stepper_forward():
    x, v = grids()
    model = StepperFNO(
        normalized_x=x,
        velocity=v,
        variant="residual",
        width=8,
        layers=2,
        modes_x=3,
        modes_v=3,
        v_padding=2,
        condition_channels=4,
        condition_hidden_dim=16,
        condition_hidden_layers=1,
        time_harmonics=1,
        phase_harmonics=1,
        x_coordinate_harmonics=1,
    )
    current = torch.randn(2, 8, 9)
    normalized = torch.zeros(2, 3)
    physical = torch.tensor([[0.45, 0.005, 0.0], [0.55, 0.025, 1.0]])
    out = model(current, normalized, physical)
    assert out["field"].shape == current.shape
    assert torch.isfinite(out["field"]).all()
