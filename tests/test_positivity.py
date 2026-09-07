import torch

from landau_surrogate.losses.positivity import positivity_losses


def test_positivity_loss_detects_negative_complete_distribution():
    prediction = torch.tensor([[[0.0, -2.0], [0.0, -2.0]]])
    background = torch.tensor([1.0, 1.0])
    absolute, tail_scaled, details = positivity_losses(
        prediction,
        background,
        delta_global_rms=1.0,
    )
    assert float(absolute) > 0.0
    assert float(tail_scaled) > 0.0
    assert float(details["negative_fraction"]) > 0.0
