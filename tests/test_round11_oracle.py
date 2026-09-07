import numpy as np
import pytest
import torch
from landau_surrogate.fluid.multimoment_1d import derivative, spectral_filter

from landau_surrogate.tools.audit_continuum_v1_closure_oracle import interpolate, electric_numpy


def test_oracle_interpolation_uses_actual_times_and_rejects_extrapolation():
    times = np.array([0., .019, .041])
    values = (3 * times[:, None, None] + np.arange(4)[None, None, :]).astype(np.float32)
    query = np.array([0., .01, .03, .041])
    np.testing.assert_allclose(interpolate(values, times, query),
                               3 * query[:, None, None] + np.arange(4)[None, None, :], atol=3e-7)
    with pytest.raises(ValueError):
        interpolate(values, times, np.array([.05]))


def test_poisson_field_has_correct_sign_and_physical_scale():
    theta = 2 * np.pi * np.arange(64) / 64
    k = np.array([.3, .5])
    state = np.zeros((2, 2, 3, 64))
    state[:, :, 0] = 1 + .1 * np.cos(theta)
    field = electric_numpy(state, k)
    np.testing.assert_allclose(field, np.broadcast_to(-.1 / k[None, :, None] * np.sin(theta), field.shape), atol=1e-14)


def test_spectral_operators_preserve_float64_for_convergence_audits():
    theta = torch.arange(64, dtype=torch.float64)*2*torch.pi/64
    value = torch.sin(3*theta)[None]
    k = torch.tensor([.35],dtype=torch.float64)
    torch.testing.assert_close(derivative(value,k),1.05*torch.cos(3*theta)[None],atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(spectral_filter(value,4),value,atol=1e-14,rtol=1e-14)
