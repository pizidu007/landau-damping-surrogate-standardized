
import numpy as np

from landau_surrogate.physics.closure import (
    phase_to_density,
    solve_periodic_poisson_from_density,
)


def test_periodic_poisson_contract():
    nx = 64
    k = 0.5
    length = 2.0 * np.pi / k
    x = np.arange(nx) * length / nx
    amplitude = 0.02
    density = 1.0 + amplitude * np.cos(k * x)
    result = solve_periodic_poisson_from_density(density, k)
    expected_e = -(amplitude / k) * np.sin(k * x)
    np.testing.assert_allclose(result["electric_field"], expected_e, atol=1e-12, rtol=1e-12)
    assert float(np.max(np.abs(result["poisson_residual"]))) < 1e-12


def test_velocity_integral():
    v = np.linspace(-5, 5, 501)
    phase = np.exp(-0.5 * v**2)[None, :]
    density = phase_to_density(phase, v)
    assert density.shape == (1,)
    assert 2.49 < density[0] < 2.52
