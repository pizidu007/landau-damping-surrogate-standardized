import numpy as np

from landau_surrogate.data.conservative import (
    conservative_m02_compress,
    trapezoid_weights,
    weighted_integral,
)


def test_conservative_m02_preserves_mass_and_kinetic_moment():
    velocity = np.linspace(-4.0, 4.0, 17)
    values = np.stack(
        [
            np.exp(-0.5 * velocity**2),
            (1.0 + 0.1 * velocity) * np.exp(-0.5 * velocity**2),
        ]
    )
    compressed, metadata = conservative_m02_compress(
        values, velocity, stride=4
    )
    coarse_velocity = metadata["target_coordinates"]

    fine_weights = trapezoid_weights(velocity)
    coarse_weights = trapezoid_weights(coarse_velocity)
    fine_mass = weighted_integral(values, fine_weights)
    coarse_mass = weighted_integral(compressed, coarse_weights)
    fine_m2 = weighted_integral(
        0.5 * velocity**2 * values, fine_weights
    )
    coarse_m2 = weighted_integral(
        0.5 * coarse_velocity**2 * compressed, coarse_weights
    )

    np.testing.assert_allclose(coarse_mass, fine_mass, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(coarse_m2, fine_m2, atol=1e-12, rtol=1e-12)
