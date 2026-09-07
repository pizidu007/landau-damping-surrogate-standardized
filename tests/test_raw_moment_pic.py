from __future__ import annotations

import numpy as np

from landau_surrogate.data.raw_moment_pic import (
    interleaved_indices,
    spectral_derivative,
    spectral_filter,
)


def test_spectral_derivative_uses_physical_wavenumber() -> None:
    nx = 128
    k = 0.35
    x_over_l = np.arange(nx) / nx
    value = np.sin(2.0 * np.pi * 3.0 * x_over_l)
    expected = 3.0 * k * np.cos(2.0 * np.pi * 3.0 * x_over_l)
    np.testing.assert_allclose(spectral_derivative(value, k), expected, atol=2.0e-14)


def test_spectral_filter_removes_modes_above_cutoff() -> None:
    x = np.arange(64) / 64
    value = np.sin(2.0 * np.pi * 3.0 * x) + 0.4 * np.cos(2.0 * np.pi * 12.0 * x)
    expected = np.sin(2.0 * np.pi * 3.0 * x)
    np.testing.assert_allclose(spectral_filter(value, 8), expected, atol=2.0e-14)


def test_interleaved_split_is_disjoint_and_complete() -> None:
    splits = interleaved_indices(601)
    merged = np.concatenate(splits)
    assert tuple(map(len, splits)) == (481, 60, 60)
    np.testing.assert_array_equal(np.sort(merged), np.arange(601))
    assert len(np.unique(merged)) == len(merged)
