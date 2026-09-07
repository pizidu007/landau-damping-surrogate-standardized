import h5py
import numpy as np

from landau_surrogate.data.huang2025 import (
    causal_indices, load_huang_mat, paper_like_indices, spectral_derivative,
)


def test_huang_spectral_derivative() -> None:
    nx = 64
    phase = 2.0 * np.pi * np.arange(nx) / nx
    value = np.sin(3.0 * phase)[None]
    result = spectral_derivative(value, 0.35)[0]
    np.testing.assert_allclose(result, 3.0 * 0.35 * np.cos(3.0 * phase), atol=1.0e-12)


def test_huang_split_contracts() -> None:
    train, validation, test = paper_like_indices(20_000)
    assert (len(train), len(validation), len(test)) == (6_400, 800, 800)
    assert len(set(train) & set(validation)) == 0
    train, validation, test = causal_indices(20_000)
    assert (train[-1], validation[0], test[0]) == (4_799, 4_800, 6_000)


def test_load_huang_consolidated_hdf5(tmp_path) -> None:
    path = tmp_path / "strict.h5"
    nx, nt, k = 8, 3, 0.35
    x = (np.arange(nx) + 0.5) * (2.0 * np.pi / k) / nx
    phase = k * x
    heat_flux = np.stack([np.sin(phase + 0.1 * t) for t in range(nt)])
    with h5py.File(path, "w") as handle:
        handle.attrs["k"] = k
        handle.attrs["alpha"] = 0.1
        handle.create_dataset("time", data=np.arange(nt) * 0.005)
        handle.create_dataset("x", data=x)
        central = handle.create_group("central_moments")
        central.create_dataset("density", data=np.ones((nt, nx)))
        central.create_dataset("velocity", data=np.zeros((nt, nx)))
        central.create_dataset("pressure", data=2.0 * np.ones((nt, nx)))
        central.create_dataset("heat_flux", data=heat_flux)
    trajectory = load_huang_mat(path)
    assert trajectory.state.shape == (nt, 3, nx)
    assert trajectory.moment_definition == "central"
    assert trajectory.source_kind == "gkeyll_hdf5"
    np.testing.assert_allclose(trajectory.heat_flux_gradient[0], k * np.cos(phase), atol=1e-6)
