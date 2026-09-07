from __future__ import annotations

import math
from pathlib import Path

import torch

from landau_surrogate.pic.cuda_pic import (
    CudaPICConfig,
    create_quiet_start,
    deposit_raw_moments,
    fluid_moments_from_raw,
    inverse_sinusoidal_density_cdf,
    particle_grid_geometry,
)
from landau_surrogate.tools.generate_nonlinear_pic import load_config


def tiny_config() -> CudaPICConfig:
    return CudaPICConfig(
        field_nx=8,
        n_x_load=16,
        n_v_load=32,
        dt=0.1,
        t_end=1.0,
        moment_dt=0.2,
        phase_dt=0.5,
        phase_nx=8,
        phase_nv=17,
        phase_v_min=-4.0,
        phase_v_max=4.0,
    )


def test_exact_initial_density_inverse_cdf() -> None:
    k_value = 0.35
    alpha = 0.125
    domain_length = 2.0 * math.pi / k_value
    uniform = (torch.arange(4096, dtype=torch.float64) + 0.5) * (
        domain_length / 4096
    )
    position = inverse_sinusoidal_density_cdf(uniform, k_value, alpha)
    reconstructed = position + (alpha / k_value) * torch.sin(k_value * position)
    assert torch.max(torch.abs(reconstructed - uniform)).item() < 1.0e-12


def test_quiet_start_is_device_native_and_stratified() -> None:
    config = tiny_config()
    position, velocity, domain_length = create_quiet_start(
        0.35, 0.1, 0, config, torch.device("cpu")
    )
    assert position.shape == (config.particle_count,)
    assert velocity.shape == position.shape
    assert position.dtype == torch.float64
    assert 0.0 <= float(torch.min(position))
    assert float(torch.max(position)) < domain_length
    assert abs(float(torch.mean(velocity))) < 1.0e-12


def test_direct_m03_deposition_has_zero_symmetric_heat_flux() -> None:
    nx = 8
    domain_length = 2.0 * math.pi
    position = (
        (torch.arange(nx, dtype=torch.float64) + 0.5)
        * (domain_length / nx)
    ).repeat(4)
    velocity = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float64).repeat_interleave(nx)
    left, right, fraction = particle_grid_geometry(
        position, nx, domain_length / nx
    )
    density = torch.ones(nx, dtype=torch.float64)
    raw = deposit_raw_moments(
        velocity, left, right, fraction, density, nx
    )
    fluid = fluid_moments_from_raw(raw, domain_length)
    assert torch.max(torch.abs(fluid["velocity"])).item() < 1.0e-12
    assert torch.max(torch.abs(fluid["heat_flux"])).item() < 1.0e-12
    assert torch.max(torch.abs(fluid["heat_flux_gradient"])).item() < 1.0e-12
    assert torch.min(fluid["pressure"]).item() > 0.0


def test_formal_nonlinear_config_has_trajectory_level_splits() -> None:
    path = Path("configs/data/nonlinear_pic_v1.json")
    solver, cases = load_config(path, "formal")
    assert solver.t_end == 60.0
    assert solver.dt == 0.05
    assert len(cases) == 60
    counts = {
        name: sum(case["split"] == name for case in cases)
        for name in ("train", "validation", "test")
    }
    assert counts == {"train": 30, "validation": 15, "test": 15}
    pair_splits: dict[tuple[float, float], set[str]] = {}
    for case in cases:
        pair_splits.setdefault((case["k"], case["alpha"]), set()).add(case["split"])
    assert all(len(values) == 1 for values in pair_splits.values())
