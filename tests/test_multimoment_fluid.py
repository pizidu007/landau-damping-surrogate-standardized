from __future__ import annotations

import torch

from landau_surrogate.fluid.multimoment_1d import (
    derivative, fluid_rhs, poisson_electric, raw_moment_rhs,
    rk4_step, rk4_step_ampere, rk4_step_raw_moments,
)


def test_uniform_equilibrium_is_stationary() -> None:
    state = torch.zeros(2, 3, 32)
    state[:, 0] = 1.0
    state[:, 2] = 1.0
    k = torch.tensor([0.35, 0.55])
    closure = lambda _state: torch.zeros(2, 32)
    rhs = fluid_rhs(state, k, closure)
    assert torch.max(torch.abs(rhs)).item() < 1.0e-7
    result = rk4_step(state, k, 0.05, lambda _stage: closure)
    assert torch.max(torch.abs(result - state)).item() < 1.0e-7


def test_poisson_field_has_zero_mean() -> None:
    density = torch.ones(1, 64)
    density[0] += 0.1 * torch.cos(2.0 * torch.pi * torch.arange(64) / 64)
    electric = poisson_electric(density, torch.tensor([0.35]))
    assert abs(float(electric.mean())) < 1.0e-7


def test_ampere_uniform_equilibrium_is_stationary() -> None:
    state = torch.zeros(1, 4, 32)
    state[:, 0] = 1.0
    state[:, 2] = 1.0
    result = rk4_step_ampere(
        state, torch.tensor([0.35]), 0.01,
        lambda _stage: lambda value: torch.zeros_like(value[:, 0]),
    )
    assert torch.max(torch.abs(result - state)).item() < 1.0e-7


def test_raw_moment_uniform_equilibrium_is_stationary() -> None:
    state = torch.zeros(2, 3, 32)
    state[:, 0] = 1.0
    state[:, 2] = 1.0
    k = torch.tensor([0.35, 0.55])
    closure = lambda value: torch.zeros_like(value[:, 0])
    rhs = raw_moment_rhs(state, k, closure)
    assert torch.max(torch.abs(rhs)).item() < 1.0e-7
    result = rk4_step_raw_moments(state, k, 0.01, lambda _stage: closure)
    assert torch.max(torch.abs(result - state)).item() < 1.0e-7


def test_primitive_and_raw_moment_equations_are_equivalent() -> None:
    nx = 64
    phase = 2.0 * torch.pi * torch.arange(nx) / nx
    density = (1.0 + 0.08 * torch.cos(phase))[None]
    velocity = (0.05 * torch.sin(phase) + 0.02 * torch.sin(2.0 * phase))[None]
    pressure = (1.0 + 0.06 * torch.cos(phase + 0.3))[None]
    heat_flux = (0.04 * torch.sin(phase - 0.2))[None]
    k = torch.tensor([0.35])
    primitive = torch.stack((density, velocity, pressure), dim=1)
    heat_flux_gradient = derivative(heat_flux, k)
    primitive_time = fluid_rhs(primitive, k, lambda _state: heat_flux_gradient)

    momentum = density * velocity
    second_moment = pressure + density * velocity.square()
    third_moment = heat_flux + 3.0 * velocity * pressure + density * velocity.pow(3)
    raw = torch.stack((density, momentum, second_moment), dim=1)
    raw_time = raw_moment_rhs(raw, k, lambda _state: derivative(third_moment, k))
    transformed = torch.stack(
        (
            raw_time[:, 0],
            (raw_time[:, 1] - velocity * raw_time[:, 0]) / density,
            raw_time[:, 2] - 2.0 * velocity * raw_time[:, 1]
            + velocity.square() * raw_time[:, 0],
        ),
        dim=1,
    )
    torch.testing.assert_close(primitive_time, transformed, atol=2.0e-6, rtol=2.0e-5)
