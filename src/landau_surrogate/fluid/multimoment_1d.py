"""Differentiable periodic 1D three-moment electron fluid system."""
from __future__ import annotations

from collections.abc import Callable

import torch


Closure = Callable[[torch.Tensor], torch.Tensor]


def spectral_filter(value: torch.Tensor, maximum_mode: int | None) -> torch.Tensor:
    if maximum_mode is None:
        return value
    fft_value = value if value.dtype in (torch.float32, torch.float64) else value.float()
    transformed = torch.fft.rfft(fft_value, dim=-1)
    transformed[..., maximum_mode + 1 :] = 0.0
    return torch.fft.irfft(transformed, n=value.shape[-1], dim=-1).to(value.dtype)


def wave_numbers(nx: int, k_fundamental: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mode = torch.fft.fftfreq(nx, d=1.0 / nx, device=k_fundamental.device, dtype=dtype)
    return k_fundamental.reshape(-1, 1) * mode.reshape(1, -1)


def derivative(
    value: torch.Tensor,
    k_fundamental: torch.Tensor,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    modes = wave_numbers(value.shape[-1], k_fundamental, value.dtype)
    fft_value = value if value.dtype in (torch.float32, torch.float64) else value.float()
    transformed = torch.fft.fft(fft_value, dim=-1)
    if maximum_mode is not None:
        signed_mode = torch.fft.fftfreq(
            value.shape[-1], d=1.0 / value.shape[-1], device=value.device
        )
        transformed = transformed * (torch.abs(signed_mode) <= maximum_mode)
    return torch.fft.ifft(
        1j * modes * transformed, dim=-1
    ).real.to(value.dtype)


def poisson_electric(density: torch.Tensor, k_fundamental: torch.Tensor) -> torch.Tensor:
    nx = density.shape[-1]
    mode = torch.arange(nx // 2 + 1, device=density.device, dtype=density.dtype)
    modes = k_fundamental.reshape(-1, 1) * mode.reshape(1, -1)
    fft_density = density if density.dtype in (torch.float32, torch.float64) else density.float()
    charge_hat = torch.fft.rfft(1.0 - fft_density, dim=-1)
    electric_hat = torch.zeros_like(charge_hat)
    electric_hat[:, 1:] = charge_hat[:, 1:] / (1j * modes[:, 1:])
    return torch.fft.irfft(electric_hat, n=nx, dim=-1).to(density.dtype)


def fluid_rhs(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    closure: Closure,
    density_floor: float | None = 1.0e-4,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    density, velocity, pressure = state[:, 0], state[:, 1], state[:, 2]
    safe_density = density if density_floor is None else torch.clamp(density, min=density_floor)
    electric = poisson_electric(density, k_fundamental)
    density_rhs = -derivative(density * velocity, k_fundamental, maximum_mode)
    velocity_rhs = (
        -velocity * derivative(velocity, k_fundamental, maximum_mode)
        - derivative(pressure, k_fundamental, maximum_mode) / safe_density
        - electric
    )
    pressure_rhs = (
        -velocity * derivative(pressure, k_fundamental, maximum_mode)
        - 3.0 * pressure * derivative(velocity, k_fundamental, maximum_mode)
        - spectral_filter(closure(state), maximum_mode)
    )
    return torch.stack((density_rhs, velocity_rhs, pressure_rhs), dim=1)


def fluid_rhs_ampere(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    closure: Closure,
    density_floor: float | None = 1.0e-4,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    """Four-field formulation that evolves the electric field by Ampere's law."""
    density, velocity, pressure, electric = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    safe_density = density if density_floor is None else torch.clamp(density, min=density_floor)
    current = density * velocity
    density_rhs = -derivative(current, k_fundamental, maximum_mode)
    velocity_rhs = (
        -velocity * derivative(velocity, k_fundamental, maximum_mode)
        - derivative(pressure, k_fundamental, maximum_mode) / safe_density
        - electric
    )
    pressure_rhs = (
        -velocity * derivative(pressure, k_fundamental, maximum_mode)
        - 3.0 * pressure * derivative(velocity, k_fundamental, maximum_mode)
        - spectral_filter(closure(state[:, :3]), maximum_mode)
    )
    # Subtract the box-mean current to preserve mean(E)=0 and the Poisson
    # constraint for a periodic neutral system.
    electric_rhs = current - current.mean(dim=-1, keepdim=True)
    return torch.stack((density_rhs, velocity_rhs, pressure_rhs, electric_rhs), dim=1)


def raw_moment_rhs(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    closure: Closure,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    """First three raw Vlasov moments ``(M0, M1, M2)`` with Poisson E.

    This is distinct from :func:`fluid_rhs`, whose state is the primitive
    central-moment tuple ``(n, u, p)``.  Several exported Gkeyll arrays use
    raw moments even when their filenames use pressure/heat-flux notation.
    """
    density, momentum, second_moment = state[:, 0], state[:, 1], state[:, 2]
    electric = poisson_electric(density, k_fundamental)
    return torch.stack(
        (
            -derivative(momentum, k_fundamental, maximum_mode),
            -derivative(second_moment, k_fundamental, maximum_mode) - density * electric,
            -spectral_filter(closure(state), maximum_mode) - 2.0 * momentum * electric,
        ),
        dim=1,
    )


def raw_moment_rhs_ampere(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    closure: Closure,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    """Raw-moment system with the electric field evolved by Ampere's law."""
    density, momentum, second_moment, electric = (
        state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    )
    electric_rhs = momentum - momentum.mean(dim=-1, keepdim=True)
    return torch.stack(
        (
            -derivative(momentum, k_fundamental, maximum_mode),
            -derivative(second_moment, k_fundamental, maximum_mode) - density * electric,
            -spectral_filter(closure(state[:, :3]), maximum_mode) - 2.0 * momentum * electric,
            electric_rhs,
        ),
        dim=1,
    )


def rk4_step(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    dt: float,
    closure_factory: Callable[[torch.Tensor], Closure],
    density_floor: float = 1.0e-4,
    pressure_floor: float = 1.0e-5,
    clamp_output: bool = True,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    def rhs(stage: torch.Tensor) -> torch.Tensor:
        filtered_stage = spectral_filter(stage, maximum_mode)
        return fluid_rhs(
            filtered_stage,
            k_fundamental,
            closure_factory(filtered_stage),
            density_floor,
            maximum_mode,
        )

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result = spectral_filter(result, maximum_mode)
    if clamp_output:
        result = torch.stack(
            (
                torch.clamp(result[:, 0], min=density_floor),
                result[:, 1],
                torch.clamp(result[:, 2], min=pressure_floor),
            ),
            dim=1,
        )
    return result


def rk4_step_ampere(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    dt: float,
    closure_factory: Callable[[torch.Tensor], Closure],
    density_floor: float = 1.0e-4,
    pressure_floor: float = 1.0e-5,
    clamp_output: bool = True,
    maximum_mode: int | None = None,
) -> torch.Tensor:
    def rhs(stage: torch.Tensor) -> torch.Tensor:
        filtered_stage = spectral_filter(stage, maximum_mode)
        return fluid_rhs_ampere(
            filtered_stage, k_fundamental,
            closure_factory(filtered_stage[:, :3]), density_floor, maximum_mode,
        )

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    result = spectral_filter(
        state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), maximum_mode
    )
    if clamp_output:
        result = torch.stack(
            (
                torch.clamp(result[:, 0], min=density_floor), result[:, 1],
                torch.clamp(result[:, 2], min=pressure_floor), result[:, 3],
            ), dim=1,
        )
    return result


def rk4_step_raw_moments(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    dt: float,
    closure_factory: Callable[[torch.Tensor], Closure],
    maximum_mode: int | None = None,
) -> torch.Tensor:
    """RK4 update of ``(M0, M1, M2)`` using Poisson's equation."""
    def rhs(stage: torch.Tensor) -> torch.Tensor:
        filtered = spectral_filter(stage, maximum_mode)
        return raw_moment_rhs(
            filtered, k_fundamental, closure_factory(filtered), maximum_mode
        )

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return spectral_filter(
        state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), maximum_mode
    )


def rk4_step_raw_moments_ampere(
    state: torch.Tensor,
    k_fundamental: torch.Tensor,
    dt: float,
    closure_factory: Callable[[torch.Tensor], Closure],
    maximum_mode: int | None = None,
) -> torch.Tensor:
    """RK4 update of ``(M0, M1, M2, E)`` using Ampere's law."""
    def rhs(stage: torch.Tensor) -> torch.Tensor:
        filtered = spectral_filter(stage, maximum_mode)
        return raw_moment_rhs_ampere(
            filtered, k_fundamental, closure_factory(filtered[:, :3]), maximum_mode
        )

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return spectral_filter(
        state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), maximum_mode
    )


def fluid_energies(state: torch.Tensor, k_fundamental: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    density, velocity, pressure = state[:, 0], state[:, 1], state[:, 2]
    electric = poisson_electric(density, k_fundamental)
    domain_length = 2.0 * torch.pi / k_fundamental
    field = 0.5 * domain_length * torch.mean(electric.square(), dim=-1)
    kinetic = 0.5 * domain_length * torch.mean(pressure + density * velocity.square(), dim=-1)
    return field, kinetic, field + kinetic


def fluid_energies_ampere(state: torch.Tensor, k_fundamental: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    density, velocity, pressure, electric = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    domain_length = 2.0 * torch.pi / k_fundamental
    field = 0.5 * domain_length * torch.mean(electric.square(), dim=-1)
    kinetic = 0.5 * domain_length * torch.mean(pressure + density * velocity.square(), dim=-1)
    return field, kinetic, field + kinetic


__all__ = [
    "derivative",
    "fluid_energies",
    "fluid_energies_ampere",
    "fluid_rhs",
    "fluid_rhs_ampere",
    "poisson_electric",
    "raw_moment_rhs",
    "raw_moment_rhs_ampere",
    "rk4_step",
    "rk4_step_ampere",
    "rk4_step_raw_moments",
    "rk4_step_raw_moments_ampere",
    "spectral_filter",
]
