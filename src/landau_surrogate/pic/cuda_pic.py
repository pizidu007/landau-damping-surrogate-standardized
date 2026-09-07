"""CUDA-resident periodic 1D electrostatic PIC for nonlinear Landau data.

The production loop keeps particles and saved histories on the GPU.  Host
transfers happen once per case, after time integration, instead of once per
PIC step.  Dense M0--M3 moments are deposited directly from particles; sparse
x-v histograms are optional and are intended for phase-space diagnostics.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class CudaPICConfig:
    field_nx: int
    n_x_load: int
    n_v_load: int
    dt: float
    t_end: float
    moment_dt: float
    phase_dt: float | None = None
    phase_nx: int = 0
    phase_nv: int = 0
    phase_v_min: float = -6.0
    phase_v_max: float = 6.0
    dtype: str = "float64"

    def validate(self) -> None:
        if min(self.field_nx, self.n_x_load, self.n_v_load) < 2:
            raise ValueError("PIC grid and quiet-start dimensions must be >= 2")
        if self.dt <= 0.0 or self.t_end <= 0.0 or self.moment_dt <= 0.0:
            raise ValueError("dt, t_end and moment_dt must be positive")
        _cadence_steps(self.t_end, self.dt, "t_end")
        _cadence_steps(self.moment_dt, self.dt, "moment_dt")
        if self.phase_dt is not None:
            _cadence_steps(self.phase_dt, self.dt, "phase_dt")
            if self.phase_nx < 2 or self.phase_nv < 3:
                raise ValueError("phase_nx >= 2 and phase_nv >= 3 are required")
            if self.phase_v_max <= self.phase_v_min:
                raise ValueError("Invalid phase-space velocity range")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    @property
    def particle_count(self) -> int:
        return self.n_x_load * self.n_v_load

    @property
    def step_count(self) -> int:
        return _cadence_steps(self.t_end, self.dt, "t_end")


def _cadence_steps(value: float, dt: float, name: str) -> int:
    steps = int(round(value / dt))
    if steps < 1 or not math.isclose(steps * dt, value, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{name}={value} is not an integer multiple of dt={dt}")
    return steps


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _quiet_offsets(seed: int) -> tuple[float, float]:
    if seed == 0:
        return 0.5, 0.5
    generator = random.Random(int(seed))
    # Stratified offsets retain quiet loading while providing independent
    # discretization replicas without changing the target continuum state.
    return generator.random(), generator.random()


def inverse_sinusoidal_density_cdf(
    uniform_position: torch.Tensor,
    k_value: float,
    alpha: float,
    iterations: int = 8,
) -> torch.Tensor:
    """Map uniform samples to n(x)=1+alpha*cos(k*x) exactly (to tolerance)."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("The sinusoidal density requires 0 <= alpha < 1")
    if k_value <= 0.0:
        raise ValueError("k must be positive")
    if alpha == 0.0:
        return uniform_position.clone()
    position = uniform_position - (alpha / k_value) * torch.sin(
        k_value * uniform_position
    )
    for _ in range(iterations):
        residual = (
            position
            + (alpha / k_value) * torch.sin(k_value * position)
            - uniform_position
        )
        derivative = 1.0 + alpha * torch.cos(k_value * position)
        position = position - residual / derivative
    return position


def create_quiet_start(
    k_value: float,
    alpha: float,
    seed: int,
    config: CudaPICConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Create stratified Maxwellian particles directly on the target device."""
    config.validate()
    dtype = torch_dtype(config.dtype)
    domain_length = 2.0 * math.pi / k_value
    x_offset, v_offset = _quiet_offsets(seed)
    x_uniform = (
        torch.arange(config.n_x_load, device=device, dtype=dtype) + x_offset
    ) * (domain_length / config.n_x_load)
    x_loaded = inverse_sinusoidal_density_cdf(x_uniform, k_value, alpha)

    quantiles = (
        torch.arange(config.n_v_load, device=device, dtype=dtype) + v_offset
    ) / config.n_v_load
    epsilon = torch.finfo(dtype).eps
    quantiles = torch.clamp(quantiles, epsilon, 1.0 - epsilon)
    velocity_streams = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)

    position = x_loaded.repeat(config.n_v_load)
    velocity = velocity_streams.repeat_interleave(config.n_x_load)
    return position, velocity, domain_length


def particle_grid_geometry(
    position: torch.Tensor, nx: int, dx: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinate = position / dx
    left_floor = torch.floor(coordinate)
    left = torch.remainder(left_floor.to(torch.int64), nx)
    fraction = coordinate - left_floor
    right = torch.remainder(left + 1, nx)
    return left, right, fraction


def deposit_weighted_cic(
    left: torch.Tensor,
    right: torch.Tensor,
    fraction: torch.Tensor,
    values: torch.Tensor,
    nx: int,
    normalization: float,
) -> torch.Tensor:
    output = torch.zeros(nx, device=values.device, dtype=values.dtype)
    output.scatter_add_(0, left, (1.0 - fraction) * values)
    output.scatter_add_(0, right, fraction * values)
    return output * normalization


def solve_electric_field(
    left: torch.Tensor,
    right: torch.Tensor,
    fraction: torch.Tensor,
    particle_dtype: torch.dtype,
    particle_count: int,
    nx: int,
    wavenumbers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(particle_count, device=left.device, dtype=particle_dtype)
    density = deposit_weighted_cic(
        left, right, fraction, ones, nx, nx / particle_count
    )
    charge_hat = torch.fft.rfft(1.0 - density)
    electric_hat = torch.zeros_like(charge_hat)
    electric_hat[1:] = charge_hat[1:] / (1j * wavenumbers[1:])
    electric = torch.fft.irfft(electric_hat, n=nx)
    return electric, density


def interpolate_field(
    field: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    return (1.0 - fraction) * field[left] + fraction * field[right]


def deposit_raw_moments(
    velocity: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    fraction: torch.Tensor,
    density: torch.Tensor,
    nx: int,
) -> torch.Tensor:
    moments = [density]
    value = velocity
    normalization = nx / velocity.numel()
    for _order in range(1, 4):
        moments.append(
            deposit_weighted_cic(
                left, right, fraction, value, nx, normalization
            )
        )
        value = value * velocity
    return torch.stack(moments, dim=0)


def fluid_moments_from_raw(
    raw: torch.Tensor, domain_length: float
) -> dict[str, torch.Tensor]:
    density = raw[0]
    safe_density = torch.clamp(density, min=torch.finfo(raw.dtype).eps)
    velocity = raw[1] / safe_density
    pressure = raw[2] - 2.0 * velocity * raw[1] + velocity.square() * density
    pressure = torch.clamp(pressure, min=0.0)
    heat_flux = (
        raw[3]
        - 3.0 * velocity * raw[2]
        + 3.0 * velocity.square() * raw[1]
        - velocity.pow(3) * density
    )
    nx = density.numel()
    dx = domain_length / nx
    wave_numbers = 2.0 * math.pi * torch.fft.rfftfreq(
        nx, d=dx, device=raw.device, dtype=raw.dtype
    )
    heat_flux_gradient = torch.fft.irfft(
        1j * wave_numbers * torch.fft.rfft(heat_flux), n=nx
    )
    temperature = pressure / safe_density
    return {
        "density": density,
        "velocity": velocity,
        "pressure": pressure,
        "temperature": temperature,
        "heat_flux": heat_flux,
        "heat_flux_gradient": heat_flux_gradient,
    }


def phase_space_cic(
    position: torch.Tensor,
    velocity: torch.Tensor,
    domain_length: float,
    config: CudaPICConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase_dx = domain_length / config.phase_nx
    phase_dv = (config.phase_v_max - config.phase_v_min) / (config.phase_nv - 1)
    x_left, x_right, x_fraction = particle_grid_geometry(
        position, config.phase_nx, phase_dx
    )
    v_coordinate = (velocity - config.phase_v_min) / phase_dv
    inside = (v_coordinate >= 0.0) & (v_coordinate <= config.phase_nv - 1)
    v_coordinate = v_coordinate[inside]
    v_left = torch.clamp(
        torch.floor(v_coordinate).to(torch.int64), 0, config.phase_nv - 2
    )
    v_fraction = v_coordinate - v_left.to(velocity.dtype)
    v_right = v_left + 1
    x_left = x_left[inside]
    x_right = x_right[inside]
    x_fraction = x_fraction[inside]

    flat_size = config.phase_nx * config.phase_nv
    counts = torch.zeros(flat_size, device=position.device, dtype=position.dtype)
    for x_index, x_weight in (
        (x_left, 1.0 - x_fraction),
        (x_right, x_fraction),
    ):
        counts.scatter_add_(
            0,
            x_index * config.phase_nv + v_left,
            x_weight * (1.0 - v_fraction),
        )
        counts.scatter_add_(
            0,
            x_index * config.phase_nv + v_right,
            x_weight * v_fraction,
        )
    f_phase = counts.reshape(config.phase_nx, config.phase_nv)
    f_phase = f_phase * domain_length / (
        position.numel() * phase_dx * phase_dv
    )
    normalized_mass = torch.sum(f_phase) * phase_dx * phase_dv / domain_length
    overflow_fraction = 1.0 - torch.mean(inside.to(position.dtype))
    return f_phase, normalized_mass, overflow_fraction


def run_case(
    k_value: float,
    alpha: float,
    seed: int,
    config: CudaPICConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Run one case. Execution is refused unless the selected device is CUDA."""
    config.validate()
    if device.type != "cuda":
        raise RuntimeError("Production PIC generation requires a CUDA device")
    torch.cuda.set_device(device)
    dtype = torch_dtype(config.dtype)
    position, velocity_initial, domain_length = create_quiet_start(
        k_value, alpha, seed, config, device
    )
    dx = domain_length / config.field_nx
    wavenumbers = 2.0 * math.pi * torch.fft.rfftfreq(
        config.field_nx, d=dx, device=device, dtype=dtype
    )
    left, right, fraction = particle_grid_geometry(position, config.field_nx, dx)
    electric, _density = solve_electric_field(
        left,
        right,
        fraction,
        dtype,
        config.particle_count,
        config.field_nx,
        wavenumbers,
    )
    acceleration = -interpolate_field(electric, left, right, fraction)
    velocity_half = velocity_initial - 0.5 * config.dt * acceleration

    moment_stride = _cadence_steps(config.moment_dt, config.dt, "moment_dt")
    moment_count = config.step_count // moment_stride + 1
    output_dtype = torch.float32
    field_history = torch.empty(
        (moment_count, config.field_nx), device=device, dtype=output_dtype
    )
    raw_history = torch.empty(
        (moment_count, 4, config.field_nx), device=device, dtype=output_dtype
    )
    fluid_names = (
        "density",
        "velocity",
        "pressure",
        "temperature",
        "heat_flux",
        "heat_flux_gradient",
    )
    fluid_history = {
        name: torch.empty(
            (moment_count, config.field_nx), device=device, dtype=output_dtype
        )
        for name in fluid_names
    }
    field_energy = torch.empty(moment_count, device=device, dtype=torch.float64)
    kinetic_energy = torch.empty_like(field_energy)

    phase_stride = (
        _cadence_steps(config.phase_dt, config.dt, "phase_dt")
        if config.phase_dt is not None
        else None
    )
    phase_frames: list[torch.Tensor] = []
    phase_mass: list[torch.Tensor] = []
    phase_overflow: list[torch.Tensor] = []

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    moment_index = 0
    with torch.no_grad():
        for step in range(config.step_count + 1):
            left, right, fraction = particle_grid_geometry(
                position, config.field_nx, dx
            )
            electric, density = solve_electric_field(
                left,
                right,
                fraction,
                dtype,
                config.particle_count,
                config.field_nx,
                wavenumbers,
            )
            acceleration = -interpolate_field(electric, left, right, fraction)
            velocity_full = velocity_half + 0.5 * config.dt * acceleration

            if step % moment_stride == 0:
                raw = deposit_raw_moments(
                    velocity_full,
                    left,
                    right,
                    fraction,
                    density,
                    config.field_nx,
                )
                fluid = fluid_moments_from_raw(raw, domain_length)
                field_history[moment_index] = electric.to(output_dtype)
                raw_history[moment_index] = raw.to(output_dtype)
                for name in fluid_names:
                    fluid_history[name][moment_index] = fluid[name].to(output_dtype)
                field_energy[moment_index] = 0.5 * dx * torch.sum(electric.square())
                kinetic_energy[moment_index] = (
                    0.5 * domain_length * torch.mean(raw[2])
                )
                moment_index += 1

            if phase_stride is not None and step % phase_stride == 0:
                frame, mass, overflow = phase_space_cic(
                    position, velocity_full, domain_length, config
                )
                phase_frames.append(frame.to(output_dtype))
                phase_mass.append(mass)
                phase_overflow.append(overflow)

            if step == config.step_count:
                break
            velocity_half.add_(config.dt * acceleration)
            position = torch.remainder(
                position + config.dt * velocity_half, domain_length
            )

    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    total_energy = field_energy + kinetic_energy

    def host(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    result: dict[str, Any] = {
        "case": {"k": k_value, "alpha": alpha, "seed": int(seed)},
        "config": asdict(config),
        "domain_length": domain_length,
        "moment_time": np.arange(moment_count, dtype=np.float64)
        * config.moment_dt,
        "electric": host(field_history),
        "raw_moments": host(raw_history),
        "field_energy": host(field_energy),
        "kinetic_energy": host(kinetic_energy),
        "total_energy": host(total_energy),
        "runtime_seconds": runtime_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }
    result.update({name: host(value) for name, value in fluid_history.items()})
    if phase_frames:
        result.update(
            {
                "phase_time": np.arange(len(phase_frames), dtype=np.float64)
                * float(config.phase_dt),
                "f_phase": host(torch.stack(phase_frames, dim=0)),
                "phase_normalized_mass": host(torch.stack(phase_mass)),
                "phase_overflow_fraction": host(torch.stack(phase_overflow)),
                "phase_velocity": np.linspace(
                    config.phase_v_min,
                    config.phase_v_max,
                    config.phase_nv,
                    dtype=np.float64,
                ),
            }
        )
    return result


def save_case_hdf5(
    result: dict[str, Any],
    path: Path,
    split: str,
    compression: str = "lzf",
) -> None:
    """Atomically save one independently resumable case file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    if temporary.exists():
        temporary.unlink()
    kwargs: dict[str, Any] = {}
    if compression == "lzf":
        kwargs = {"compression": "lzf", "shuffle": True}
    elif compression == "gzip":
        kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    elif compression != "none":
        raise ValueError(compression)

    with h5py.File(temporary, "w", libver="latest") as handle:
        handle.attrs["status"] = "INCOMPLETE"
        handle.attrs["dataset_version"] = "nonlinear_cuda_pic_m03_v1"
        handle.attrs["case_json"] = json.dumps(result["case"], sort_keys=True)
        handle.attrs["config_json"] = json.dumps(result["config"], sort_keys=True)
        handle.attrs["split"] = split
        handle.attrs["domain_length"] = float(result["domain_length"])
        handle.attrs["runtime_seconds"] = float(result["runtime_seconds"])
        handle.attrs["peak_memory_bytes"] = int(result["peak_memory_bytes"])
        handle.attrs["initial_density_loading"] = "exact_inverse_cdf"
        handle.attrs["compute_backend"] = "torch_cuda"

        grids = handle.create_group("grids")
        grids.create_dataset("moment_time", data=result["moment_time"])
        grids.create_dataset(
            "normalized_x",
            data=np.arange(result["electric"].shape[1], dtype=np.float64)
            / result["electric"].shape[1],
        )
        if "phase_time" in result:
            grids.create_dataset("phase_time", data=result["phase_time"])
            grids.create_dataset("phase_velocity", data=result["phase_velocity"])

        fields = handle.create_group("fields")
        fields.create_dataset("electric", data=result["electric"], **kwargs)
        raw = handle.create_group("raw_moments")
        for order in range(4):
            raw.create_dataset(
                f"m{order}", data=result["raw_moments"][:, order], **kwargs
            )
        fluid = handle.create_group("fluid")
        for name in (
            "density",
            "velocity",
            "pressure",
            "temperature",
            "heat_flux",
            "heat_flux_gradient",
        ):
            fluid.create_dataset(name, data=result[name], **kwargs)
        energies = handle.create_group("energies")
        for name in ("field_energy", "kinetic_energy", "total_energy"):
            energies.create_dataset(name, data=result[name])
        if "f_phase" in result:
            phase = handle.create_group("phase_space")
            phase.create_dataset(
                "f", data=result["f_phase"], chunks=(1,) + result["f_phase"].shape[1:], **kwargs
            )
            phase.create_dataset(
                "normalized_mass", data=result["phase_normalized_mass"]
            )
            phase.create_dataset(
                "overflow_fraction", data=result["phase_overflow_fraction"]
            )

        arrays = [
            result["electric"],
            result["raw_moments"],
            result["pressure"],
            result["heat_flux"],
            result["heat_flux_gradient"],
            result["total_energy"],
        ]
        if not all(np.isfinite(value).all() for value in arrays):
            raise FloatingPointError("Refusing to save non-finite PIC output")
        handle.attrs["heat_flux_gradient_mean_abs_max"] = float(
            np.max(np.abs(np.mean(result["heat_flux_gradient"], axis=-1)))
        )
        handle.attrs["total_energy_relative_span"] = float(
            np.ptp(result["total_energy"] / result["total_energy"][0] - 1.0)
        )
        handle.attrs["status"] = "COMPLETE"
        handle.flush()
    temporary.replace(path)
