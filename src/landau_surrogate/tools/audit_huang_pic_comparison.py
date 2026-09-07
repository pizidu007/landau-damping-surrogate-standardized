"""Compare time/grid-matched CUDA-PIC moments with the Huang Gkeyll record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from landau_surrogate.data.huang2025 import load_huang_mat, spectral_derivative


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-12))


def interpolate_time(value: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(target_time, source_time, value[:, j]) for j in range(value.shape[1])], axis=1)


def lowpass(value: np.ndarray, maximum_mode: int) -> np.ndarray:
    transformed = np.fft.rfft(value, axis=-1)
    transformed[..., maximum_mode + 1 :] = 0.0
    return np.fft.irfft(transformed, n=value.shape[-1], axis=-1)


def infer_snapshot_dt_from_continuity(state: np.ndarray, k_value: float, count: int = 8_000) -> float:
    """Infer file cadence from dn/dt + d(nu)/dx = 0 without metadata."""
    selected = state[: min(len(state), count)]
    density, velocity = selected[:, 0], selected[:, 1]
    half_difference = 0.5 * (density[2:] - density[:-2])
    current = density[1:-1] * velocity[1:-1]
    modes = np.arange(current.shape[-1] // 2 + 1, dtype=np.float64) * k_value
    negative_current_gradient = -np.fft.irfft(
        1j * modes * np.fft.rfft(current, axis=-1), n=current.shape[-1], axis=-1
    )
    return float(
        np.sum(half_difference * negative_current_gradient)
        / np.sum(negative_current_gradient ** 2)
    )


def fit_pressure_closure_scale(
    state: np.ndarray, heat_flux_gradient: np.ndarray, k_value: float,
    dt: float, count: int = 4_800,
) -> float:
    """Fit the missing public-data q normalization on the causal training interval."""
    selected = state[: min(len(state), count)]
    density, velocity, pressure = selected[:, 0], selected[:, 1], selected[:, 2]
    del density  # retained in the signature to make the primitive-state contract explicit
    modes = np.arange(pressure.shape[-1] // 2 + 1, dtype=np.float64) * k_value
    def derivative(value: np.ndarray) -> np.ndarray:
        return np.fft.irfft(
            1j * modes * np.fft.rfft(value, axis=-1), n=value.shape[-1], axis=-1
        )
    pressure_lhs = (
        (pressure[2:] - pressure[:-2]) / (2.0 * dt)
        + velocity[1:-1] * derivative(pressure[1:-1])
        + 3.0 * pressure[1:-1] * derivative(velocity[1:-1])
    )
    unit_source = -heat_flux_gradient[1 : len(selected) - 1]
    return float(np.sum(unit_source * pressure_lhs) / np.sum(unit_source ** 2))


def raw_moment_equation_audit(
    state: np.ndarray, heat_flux_gradient: np.ndarray, k_value: float,
    dt: float = 0.005, count: int = 4_800,
) -> dict[str, dict[str, float]]:
    """Check whether exported ``p``/``q`` behave as raw M2/M3 moments."""
    selected = state[: min(len(state), count)]
    density, velocity, second_moment = selected[:, 0], selected[:, 1], selected[:, 2]
    momentum = density * velocity
    modes = np.arange(density.shape[-1] // 2 + 1, dtype=np.float64) * k_value
    def derivative(value: np.ndarray) -> np.ndarray:
        return np.fft.irfft(
            1j * modes * np.fft.rfft(value, axis=-1), n=value.shape[-1], axis=-1
        )
    charge_hat = np.fft.rfft(1.0 - density[1:-1], axis=-1)
    electric_hat = np.zeros_like(charge_hat)
    electric_hat[:, 1:] = charge_hat[:, 1:] / (1j * modes[1:])
    electric = np.fft.irfft(electric_hat, n=density.shape[-1], axis=-1)
    left = (
        (density[2:] - density[:-2]) / (2.0 * dt),
        (momentum[2:] - momentum[:-2]) / (2.0 * dt),
        (second_moment[2:] - second_moment[:-2]) / (2.0 * dt),
    )
    right = (
        -derivative(momentum[1:-1]),
        -derivative(second_moment[1:-1]) - density[1:-1] * electric,
        -heat_flux_gradient[1 : len(selected) - 1] - 2.0 * momentum[1:-1] * electric,
    )
    result = {}
    for name, lhs, rhs in zip(("M0", "M1", "M2"), left, right):
        result[name] = {
            "relative_residual": relative_l2(lhs, rhs),
            "correlation": float(np.corrcoef(lhs.ravel(), rhs.ravel())[0, 1]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gkeyll-mat", type=Path, required=True)
    parser.add_argument("--pic-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gkeyll = load_huang_mat(args.gkeyll_mat)
    paths = sorted((args.pic_run_dir / "cases").glob("*.h5"))
    if len(paths) < 2:
        raise ValueError("At least two completed PIC replicas are required")
    pic_values: dict[str, list[np.ndarray]] = {name: [] for name in ("density", "velocity", "pressure", "heat_flux", "heat_flux_gradient")}
    pic_time = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            current_time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
            if pic_time is None:
                pic_time = current_time
            elif not np.array_equal(pic_time, current_time):
                raise ValueError(f"PIC time mismatch in {path}")
            for name in pic_values:
                pic_values[name].append(np.asarray(handle[f"fluid/{name}"], dtype=np.float64))
    assert pic_time is not None
    stacks = {name: np.stack(values) for name, values in pic_values.items()}
    means = {name: values.mean(axis=0) for name, values in stacks.items()}
    stds = {name: values.std(axis=0) for name, values in stacks.items()}

    # The public arrays satisfy raw-moment equations: p_new=M2 and
    # q_new=M3.  Convert them to the central moments emitted by our PIC
    # generator before making a solver-to-solver comparison.
    g_density, g_velocity, g_second = (
        gkeyll.state[:, 0], gkeyll.state[:, 1], gkeyll.state[:, 2]
    )
    g_pressure = g_second - g_density * g_velocity ** 2
    g_heat_flux = (
        gkeyll.heat_flux - 3.0 * g_velocity * g_second
        + 2.0 * g_density * g_velocity ** 3
    )
    g_state = {
        "density": g_density, "velocity": g_velocity,
        "pressure": g_pressure, "heat_flux": g_heat_flux,
        "heat_flux_gradient": spectral_derivative(g_heat_flux, gkeyll.k),
    }
    interpolated = {
        name: interpolate_time(value, gkeyll.time, pic_time) for name, value in g_state.items()
    }
    initial_pic = means["density"][0]
    initial_gkeyll = interpolated["density"][0]
    shift = max(
        range(initial_pic.shape[-1]),
        key=lambda candidate: np.corrcoef(np.roll(initial_pic, candidate), initial_gkeyll)[0, 1],
    )
    means = {name: np.roll(value, shift, axis=-1) for name, value in means.items()}
    stds = {name: np.roll(value, shift, axis=-1) for name, value in stds.items()}
    stacks = {name: np.roll(value, shift, axis=-1) for name, value in stacks.items()}

    intervals = {"linear_0_15": (0.0, 15.0), "transition_15_30": (15.0, 30.0), "nonlinear_30_40": (30.0, 40.0)}
    field_metrics = {}
    for name in means:
        rows = {}
        for label, (start, end) in intervals.items():
            selected = (pic_time >= start) & (pic_time <= end)
            rows[label] = {
                "relative_l2_pic_vs_gkeyll": relative_l2(means[name][selected], interpolated[name][selected]),
                "correlation": float(np.corrcoef(means[name][selected].ravel(), interpolated[name][selected].ravel())[0, 1]),
                "pic_seed_noise_to_pic_signal": float(
                    np.sqrt(np.mean(stds[name][selected] ** 2)) /
                    max(np.sqrt(np.mean(means[name][selected] ** 2)), 1.0e-12)
                ),
            }
        field_metrics[name] = rows

    spectral = {}
    for maximum_mode in (1, 4, 8, 16, 24, 32):
        spectral[str(maximum_mode)] = {
            "pic_vs_gkeyll_relative_l2": relative_l2(
                lowpass(means["heat_flux_gradient"], maximum_mode),
                lowpass(interpolated["heat_flux_gradient"], maximum_mode),
            ),
            "pic_seed_noise_to_signal": float(
                np.sqrt(np.mean(lowpass(
                    stacks["heat_flux_gradient"] - means["heat_flux_gradient"][None],
                    maximum_mode,
                ) ** 2)) /
                max(np.sqrt(np.mean(lowpass(means["heat_flux_gradient"], maximum_mode) ** 2)), 1.0e-12)
            ),
        }
    inferred_dt = infer_snapshot_dt_from_continuity(gkeyll.state, gkeyll.k)
    summary = {
        "gkeyll_source": str(args.gkeyll_mat),
        "pic_run_dir": str(args.pic_run_dir),
        "pic_replica_count": len(paths),
        "gkeyll_continuity_inferred_dt": inferred_dt,
        "gkeyll_primitive_interpretation_best_fit_closure_scale_t0_t24": fit_pressure_closure_scale(
            gkeyll.state, gkeyll.heat_flux_gradient, gkeyll.k, inferred_dt
        ),
        "gkeyll_raw_moment_equation_audit_t0_t24": raw_moment_equation_audit(
            gkeyll.state, gkeyll.heat_flux_gradient, gkeyll.k
        ),
        "spatial_roll_cells_pic_to_gkeyll": shift,
        "time_count": len(pic_time),
        "field_metrics": field_metrics,
        "heat_flux_gradient_spectral_metrics": spectral,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "comparison.npz", time=pic_time,
        pic_dqdx=means["heat_flux_gradient"], gkeyll_dqdx=interpolated["heat_flux_gradient"],
        pic_dqdx_seed_std=stds["heat_flux_gradient"], pic_state=np.stack([means[n] for n in ("density", "velocity", "pressure")], axis=1),
        gkeyll_state=np.stack([interpolated[n] for n in ("density", "velocity", "pressure")], axis=1),
    )
    extent = (pic_time[0], pic_time[-1], 0.0, 1.0)
    scale = float(np.quantile(np.abs(interpolated["heat_flux_gradient"]), 0.995))
    difference = np.abs(means["heat_flux_gradient"] - interpolated["heat_flux_gradient"])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for axis, value, title in zip(
        axes[:3],
        (interpolated["heat_flux_gradient"], means["heat_flux_gradient"], stds["heat_flux_gradient"]),
        ("Gkeyll", "PIC three-seed mean", "PIC seed std"),
    ):
        axis.imshow(value.T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set_title(title); axis.set_xlabel("time")
    image = axes[3].imshow(difference.T, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0, vmax=float(np.quantile(difference, 0.995)))
    axes[3].set_title("absolute PIC-Gkeyll difference"); axes[3].set_xlabel("time")
    axes[0].set_ylabel("x/L")
    fig.colorbar(image, ax=axes[3], shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "pic_gkeyll_dqdx_xt.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
