"""Consolidate and validate the strict Huang-2025 Gkeyll single case."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import postgkyl as pg
from scipy.io import loadmat


FRAME_RE = re.compile(r"_M0_(\d+)\.gkyl$")
DISTRIBUTION_FRAME_RE = re.compile(r"-elc_(\d+)\.gkyl$")


def rel_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-14))


def correlation(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.corrcoef(value.ravel(), reference.ravel())[0, 1])


def modal_centers(path: Path, component: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate a p2 modal field at each cell center."""
    data = pg.GData(str(path))
    coefficients = np.asarray(data.get_values(), dtype=np.float64)
    lo = 3 * component
    # Gkeyll's orthonormal 1D p2 basis at xi=0 is
    # [1/sqrt(2), 0, -sqrt(5/8)].  Evaluating it directly avoids
    # rebuilding a symbolic Postgkyl interpolation matrix for every frame.
    values = coefficients[:, lo] / np.sqrt(2.0) - np.sqrt(5.0 / 8.0) * coefficients[:, lo + 2]
    edges = np.asarray(data.get_grid()[0], dtype=np.float64)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, values, float(data.ctx["time"])


def modal_phase(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = pg.GData(str(path))
    grid, values = pg.GInterpModal(data, num_interp=3).interpolate(0)
    x = 0.5 * (grid[0][1:] + grid[0][:-1])
    v = 0.5 * (grid[1][1:] + grid[1][:-1])
    return x, v, np.asarray(values[..., 0], dtype=np.float64), float(data.ctx["time"])


def frame_map(run_dir: Path) -> tuple[str, dict[int, Path]]:
    paths = sorted(run_dir.glob("*-elc_M0_*.gkyl"))
    if not paths:
        raise FileNotFoundError(f"No M0 frames found in {run_dir}")
    result: dict[int, Path] = {}
    for path in paths:
        match = FRAME_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    prefix = paths[0].name.split("-elc_M0_")[0]
    return prefix, result


def load_dynvec(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pg.GData(str(path))
    return np.asarray(data.get_grid()[0], dtype=np.float64), np.asarray(data.get_values(), dtype=np.float64)


def consolidate(
    run_dir: Path,
    output: Path,
    phase_targets: list[float],
    *,
    k: float = 0.35,
    alpha: float = 0.10,
) -> dict:
    prefix, frames = frame_map(run_dir)
    frame_ids = np.array(sorted(frames), dtype=np.int64)
    first_x, first_m0, first_time = modal_centers(frames[int(frame_ids[0])])
    count, nx = len(frame_ids), len(first_x)
    raw = {name: np.empty((count, nx), dtype=np.float64) for name in ("M0", "M1", "M2", "M3")}
    time = np.empty(count, dtype=np.float64)

    for row, frame in enumerate(frame_ids):
        for name in raw:
            path = run_dir / f"{prefix}-elc_{name}_{frame}.gkyl"
            x, raw[name][row], current_time = modal_centers(path)
            if not np.allclose(x, first_x, rtol=0.0, atol=1.0e-13):
                raise ValueError(f"Grid mismatch in {path}")
            if name == "M0":
                time[row] = current_time

    density = raw["M0"]
    velocity = raw["M1"] / density
    pressure = raw["M2"] - density * velocity**2
    heat_flux = raw["M3"] - 3.0 * velocity * raw["M2"] + 2.0 * density * velocity**3

    distribution_ids = []
    for path in run_dir.glob(f"{prefix}-elc_*.gkyl"):
        match = DISTRIBUTION_FRAME_RE.search(path.name)
        if match:
            distribution_ids.append(int(match.group(1)))
    if not distribution_ids:
        raise FileNotFoundError(f"No distribution-function frames found in {run_dir}")
    time_by_frame = {int(frame): float(current) for frame, current in zip(frame_ids, time)}
    available_phase_ids = [frame for frame in sorted(distribution_ids) if frame in time_by_frame]
    phase_ids = [
        min(available_phase_ids, key=lambda frame: abs(time_by_frame[frame] - target))
        for target in phase_targets
    ]
    phase_x = phase_v = None
    phase_f, phase_time = [], []
    for frame in phase_ids:
        path = run_dir / f"{prefix}-elc_{frame}.gkyl"
        px, pv, pf, pt = modal_phase(path)
        if phase_x is None:
            phase_x, phase_v = px, pv
        phase_f.append(pf)
        phase_time.append(pt)

    energy_path = run_dir / f"{prefix}-field-energy.gkyl"
    energy_time, energy = load_dynvec(energy_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs.update(
            solver="Gkeyll Vlasov-Ampere CUDA",
            prefix=prefix,
            k=k,
            alpha=alpha,
            nx=64,
            nv=64,
            velocity_min=-6.0,
            velocity_max=6.0,
            polynomial_order=2,
            basis="serendipity",
        )
        handle.create_dataset("frame", data=frame_ids)
        handle.create_dataset("time", data=time)
        handle.create_dataset("x", data=first_x)
        raw_group = handle.create_group("raw_moments")
        for name, value in raw.items():
            raw_group.create_dataset(name, data=value, compression="gzip", shuffle=True)
        central = handle.create_group("central_moments")
        central.create_dataset("density", data=density, compression="gzip", shuffle=True)
        central.create_dataset("velocity", data=velocity, compression="gzip", shuffle=True)
        central.create_dataset("pressure", data=pressure, compression="gzip", shuffle=True)
        central.create_dataset("heat_flux", data=heat_flux, compression="gzip", shuffle=True)
        field = handle.create_group("field")
        field.create_dataset("energy_time", data=energy_time)
        field.create_dataset("energy", data=energy)
        phase = handle.create_group("phase")
        phase.create_dataset("frame", data=phase_ids)
        phase.create_dataset("time", data=phase_time)
        phase.create_dataset("x", data=phase_x)
        phase.create_dataset("v", data=phase_v)
        phase.create_dataset("f", data=np.asarray(phase_f), compression="gzip", shuffle=True)

    analytic_m0 = 1.0 + alpha * np.cos(k * first_x)
    return {
        "prefix": prefix,
        "frame_count": count,
        "first_frame": int(frame_ids[0]),
        "last_frame": int(frame_ids[-1]),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "dt_median": float(np.median(np.diff(time))) if count > 1 else None,
        "dt_min": float(np.min(np.diff(time))) if count > 1 else None,
        "dt_max": float(np.max(np.diff(time))) if count > 1 else None,
        "initial_M0_relative_l2_vs_analytic_cell_centers": rel_l2(raw["M0"][0], analytic_m0),
        "initial_M1_rms": float(np.sqrt(np.mean(raw["M1"][0] ** 2))),
        "initial_M2_relative_l2_vs_M0": rel_l2(raw["M2"][0], raw["M0"][0]),
        "initial_M3_rms": float(np.sqrt(np.mean(raw["M3"][0] ** 2))),
        "phase_frames": phase_ids,
        "phase_times": phase_time,
        "consolidated_h5": str(output),
        "raw_bytes": sum(path.stat().st_size for path in run_dir.glob("*.gkyl")),
    }


def interpolate_series(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(target_time, source_time, values[:, j]) for j in range(values.shape[1])], axis=1)


def compare_public(h5_path: Path, public_mat: Path, output_dir: Path) -> dict:
    public = loadmat(public_mat)
    public_time = np.arange(public["n_new"].shape[0], dtype=np.float64) * 0.005
    with h5py.File(h5_path, "r") as handle:
        time = np.asarray(handle["time"])
        x = np.asarray(handle["x"])
        gkeyll = {
            "density": np.asarray(handle["raw_moments/M0"]),
            "velocity": np.asarray(handle["raw_moments/M1"]) / np.asarray(handle["raw_moments/M0"]),
            "M2": np.asarray(handle["raw_moments/M2"]),
            "M3": np.asarray(handle["raw_moments/M3"]),
        }
        energy_time = np.asarray(handle["field/energy_time"])
        energy = np.asarray(handle["field/energy"])
        phase_time = np.asarray(handle["phase/time"])
        phase_x = np.asarray(handle["phase/x"])
        phase_v = np.asarray(handle["phase/v"])
        phase_f = np.asarray(handle["phase/f"])

    selected = public_time <= time[-1] + 1.0e-10
    target_time = public_time[selected]
    reference = {
        "density": np.asarray(public["n_new"], dtype=np.float64)[selected],
        "velocity": np.asarray(public["u_new"], dtype=np.float64)[selected],
        "M2": np.asarray(public["p_new"], dtype=np.float64)[selected],
        "M3": np.asarray(public["q_new"], dtype=np.float64)[selected],
    }
    matched = {name: interpolate_series(time, value, target_time) for name, value in gkeyll.items()}
    # The public A01 record starts with the same single Fourier mode but an
    # undocumented spatial phase (five cells relative to cos(kx) on our grid).
    # Translation is a symmetry of the periodic problem, so determine one
    # integer roll from M0(t=0) and apply it unchanged to every field/time.
    spatial_roll = min(
        range(reference["density"].shape[1]),
        key=lambda shift: np.linalg.norm(np.roll(matched["density"][0], shift) - reference["density"][0]),
    )
    matched = {name: np.roll(value, spatial_roll, axis=1) for name, value in matched.items()}
    metrics = {
        name: {"relative_l2": rel_l2(matched[name], reference[name]), "correlation": correlation(matched[name], reference[name])}
        for name in matched
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    extent = (target_time[0], target_time[-1], 0.0, 1.0)
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey=True)
    for row, name in enumerate(("density", "velocity", "M2", "M3")):
        scale = float(np.quantile(np.abs(reference[name] - (1.0 if name in ("density", "M2") else 0.0)), 0.995))
        offset = 1.0 if name in ("density", "M2") else 0.0
        difference = matched[name] - reference[name]
        axes[row, 0].imshow((reference[name] - offset).T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        axes[row, 1].imshow((matched[name] - offset).T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        dscale = float(np.quantile(np.abs(difference), 0.995))
        axes[row, 2].imshow(difference.T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-dscale, vmax=dscale)
        axes[row, 0].set_ylabel(f"{name}\nx/L")
    for axis, title in zip(axes[0], ("Huang public", "new CUDA Gkeyll", "new - public")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output_dir / "moments_public_comparison_xt.png", dpi=180)
    plt.close(fig)

    modes = np.arange(reference["density"].shape[1] // 2 + 1) * 0.35
    rho_hat = np.fft.rfft(1.0 - reference["density"], axis=1)
    e_hat = np.zeros_like(rho_hat)
    e_hat[:, 1:] = rho_hat[:, 1:] / (1j * modes[1:])
    public_e = np.fft.irfft(e_hat, n=reference["density"].shape[1], axis=1)
    # Huang Fig. 4 and Gkeyll's field-energy diagnostic use integral |Ex|^2 dx,
    # without the conventional electromagnetic-energy factor of 1/2.
    public_energy = (2.0 * np.pi / 0.35) * np.mean(public_e**2, axis=1)
    g_energy = energy[:, 0] if energy.ndim == 2 else energy
    matched_g_energy = np.interp(target_time, energy_time, g_energy)
    field_energy_metrics = {
        "relative_l2": rel_l2(matched_g_energy, public_energy),
        "correlation": correlation(matched_g_energy, public_energy),
        "log10_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        np.log10(np.maximum(matched_g_energy, 1.0e-12))
                        - np.log10(np.maximum(public_energy, 1.0e-12))
                    )
                    ** 2
                )
            )
        ),
        "initial_new": float(matched_g_energy[0]),
        "initial_public_reconstructed": float(public_energy[0]),
    }
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogy(target_time, public_energy, label="public moments: Poisson-reconstructed")
    axis.semilogy(energy_time, g_energy, label="new CUDA Gkeyll", alpha=0.85)
    axis.set(xlabel="time", ylabel=r"$\int |E_x|^2\,dx$", xlim=(0.0, target_time[-1]))
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "long_time_field_energy.png", dpi=180)
    plt.close(fig)

    ncol = len(phase_time)
    fig, axes = plt.subplots(3, ncol, figsize=(3.1 * ncol, 8.8), sharex=True)
    equilibrium = phase_f[0].mean(axis=0, keepdims=True)
    perturbation = phase_f - equilibrium[None, ...]
    # Huang Fig. 4 crops the nonlinear snapshots to the positive phase-speed
    # branch, approximately vx in [2, 4.5].  A symmetric [-3, 3] crop hides
    # the trapped-particle island centered near vx=3.5.
    resonant = (phase_v >= 2.0) & (phase_v <= 4.5)
    row_maxima = (
        float(np.quantile(phase_f, 0.999)),
        float(np.quantile(phase_f[:, :, resonant], 0.999)),
    )
    perturbation_scale = float(np.quantile(np.abs(perturbation[:, :, resonant]), 0.995))
    for col, current_time in enumerate(phase_time):
        for row, (vmin, vmax, label) in enumerate(((-6.0, 6.0, "f, full velocity range"), (2.0, 4.5, "f, paper resonant range"))):
            mask = (phase_v >= vmin) & (phase_v <= vmax)
            image = axes[row, col].pcolormesh(
                phase_x / (2.0 * np.pi / 0.35),
                phase_v[mask],
                phase_f[col][:, mask].T,
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=row_maxima[row],
            )
            axes[row, col].set_title(f"t={current_time:.2f}")
            axes[row, col].set_ylim(vmin, vmax)
            if col == 0:
                axes[row, col].set_ylabel(f"v\n{label}")
            axes[row, col].set_xlabel("x/L")
            fig.colorbar(image, ax=axes[row, col], shrink=0.72)
        image = axes[2, col].pcolormesh(
            phase_x / (2.0 * np.pi / 0.35),
            phase_v[resonant],
            perturbation[col][:, resonant].T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-perturbation_scale,
            vmax=perturbation_scale,
        )
        axes[2, col].set_ylim(2.0, 4.5)
        axes[2, col].set_xlabel("x/L")
        if col == 0:
            axes[2, col].set_ylabel("v\nδf, resonant range")
        fig.colorbar(image, ax=axes[2, col], shrink=0.72)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_space_full_and_resonant.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(output_dir / "public_comparison.npz", time=target_time, x=x, **{f"new_{k}": v for k, v in matched.items()}, **{f"public_{k}": v for k, v in reference.items()})
    return {
        "public_mat": str(public_mat),
        "matched_time_count": len(target_time),
        "matched_time_end": float(target_time[-1]),
        "spatial_roll_cells_new_to_public": spatial_roll,
        "metrics": metrics,
        "field_energy_metrics": field_energy_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--public-mat", type=Path)
    parser.add_argument("--k", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--phase-times", type=float, nargs="*", default=[0.0, 10.0, 20.0, 30.0, 35.0, 40.0])
    args = parser.parse_args()
    summary = consolidate(
        args.run_dir, args.output, args.phase_times, k=args.k, alpha=args.alpha
    )
    if args.public_mat and args.figures:
        summary["public_comparison"] = compare_public(args.output, args.public_mat, args.figures)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
