"""Expose nonlinear phase-space structure hidden by the Maxwellian background."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=(20, 30, 40, 50, 60))
    args = parser.parse_args()
    with h5py.File(args.case, "r") as handle:
        phase_time = np.asarray(handle["grids/phase_time"], dtype=np.float64)
        velocity = np.asarray(handle["grids/phase_velocity"], dtype=np.float64)
        moment_time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
        field_energy = np.asarray(handle["energies/field_energy"], dtype=np.float64)
        indices = [int(np.argmin(np.abs(phase_time - value))) for value in args.times]
        phase = np.asarray(handle["phase_space/f"][indices], dtype=np.float32)
    delta = phase - phase.mean(axis=1, keepdims=True)
    full_maximum = float(np.quantile(phase, 0.9995))
    delta_maximum = float(np.quantile(np.abs(delta), 0.995))
    columns = len(indices)
    figure, axes = plt.subplots(2, columns, figsize=(3.15 * columns, 6.3), sharex=True, sharey=True)
    extent = (0.0, 1.0, float(velocity[0]), float(velocity[-1]))
    for column, index in enumerate(indices):
        full_image = axes[0, column].imshow(
            phase[column].T, origin="lower", aspect="auto", extent=extent,
            cmap="magma", vmin=0.0, vmax=full_maximum,
        )
        delta_image = axes[1, column].imshow(
            delta[column].T, origin="lower", aspect="auto", extent=extent,
            cmap="RdBu_r", vmin=-delta_maximum, vmax=delta_maximum,
        )
        axes[0, column].set_title(f"t={phase_time[index]:.0f}")
        axes[1, column].set_xlabel("x/L")
    axes[0, 0].set_ylabel("full f\nvelocity v")
    axes[1, 0].set_ylabel(r"$f-\langle f\rangle_x$" + "\nvelocity v")
    figure.colorbar(full_image, ax=axes[0].tolist(), shrink=0.78, pad=0.015, label="f")
    figure.colorbar(delta_image, ax=axes[1].tolist(), shrink=0.78, pad=0.015, label=r"$\delta f$")
    figure.suptitle("Nonlinear phase-space audit: full distribution versus background-subtracted structure")
    figure.subplots_adjust(left=0.07, right=0.92, bottom=0.08, top=0.89, wspace=0.12, hspace=0.14)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "phase_space_full_vs_delta.png", dpi=190)
    plt.close(figure)

    # Huang et al. show the nonlinear panels only around the positive resonant
    # velocity (roughly v=2..4.5), rather than on the full v=-6..6 domain.
    resonant = (velocity >= 2.0) & (velocity <= 4.5)
    resonant_phase = phase[:, :, resonant]
    resonant_maximum = float(np.quantile(resonant_phase, 0.997))
    zoom_figure, zoom_axes = plt.subplots(
        1, columns, figsize=(3.15 * columns, 3.25), sharex=True, sharey=True,
    )
    if columns == 1:
        zoom_axes = np.asarray([zoom_axes])
    zoom_extent = (0.0, 1.0, float(velocity[resonant][0]), float(velocity[resonant][-1]))
    for column, index in enumerate(indices):
        zoom_image = zoom_axes[column].imshow(
            resonant_phase[column].T, origin="lower", aspect="auto", extent=zoom_extent,
            cmap="magma", vmin=0.0, vmax=resonant_maximum,
        )
        zoom_axes[column].set_title(f"t={phase_time[index]:.0f}")
        zoom_axes[column].set_xlabel("x/L")
    zoom_axes[0].set_ylabel("resonant velocity v")
    zoom_figure.colorbar(
        zoom_image, ax=zoom_axes.tolist(), shrink=0.78, pad=0.015, label="f"
    )
    zoom_figure.suptitle("Positive resonant-velocity zoom (same window used qualitatively in the paper)")
    zoom_figure.subplots_adjust(left=0.07, right=0.92, bottom=0.17, top=0.82, wspace=0.12)
    zoom_figure.savefig(args.output_dir / "phase_space_resonant_velocity_zoom.png", dpi=190)
    plt.close(zoom_figure)

    minimum_index = int(np.argmin(field_energy))
    summary = {
        "case": str(args.case),
        "field_energy_initial": float(field_energy[0]),
        "field_energy_minimum": float(field_energy[minimum_index]),
        "field_energy_minimum_time": float(moment_time[minimum_index]),
        "field_energy_at_requested_times": {
            f"{phase_time[index]:g}": float(field_energy[np.argmin(np.abs(moment_time - phase_time[index]))])
            for index in indices
        },
        "revival_factor_minimum_to_t50": float(
            field_energy[np.argmin(np.abs(moment_time - 50.0))] / field_energy[minimum_index]
        ),
        "delta_f_relative_rms": {
            f"{phase_time[index]:g}": float(
                np.sqrt(np.mean(delta[column] ** 2)) / np.sqrt(np.mean(phase[column] ** 2))
            ) for column, index in enumerate(indices)
        },
        "interpretation": "Full f is dominated by the Maxwellian background; delta f reveals trapped-particle/filamentary structure.",
    }
    (args.output_dir / "nonlinearity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
