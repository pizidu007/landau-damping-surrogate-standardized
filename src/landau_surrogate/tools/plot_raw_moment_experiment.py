"""Plot the long rollout and resonant phase-space audit for the raw-moment FNO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, resample


def _load_pic(case_dir: Path, pair_id: str) -> dict[str, np.ndarray]:
    paths = sorted(case_dir.glob(f"{pair_id}_s*.h5"))
    if not paths:
        raise FileNotFoundError(f"No PIC cases matching {pair_id} in {case_dir}")
    fields = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            fields.append(np.asarray(handle["energies/field_energy"], dtype=np.float64))
            if path == paths[0]:
                moment_time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
                phase_time = np.asarray(handle["grids/phase_time"], dtype=np.float64)
                velocity = np.asarray(handle["grids/phase_velocity"], dtype=np.float64)
                phase = np.asarray(handle["phase_space/f"], dtype=np.float32)
    return {
        "moment_time": moment_time,
        "field_energy": np.stack(fields),
        "phase_time": phase_time,
        "velocity": velocity,
        "phase": phase,
        "replicas": np.asarray([len(paths)]),
    }


def _maxwellian(public_state: np.ndarray, velocity: np.ndarray, nx: int) -> np.ndarray:
    density = np.maximum(public_state[0], 1.0e-6)
    flow = public_state[1]
    second = public_state[2]
    temperature = np.maximum(second / density - flow * flow, 1.0e-5)
    density = resample(density, nx)
    flow = resample(flow, nx)
    temperature = np.maximum(resample(temperature, nx), 1.0e-5)
    return density[:, None] / np.sqrt(2.0 * np.pi * temperature[:, None]) * np.exp(
        -0.5 * (velocity[None, :] - flow[:, None]) ** 2 / temperature[:, None]
    )


def _plot_field(
    pic: dict[str, np.ndarray],
    rollouts: list[tuple[str, dict[str, np.ndarray]]],
    output: Path,
) -> dict:
    time = pic["moment_time"]
    curves = pic["field_energy"]
    mean = curves.mean(axis=0)
    low, high = curves.min(axis=0), curves.max(axis=0)
    distance = max(1, int(round(1.5 / np.median(np.diff(time)))))
    pic_peaks, _ = find_peaks(mean, distance=distance)
    colors = ("#d95f02", "#1b9e77", "#7570b3", "#e7298a")

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.35))
    axes[0].fill_between(time, low, high, color="0.7", alpha=0.28, label="PIC 3-seed range")
    axes[0].semilogy(time, mean, color="black", linewidth=1.8, label="PIC mean")
    peak_counts: dict[str, int] = {}
    completed_times: dict[str, float] = {}
    for (label, rollout), color in zip(rollouts, colors):
        model_time = rollout["time"]
        model = rollout["field_energy"]
        axes[0].semilogy(model_time, model, color=color, linewidth=1.65, label=label)
        axes[0].axvline(model_time[-1], color=color, linestyle=":", linewidth=1.0, alpha=0.75)
        model_distance = max(1, int(round(1.5 / np.median(np.diff(model_time)))))
        model_peaks, _ = find_peaks(model, distance=model_distance)
        axes[1].semilogy(model_time[model_peaks], model[model_peaks], "o-", color=color,
                        markersize=3.2, linewidth=1.5, label=label)
        peak_counts[label] = int(len(model_peaks))
        completed_times[label] = float(model_time[-1])
    axes[0].set(xlabel="time", ylabel=r"electric-field energy $W_E$", xlim=(0, 60))
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].semilogy(time[pic_peaks], mean[pic_peaks], "o-", color="black", markersize=3.2,
                    linewidth=1.5, label="PIC peak envelope")
    axes[1].set(xlabel="time", ylabel=r"peak envelope of $W_E$", xlim=(0, 60))
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle(r"Long-time field-energy evolution: $k=0.35$, $A=0.10$")
    figure.tight_layout()
    figure.savefig(output / "long_time_field_energy.png", dpi=210)
    plt.close(figure)
    return {
        "pic_peak_count": int(len(pic_peaks)),
        "fno_peak_count": peak_counts,
        "fno_completed_time": completed_times,
    }


def _plot_phase(
    pic: dict[str, np.ndarray], rollout: dict[str, np.ndarray], output: Path, requested: list[float]
) -> dict:
    phase_time = pic["phase_time"]
    velocity = pic["velocity"]
    mask = (velocity >= 2.0) & (velocity <= 4.5)
    cropped_velocity = velocity[mask]
    phase_indices = [int(np.argmin(np.abs(phase_time - value))) for value in requested]
    true_phase = pic["phase"][phase_indices, :, :][:, :, mask]
    model_time = rollout["time"]
    public = rollout["public_prediction"]
    reconstructed: list[np.ndarray | None] = []
    model_times: list[float | None] = []
    for value in requested:
        if value > model_time[-1] + 0.05:
            reconstructed.append(None)
            model_times.append(None)
        else:
            index = int(np.argmin(np.abs(model_time - value)))
            reconstructed.append(_maxwellian(public[index], cropped_velocity, true_phase.shape[1]))
            model_times.append(float(model_time[index]))

    upper = float(np.quantile(true_phase, 0.997))
    available = [value for value in reconstructed if value is not None]
    if available:
        upper = max(upper, float(np.quantile(np.stack(available), 0.997)))
    extent = (0.0, 1.0, float(cropped_velocity[0]), float(cropped_velocity[-1]))
    columns = len(requested)
    figure, axes = plt.subplots(2, columns, figsize=(3.15 * columns, 6.0), sharex=True, sharey=True)
    if columns == 1:
        axes = np.asarray(axes).reshape(2, 1)
    image = None
    for column, (target, phase_index) in enumerate(zip(requested, phase_indices)):
        image = axes[0, column].imshow(
            true_phase[column].T, origin="lower", aspect="auto", extent=extent,
            cmap="magma", vmin=0.0, vmax=upper,
        )
        axes[0, column].set_title(f"t={phase_time[phase_index]:.0f}")
        if reconstructed[column] is None:
            axes[1, column].set_facecolor("0.94")
            axes[1, column].text(0.5, 0.5, f"rollout stopped\nbefore t={target:g}",
                                 ha="center", va="center", transform=axes[1, column].transAxes)
        else:
            axes[1, column].imshow(
                reconstructed[column].T, origin="lower", aspect="auto", extent=extent,
                cmap="magma", vmin=0.0, vmax=upper,
            )
        axes[1, column].set_xlabel("x/L")
    axes[0, 0].set_ylabel("PIC particle distribution\nresonant velocity v")
    axes[1, 0].set_ylabel("FNO moments -> local Maxwellian\nresonant velocity v")
    if image is not None:
        color_axis = figure.add_axes((0.925, 0.19, 0.012, 0.62))
        figure.colorbar(image, cax=color_axis, label="f")
    figure.suptitle(
        "Positive resonant-velocity window (2 <= v <= 4.5); lower row is moment reconstruction"
    )
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.09, top=0.88, wspace=0.10, hspace=0.17)
    figure.savefig(output / "phase_space_resonant_pic_vs_fno_moments.png", dpi=210)
    plt.close(figure)

    pic_figure, pic_axes = plt.subplots(1, columns, figsize=(3.15 * columns, 3.25), sharex=True, sharey=True)
    if columns == 1:
        pic_axes = np.asarray([pic_axes])
    for column, phase_index in enumerate(phase_indices):
        pic_image = pic_axes[column].imshow(
            true_phase[column].T, origin="lower", aspect="auto", extent=extent,
            cmap="magma", vmin=0.0, vmax=float(np.quantile(true_phase, 0.997)),
        )
        pic_axes[column].set_title(f"t={phase_time[phase_index]:.0f}")
        pic_axes[column].set_xlabel("x/L")
    pic_axes[0].set_ylabel("resonant velocity v")
    pic_color_axis = pic_figure.add_axes((0.925, 0.20, 0.012, 0.55))
    pic_figure.colorbar(pic_image, cax=pic_color_axis, label="PIC f")
    pic_figure.suptitle(r"PIC phase-space evolution: $k=0.35$, $A=0.10$, $2\leq v\leq4.5$")
    pic_figure.subplots_adjust(left=0.07, right=0.90, bottom=0.17, top=0.82, wspace=0.10)
    pic_figure.savefig(output / "pic_phase_space_resonant_velocity.png", dpi=210)
    plt.close(pic_figure)
    return {
        "velocity_window": [float(cropped_velocity[0]), float(cropped_velocity[-1])],
        "requested_times": requested,
        "pic_times": [float(phase_time[index]) for index in phase_indices],
        "fno_reconstruction_times": model_times,
        "note": "FNO predicts only M0, M1, M2; its plotted phase space is a moment-matched local Maxwellian, not a kinetic distribution prediction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default="k0p350_a0p100")
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--rollout-label", default="FNO mode 8 (strong low-pass)")
    parser.add_argument(
        "--comparison-rollout", action="append", default=[], metavar="LABEL=PATH",
        help="Additional curve for the long-time field-energy comparison",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=(30.0, 40.0, 50.0, 60.0))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pic = _load_pic(args.case_dir, args.pair_id)
    rollout = dict(np.load(args.rollout))
    rollouts = [(args.rollout_label, rollout)]
    for specification in args.comparison_rollout:
        label, separator, path = specification.partition("=")
        if not separator:
            raise ValueError("--comparison-rollout must be LABEL=PATH")
        rollouts.append((label, dict(np.load(path))))
    summary = {
        "pair_id": args.pair_id,
        "field_energy": _plot_field(pic, rollouts, args.output_dir),
        "phase_space": _plot_phase(pic, rollout, args.output_dir, list(args.times)),
    }
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
