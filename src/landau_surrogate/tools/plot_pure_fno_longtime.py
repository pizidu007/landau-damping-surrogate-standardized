"""Plot pure-FNO long rollouts and animate their moment-based phase reconstruction."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pic_field_mean(case_dir: Path, pair_id: str) -> tuple[np.ndarray, np.ndarray]:
    time = None
    values = []
    for path in sorted(case_dir.glob(f"{pair_id}_s*.h5")):
        with h5py.File(path, "r") as handle:
            current_time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
            if time is None:
                time = current_time
            elif not np.array_equal(time, current_time):
                raise ValueError(f"Time mismatch in {path}")
            values.append(np.asarray(handle["energies/field_energy"], dtype=np.float64))
    if time is None or not values:
        raise FileNotFoundError(f"No PIC cases for {pair_id}")
    return time, np.mean(values, axis=0)


def make_energy_figure(rollout_dir: Path, case_dir: Path, output: Path) -> dict[str, float]:
    rows = read_rows(rollout_dir / "pair_metrics.csv")
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True)
    axes_flat = axes.ravel()
    completed = {}
    for axis, row in zip(axes_flat, rows):
        pair_id = row["pair_id"]
        data = np.load(rollout_dir / f"{pair_id}.npz")
        pic_time, pic_energy = pic_field_mean(case_dir, pair_id)
        axis.semilogy(pic_time, pic_energy, color="black", linewidth=1.8, label="PIC truth")
        axis.semilogy(data["time"], data["field_energy"], color="#e66101", linewidth=1.6, label="pure FNO fluid")
        stop = float(data["time"][-1])
        completed[pair_id] = stop
        axis.axvline(stop, color="#b2182b", linestyle="--", linewidth=1.0)
        axis.text(
            stop, 0.04, f"stop {stop:.1f}", rotation=90, color="#b2182b",
            transform=axis.get_xaxis_transform(), va="bottom", ha="right", fontsize=8,
        )
        axis.set_xlim(0.0, 60.0)
        axis.set_title(f"k={float(row['k']):.2f}, A={float(row['alpha']):.3f}")
        axis.grid(alpha=0.2)
    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[5].legend(handles, labels, loc="center", frameon=False)
    axes_flat[5].text(
        0.5, 0.35, "Dashed line: pure-FNO rollout termination\nPIC truth continues to t=60",
        transform=axes_flat[5].transAxes, ha="center", va="center",
    )
    for axis in axes[1]:
        axis.set_xlabel("time")
    axes[0, 0].set_ylabel("electric-field energy")
    axes[1, 0].set_ylabel("electric-field energy")
    figure.suptitle("Pure FNO long-time electric-field energy", y=0.995)
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)
    return completed


def moment_maxwellian(state: np.ndarray, velocity: np.ndarray, target_nx: int) -> np.ndarray:
    density = np.maximum(resample(state[0], target_nx).real, 1.0e-4)
    flow = resample(state[1], target_nx).real
    pressure = np.maximum(resample(state[2], target_nx).real, 1.0e-5)
    temperature = np.maximum(pressure / density, 1.0e-4)
    delta = velocity[None, :] - flow[:, None]
    return density[:, None] / np.sqrt(2.0 * np.pi * temperature[:, None]) * np.exp(
        -0.5 * delta ** 2 / temperature[:, None]
    )


def make_phase_animation(
    rollout_dir: Path, case_dir: Path, pair_id: str, output: Path,
) -> dict[str, object]:
    rollout = np.load(rollout_dir / f"{pair_id}.npz")
    pic_path = sorted(case_dir.glob(f"{pair_id}_s00.h5"))[0]
    with h5py.File(pic_path, "r") as handle:
        all_phase_time = np.asarray(handle["grids/phase_time"], dtype=np.float64)
        velocity_full = np.asarray(handle["grids/phase_velocity"], dtype=np.float64)
        selected = np.flatnonzero(
            (all_phase_time >= rollout["time"][0]) & (all_phase_time <= rollout["time"][-1])
        )
        velocity_index = np.arange(0, len(velocity_full), 4)
        velocity = velocity_full[velocity_index]
        phase_time = all_phase_time[selected]
        pic_phase = np.asarray(handle["phase_space/f"][selected, ::2, ::4], dtype=np.float32)
    state_index = np.array(
        [int(np.argmin(np.abs(rollout["time"] - value))) for value in phase_time], dtype=np.int64
    )
    reconstructed = np.stack(
        [moment_maxwellian(rollout["prediction"][index], velocity, pic_phase.shape[1]) for index in state_index]
    ).astype(np.float32)
    maximum = float(np.quantile(np.concatenate((pic_phase.ravel(), reconstructed.ravel())), 0.999))
    pic_time, pic_energy = pic_field_mean(case_dir, pair_id)

    figure = plt.figure(figsize=(12, 7.2))
    grid = figure.add_gridspec(2, 2, height_ratios=(3.2, 1.25), hspace=0.3, wspace=0.18)
    axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    energy_axis = figure.add_subplot(grid[1, :])
    extent = (0.0, 1.0, float(velocity[0]), float(velocity[-1]))
    images = [
        axes[0].imshow(pic_phase[0].T, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0.0, vmax=maximum),
        axes[1].imshow(reconstructed[0].T, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0.0, vmax=maximum),
    ]
    axes[0].set_title("PIC truth: f(x,v)")
    axes[1].set_title("Pure FNO moments → local Maxwellian reconstruction")
    axes[0].set_ylabel("velocity v")
    for axis in axes:
        axis.set_xlabel("x/L")
    colorbar = figure.colorbar(images[1], ax=list(axes), shrink=0.82, pad=0.02)
    colorbar.set_label("distribution f")
    energy_axis.semilogy(pic_time, pic_energy, color="black", linewidth=1.8, label="PIC truth")
    energy_axis.semilogy(rollout["time"], rollout["field_energy"], color="#e66101", linewidth=1.6, label="pure FNO fluid")
    cursor = energy_axis.axvline(float(phase_time[0]), color="#2166ac", linewidth=1.5)
    energy_axis.set_xlim(0.0, 60.0)
    energy_axis.set_xlabel("time")
    energy_axis.set_ylabel("electric-field energy")
    energy_axis.legend(loc="best", ncol=2)
    time_label = figure.suptitle(f"{pair_id}: t={phase_time[0]:.1f}")

    def update(frame: int):
        images[0].set_data(pic_phase[frame].T)
        images[1].set_data(reconstructed[frame].T)
        cursor.set_xdata([phase_time[frame], phase_time[frame]])
        time_label.set_text(f"{pair_id}: t={phase_time[frame]:.1f}")
        return images[0], images[1], cursor, time_label

    movie = animation.FuncAnimation(
        figure, update, frames=len(phase_time), interval=180, blit=False, repeat=True,
    )
    movie.save(output, writer=animation.PillowWriter(fps=6), dpi=105)
    plt.close(figure)
    return {
        "pair_id": pair_id,
        "pic_phase_source": str(pic_path),
        "frame_count": int(len(phase_time)),
        "start_time": float(phase_time[0]),
        "stop_time": float(phase_time[-1]),
        "fno_completed_time": float(rollout["time"][-1]),
        "right_panel": "Moment-matched local Maxwellian reconstruction; not a kinetic FNO prediction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default="k0p400_a0p050")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = make_energy_figure(
        args.rollout_dir, args.case_dir, args.output_dir / "pure_fno_field_energy_t60.png"
    )
    animation_summary = make_phase_animation(
        args.rollout_dir, args.case_dir, args.pair_id,
        args.output_dir / f"{args.pair_id}_pic_vs_pure_fno_maxwellian.gif",
    )
    summary = {
        "rollout_dir": str(args.rollout_dir),
        "completed_times": completed,
        "animation": animation_summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
