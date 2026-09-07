"""Quantify seed noise and spectral signal quality of PIC closure labels."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from landau_surrogate.data.closure_dataset import load_pair_trajectories
from landau_surrogate.data.paths import nonlinear_runs_root


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=nonlinear_runs_root() / "nonlinear_formal_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []
    spectral_signal: list[np.ndarray] = []
    spectral_noise: list[np.ndarray] = []
    for trajectory in load_pair_trajectories(args.run_dir):
        raw: list[np.ndarray] = []
        for source in trajectory.source_paths:
            with h5py.File(source, "r") as handle:
                raw.append(np.asarray(handle["fluid/heat_flux_gradient"], dtype=np.float64))
        stack = np.stack(raw)
        mean = stack.mean(axis=0)
        residual = stack - mean[None]
        signal_rms = float(np.sqrt(np.mean(mean**2)))
        noise_rms = float(np.sqrt(np.mean(residual**2)))
        signal_power = np.mean(np.abs(np.fft.rfft(mean, axis=-1)) ** 2, axis=0)
        noise_power = np.mean(np.abs(np.fft.rfft(residual, axis=-1)) ** 2, axis=(0, 1))
        spectral_signal.append(signal_power)
        spectral_noise.append(noise_power)
        snr = signal_power / np.maximum(noise_power, 1.0e-30)
        trustworthy = np.flatnonzero(snr >= 1.0)
        rows.append({
            "pair_id": trajectory.pair_id,
            "split": trajectory.split,
            "k": trajectory.k,
            "alpha": trajectory.alpha,
            "signal_rms": signal_rms,
            "seed_noise_rms": noise_rms,
            "noise_to_signal": noise_rms / max(signal_rms, 1.0e-30),
            "last_mode_snr_ge_1": int(trustworthy[-1]) if trustworthy.size else 0,
        })

    csv_path = args.output_dir / "pair_label_quality.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    mean_signal = np.mean(spectral_signal, axis=0)
    mean_noise = np.mean(spectral_noise, axis=0)
    mode = np.arange(len(mean_signal))
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(mode[1:], mean_signal[1:], label="seed-mean signal")
    plt.semilogy(mode[1:], mean_noise[1:], label="inter-seed noise")
    plt.xlabel("Fourier mode")
    plt.ylabel("Mean power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "dqdx_signal_noise_spectrum.png", dpi=180)
    plt.close()

    matrix = np.asarray([[float(row["k"]), float(row["alpha"]), float(row["noise_to_signal"])] for row in rows])
    plt.figure(figsize=(6, 4.5))
    scatter = plt.scatter(matrix[:, 0], matrix[:, 1], c=matrix[:, 2], s=130, cmap="viridis")
    plt.colorbar(scatter, label="seed noise / signal RMS")
    plt.xlabel("k")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(plot_dir / "seed_noise_parameter_map.png", dpi=180)
    plt.close()

    snr = mean_signal / np.maximum(mean_noise, 1.0e-30)
    good = np.flatnonzero(snr >= 1.0)
    summary = {
        "pair_count": len(rows),
        "maximum_noise_to_signal": max(float(row["noise_to_signal"]) for row in rows),
        "median_noise_to_signal": float(np.median([float(row["noise_to_signal"]) for row in rows])),
        "recommended_max_mode": int(good[-1]) if good.size else 0,
        "recommendation": "seed_mean_with_spectral_loss",
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
