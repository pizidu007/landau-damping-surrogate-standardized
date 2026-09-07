"""Compare one pure-FNO field-energy rollout with all matching PIC replicas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rollout = np.load(args.rollout_dir / f"{args.pair_id}.npz")
    pic_values = []
    pic_time = None
    for path in sorted(args.case_dir.glob(f"{args.pair_id}_s*.h5")):
        with h5py.File(path, "r") as handle:
            time = np.asarray(handle["grids/moment_time"], dtype=np.float64)
            energy = np.asarray(handle["energies/field_energy"], dtype=np.float64)
        if pic_time is None:
            pic_time = time
        elif not np.array_equal(pic_time, time):
            raise ValueError(f"PIC time mismatch in {path}")
        pic_values.append(energy)
    if pic_time is None or not pic_values:
        raise FileNotFoundError(args.pair_id)
    pic_stack = np.stack(pic_values)
    pic_mean = pic_stack.mean(axis=0)
    pic_peaks, _ = find_peaks(pic_mean, distance=15)
    fno_peaks, _ = find_peaks(rollout["field_energy"], distance=15)
    pic_peaks = np.unique(np.concatenate(([0], pic_peaks)))
    fno_peaks = np.unique(np.concatenate(([0], fno_peaks)))
    stop = float(rollout["time"][-1])

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for index, value in enumerate(pic_stack):
        axes[0].semilogy(pic_time, value, color="0.65", alpha=0.55, linewidth=0.8,
                         label="PIC replicas" if index == 0 else None)
    axes[0].semilogy(pic_time, pic_mean, color="black", linewidth=1.8, label="PIC three-seed mean")
    axes[0].semilogy(rollout["time"], rollout["field_energy"], color="#e66101", linewidth=1.7,
                     label="pure FNO fluid")
    axes[0].axvline(stop, color="#b2182b", linestyle="--", linewidth=1.1, label=f"FNO stop t={stop:.1f}")
    axes[0].set_xlim(0.0, 60.0)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("electric-field energy")
    axes[0].set_title("Instantaneous field energy")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].semilogy(pic_time[pic_peaks], pic_mean[pic_peaks], "o-", color="black",
                     markersize=3, linewidth=1.5, label="PIC peak envelope")
    axes[1].semilogy(rollout["time"][fno_peaks], rollout["field_energy"][fno_peaks], "o-",
                     color="#e66101", markersize=3, linewidth=1.5, label="pure FNO peak envelope")
    axes[1].axvline(stop, color="#b2182b", linestyle="--", linewidth=1.1)
    axes[1].set_xlim(0.0, 60.0)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("peak electric-field energy")
    axes[1].set_title("Oscillation envelope")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.suptitle(f"Pure FNO long rollout: {args.pair_id} (k=0.35, A=0.10)")
    figure.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / f"{args.pair_id}_field_energy.png", dpi=190)
    plt.close(figure)

    with (args.rollout_dir / "pair_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = next(csv.DictReader(handle))
    summary = {
        "pair_id": args.pair_id,
        "pic_replica_count": len(pic_values),
        "pure_fno_completed_time": stop,
        "pure_fno_clamp_count": int(metrics["clamp_count"]),
        "pure_fno_finite": bool(float(metrics["finite"])),
        "pure_fno_field_energy_log10_rmse_until_stop": float(metrics["field_energy_log10_rmse"]),
        "pic_field_energy": {
            f"t{time:g}": float(pic_mean[np.argmin(np.abs(pic_time - time))])
            for time in (20.0, 30.0, 35.0, 40.0, 50.0, 60.0)
        },
        "visualization_note": "Velocity cropping affects phase-space visibility, not this integrated field-energy diagnostic.",
    }
    (args.output_dir / f"{args.pair_id}_field_energy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
