"""Label damping, turnover, and nonlinear-rebound frames from field energy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks

from landau_surrogate.training.gkeyll_multicase import case_id


REGIME_NAMES = {0: "damping", 1: "turnover_or_late_weak", 2: "nonlinear_rebound"}


def peak_envelope(time: np.ndarray, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return peak locations and their linearly interpolated envelope."""
    dt = float(np.median(np.diff(time)))
    # Langmuir-energy peaks are separated by O(1); suppress numerical ripples.
    distance = max(1, int(round(0.5 / dt)))
    peaks, _ = find_peaks(energy, distance=distance)
    peaks = peaks[(time[peaks] >= 1.0) & (time[peaks] <= 39.9)]
    if len(peaks) < 3:
        raise ValueError("Too few field-energy peaks for a regime label")
    envelope = np.interp(time, time[peaks], energy[peaks])
    return peaks, energy[peaks], envelope


def label_case(path: Path, rebound_factor: float, latest_minimum: float) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as handle:
        time = np.asarray(handle["time"], dtype=np.float64)
        energy_time = np.asarray(handle["field/energy_time"], dtype=np.float64)
        raw_energy = np.asarray(handle["field/energy"], dtype=np.float64)
        energy = raw_energy[:, 0] if raw_energy.ndim == 2 else raw_energy
        k = float(handle.attrs["k"])
        alpha = float(handle.attrs["alpha"])
    peaks, peak_energy, envelope = peak_envelope(energy_time, energy)
    eligible = peaks[energy_time[peaks] < latest_minimum]
    if not len(eligible):
        raise ValueError(f"No envelope minimum before t={latest_minimum:g}: {path}")
    minimum_index = int(eligible[np.argmin(energy[eligible])])
    later_peaks = peaks[peaks > minimum_index]
    rebound_peak = float(np.max(energy[later_peaks])) if len(later_peaks) else float(energy[minimum_index])
    ratio = rebound_peak / max(float(energy[minimum_index]), np.finfo(float).tiny)
    is_strong = bool(energy_time[minimum_index] < latest_minimum and ratio >= rebound_factor)

    # Reserve the final 35% of the damping interval for the nonlinear turnover.
    # Strong cases receive a distinct post-minimum class; weak cases stay in class 1.
    turnover_start = 0.65 * float(energy_time[minimum_index])
    labels = np.zeros(len(time), dtype=np.uint8)
    labels[time >= turnover_start] = 1
    if is_strong:
        labels[time >= energy_time[minimum_index]] = 2
    counts = {REGIME_NAMES[key]: int(np.sum(labels == key)) for key in REGIME_NAMES}
    summary = {
        "case_id": case_id(k, alpha),
        "k": k,
        "alpha": alpha,
        "minimum_time": float(energy_time[minimum_index]),
        "minimum_envelope_energy": float(energy[minimum_index]),
        "maximum_postminimum_peak_energy": rebound_peak,
        "rebound_ratio": float(ratio),
        "strong_nonlinear_rebound": is_strong,
        "turnover_start_time": turnover_start,
        "counts": counts,
        "fractions": {name: count / len(time) for name, count in counts.items()},
        "peak_count": int(len(peaks)),
        "envelope_final": float(envelope[-1]),
    }
    return labels, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-factor", type=float, default=1.5)
    parser.add_argument("--latest-minimum", type=float, default=35.0)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for spec in plan["cases"]:
        identifier = case_id(float(spec["k"]), float(spec["alpha"]))
        path = args.dataset_root / "cases" / identifier / "processed" / "trajectory.h5"
        labels, summary = label_case(path, args.rebound_factor, args.latest_minimum)
        arrays[identifier] = labels
        rows.append({**summary, "split": spec["split"], "source": spec.get("source")})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_temp = args.output_dir / "regime_labels.npz.tmp"
    with npz_temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(npz_temp, args.output_dir / "regime_labels.npz")
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        split_counts[split] = {
            name: sum(row["counts"][name] for row in selected) for name in REGIME_NAMES.values()
        }
    report = {
        "criterion": {
            "description": "peak-envelope minimum before cutoff and later peak / minimum >= factor",
            "latest_minimum": args.latest_minimum,
            "rebound_factor": args.rebound_factor,
            "turnover_start_fraction_of_minimum_time": 0.65,
        },
        "regime_ids": {str(key): value for key, value in REGIME_NAMES.items()},
        "strong_case_count": sum(row["strong_nonlinear_rebound"] for row in rows),
        "split_counts": split_counts,
        "cases": rows,
    }
    temporary = args.output_dir / "regime_summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.output_dir / "regime_summary.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
