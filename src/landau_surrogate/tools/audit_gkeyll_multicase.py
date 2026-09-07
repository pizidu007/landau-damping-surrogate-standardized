"""Audit the generated Gkeyll multi-case pilot before training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from landau_surrogate.data.huang2025 import load_huang_mat
from landau_surrogate.training.gkeyll_multicase import case_id, lowpass_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-mode", type=int, default=8)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for spec in plan["cases"]:
        identifier = case_id(float(spec["k"]), float(spec["alpha"]))
        path = args.dataset_root / "cases" / identifier / "processed" / "trajectory.h5"
        trajectory = load_huang_mat(path)
        filtered = lowpass_numpy(trajectory.heat_flux_gradient, args.maximum_mode)
        discarded = trajectory.heat_flux_gradient - filtered
        with h5py.File(path, "r") as handle:
            energy_time = np.asarray(handle["field/energy_time"])
            field_energy = np.asarray(handle["field/energy"])
            # Gkeyll writes the six electromagnetic energy components.  This
            # 1D electrostatic setup only has Ex, so the other five columns are
            # identically zero and must not participate in the minimum search.
            energy = field_energy if field_energy.ndim == 1 else field_energy[:, 0]
            phase_time = np.asarray(handle["phase/time"])
        minimum = int(np.argmin(energy))
        rows.append(
            {
                "case_id": identifier,
                "split": spec["split"],
                "k": trajectory.k,
                "alpha": trajectory.alpha,
                "frame_count": len(trajectory.time),
                "time_end": float(trajectory.time[-1]),
                "minimum_density": float(trajectory.state[:, 0].min()),
                "minimum_pressure": float(trajectory.state[:, 2].min()),
                "target_fraction_above_mode": float(
                    np.linalg.norm(discarded)
                    / max(np.linalg.norm(trajectory.heat_flux_gradient), 1.0e-12)
                ),
                "field_energy_minimum_time": float(energy_time[minimum]),
                "field_energy_final_over_minimum": float(
                    energy[-1] / max(energy[minimum], 1.0e-14)
                ),
                "phase_time_count": len(phase_time),
            }
        )
    complete = all(
        row["frame_count"] == 8001
        and row["time_end"] >= 39.99
        and row["phase_time_count"] >= 5
        and row["minimum_density"] > 0.0
        and row["minimum_pressure"] > 0.0
        for row in rows
    )
    summary = {
        "complete": complete,
        "case_count": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "maximum_target_fraction_above_mode": max(
            row["target_fraction_above_mode"] for row in rows
        ),
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
