"""Compare low/mid/high continuum_v1 anchors and recommend production fidelity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from landau_surrogate.tools.select_gkeyll_continuum_v1 import label_case


DEFAULT_ROOT = Path(
    "/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1"
)
PROFILES = ("anchors_low", "anchors", "anchors_high")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _relative_delta(candidate: float, reference: float) -> float:
    return float(abs(candidate - reference) / max(abs(reference), 1.0e-30))


def _load_field(path: Path, K: float) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        time = np.asarray(handle["diagnostics/time"], dtype=np.float64)
        x = np.asarray(handle["coordinates/x_cell"], dtype=np.float64)
        field = np.asarray(handle["diagnostics/electric_field"], dtype=np.float64)
    mode = 2.0 * np.mean(field * np.exp(-1j * K * x)[None, :], axis=1)
    return time, mode


def _field_metrics(candidate: Path, reference: Path, K: float) -> dict[str, float]:
    candidate_time, candidate_mode = _load_field(candidate, K)
    reference_time, reference_mode = _load_field(reference, K)
    real = np.interp(reference_time, candidate_time, candidate_mode.real)
    imag = np.interp(reference_time, candidate_time, candidate_mode.imag)
    aligned = real + 1j * imag
    scale = np.sqrt(np.mean(np.abs(reference_mode) ** 2))
    error = np.sqrt(np.mean(np.abs(aligned - reference_mode) ** 2))
    amplitude_scale = np.sqrt(np.mean(np.abs(reference_mode) ** 2))
    amplitude_error = np.sqrt(
        np.mean((np.abs(aligned) - np.abs(reference_mode)) ** 2)
    )
    return {
        "complex_E1_relative_l2": float(error / max(scale, 1.0e-30)),
        "E1_amplitude_relative_l2": float(
            amplitude_error / max(amplitude_scale, 1.0e-30)
        ),
    }


def _periodic_resample(
    source_x: np.ndarray,
    source_u: np.ndarray,
    source: np.ndarray,
    target_x: np.ndarray,
    target_u: np.ndarray,
) -> np.ndarray:
    length = float((source_x[-1] - source_x[0]) + np.median(np.diff(source_x)))
    extended_x = np.concatenate(
        ([source_x[-1] - length], source_x, [source_x[0] + length])
    )
    extended = np.concatenate((source[-1:], source, source[:1]), axis=0)
    interpolator = RegularGridInterpolator(
        (extended_x, source_u), extended, bounds_error=True
    )
    xx, uu = np.meshgrid(target_x, target_u, indexing="ij")
    return interpolator(np.stack((xx.ravel(), uu.ravel()), axis=-1)).reshape(
        len(target_x), len(target_u)
    )


def _kinetic_metrics(
    candidate: Path,
    reference: Path,
    K: float,
    event_times: list[float],
) -> dict[str, Any]:
    with h5py.File(candidate, "r") as candidate_handle, h5py.File(
        reference, "r"
    ) as reference_handle:
        candidate_time = np.asarray(candidate_handle["kinetic/time"], dtype=np.float64)
        reference_time = np.asarray(reference_handle["kinetic/time"], dtype=np.float64)
        candidate_x = np.asarray(candidate_handle["coordinates/x_phase"], dtype=np.float64)
        candidate_u = np.asarray(candidate_handle["coordinates/u_phase"], dtype=np.float64)
        reference_x = np.asarray(reference_handle["coordinates/x_phase"], dtype=np.float64)
        reference_u_all = np.asarray(
            reference_handle["coordinates/u_phase"], dtype=np.float64
        )
        phase_speed = 1.0 / K
        resonant = np.abs(np.abs(reference_u_all) - phase_speed) <= 1.0
        reference_u = reference_u_all[resonant]
        candidate_initial = np.asarray(candidate_handle["kinetic/g"][0], dtype=np.float64)
        reference_initial = np.asarray(
            reference_handle["kinetic/g"][0, :, resonant], dtype=np.float64
        )
        candidate_initial_on_reference = _periodic_resample(
            candidate_x,
            candidate_u,
            candidate_initial,
            reference_x,
            reference_u,
        )
        rows = []
        for event_time in event_times:
            candidate_index = int(np.argmin(np.abs(candidate_time - event_time)))
            reference_index = int(np.argmin(np.abs(reference_time - event_time)))
            candidate_value = np.asarray(
                candidate_handle["kinetic/g"][candidate_index], dtype=np.float64
            )
            reference_value = np.asarray(
                reference_handle["kinetic/g"][reference_index, :, resonant],
                dtype=np.float64,
            )
            candidate_delta = _periodic_resample(
                candidate_x,
                candidate_u,
                candidate_value - candidate_initial,
                reference_x,
                reference_u,
            )
            reference_delta = reference_value - reference_initial
            numerator = np.linalg.norm(candidate_delta - reference_delta)
            denominator = np.linalg.norm(reference_delta)
            rows.append(
                {
                    "requested_time": float(event_time),
                    "candidate_time": float(candidate_time[candidate_index]),
                    "reference_time": float(reference_time[reference_index]),
                    "resonant_delta_g_relative_l2": float(
                        numerator / max(denominator, 1.0e-30)
                    ),
                    "equilibrium_resampling_relative_l2": float(
                        np.linalg.norm(candidate_initial_on_reference - reference_initial)
                        / max(np.linalg.norm(reference_initial), 1.0e-30)
                    ),
                }
            )
    return {
        "phase_speed_proxy": phase_speed,
        "resonant_windows": [
            [-phase_speed - 1.0, -phase_speed + 1.0],
            [phase_speed - 1.0, phase_speed + 1.0],
        ],
        "event_snapshots": rows,
        "resonant_delta_g_relative_l2_max": max(
            row["resonant_delta_g_relative_l2"] for row in rows
        ),
    }


def _case_root(root: Path, profile: str, case_id: str) -> Path:
    return root / "profiles" / profile / "cases" / case_id


def _processed_path(root: Path, profile: str, case_id: str) -> Path:
    return _case_root(root, profile, case_id) / "processed" / "trajectory.h5"


def _compare_pair(
    root: Path,
    candidate_profile: str,
    reference_profile: str,
    case_id: str,
    K: float,
    alpha: float,
) -> dict[str, Any]:
    candidate_root = _case_root(root, candidate_profile, case_id)
    reference_root = _case_root(root, reference_profile, case_id)
    candidate_label = label_case(candidate_root, K, alpha)
    reference_label = label_case(reference_root, K, alpha)
    event_metrics = {
        name: _relative_delta(candidate_label[name], reference_label[name])
        for name in (
            "minimum_time",
            "rebound_energy_ratio",
            "post_peak_time",
            "bounce_cycles_postminimum",
        )
    }
    event_times = sorted(
        {
            reference_label["minimum_time"],
            reference_label["post_peak_time"],
            60.0,
        }
    )
    return {
        "candidate_profile": candidate_profile,
        "reference_profile": reference_profile,
        "field": _field_metrics(
            _processed_path(root, candidate_profile, case_id),
            _processed_path(root, reference_profile, case_id),
            K,
        ),
        "events": event_metrics,
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "kinetic": _kinetic_metrics(
            _processed_path(root, candidate_profile, case_id),
            _processed_path(root, reference_profile, case_id),
            K,
            event_times,
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = args.dataset_root.resolve()
    plans = {
        profile: json.loads(
            (root / "manifests" / f"{profile}_plan.json").read_text(
                encoding="utf-8"
            )
        )
        for profile in PROFILES
    }
    case_rows = plans["anchors"]["cases"]
    for profile in PROFILES:
        if [item["case_id"] for item in plans[profile]["cases"]] != [
            item["case_id"] for item in case_rows
        ]:
            raise RuntimeError(f"anchor case order differs for {profile}")
        for item in case_rows:
            processed = _processed_path(root, profile, item["case_id"])
            if not processed.exists():
                raise RuntimeError(f"missing processed anchor: {processed}")

    comparisons = []
    for item in case_rows:
        case_id = item["case_id"]
        K = float(item["K"])
        alpha = float(item["alpha"])
        comparisons.append(
            {
                "case_id": case_id,
                "K": K,
                "alpha": alpha,
                "low_vs_mid": _compare_pair(
                    root, "anchors_low", "anchors", case_id, K, alpha
                ),
                "mid_vs_high": _compare_pair(
                    root, "anchors", "anchors_high", case_id, K, alpha
                ),
            }
        )

    thresholds = {
        "complex_E1_relative_l2_max": 0.03,
        "E1_amplitude_relative_l2_max": 0.03,
        "minimum_time_relative_error_max": 0.03,
        "rebound_energy_ratio_relative_error_max": 0.05,
        "bounce_cycles_relative_error_max": 0.05,
        "resonant_delta_g_relative_l2_max": 0.05,
    }
    mid_high_summary = {
        "complex_E1_relative_l2_max": max(
            row["mid_vs_high"]["field"]["complex_E1_relative_l2"]
            for row in comparisons
        ),
        "E1_amplitude_relative_l2_max": max(
            row["mid_vs_high"]["field"]["E1_amplitude_relative_l2"]
            for row in comparisons
        ),
        "minimum_time_relative_error_max": max(
            row["mid_vs_high"]["events"]["minimum_time"]
            for row in comparisons
        ),
        "rebound_energy_ratio_relative_error_max": max(
            row["mid_vs_high"]["events"]["rebound_energy_ratio"]
            for row in comparisons
        ),
        "bounce_cycles_relative_error_max": max(
            row["mid_vs_high"]["events"]["bounce_cycles_postminimum"]
            for row in comparisons
        ),
        "resonant_delta_g_relative_l2_max": max(
            row["mid_vs_high"]["kinetic"]["resonant_delta_g_relative_l2_max"]
            for row in comparisons
        ),
    }
    accepted = all(
        mid_high_summary[name] <= threshold for name, threshold in thresholds.items()
    )
    report = {
        "schema_version": 1,
        "profiles": {
            "low": {"name": "anchors_low", "nx": 64, "nv": 128},
            "mid": {"name": "anchors", "nx": 96, "nv": 192},
            "high": {"name": "anchors_high", "nx": 128, "nv": 256},
        },
        "thresholds": thresholds,
        "mid_vs_high_summary": mid_high_summary,
        "mid_resolution_accepted": accepted,
        "recommended_production_resolution": (
            {"nx": 96, "nv": 192} if accepted else {"nx": 128, "nv": 256}
        ),
        "note": (
            "A high-resolution recommendation means the conservative automated "
            "gate failed; inspect per-case metrics before freezing production."
        ),
        "cases": comparisons,
    }
    output = root / "audit" / "anchor_convergence.json"
    _atomic_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "mid_resolution_accepted": accepted,
                "recommended_production_resolution": report[
                    "recommended_production_resolution"
                ],
                "mid_vs_high_summary": mid_high_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
