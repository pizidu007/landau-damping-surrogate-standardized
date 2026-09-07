"""Label the scout phase diagram and create a deterministic production design."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import postgkyl as pg
from scipy.signal import find_peaks
from scipy.stats import qmc


PROJECT_ROOT = Path(
    "/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized"
)
DEFAULT_ROOT = Path(
    "/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gkeyll/continuum_v1.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_energy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pg.GData(str(path))
    time = np.asarray(data.get_grid()[0], dtype=np.float64)
    energy = np.asarray(data.get_values(), dtype=np.float64)
    if energy.ndim == 2:
        energy = energy[:, 0]
    return time, energy


def label_case(case_root: Path, K: float, alpha: float) -> dict[str, Any]:
    complete = json.loads(
        (case_root / "provenance" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    raw_dir = case_root / "raw"
    energy_paths = list(raw_dir.glob("*-field-energy.gkyl"))
    if len(energy_paths) != 1:
        raise RuntimeError(f"expected one field-energy file in {raw_dir}")
    time, energy = _load_energy(energy_paths[0])
    energy = np.maximum(energy, np.finfo(np.float64).tiny)
    amplitude = np.sqrt(energy)
    dt = float(np.median(np.diff(time)))
    peaks, _ = find_peaks(energy, distance=max(1, int(round(0.5 / dt))))
    peaks = peaks[(time[peaks] >= 1.0) & (time[peaks] <= time[-1] - dt)]
    eligible = peaks[time[peaks] <= time[-1] - 10.0]
    if len(eligible) < 3:
        raise RuntimeError(f"too few eligible field-energy peaks in {energy_paths[0]}")
    minimum = int(eligible[np.argmin(energy[eligible])])
    later = peaks[peaks > minimum]
    post_peak = int(later[np.argmax(energy[later])]) if len(later) else minimum
    rebound_energy_ratio = float(energy[post_peak] / energy[minimum])
    rebound_amplitude_ratio = float(amplitude[post_peak] / amplitude[minimum])
    post_peak_fraction_initial = float(energy[post_peak] / energy[0])

    domain_length = 2.0 * np.pi / K
    electric_mode_amplitude = np.sqrt(2.0 * energy / domain_length)
    bounce_frequency = np.sqrt(K * np.maximum(electric_mode_amplitude, 0.0))
    post = time >= time[minimum]
    bounce_cycles_postminimum = float(
        np.trapz(bounce_frequency[post], time[post]) / (2.0 * np.pi)
    )

    early_peaks = peaks[(time[peaks] >= 1.0) & (time[peaks] <= min(15.0, time[minimum]))]
    if len(early_peaks) >= 3:
        damping_rate = float(
            np.polyfit(time[early_peaks], np.log(amplitude[early_peaks]), 1)[0]
        )
    else:
        damping_rate = float("nan")

    reliable_rebound = bool(
        rebound_energy_ratio >= 1.5
        and (
            bounce_cycles_postminimum >= 0.75
            or rebound_energy_ratio >= 3.0
        )
        and post_peak_fraction_initial >= 1.0e-8
    )
    if reliable_rebound:
        regime = "strong_nonlinear"
    elif rebound_energy_ratio >= 1.25 or bounce_cycles_postminimum >= 0.5:
        regime = "transition"
    else:
        regime = "weak"
    return {
        "case_id": complete["case_id"],
        "K": K,
        "alpha": alpha,
        "regime": regime,
        "strong_nonlinear": reliable_rebound,
        "early_amplitude_damping_rate": damping_rate,
        "minimum_time": float(time[minimum]),
        "minimum_energy_fraction_initial": float(energy[minimum] / energy[0]),
        "rebound_energy_ratio": rebound_energy_ratio,
        "rebound_amplitude_ratio": rebound_amplitude_ratio,
        "post_peak_time": float(time[post_peak]),
        "post_peak_energy_fraction_initial": post_peak_fraction_initial,
        "bounce_cycles_postminimum": bounce_cycles_postminimum,
        "source_use_gpu": complete["gkeyll_stat"]["use_gpu"],
    }


def _monotone_threshold(rows: list[dict[str, Any]]) -> float:
    ordered = sorted(rows, key=lambda item: item["alpha"])
    alpha = np.asarray([item["alpha"] for item in ordered], dtype=np.float64)
    strong = np.asarray([item["strong_nonlinear"] for item in ordered], dtype=bool)
    if np.all(strong):
        return float(alpha[0])
    if not np.any(strong):
        return float(alpha[-1])
    candidates = 0.5 * (alpha[:-1] + alpha[1:])
    errors = [
        int(np.sum((alpha >= candidate) != strong)) for candidate in candidates
    ]
    return float(candidates[int(np.argmin(errors))])


def _canonical(K: float, alpha: float) -> tuple[float, float]:
    return round(float(K), 3), round(float(alpha), 3)


def _production_cases(
    labels: list[dict[str, Any]], threshold_by_K: dict[float, float], count: int
) -> list[dict[str, Any]]:
    K_values = np.asarray(sorted(threshold_by_K), dtype=np.float64)
    thresholds = np.asarray([threshold_by_K[K] for K in K_values], dtype=np.float64)
    K_min, K_max = float(K_values[0]), float(K_values[-1])
    alpha_min = float(min(item["alpha"] for item in labels))
    alpha_max = float(max(item["alpha"] for item in labels))
    anchor_pairs = {
        _canonical(float(item["K"]), float(item["alpha"]))
        for item in labels
        if item["case_id"]
        in {
            "K0p280_a0p030",
            "K0p280_a0p140",
            "K0p350_a0p050",
            "K0p350_a0p075",
            "K0p350_a0p150",
            "K0p420_a0p080",
            "K0p420_a0p120",
            "K0p500_a0p100",
            "K0p500_a0p200",
        }
    }
    selected: dict[tuple[float, float], str] = {}

    def add(K: float, alpha: float, source: str) -> None:
        pair = _canonical(
            np.clip(K, K_min, K_max), np.clip(alpha, alpha_min, alpha_max)
        )
        if pair not in anchor_pairs:
            selected.setdefault(pair, source)

    for K in np.linspace(K_min, K_max, 20):
        alpha_c = float(np.interp(K, K_values, thresholds))
        for offset in (-0.025, -0.012, 0.0, 0.012, 0.025):
            add(float(K), alpha_c + offset, "transition_band")
    for alpha in np.linspace(alpha_min, alpha_max, 5):
        add(K_min, float(alpha), "parameter_edge")
        add(K_max, float(alpha), "parameter_edge")
    for K in np.linspace(K_min, K_max, 5):
        add(float(K), alpha_min, "parameter_edge")
        add(float(K), alpha_max, "parameter_edge")

    sampler = qmc.LatinHypercube(d=2, seed=20260831)
    lhs = sampler.random(max(4 * count, 512))
    for point in lhs:
        K = K_min + (K_max - K_min) * point[0]
        alpha = alpha_min + (alpha_max - alpha_min) * point[1]
        add(float(K), float(alpha), "space_filling")
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"could only construct {len(selected)} unique production cases")
    result = []
    for (K, alpha), source in list(selected.items())[:count]:
        alpha_c = float(np.interp(K, K_values, thresholds))
        predicted = (
            "predicted_strong" if alpha >= alpha_c else "predicted_weak"
        )
        result.append(
            {
                "K": K,
                "alpha": alpha,
                "selection_source": source,
                "predicted_regime": predicted,
                "alpha_c_interpolated": alpha_c,
            }
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--production-count", type=int, default=200)
    parser.add_argument(
        "--production-t-end",
        type=float,
        default=80.0,
        help="production horizon; diagnostics remain at Delta-tau=0.02",
    )
    args = parser.parse_args(argv)
    root = args.dataset_root.resolve()
    scout_plan = json.loads(
        (root / "manifests" / "scout_plan.json").read_text(encoding="utf-8")
    )
    labels = []
    for item in scout_plan["cases"]:
        case_root = root / "profiles" / "scout" / "cases" / item["case_id"]
        if not (case_root / "provenance" / "COMPLETE.json").exists():
            raise RuntimeError(f"scout is incomplete: {item['case_id']}")
        labels.append(label_case(case_root, float(item["K"]), float(item["alpha"])))

    unique_K = sorted({float(item["K"]) for item in labels})
    threshold_by_K = {
        K: _monotone_threshold([item for item in labels if float(item["K"]) == K])
        for K in unique_K
    }
    production_cases = _production_cases(
        labels, threshold_by_K, args.production_count
    )
    counts: dict[str, int] = {}
    for item in labels:
        counts[item["regime"]] = counts.get(item["regime"], 0) + 1
    selection_report = {
        "schema_version": 1,
        "criterion": {
            "strong": "field-energy rebound >=1.5, post-peak energy/W0 >=1e-8, and either post-minimum bounce cycles >=0.75 or a large rebound >=3",
            "transition": "not strong and field-energy rebound >=1.25 or bounce cycles >=0.5",
            "note": "These are candidate physics labels; resolution anchors remain authoritative for acceptance.",
        },
        "scout_regime_counts": counts,
        "alpha_c_by_K": {f"{K:.3f}": value for K, value in threshold_by_K.items()},
        "scout_cases": labels,
        "production_case_count": len(production_cases),
        "production_predicted_counts": {
            name: sum(item["predicted_regime"] == name for item in production_cases)
            for name in ("predicted_weak", "predicted_strong")
        },
        "production_cases": production_cases,
    }
    _atomic_json(root / "audit" / "scout_physics_labels.json", selection_report)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    production_config = copy.deepcopy(config)
    production_profile = production_config["profiles"]["production"]
    convergence_path = root / "audit" / "anchor_convergence.json"
    if convergence_path.exists():
        convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
        resolution = convergence["recommended_production_resolution"]
        production_profile["nx"] = int(resolution["nx"])
        production_profile["nv"] = int(resolution["nv"])
        resolution_source = str(convergence_path)
    else:
        resolution_source = "base config (anchor convergence report unavailable)"
    production_profile["t_end"] = float(args.production_t_end)
    production_profile["num_frames"] = int(round(args.production_t_end / 0.02))
    production_profile["distribution_frame_stride"] = 5
    maximum_K = max(float(item["K"]) for item in production_cases)
    recurrence_time = (
        2.0
        * np.pi
        / (
            maximum_K
            * 2.0
            * float(production_profile["vmax"])
            / int(production_profile["nv"])
        )
    )
    if recurrence_time < 2.0 * args.production_t_end:
        raise RuntimeError(
            "selected production resolution violates the recurrence safety gate: "
            f"tau_rec={recurrence_time:.3f} < 2*t_end={2*args.production_t_end:.3f}"
        )
    production_profile["resolution_selection_source"] = resolution_source
    production_profile["classical_recurrence_time_at_maximum_K"] = recurrence_time
    production_profile["time_horizon_policy"] = (
        "tau=80 captures the scout turnover/rebound beyond the tau=60 screening "
        "window while retaining tau_rec >= 2*t_end"
    )
    production_profile["cases"] = production_cases
    production_config_path = root / "manifests" / "production_config.json"
    _atomic_json(production_config_path, production_config)
    print(
        json.dumps(
            {
                "scout_counts": counts,
                "alpha_c_by_K": selection_report["alpha_c_by_K"],
                "production_cases": len(production_cases),
                "production_resolution": {
                    "nx": production_profile["nx"],
                    "nv": production_profile["nv"],
                },
                "production_t_end": production_profile["t_end"],
                "production_recurrence_time_at_maximum_K": recurrence_time,
                "production_config": str(production_config_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
