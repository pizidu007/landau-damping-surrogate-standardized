"""Materialize SI scale metadata views without duplicating normalized tensors."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = args.dataset_root.resolve()
    scale_plan = json.loads(
        (root / "manifests" / "scale_equivariance_plan.json").read_text(
            encoding="utf-8"
        )
    )
    splits = json.loads(
        (root / "manifests" / "splits_v1.json").read_text(encoding="utf-8")
    )
    constants = scale_plan["si_constants"]
    charge = float(constants["elementary_charge_C"])
    mass = float(constants["electron_mass_kg"])
    epsilon0 = float(constants["vacuum_permittivity_F_per_m"])
    views = []
    for assignment in splits["assignments"]:
        for scale in scale_plan["physical_scales"]:
            density = float(scale["n0_per_m3"])
            temperature_eV = float(scale["T0_eV"])
            thermal_speed = math.sqrt(charge * temperature_eV / mass)
            plasma_frequency = math.sqrt(
                density * charge * charge / (mass * epsilon0)
            )
            debye_length = thermal_speed / plasma_frequency
            K = float(assignment["K"])
            view_id = (
                f"{assignment['case_id']}__n{density:.0e}_T{temperature_eV:g}eV"
            )
            views.append(
                {
                    "view_id": view_id,
                    "case_id": assignment["case_id"],
                    "canonical_split": assignment["canonical_split"],
                    "equivalence_group": assignment["equivalence_group"],
                    "K": K,
                    "alpha": float(assignment["alpha"]),
                    "n0_per_m3": density,
                    "T0_eV": temperature_eV,
                    "thermal_speed_m_per_s": thermal_speed,
                    "plasma_frequency_rad_per_s": plasma_frequency,
                    "debye_length_m": debye_length,
                    "physical_wavenumber_rad_per_m": K / debye_length,
                    "time_seconds_per_tau": 1.0 / plasma_frequency,
                    "distribution_scale_n0_over_vth": density / thermal_speed,
                    "electric_field_scale_V_per_m": (
                        mass * thermal_speed * plasma_frequency / charge
                    ),
                    "normalized_trajectory": str(
                        root
                        / "profiles"
                        / "production"
                        / "cases"
                        / assignment["case_id"]
                        / "processed"
                        / "trajectory.h5"
                    ),
                }
            )
    payload = {
        "schema_version": 1,
        "dataset_version": "continuum_v1",
        "materialization": "metadata-only; normalized tensors are not copied",
        "leakage_policy": "every scale view inherits its normalized case split",
        "view_count": len(views),
        "views": views,
    }
    output = root / "manifests" / "dimensional_views_v1.json"
    if output.exists():
        current = json.loads(output.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError(f"refusing to change dimensional views: {output}")
    else:
        _atomic_json(output, payload)
    print(json.dumps({"output": str(output), "view_count": len(views)}, indent=2))


if __name__ == "__main__":
    main()
