"""Convert completed continuum_v1 Gkeyll cases to training-ready HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import postgkyl as pg


DEFAULT_ROOT = Path(
    "/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1"
)
MOMENT_RE = re.compile(r"-elc_M0_(\d+)\.gkyl$")
DISTRIBUTION_RE = re.compile(r"-elc_(\d+)\.gkyl$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _modal_centers(path: Path, component: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    data = pg.GData(str(path))
    coefficients = np.asarray(data.get_values(), dtype=np.float64)
    lo = 3 * component
    values = (
        coefficients[:, lo] / np.sqrt(2.0)
        - np.sqrt(5.0 / 8.0) * coefficients[:, lo + 2]
    )
    edges = np.asarray(data.get_grid()[0], dtype=np.float64)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, values, float(data.ctx["time"])


def _phase_frame(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = pg.GData(str(path))
    grid, values = pg.GInterpModal(data, num_interp=3).interpolate(0)
    x = 0.5 * (np.asarray(grid[0][1:]) + np.asarray(grid[0][:-1]))
    velocity = 0.5 * (np.asarray(grid[1][1:]) + np.asarray(grid[1][:-1]))
    return x, velocity, np.asarray(values[..., 0], dtype=np.float32), float(data.ctx["time"])


def _dynamic_vector(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pg.GData(str(path))
    return (
        np.asarray(data.get_grid()[0], dtype=np.float64),
        np.asarray(data.get_values(), dtype=np.float64),
    )


def _frame_map(raw_dir: Path) -> tuple[str, list[int]]:
    result: list[int] = []
    prefix: str | None = None
    for path in raw_dir.glob("*-elc_M0_*.gkyl"):
        match = MOMENT_RE.search(path.name)
        if match:
            result.append(int(match.group(1)))
            prefix = path.name.split("-elc_M0_")[0]
    if prefix is None or not result:
        raise FileNotFoundError(f"no M0 frames in {raw_dir}")
    return prefix, sorted(result)


def _distribution_frames(raw_dir: Path, prefix: str) -> list[int]:
    result = []
    for path in raw_dir.glob(f"{prefix}-elc_*.gkyl"):
        match = DISTRIBUTION_RE.search(path.name)
        if match:
            result.append(int(match.group(1)))
    if not result:
        raise FileNotFoundError(f"no distribution frames in {raw_dir}")
    return sorted(result)


def process_case(dataset_root: Path, profile: str, identifier: str) -> str:
    case_root = dataset_root / "profiles" / profile / "cases" / identifier
    raw_dir = case_root / "raw"
    provenance_dir = case_root / "provenance"
    processed_dir = case_root / "processed"
    complete_path = provenance_dir / "COMPLETE.json"
    if not complete_path.exists():
        return "incomplete"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("gkeyll_stat", {}).get("use_gpu") != 1:
        raise RuntimeError(f"refusing non-GPU source case {profile}/{identifier}")
    output = processed_dir / "trajectory.h5"
    process_status = processed_dir / "COMPLETE.json"
    if output.exists() and process_status.exists():
        current_status = json.loads(process_status.read_text(encoding="utf-8"))
        if current_status.get("processing_schema_version", 1) >= 2:
            return "skipped"
        archived_output = processed_dir / "trajectory.schema1.h5"
        archived_status = processed_dir / "COMPLETE.schema1.json"
        if archived_output.exists() or archived_status.exists():
            raise RuntimeError(f"schema-1 archive already exists in {processed_dir}")
        os.replace(output, archived_output)
        os.replace(process_status, archived_status)
    processed_dir.mkdir(parents=True, exist_ok=True)
    temporary = processed_dir / ".trajectory.h5.incomplete"
    if temporary.exists():
        raise RuntimeError(f"incomplete processed output already exists: {temporary}")

    prefix, moment_frames = _frame_map(raw_dir)
    phase_frames = _distribution_frames(raw_dir, prefix)
    nx = int(complete["numerics"]["nx"])
    moment_count = len(moment_frames)
    moments = np.empty((moment_count, nx, 4), dtype=np.float32)
    electric_field = np.empty((moment_count, nx), dtype=np.float32)
    moment_time = np.empty(moment_count, dtype=np.float64)
    x_cell: np.ndarray | None = None
    for row, frame in enumerate(moment_frames):
        for column, name in enumerate(("M0", "M1", "M2", "M3")):
            path = raw_dir / f"{prefix}-elc_{name}_{frame}.gkyl"
            x, value, current_time = _modal_centers(path)
            if x_cell is None:
                x_cell = x
            elif not np.allclose(x, x_cell, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"moment grid mismatch in {path}")
            moments[row, :, column] = value
            if column == 0:
                moment_time[row] = current_time
        field_path = raw_dir / f"{prefix}-field_{frame}.gkyl"
        field_x, electric_field[row], field_time = _modal_centers(field_path, component=0)
        if not np.allclose(field_x, x_cell, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"field grid mismatch in {field_path}")
        if not np.isclose(field_time, moment_time[row], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"field time mismatch in {field_path}")

    density = moments[..., 0].astype(np.float64)
    flow = moments[..., 1].astype(np.float64) / density
    pressure = moments[..., 2].astype(np.float64) - density * flow**2
    heat_flux = (
        moments[..., 3].astype(np.float64)
        - 3.0 * flow * moments[..., 2]
        + 2.0 * density * flow**3
    )
    temperature = pressure / density
    central = np.stack((density, flow, temperature, heat_flux), axis=-1).astype(
        np.float32
    )
    if not np.all(np.isfinite(moments)) or not np.all(np.isfinite(central)):
        raise RuntimeError(f"non-finite moments in {profile}/{identifier}")

    first_phase = raw_dir / f"{prefix}-elc_{phase_frames[0]}.gkyl"
    x_phase, velocity, first_g, first_phase_time = _phase_frame(first_phase)
    phase_shape = (len(phase_frames), len(x_phase), len(velocity))
    phase_time = np.empty(len(phase_frames), dtype=np.float64)
    valid = np.ones(len(phase_frames), dtype=np.uint8)
    negative_fraction = np.empty(len(phase_frames), dtype=np.float64)
    edge_fraction = np.empty(len(phase_frames), dtype=np.float64)

    energy_time, energy_value = _dynamic_vector(raw_dir / f"{prefix}-field-energy.gkyl")
    integrated_time, integrated_moments = _dynamic_vector(
        raw_dir / f"{prefix}-elc-imom.gkyl"
    )
    l2_time, integrated_l2 = _dynamic_vector(raw_dir / f"{prefix}-elc-L2.gkyl")
    mass = integrated_moments[:, 0]
    momentum = integrated_moments[:, 1]
    field_energy_scalar = (
        energy_value[:, 0] if energy_value.ndim == 2 else energy_value
    )
    total_energy = 0.5 * integrated_moments[:, 2] + 0.5 * field_energy_scalar
    if not np.allclose(integrated_time, energy_time, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"integrated-moment/field-energy time mismatch for {identifier}")

    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            schema_version=2,
            dataset_version="continuum_v1",
            source_solver="Gkeyll Vlasov-Ampere CUDA",
            source_solver_use_gpu=1,
            case_id=identifier,
            profile=profile,
            K=float(complete["parameters"]["K"]),
            alpha=float(complete["parameters"]["alpha"]),
            nx=int(complete["numerics"]["nx"]),
            nv=int(complete["numerics"]["nv"]),
            velocity_max=float(complete["numerics"]["vmax"]),
            polynomial_order=2,
            basis="serendipity",
            normalization="tau=omega_pe*t, x/lambda_D, u=v/v_th, g=v_th*f/n0",
            source_complete_sha256=_sha256(complete_path),
        )
        coordinates = handle.create_group("coordinates")
        coordinates.create_dataset("x_cell", data=x_cell)
        coordinates.create_dataset("x_phase", data=x_phase)
        coordinates.create_dataset("u_phase", data=velocity)

        diagnostics = handle.create_group("diagnostics")
        diagnostics.create_dataset("frame", data=np.asarray(moment_frames, dtype=np.int64))
        diagnostics.create_dataset("time", data=moment_time)
        diagnostics.create_dataset(
            "raw_moments",
            data=moments,
            chunks=(min(128, moment_count), nx, 4),
            compression="lzf",
            shuffle=True,
        )
        diagnostics.create_dataset(
            "central_moments",
            data=central,
            chunks=(min(128, moment_count), nx, 4),
            compression="lzf",
            shuffle=True,
        )
        diagnostics.create_dataset(
            "electric_field",
            data=electric_field,
            chunks=(min(128, moment_count), nx),
            compression="lzf",
            shuffle=True,
        )
        diagnostics.create_dataset("mass", data=mass)
        diagnostics.create_dataset("momentum", data=momentum)
        diagnostics.create_dataset("field_energy_time", data=energy_time)
        diagnostics.create_dataset("field_energy", data=energy_value)
        integrated = handle.create_group("integrated")
        integrated.create_dataset("time", data=integrated_time)
        integrated.create_dataset("moments", data=integrated_moments)
        integrated.create_dataset("total_energy", data=total_energy)
        integrated.create_dataset("l2_time", data=l2_time)
        integrated.create_dataset("l2", data=integrated_l2)

        kinetic = handle.create_group("kinetic")
        kinetic.create_dataset("frame", data=np.asarray(phase_frames, dtype=np.int64))
        kinetic.create_dataset("time", data=phase_time)
        kinetic.create_dataset("valid", data=valid)
        kinetic.create_dataset("negative_sample_mass_fraction", data=negative_fraction)
        kinetic.create_dataset("velocity_edge_sample_mass_fraction", data=edge_fraction)
        distribution = kinetic.create_dataset(
            "g",
            shape=phase_shape,
            dtype=np.float32,
            chunks=(1, min(48, len(x_phase)), min(192, len(velocity))),
            compression="lzf",
            shuffle=True,
        )

        for row, frame in enumerate(phase_frames):
            if row == 0:
                current_x, current_v, value, current_time = (
                    x_phase,
                    velocity,
                    first_g,
                    first_phase_time,
                )
            else:
                current_x, current_v, value, current_time = _phase_frame(
                    raw_dir / f"{prefix}-elc_{frame}.gkyl"
                )
            if not np.allclose(current_x, x_phase, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"phase x grid mismatch at frame {frame}")
            if not np.allclose(current_v, velocity, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"phase velocity grid mismatch at frame {frame}")
            finite = bool(np.all(np.isfinite(value)))
            valid[row] = finite
            if not finite:
                raise RuntimeError(f"non-finite phase distribution at frame {frame}")
            distribution[row] = value
            phase_time[row] = current_time
            positive_mass = float(np.sum(np.maximum(value, 0.0)))
            negative_mass = float(np.sum(np.maximum(-value, 0.0)))
            negative_fraction[row] = negative_mass / max(positive_mass, 1.0e-30)
            edge_width = max(1, len(velocity) // 32)
            edge_mass = float(
                np.sum(np.abs(value[:, :edge_width]))
                + np.sum(np.abs(value[:, -edge_width:]))
            )
            edge_fraction[row] = edge_mass / max(float(np.sum(np.abs(value))), 1.0e-30)

        kinetic["time"][:] = phase_time
        kinetic["valid"][:] = valid
        kinetic["negative_sample_mass_fraction"][:] = negative_fraction
        kinetic["velocity_edge_sample_mass_fraction"][:] = edge_fraction
        handle.flush()

    os.replace(temporary, output)
    status = {
        "schema_version": 1,
        "processing_schema_version": 2,
        "status": "complete",
        "processed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": identifier,
        "profile": profile,
        "source_solver_use_gpu": 1,
        "trajectory": str(output),
        "trajectory_sha256": _sha256(output),
        "diagnostic_frames": moment_count,
        "kinetic_frames": len(phase_frames),
        "phase_shape": list(phase_shape),
        "mass_relative_drift": float(
            np.max(np.abs(mass - mass[0])) / max(abs(mass[0]), 1.0e-30)
        ),
        "momentum_absolute_max": float(np.max(np.abs(momentum))),
        "total_energy_relative_drift": float(
            np.max(np.abs(total_energy - total_energy[0]))
            / max(abs(total_energy[0]), 1.0e-30)
        ),
        "integrated_l2_relative_change": float(
            np.max(np.abs(integrated_l2[:, 0] - integrated_l2[0, 0]))
            / max(abs(integrated_l2[0, 0]), 1.0e-30)
        ),
        "negative_sample_mass_fraction_max": float(np.max(negative_fraction)),
        "velocity_edge_sample_mass_fraction_max": float(np.max(edge_fraction)),
    }
    _atomic_json(process_status, status)
    return "complete"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    root = args.dataset_root.resolve()
    plan = json.loads(
        (root / "manifests" / f"{args.profile}_plan.json").read_text(
            encoding="utf-8"
        )
    )
    selected = [
        item["case_id"]
        for index, item in enumerate(plan["cases"])
        if index % args.shard_count == args.shard_index
    ]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    counts: dict[str, int] = {}
    for identifier in selected:
        outcome = process_case(root, args.profile, identifier)
        if outcome != "incomplete":
            print(f"{outcome.upper()} {args.profile}/{identifier}", flush=True)
        counts[outcome] = counts.get(outcome, 0) + 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
