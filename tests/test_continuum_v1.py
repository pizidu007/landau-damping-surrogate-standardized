from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from landau_surrogate.data.continuum_v1 import (
    ContinuumCase,
    load_continuum_case,
    load_continuum_case_index,
    spectral_derivative,
    spectral_lowpass,
)


def _write_case(root: Path, case_id: str, k: float, alpha: float) -> Path:
    path = (
        root
        / "profiles"
        / "production"
        / "cases"
        / case_id
        / "processed"
        / "trajectory.h5"
    )
    path.parent.mkdir(parents=True)
    nx, nt = 16, 5
    length = 2.0 * np.pi / k
    x = (np.arange(nx) + 0.5) * length / nx
    phase = k * x
    central = np.empty((nt, nx, 4), dtype=np.float32)
    for frame in range(nt):
        central[frame, :, 0] = 1.0 + 0.1 * np.cos(phase + 0.1 * frame)
        central[frame, :, 1] = 0.2 * np.sin(phase)
        central[frame, :, 2] = 2.0 + 0.1 * np.cos(phase)
        central[frame, :, 3] = np.sin(2.0 * phase + 0.1 * frame)
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_version"] = "continuum_v1"
        handle.attrs["case_id"] = case_id
        handle.attrs["K"] = k
        handle.attrs["alpha"] = alpha
        handle.attrs["source_solver_use_gpu"] = 1
        diagnostics = handle.create_group("diagnostics")
        diagnostics.create_dataset("central_moments", data=central)
        diagnostics.create_dataset(
            "time", data=np.asarray([0.0, 0.09, 0.20, 0.31, 0.43])
        )
        coordinates = handle.create_group("coordinates")
        coordinates.create_dataset("x_cell", data=x)
    return path


def test_spectral_helpers() -> None:
    nx = 32
    phase = 2.0 * np.pi * np.arange(nx) / nx
    value = np.sin(2.0 * phase) + 0.5 * np.sin(7.0 * phase)
    filtered = spectral_lowpass(value, 3)
    np.testing.assert_allclose(filtered, np.sin(2.0 * phase), atol=2.0e-6)
    derivative = spectral_derivative(value, 0.4, maximum_mode=3)
    np.testing.assert_allclose(
        derivative, 0.8 * np.cos(2.0 * phase), atol=2.0e-6
    )


def test_load_continuum_case_reconstructs_pressure_and_preserves_time(
    tmp_path: Path,
) -> None:
    path = _write_case(tmp_path, "case_train", 0.4, 0.1)
    case = ContinuumCase(
        case_id="case_train",
        K=0.4,
        alpha=0.1,
        split="train",
        regime="strong_nonlinear",
        path=path,
    )
    trajectory = load_continuum_case(case, temporal_stride=2)
    assert trajectory.state.shape == (3, 3, 16)
    np.testing.assert_allclose(trajectory.time, [0.0, 0.20, 0.43])
    with h5py.File(path, "r") as handle:
        central = handle["diagnostics/central_moments"][::2]
    np.testing.assert_allclose(
        trajectory.state[:, 2], central[..., 0] * central[..., 2], rtol=2e-6
    )
    phase = 0.4 * (trajectory.x_over_l * (2.0 * np.pi / 0.4))
    np.testing.assert_allclose(
        trajectory.heat_flux_gradient[0], 0.8 * np.cos(2.0 * phase), atol=2e-6
    )


def test_case_index_applies_frozen_split_and_qc_exclusion(tmp_path: Path) -> None:
    specifications = [
        ("case_train", 0.30, 0.08, "train", "strong_nonlinear"),
        ("case_validation", 0.35, 0.10, "validation", "transition"),
        ("case_test", 0.40, 0.12, "test", "weak"),
        ("case_excluded", 0.45, 0.20, "test", "strong_nonlinear"),
    ]
    for case_id, k, alpha, _split, _regime in specifications[:-1]:
        _write_case(tmp_path, case_id, k, alpha)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    assignments = [
        {
            "case_id": case_id,
            "K": k,
            "alpha": alpha,
            "canonical_split": split,
        }
        for case_id, k, alpha, split, _regime in specifications
    ]
    (manifest_dir / "splits_v1.json").write_text(
        json.dumps({"assignments": assignments}), encoding="utf-8"
    )
    (manifest_dir / "training_eligibility_v1.json").write_text(
        json.dumps(
            {
                "counts": {"eligible": 3},
                "eligible_by_canonical_split": {
                    "train": 1,
                    "validation": 1,
                    "test": 1,
                },
                "excluded_cases": [{"case_id": "case_excluded"}],
            }
        ),
        encoding="utf-8",
    )
    label_dir = tmp_path / "figures" / "rebound_audit_v1"
    label_dir.mkdir(parents=True)
    (label_dir / "production_rebound_labels.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": case_id, "regime": regime}
                    for case_id, _k, _alpha, _split, regime in specifications
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_continuum_case_index(tmp_path)
    assert [case.case_id for case in cases] == [
        "case_train",
        "case_validation",
        "case_test",
    ]
    assert [case.split for case in cases] == ["train", "validation", "test"]
