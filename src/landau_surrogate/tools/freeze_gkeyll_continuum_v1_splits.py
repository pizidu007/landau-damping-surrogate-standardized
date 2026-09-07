"""Freeze whole-case V1 splits and parameter-generalization evaluation suites."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from landau_surrogate.tools.generate_gkeyll_continuum_v1 import case_id


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


def _production_rows(root: Path) -> list[dict[str, Any]]:
    plan = root / "manifests" / "production_plan.json"
    if plan.exists():
        return json.loads(plan.read_text(encoding="utf-8"))["cases"]
    config = json.loads(
        (root / "manifests" / "production_config.json").read_text(encoding="utf-8")
    )
    return [
        {"case_id": case_id(item), **item}
        for item in config["profiles"]["production"]["cases"]
    ]


def _normalized(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    K = np.asarray([float(row["K"]) for row in rows])
    alpha = np.asarray([float(row["alpha"]) for row in rows])
    return {
        "K": K,
        "alpha": alpha,
        "z": np.column_stack(
            (
                (K - K.min()) / (K.max() - K.min()),
                (alpha - alpha.min()) / (alpha.max() - alpha.min()),
            )
        ),
    }


def _diverse_select(
    candidates: list[int], count: int, z: np.ndarray, selected: list[int]
) -> list[int]:
    candidates = sorted(set(candidates) - set(selected))
    result = []
    while candidates and len(result) < count:
        references = selected + result
        if references:
            distances = [
                float(np.min(np.linalg.norm(z[index] - z[references], axis=1)))
                for index in candidates
            ]
        else:
            distances = [float(np.linalg.norm(z[index] - 0.5)) for index in candidates]
        best = max(range(len(candidates)), key=lambda i: (distances[i], -candidates[i]))
        result.append(candidates.pop(best))
    return result


def _suite(test: list[int], all_indices: list[int], rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_set = set(test)
    return {
        "train_case_ids": [rows[index]["case_id"] for index in all_indices if index not in test_set],
        "test_case_ids": [rows[index]["case_id"] for index in all_indices if index in test_set],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    root = args.dataset_root.resolve()
    rows = _production_rows(root)
    if len(rows) < 20:
        raise RuntimeError("at least 20 production cases are required to freeze splits")
    labels = json.loads(
        (root / "audit" / "scout_physics_labels.json").read_text(encoding="utf-8")
    )
    threshold_K = np.asarray(
        sorted(float(key) for key in labels["alpha_c_by_K"]), dtype=np.float64
    )
    threshold_alpha = np.asarray(
        [labels["alpha_c_by_K"][f"{K:.3f}"] for K in threshold_K],
        dtype=np.float64,
    )
    arrays = _normalized(rows)
    K, alpha, z = arrays["K"], arrays["alpha"], arrays["z"]
    alpha_c = np.interp(K, threshold_K, threshold_alpha)
    all_indices = list(range(len(rows)))
    boundary = [
        i for i in all_indices if np.any((z[i] <= 0.04) | (z[i] >= 0.96))
    ]
    centers = np.asarray(((0.20, 0.35), (0.50, 0.72), (0.78, 0.25)))
    holes = [
        i
        for i in all_indices
        if float(np.min(np.linalg.norm(z[i] - centers, axis=1))) <= 0.11
    ]
    transition = [i for i in all_indices if abs(alpha[i] - alpha_c[i]) <= 0.008]
    test_count = max(1, int(round(args.test_fraction * len(rows))))
    selected: list[int] = []
    for candidates, quota in (
        (boundary, max(4, test_count // 4)),
        (holes, max(4, test_count // 3)),
        (transition, max(4, test_count // 4)),
        (all_indices, test_count),
    ):
        selected.extend(
            _diverse_select(candidates, min(quota, test_count - len(selected)), z, selected)
        )
        if len(selected) >= test_count:
            break
    test = sorted(selected[:test_count])
    remaining = [i for i in all_indices if i not in set(test)]
    validation_count = max(1, int(round(args.validation_fraction * len(rows))))
    validation = sorted(_diverse_select(remaining, validation_count, z, test))
    train = [i for i in remaining if i not in set(validation)]

    assignments = []
    for index, row in enumerate(rows):
        split = "test" if index in test else "validation" if index in validation else "train"
        assignments.append(
            {
                "case_id": row["case_id"],
                "K": float(row["K"]),
                "alpha": float(row["alpha"]),
                "alpha_c_interpolated": float(alpha_c[index]),
                "canonical_split": split,
                "equivalence_group": f"K{float(row['K']):.3f}_a{float(row['alpha']):.3f}",
            }
        )

    leave_K = [i for i in all_indices if abs(K[i] - 0.39) <= 0.012]
    leave_alpha = [i for i in all_indices if abs(alpha[i] - 0.115) <= 0.009]
    transition_holdout = [
        i for i in all_indices if abs(alpha[i] - alpha_c[i]) <= 0.010
    ]
    profile_roles = {
        "smoke": "contract_only",
        "scout": "screening_only",
        "anchors_low": "convergence_only",
        "anchors": "convergence_only",
        "anchors_high": "convergence_only",
        "production": "canonical_model_data",
    }
    equivalent_profiles: dict[str, list[dict[str, str]]] = {}
    for plan_path in sorted((root / "manifests").glob("*_plan.json")):
        profile = plan_path.name.removesuffix("_plan.json")
        if profile not in profile_roles:
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for item in plan["cases"]:
            group = f"K{float(item['K']):.3f}_a{float(item['alpha']):.3f}"
            equivalent_profiles.setdefault(group, []).append(
                {"profile": profile, "case_id": item["case_id"], "role": profile_roles[profile]}
            )
    recorded_production = {
        item["case_id"]
        for members in equivalent_profiles.values()
        for item in members
        if item["profile"] == "production"
    }
    for item in rows:
        if item["case_id"] in recorded_production:
            continue
        group = f"K{float(item['K']):.3f}_a{float(item['alpha']):.3f}"
        equivalent_profiles.setdefault(group, []).append(
            {
                "profile": "production",
                "case_id": item["case_id"],
                "role": profile_roles["production"],
            }
        )
    profile_order = {
        name: index
        for index, name in enumerate(
            ("smoke", "scout", "anchors_low", "anchors", "anchors_high", "production")
        )
    }
    for members in equivalent_profiles.values():
        members.sort(key=lambda item: (profile_order[item["profile"]], item["case_id"]))

    suites = {
        "interpolation_holes": {
            **_suite(holes, all_indices, rows),
            "definition": "three sealed interior disks of radius 0.11 in normalized (K,alpha)",
        },
        "leave_one_K_band": {
            **_suite(leave_K, all_indices, rows),
            "definition": "abs(K-0.39) <= 0.012",
        },
        "leave_one_alpha_band": {
            **_suite(leave_alpha, all_indices, rows),
            "definition": "abs(alpha-0.115) <= 0.009",
        },
        "weak_strong_transition": {
            **_suite(transition_holdout, all_indices, rows),
            "definition": "abs(alpha-alpha_c(K)) <= 0.010",
        },
        "boundary_extrapolation": {
            **_suite(boundary, all_indices, rows),
            "definition": "outside the inner 92% box of normalized (K,alpha)",
        },
        "time_extrapolation": {
            "case_ids": [row["case_id"] for row in rows],
            "train_time": [0.0, 40.0],
            "validation_time": [40.0, 55.0],
            "test_time": [55.0, 80.0],
            "note": "A separate chronological protocol; never mix its future frames into its training prefix.",
        },
        "scale_equivariance": {
            "source": str(root / "manifests" / "scale_equivariance_plan.json"),
            "note": "All dimensional views of one normalized trajectory remain in the same equivalence group.",
        },
    }
    manifest = {
        "schema_version": 1,
        "dataset_version": "continuum_v1",
        "frozen_before_production_labels": True,
        "split_unit": "complete normalized trajectory and all scale/fidelity replicas",
        "frame_weighting": "case-balanced; never treat adjacent frames as independent cases",
        "canonical_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "assignments": assignments,
        "profile_roles": profile_roles,
        "equivalent_profile_cases": equivalent_profiles,
        "generalization_suites": suites,
    }
    output = root / "manifests" / "splits_v1.json"
    if output.exists():
        current = json.loads(output.read_text(encoding="utf-8"))
        if current != manifest:
            raise RuntimeError(f"refusing to change frozen split manifest: {output}")
    else:
        _atomic_json(output, manifest)
    print(json.dumps({"output": str(output), **manifest["canonical_counts"]}, indent=2))


if __name__ == "__main__":
    main()
