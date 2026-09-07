"""Generate the continuum_v1 Gkeyll dataset with a hard CUDA requirement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    "/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gkeyll/continuum_v1.json"
GKEYLL_ENV = Path(
    "/wangx/home/duxinxu/software/micromamba-root-v1/envs/gkeyll-build"
)
CUDA_LIBRARY_DIR = Path("/usr/local/cuda-12.1/lib64")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _slug_number(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def case_id(case: dict[str, Any]) -> str:
    return f"K{_slug_number(float(case['K']))}_a{_slug_number(float(case['alpha']))}"


def expand_cases(profile: dict[str, Any]) -> list[dict[str, Any]]:
    if "cases" in profile:
        cases = [dict(item) for item in profile["cases"]]
    else:
        cases = [
            {"K": K, "alpha": alpha}
            for K in profile["K_values"]
            for alpha in profile["alpha_values"]
        ]
    ids = [case_id(item) for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs collide after three-decimal canonicalization")
    return cases


def _query_gpu(gpu: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "memory_used_mib": int(fields[5]),
    }


def _require_permitted_gpu(dataset_root: Path, gpu: int) -> None:
    """Enforce dataset-local GPU exclusions before launching a solver."""
    policy_path = dataset_root / "manifests" / "disallowed_gpus.json"
    if not policy_path.is_file():
        return
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    disallowed = {int(index) for index in policy.get("indices", [])}
    if gpu in disallowed:
        reason = policy.get("reason", "dataset GPU policy")
        raise RuntimeError(
            f"GPU {gpu} is disallowed by {policy_path}: {reason}"
        )


def _wait_for_gpu(gpu: int, minimum_mib: int, timeout_minutes: float) -> dict[str, Any]:
    deadline = time.monotonic() + 60.0 * timeout_minutes
    while True:
        status = _query_gpu(gpu)
        if status["memory_free_mib"] >= minimum_mib:
            return status
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU {gpu} has {status['memory_free_mib']} MiB free; "
                f"at least {minimum_mib} MiB is required. CPU fallback is disabled."
            )
        print(
            f"GPU {gpu}: {status['memory_free_mib']} MiB free; waiting for "
            f"{minimum_mib} MiB (CPU fallback disabled)",
            flush=True,
        )
        time.sleep(30.0)


def _binary_version(binary: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [str(binary), "--version"], text=True, capture_output=True, env=environment
    )
    return (result.stdout + "\n" + result.stderr).strip()


def _require_cuda_binary(binary: Path, environment: dict[str, str]) -> str:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"missing Gkeyll executable: {binary}")
    version = _binary_version(binary, environment)
    if "Built with CUDA" not in version or "Built without CUDA" in version:
        raise RuntimeError(
            f"Gkeyll binary does not report CUDA support: {binary}\n{version}"
        )
    return version


def _parse_stat(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    values: dict[str, Any] = {"raw_text_sha256": _sha256(path)}
    patterns = {
        "use_gpu": r"\buse_gpu\s*:\s*(\d+)",
        "num_ranks": r"\bnum_ranks\s*:\s*(\d+)",
        "nup": r"\bnup\s*:\s*(\d+)",
        "total_tm": r"\btotal_tm\s*:\s*([0-9.eE+-]+)",
        "io_tm": r"\bio_tm\s*:\s*([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError(f"missing {key} in {path}")
        values[key] = float(match.group(1)) if key.endswith("_tm") else int(match.group(1))
    return values


def _write_checksums(raw_dir: Path, provenance_dir: Path) -> Path:
    checksum_path = provenance_dir / "raw_files.sha256"
    temporary = checksum_path.with_name(f".{checksum_path.name}.incomplete")
    lines = []
    for path in sorted(raw_dir.glob("*.gkyl")):
        lines.append(f"{_sha256(path)}  raw/{path.name}\n")
    temporary.write_text("".join(lines), encoding="utf-8")
    os.replace(temporary, checksum_path)
    return checksum_path


def _base_environment() -> dict[str, str]:
    environment = dict(os.environ)
    old_library_path = environment.get("LD_LIBRARY_PATH", "")
    library_parts = [str(GKEYLL_ENV / "lib"), str(CUDA_LIBRARY_DIR)]
    if old_library_path:
        library_parts.append(old_library_path)
    environment["LD_LIBRARY_PATH"] = ":".join(library_parts)
    return environment


def _dataset_contract(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "dataset_version": config["dataset_version"],
        "created_at_utc": _utc_now(),
        "description": config["description"],
        "normalization": config["normalization"],
        "solver": config["solver"],
        "quality_policy": config["quality_policy"],
        "source_config": str(config_path.resolve()),
        "source_config_sha256": _sha256(config_path),
        "layout": "profiles/<profile>/cases/<case_id>/{raw,provenance,processed}",
        "immutability": "completed raw cases are never overwritten",
    }


def initialize_dataset(
    config: dict[str, Any], config_path: Path, dataset_root: Path, profile_name: str
) -> list[dict[str, Any]]:
    profile = config["profiles"][profile_name]
    cases = expand_cases(profile)
    dataset_root.mkdir(parents=True, exist_ok=True)
    for name in ("manifests", "profiles", "logs", "audit"):
        (dataset_root / name).mkdir(exist_ok=True)
    contract_path = dataset_root / "DATASET.json"
    expected = _dataset_contract(config, config_path)
    if contract_path.exists():
        current = json.loads(contract_path.read_text(encoding="utf-8"))
        if current.get("dataset_version") != expected["dataset_version"]:
            raise RuntimeError(f"dataset version mismatch in {contract_path}")
    else:
        _atomic_json(contract_path, expected)

    plan = {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "profile": profile_name,
        "profile_settings": profile,
        "case_count": len(cases),
        "cases": [{"case_id": case_id(item), **item} for item in cases],
    }
    plan_path = dataset_root / "manifests" / f"{profile_name}_plan.json"
    if plan_path.exists():
        current_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if current_plan != plan:
            raise RuntimeError(
                f"immutable profile plan differs from current config: {plan_path}"
            )
    else:
        _atomic_json(plan_path, plan)
    return cases


def _case_metadata(
    profile_name: str,
    profile: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    delta_u = 2.0 * float(profile["vmax"]) / int(profile["nv"])
    recurrence_time = 2.0 * 3.141592653589793 / (float(case["K"]) * delta_u)
    output_dt = float(profile["t_end"]) / int(profile["num_frames"])
    return {
        "case_id": case_id(case),
        "profile": profile_name,
        "fidelity": profile["fidelity"],
        "parameters": {
            "K": float(case["K"]),
            "alpha": float(case["alpha"]),
            "n0_normalized": 1.0,
            "T0_normalized": 1.0,
        },
        "numerics": {
            "nx": int(profile["nx"]),
            "nv": int(profile["nv"]),
            "vmax": float(profile["vmax"]),
            "t_end": float(profile["t_end"]),
            "num_frames": int(profile["num_frames"]),
            "distribution_frame_stride": int(profile["distribution_frame_stride"]),
            "diagnostic_output_dt": output_dt,
            "distribution_output_dt": output_dt
            * int(profile["distribution_frame_stride"]),
            "delta_u_cell": delta_u,
            "classical_recurrence_time_estimate": recurrence_time,
        },
    }


def run_case(
    *,
    config: dict[str, Any],
    profile_name: str,
    case: dict[str, Any],
    dataset_root: Path,
    gpu: int,
    wait_gpu_minutes: float,
    version_text: str,
    dry_run: bool,
) -> str:
    profile = config["profiles"][profile_name]
    identifier = case_id(case)
    case_root = dataset_root / "profiles" / profile_name / "cases" / identifier
    raw_dir = case_root / "raw"
    provenance_dir = case_root / "provenance"
    processed_dir = case_root / "processed"
    complete_path = provenance_dir / "COMPLETE.json"
    if complete_path.exists():
        print(f"SKIP complete {profile_name}/{identifier}", flush=True)
        return "skipped"

    if dry_run:
        print(f"DRY-RUN GPU {gpu}: {profile_name}/{identifier}", flush=True)
        return "dry-run"

    minimum_mib = int(config["solver"]["minimum_free_gpu_memory_mib"])
    initial_gpu = _wait_for_gpu(gpu, minimum_mib, wait_gpu_minutes)
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite incomplete raw data in {raw_dir}"
        )

    lock_path = provenance_dir / "RUNNING.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"case is already locked: {case_root}") from error
    os.write(descriptor, f"pid={os.getpid()} started={_utc_now()}\n".encode())
    os.close(descriptor)

    metadata = _case_metadata(profile_name, profile, case)
    input_source = PROJECT_ROOT / config["solver"]["input"]
    input_name = f"continuum_v1_{profile_name}_{identifier}_1x1v_p2.lua"
    input_path = raw_dir / input_name
    binary = Path(config["solver"]["binary"])
    log_path = dataset_root / "logs" / f"{profile_name}_{identifier}.log"
    environment = _base_environment()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "GKYL_K0": str(case["K"]),
            "GKYL_ALPHA": str(case["alpha"]),
            "GKYL_NX": str(profile["nx"]),
            "GKYL_NV": str(profile["nv"]),
            "GKYL_VMAX": str(profile["vmax"]),
            "GKYL_T_END": str(profile["t_end"]),
            "GKYL_NUM_FRAMES": str(profile["num_frames"]),
            "GKYL_DISTRIBUTION_FRAME_STRIDE": str(
                profile["distribution_frame_stride"]
            ),
            "GKYL_CFL_FRAC": str(config["solver"]["cfl_fraction"]),
        }
    )

    started_at = _utc_now()
    shutil.copy2(input_source, input_path)
    metadata.update(
        {
            "status": "running",
            "started_at_utc": started_at,
            "gpu_before": initial_gpu,
            "cuda_visible_devices": str(gpu),
            "cpu_fallback": False,
            "input_sha256": _sha256(input_path),
            "binary": str(binary),
            "binary_sha256": _sha256(binary),
            "gkeyll_version": version_text,
        }
    )
    _atomic_json(provenance_dir / "RUNNING.json", metadata)

    command = [str(binary), "-g", input_name]
    failure: BaseException | None = None
    try:
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps(metadata, sort_keys=True) + "\n")
            log.write(f"command={command!r}\n")
            log.flush()
            print(
                f"START GPU {gpu} ({initial_gpu['memory_free_mib']} MiB free): "
                f"{profile_name}/{identifier}",
                flush=True,
            )
            result = subprocess.run(
                command,
                cwd=raw_dir,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Gkeyll failed with exit code {result.returncode}; see {log_path}"
            )

        stem = input_path.stem
        stat_source = raw_dir / f"{stem}-stat.json"
        if not stat_source.exists():
            raise RuntimeError(f"missing Gkeyll statistics: {stat_source}")
        stat = _parse_stat(stat_source)
        if stat["use_gpu"] != 1:
            raise RuntimeError(
                f"GPU contract violation: use_gpu={stat['use_gpu']} in {stat_source}"
            )
        final_frame = int(profile["num_frames"])
        required = [
            raw_dir / f"{stem}-field_{final_frame}.gkyl",
            raw_dir / f"{stem}-elc_M0_{final_frame}.gkyl",
            raw_dir / f"{stem}-elc_M3_{final_frame}.gkyl",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"terminal-frame contract failed; missing {missing}")
        distribution_files = sorted(raw_dir.glob(f"{stem}-elc_[0-9]*.gkyl"))
        expected_distribution_files = (
            final_frame // int(profile["distribution_frame_stride"]) + 1
        )
        if len(distribution_files) != expected_distribution_files:
            raise RuntimeError(
                "distribution-frame contract failed: "
                f"found {len(distribution_files)}, expected {expected_distribution_files}"
            )

        shutil.copy2(stat_source, provenance_dir / "stat.json")
        shutil.copy2(input_path, provenance_dir / "input.lua")
        checksum_path = _write_checksums(raw_dir, provenance_dir)
        finished_gpu = _query_gpu(gpu)
        complete = {
            **metadata,
            "status": "complete",
            "finished_at_utc": _utc_now(),
            "gpu_after": finished_gpu,
            "gkeyll_stat": stat,
            "raw_gkyl_file_count": len(list(raw_dir.glob("*.gkyl"))),
            "distribution_file_count": len(distribution_files),
            "raw_checksums_sha256": _sha256(checksum_path),
            "log": str(log_path),
            "log_sha256": _sha256(log_path),
        }
        _atomic_json(complete_path, complete)
        (provenance_dir / "RUNNING.json").unlink(missing_ok=True)
        print(
            f"COMPLETE GPU {gpu}: {profile_name}/{identifier}; "
            f"solver={stat['total_tm']:.1f}s, f_frames={len(distribution_files)}",
            flush=True,
        )
        return "complete"
    except BaseException as error:
        failure = error
        failed = {
            **metadata,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error": repr(error),
            "log": str(log_path),
        }
        _atomic_json(provenance_dir / "FAILED.json", failed)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
        if failure is not None:
            print(f"FAILED {profile_name}/{identifier}: {failure}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--only-case")
    parser.add_argument("--wait-gpu-minutes", type=float, default=0.0)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.profile not in config["profiles"]:
        raise KeyError(f"unknown profile {args.profile!r}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else Path(config["dataset_root"])
    )
    cases = initialize_dataset(config, config_path, dataset_root, args.profile)
    if args.init_only:
        print(
            f"Initialized {dataset_root} profile={args.profile} cases={len(cases)}"
        )
        return

    _require_permitted_gpu(dataset_root, args.gpu)

    selected = [
        item
        for index, item in enumerate(cases)
        if index % args.shard_count == args.shard_index
    ]
    if args.only_case is not None:
        selected = [item for item in selected if case_id(item) == args.only_case]
        if not selected:
            raise ValueError(f"case {args.only_case!r} is not in this shard/profile")
    if args.case_offset < 0:
        raise ValueError("case offset must be non-negative")
    selected = selected[args.case_offset :]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]

    environment = _base_environment()
    binary = Path(config["solver"]["binary"])
    version_text = _require_cuda_binary(binary, environment)
    print(
        f"CUDA-only contract accepted for {binary}; selected={len(selected)}, "
        f"GPU={args.gpu}",
        flush=True,
    )
    counts: dict[str, int] = {}
    for item in selected:
        outcome = run_case(
            config=config,
            profile_name=args.profile,
            case=item,
            dataset_root=dataset_root,
            gpu=args.gpu,
            wait_gpu_minutes=args.wait_gpu_minutes,
            version_text=version_text,
            dry_run=args.dry_run,
        )
        counts[outcome] = counts.get(outcome, 0) + 1
    print(f"Finished shard: {json.dumps(counts, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
