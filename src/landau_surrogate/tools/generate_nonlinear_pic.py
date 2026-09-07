"""Generate strong-nonlinear Landau PIC cases on one CUDA shard."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import torch

from landau_surrogate.data.paths import nonlinear_runs_root
from landau_surrogate.pic.cuda_pic import CudaPICConfig, run_case, save_case_hdf5


def load_config(path: Path, profile: str) -> tuple[CudaPICConfig, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if profile not in payload["profiles"]:
        raise KeyError(f"Unknown profile {profile!r}")
    selected = payload["profiles"][profile]
    solver = CudaPICConfig(**selected["solver"])
    solver.validate()
    if "cases" in selected:
        cases = list(selected["cases"])
    else:
        grid = selected["case_grid"]
        split_lookup: dict[tuple[float, float], str] = {}
        for split, pairs in selected.get("split_pairs", {}).items():
            for pair in pairs:
                split_lookup[(round(float(pair[0]), 6), round(float(pair[1]), 6))] = split
        cases = []
        for k_value in grid["k_values"]:
            for alpha in grid["alpha_values"]:
                pair = (round(float(k_value), 6), round(float(alpha), 6))
                split = split_lookup.get(pair, "train")
                for seed in grid.get("seeds", [0]):
                    cases.append(
                        {
                            "k": float(k_value),
                            "alpha": float(alpha),
                            "seed": int(seed),
                            "split": split,
                        }
                    )
    identifiers: set[str] = set()
    for case in cases:
        case["case_id"] = case_id(case["k"], case["alpha"], case.get("seed", 0))
        if case["case_id"] in identifiers:
            raise ValueError(f"Duplicate case: {case['case_id']}")
        identifiers.add(case["case_id"])
        case.setdefault("split", "train")
    return solver, cases


def case_id(k_value: float, alpha: float, seed: int) -> str:
    return (
        f"k{k_value:.3f}_a{alpha:.3f}_s{int(seed):02d}".replace(".", "p")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/nonlinear_pic_v1.json"),
    )
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "paper_match", "formal"), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=nonlinear_runs_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return str(handle.attrs.get("status", "")) == "COMPLETE"
    except OSError:
        return False


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    solver, cases = load_config(args.config, args.profile)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in requested]
        missing = requested - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown requested cases: {sorted(missing)}")
    cases = [
        case for index, case in enumerate(cases) if index % args.shard_count == args.shard_index
    ]
    run_dir = args.output_root / args.run_id
    case_dir = run_dir / "cases"
    plan = {
        "profile": args.profile,
        "run_id": args.run_id,
        "output": str(run_dir),
        "device": args.device,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "solver": asdict(solver),
        "particle_count_per_case": solver.particle_count,
        "pic_steps_per_case": solver.step_count,
        "cases": cases,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is intentionally disabled")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    device_tag = args.device.replace(":", "_").replace("/", "_")
    plan_path = run_dir / (
        f"plan_shard_{args.shard_index:02d}_of_{args.shard_count:02d}_{device_tag}.json"
    )
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    for index, case in enumerate(cases, start=1):
        output = case_dir / f"{case['case_id']}.h5"
        if output.exists():
            if args.resume and is_complete(output):
                print(f"[{index}/{len(cases)}] skip complete {output.name}")
                continue
            raise FileExistsError(f"Output exists but is not resumable: {output}")
        print(f"[{index}/{len(cases)}] run {case['case_id']} on {device}")
        result = run_case(
            float(case["k"]),
            float(case["alpha"]),
            int(case.get("seed", 0)),
            solver,
            device,
        )
        save_case_hdf5(result, output, str(case["split"]), args.compression)
        print(
            f"  saved={output} runtime={result['runtime_seconds']:.3f}s "
            f"peak={result['peak_memory_bytes']/1024**3:.3f}GiB"
        )
        del result
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
