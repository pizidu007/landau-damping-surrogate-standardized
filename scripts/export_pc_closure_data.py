"""Export lossless Round 11 arrays and matched checkpoint bundles for personal PCs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from landau_surrogate.data.continuum_v1 import load_continuum_case_index, load_continuum_case, spectral_lowpass
from landau_surrogate.tools.audit_continuum_v1_closure_oracle import interpolate, electric_numpy

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_starter(cases):
    identifiers = set()
    for split, count in (("train", 8), ("validation", 2)):
        for regime in ("weak", "transition", "strong_nonlinear"):
            group = sorted((c for c in cases if c.split == split and c.regime == regime),
                           key=lambda c: (c.K, c.alpha, c.case_id))
            if len(group) < count:
                raise ValueError("Insufficient cases for the frozen starter selection")
            indices = np.rint(np.linspace(0, len(group) - 1, count)).astype(int)
            identifiers.update(group[i].case_id for i in indices)
    return identifiers


def export_case(case, starter, cache_dir, output):
    grid = np.arange(4001, dtype=np.float64) * .02
    cached = cache_dir / f"mode16_{case.case_id}.npz"
    trajectory = None
    if case.split != "test":
        if not cached.is_file():
            raise FileNotFoundError(f"Require the original Round 11 train/validation cache: {cached}")
        with np.load(cached, allow_pickle=False) as data:
            state, gradient, dt = data["state"], data["gradient"], float(data["dt"])
        assert dt == .02
        source = {"kind": "original_round11_cache", "name": cached.name, "sha256": digest(cached)}
    else:
        # The OLD diagnostic test already used in Round 10. This exports arrays;
        # it does not run a model or open/generate the independent new holdout.
        trajectory = load_continuum_case(case)
        moments = interpolate(spectral_lowpass(trajectory.state, 16), trajectory.time, grid)
        electric = electric_numpy(moments[:, None], np.array([case.K]))[:, 0]
        state = np.concatenate((moments, electric[:, None]), axis=1).astype(np.float32)
        gradient = interpolate(spectral_lowpass(trajectory.heat_flux_gradient, 16), trajectory.time, grid)
        source = {"kind": "existing_continuum_v1_diagnostics", "case_id": case.case_id}
    assert state.shape == (4001, 4, 128) and gradient.shape == (4001, 128)
    assert state.dtype == gradient.dtype == np.float32
    assert np.isfinite(state).all() and np.isfinite(gradient).all()
    arrays = {"state": state, "gradient": gradient, "time": grid,
              "dt": np.array(.02), "x_over_l": (np.arange(128, dtype=np.float64) + .5) / 128}
    if case.split != "train":
        trajectory = trajectory or load_continuum_case(case)
        reference_time = np.arange(801, dtype=np.float64) * .1
        moments = interpolate(trajectory.state, trajectory.time, reference_time)
        electric = electric_numpy(moments[:, None], np.array([case.K]))[:, 0]
        arrays["reference_state"] = np.concatenate((moments, electric[:, None]), axis=1).astype(np.float32)
        arrays["reference_time"] = reference_time
    relative = Path("cases") / case.split / f"{case.case_id}.npz"
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_LZMA) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w") as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    # Verify the *decompressed arrays*, not just the new container checksum.
    with np.load(temporary, allow_pickle=False) as saved:
        for name, value in arrays.items():
            assert saved[name].dtype == value.dtype and np.array_equal(saved[name], value), (case.case_id, name)
    temporary.replace(target)
    record = {"case_id": case.case_id, "K": case.K, "alpha": case.alpha,
              "split": case.split, "regime": case.regime, "starter": case.case_id in starter,
              "path": relative.as_posix(), "shape": list(state.shape), "size_bytes": target.stat().st_size,
              "sha256": digest(target), "source": source,
              "arrays_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
    sufficient = None
    if case.split == "train":
        values = state.astype(np.float64)
        y = gradient.astype(np.float64)
        wave = np.arange(65) * case.K
        wave[17:] = 0
        hp = np.fft.irfft(np.fft.rfft(values[:, 2] - values[:, 0], axis=-1) * wave, n=128)
        sufficient = {"sum": values.sum(axis=(0, 2)), "sum2": (values * values).sum(axis=(0, 2)),
                      "count": values.shape[0] * values.shape[2], "gradient_sum": y.sum(),
                      "gradient_sum2": (y * y).sum(), "hp_xy": (hp * y).sum(), "hp_xx": (hp * hp).sum()}
    return record, sufficient


def statistics(rows, subset):
    selected = [(r, s) for r, s in rows if s is not None and (subset == "full" or r["starter"])]
    count = sum(s["count"] for _, s in selected)
    mean = sum(s["sum"] for _, s in selected) / count
    std = np.sqrt(np.maximum(sum(s["sum2"] for _, s in selected) / count - mean * mean, 1e-12))
    gradient_mean = sum(s["gradient_sum"] for _, s in selected) / count
    gradient_std = np.sqrt(max(sum(s["gradient_sum2"] for _, s in selected) / count - gradient_mean**2, 1e-12))
    k = np.array([r["K"] for r, _ in selected]); alpha = np.array([r["alpha"] for r, _ in selected])
    unconstrained = sum(s["hp_xy"] for _, s in selected) / max(sum(s["hp_xx"] for _, s in selected), 1e-30)
    return {"source_split": "train", "subset": subset, "case_count": len(selected),
            "case_ids": [r["case_id"] for r, _ in selected], "input_mean": mean.tolist(), "input_std": std.tolist(),
            "gradient_std": float(gradient_std), "gradient_mean": 0., "k_mean": float(k.mean()),
            "k_std": float(k.std()), "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std()),
            "hp_scale": float(max(0., unconstrained)), "hp_scale_unconstrained": float(unconstrained),
            "hp_definition": "scale * abs(d/dx)(p-n), modes <= 16; nonnegative least squares, no intercept",
            "method": "float64 streaming sufficient statistics; no validation or test fitting"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; exports never overwrite a prior bundle")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "landau-pc-closure-v1"
    output.mkdir()
    cases = load_continuum_case_index(args.dataset_root)
    starter = select_starter(cases)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = pool.map(lambda c: export_case(c, starter, args.cache_dir, output), cases)
        for item in jobs:
            rows.append(item)
            if len(rows) % 10 == 0 or len(rows) == len(cases):
                print(f"Exported and verified {len(rows)}/{len(cases)} trajectories", flush=True)
    for subset in ("starter", "full"):
        atomic_json(output / f"statistics_{subset}.json", statistics(rows, subset))
    checkpoint_manifest = json.loads((ROOT / "artifacts/manifests/public_release_2026-09-09.json").read_text())
    checkpoints = []
    (output / "checkpoints").mkdir()
    for arm in "ABC":
        item = next(c for c in checkpoint_manifest["checkpoints"] if c["id"] == f"round11_{arm}_seed0")
        source = ROOT / item["path"]
        assert digest(source) == item["sha256"]
        relative = f"checkpoints/{item['asset_name']}"
        shutil.copyfile(source, output / relative)
        checkpoints.append({**item, "bundle_path": relative,
                            "comparison_role": "full139-trained historical reference; not trained on the starter24 subset"})
    manifest = {"schema": "landau_pc_closure_v1", "source_dataset": "continuum_v1",
                "state_channels": ["n", "u", "p", "E"], "target": "dq/dx", "maximum_mode": 16,
                "dt": .02, "t_end": 80., "nx": 128, "dtype": "float32", "compression": "NPZ ZIP_LZMA, lossless",
                "reference": "validation/old diagnostic test contain UNFILTERED [n,u,p,E] at dt=0.1",
                "raw_phase_space_included": False, "new_independent_holdout_included": False,
                "starter_selection": "Within train/validation and each regime, sort by (K,alpha,case_id); take 8/2 evenly spaced ranks. No model error used.",
                "full_counts": {"train": 139, "validation": 20, "test": 36},
                "starter_counts": {"train": 24, "validation": 6},
                "cases": [r for r, _ in rows], "checkpoints": checkpoints}
    atomic_json(output / "manifest.json", manifest)
    guide = ROOT / "docs/onboarding/05_PC_FNO_SSM_TASK_zh-CN.md"
    if guide.is_file():
        shutil.copyfile(guide, output / "README_zh-CN.md")
    groups = {"starter": [r for r, _ in rows if r["starter"]],
              "validation_extra": [r for r, _ in rows if r["split"] == "validation" and not r["starter"]],
              "diagnostic_test_optional": [r for r, _ in rows if r["split"] == "test"]}
    remaining = [r for r, _ in rows if r["split"] == "train" and not r["starter"]]
    for i, offset in enumerate(range(0, len(remaining), 40), 1):
        groups[f"train_extra_{i}"] = remaining[offset:offset + 40]
    archives = []
    for name, selected in groups.items():
        archive_path = args.output_dir / f"landau_pc_closure_v1_{name}.zip"
        paths = [output / r["path"] for r in selected]
        if name == "starter":
            paths.extend(p for p in output.iterdir() if p.is_file())
            paths.extend((output / "checkpoints").iterdir())
        # Each NPZ is already compressed; store outer members to avoid another
        # full decompression/recompression pass and preserve independent files.
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in paths:
                archive.write(path, path.relative_to(args.output_dir).as_posix())
        archives.append({"group": name, "filename": archive_path.name,
                         "size_bytes": archive_path.stat().st_size, "sha256": digest(archive_path),
                         "case_ids": [r["case_id"] for r in selected]})
        print(f"Archive {name}: {archive_path.stat().st_size/2**20:.1f} MiB", flush=True)
    atomic_json(args.output_dir / "archives.json", {"schema": "landau_pc_closure_archives_v1", "archives": archives})
    (args.output_dir / "SHA256SUMS").write_text("".join(f"{a['sha256']}  {a['filename']}\n" for a in archives))
    print(f"Completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
