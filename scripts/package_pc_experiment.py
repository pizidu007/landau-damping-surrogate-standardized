"""Assemble ONE self-contained handoff ZIP from the verified portable export."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-root", type=Path, required=True, help="Directory containing the exported dataset manifest.json")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    kit = args.output_dir / "landau-fno-ssm-kit"
    kit.mkdir(parents=True)

    def copy(source, relative):
        target = kit / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    data_manifest = json.loads((args.export_root / "manifest.json").read_text())
    for i, case in enumerate(data_manifest["cases"]):
        source = args.export_root / case["path"]
        if digest(source) != case["sha256"]:
            raise ValueError(f"Export changed: {case['case_id']}")
        copy(source, "data/" + case["path"])
    data_manifest["checkpoint_paths_relative_to"] = "kit root, not data root"
    (kit / "data/manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("statistics_full.json", "statistics_starter.json"):
        copy(args.export_root / name, "data/" + name)
    for checkpoint in data_manifest["checkpoints"]:
        source = args.export_root / checkpoint["bundle_path"]
        if digest(source) != checkpoint["sha256"]:
            raise ValueError("Checkpoint differs from published original")
        copy(source, checkpoint["bundle_path"])
    for path in (ROOT / "src").rglob("*.py"):
        copy(path, path.relative_to(ROOT))
    for name in ("train_pc_ssm.py", "compare_pc_closures.py", "evaluate_pc_closure.py",
                 "inspect_pc_closure_data.py", "summarize_round11_history.py"):
        copy(ROOT / "scripts" / name, "scripts/" + name)
    for name in ("test_pc_fno_ssm.py", "test_portable_closure.py"):
        copy(ROOT / "tests" / name, "tests/" + name)
    for name in ("pyproject.toml", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        copy(ROOT / name, name)
    copy(ROOT / "reference/DIFFERENCES.md", "docs/upstream/DIFFERENCES.md")
    copy(ROOT / "reference/upstream/landau-damping-surrogate-original/LICENSE.md", "docs/upstream/LICENSE.md")
    # The historical upstream snapshot is unnecessary for running these tasks;
    # retain its authorship, license and a link to the fixed reference instead.
    notices = (kit / "THIRD_PARTY_NOTICES.md").read_text()
    notices = notices.replace("`reference/upstream/`", "https://github.com/kaneman777/landau-damping-surrogate/tree/9490dbff6b322bf7f43bbcff49af48bb9ff65fd7")
    notices = notices.replace("`reference/DIFFERENCES.md`", "`docs/upstream/DIFFERENCES.md`")
    (kit / "THIRD_PARTY_NOTICES.md").write_text(notices)
    readme = (ROOT / "docs/onboarding/PC_KIT_README_zh-CN.md").read_text()
    (kit / "README.md").write_text(readme.replace("](05_PC_FNO_SSM_TASK_zh-CN.md)", "](docs/05_PC_FNO_SSM_TASK_zh-CN.md)"), encoding="utf-8")
    for name in ("01_BACKGROUND_zh-CN.md", "02_PROJECT_LOGIC_zh-CN.md", "03_TASK_DEFINITIONS_zh-CN.md",
                 "04_GLOSSARY_zh-CN.md", "05_PC_FNO_SSM_TASK_zh-CN.md"):
        content = (ROOT / "docs/onboarding" / name).read_text()
        # These optional background chapters link into the larger repository.
        # Make every unavailable relative target an explicit GitHub link.
        import re
        def link(match):
            value = match.group(1)
            if value.startswith(("https://", "http://", "#")):
                return match.group(0)
            path, _, anchor = value.partition("#")
            resolved = (ROOT / "docs/onboarding" / path).resolve()
            relative = resolved.relative_to(ROOT)
            if relative.parts[:2] == ("docs", "onboarding") and relative.name in (
                    "01_BACKGROUND_zh-CN.md", "02_PROJECT_LOGIC_zh-CN.md", "03_TASK_DEFINITIONS_zh-CN.md",
                    "04_GLOSSARY_zh-CN.md", "05_PC_FNO_SSM_TASK_zh-CN.md"):
                return "](" + relative.name + ("#" + anchor if anchor else "") + ")"
            return "](https://github.com/pizidu007/landau-damping-surrogate-standardized/blob/main/" + relative.as_posix() + ("#" + anchor if anchor else "") + ")"
        content = re.sub(r"\]\(([^)]+)\)", link, content)
        (kit / "docs" / name).write_text(content, encoding="utf-8")
    frozen = ROOT / "results/published/2026-09-09/round11"
    for name in ("summary.json", "quality.json"):
        copy(frozen / name, "docs/previous_results/" + name)
    for name in ("01_overview.png", "02_paired_differences.png", "03_representative_field_energy.png"):
        copy(frozen / "figures" / name, "docs/previous_results/" + name)
    copy(ROOT / "configs/training/continuum_history_closure_round11_quality.json", "configs/training/continuum_history_closure_round11_quality.json")
    entries = []
    for path in sorted(kit.rglob("*")):
        if path.is_file():
            entries.append({"path": path.relative_to(kit).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"package": "landau-fno-ssm-kit", "date": "2026-09-09",
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_was_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "cases": data_manifest["full_counts"], "old_checkpoint_count": 3,
                "new_model_status": "trainable experimental prototype; no full accuracy claim",
                "tested_runtime": {"python": "3.11", "torch": "2.5.1+cu121", "test_device": "cpu"},
                "files": entries}
    (kit / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive_path = args.output_dir / "landau-fno-ssm-kit-2026-09-09.zip"
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        for path in sorted(kit.rglob("*")):
            if path.is_file():
                kind = zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED
                archive.write(path, path.relative_to(args.output_dir).as_posix(), compress_type=kind)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Final ZIP CRC verification failed")
    summary = {"path": str(archive_path.resolve()), "size_bytes": archive_path.stat().st_size,
               "sha256": digest(archive_path), "file_count": len(entries) + 1, "case_count": len(data_manifest["cases"])}
    archive_path.with_suffix(".sha256").write_text(summary["sha256"] + "  " + archive_path.name + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
