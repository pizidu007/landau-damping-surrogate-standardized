"""Select a supervised closure using validation metrics only."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for summary_path in sorted(args.runs_dir.glob("*/validation_summary.json")):
        summary = json.loads(summary_path.read_text())
        candidates.append({
            "run": summary_path.parent.name,
            "validation_relative_l2": float(summary["metrics"]["relative_l2"]),
            "validation_spectral_relative_l2": float(summary["metrics"]["spectral_relative_l2"]),
            "checkpoint": str(summary_path.parent / "best.pt"),
        })
    if not candidates:
        raise FileNotFoundError(f"No completed candidates in {args.runs_dir}")
    candidates.sort(key=lambda row: (row["validation_relative_l2"], row["validation_spectral_relative_l2"]))
    selected = candidates[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["checkpoint"], args.output_dir / "best_supervised.pt")
    payload = {"selection_metric": "validation_pair_macro_relative_l2", "selected": selected, "ranking": candidates}
    temporary = args.output_dir / "selection.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output_dir / "selection.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
