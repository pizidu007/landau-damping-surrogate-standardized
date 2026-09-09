"""Run the provided A/B/C, HP and a new small-model checkpoint on identical cases."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--checkpoint", type=Path, help="A new checkpoint from train_pc_ssm.py")
    p.add_argument("--output-dir", type=Path, default=Path("runs/comparison"))
    p.add_argument("--horizon", type=float, default=80.)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    models = [("hp", "hp", None)]
    models += [(f"round11_{a}", "round11", args.checkpoint_dir / f"round11_{a}_seed0.pt") for a in "ABC"]
    if args.checkpoint:
        models.append(("new_model", "ssm", args.checkpoint))
    for name, _, checkpoint in models:
        if checkpoint and not checkpoint.is_file():
            p.error(f"Missing checkpoint for {name}: {checkpoint}")
    args.output_dir.mkdir(parents=True)
    summaries = {}
    for name, kind, checkpoint in models:
        command = [sys.executable, str(Path(__file__).with_name("evaluate_pc_closure.py")),
                   "--data-root", str(args.data_root), "--baseline", kind, "--subset", "full",
                   "--split", "validation", "--horizon", str(args.horizon), "--limit", str(args.limit),
                   "--device", device, "--output-dir", str(args.output_dir / name)]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        print(f"Comparing {name} on {device}", flush=True)
        subprocess.run(command, check=True)
        summaries[name] = json.loads((args.output_dir / name / "summary.json").read_text())
    rows = []
    for name, summary in summaries.items():
        cases = summary["case_metrics"]
        complete = [r for r in cases if r.get("complete", r.get("complete_to_requested_horizon", False))]
        row = {"model": name, "horizon": args.horizon, "case_count": len(cases), "completed": len(complete),
               "wall_seconds": sum(r["wall_seconds"] for r in cases)}
        for key in ("field_log10_rmse", "perturbation_relative_l2", "electric_mode1_phase_mae"):
            values = [r["late"][key] for r in complete if r.get("late") and r["late"][key] is not None]
            row["late_" + key + "_median_completed_only"] = float(np.median(values)) if values else None
        rows.append(row)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    # Per-case differences avoid mistaking different survivor sets for a ranking.
    paired = []
    if "new_model" in summaries and args.horizon == 80.:
        new = {r["case_id"]: r for r in summaries["new_model"]["case_metrics"]}
        for name, summary in summaries.items():
            if name == "new_model":
                continue
            for old in summary["case_metrics"]:
                current = new[old["case_id"]]
                if old["complete"] and current["complete"]:
                    row = {"reference": name, "case_id": old["case_id"]}
                    for key in ("field_log10_rmse", "perturbation_relative_l2", "electric_mode1_phase_mae"):
                        a, b = old["late"][key], current["late"][key]
                        row[key + "_new_minus_reference"] = b - a if a is not None and b is not None else None
                    paired.append(row)
    (args.output_dir / "paired.json").write_text(json.dumps({"warning": "Common completed cases only; read completion counts alongside this table", "rows": paired}, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    representative = list(summaries.values())[0]["cases"][:3]
    fig, axes = plt.subplots(len(representative), 1, figsize=(9, 3 * len(representative)), squeeze=False)
    for ax, case_id in zip(axes[:, 0], representative):
        for i, name in enumerate(summaries):
            with np.load(args.output_dir / name / f"{case_id}.npz") as values:
                t = values["time"]
                if i == 0:
                    ax.semilogy(t, .5 * (values["truth"][:, 3].astype(np.float64)**2).mean(-1), "k--", label="kinetic reference")
                energy = .5 * (values["prediction"][:, 3].astype(np.float64)**2).mean(-1)
                ax.semilogy(t, np.where(energy > 0, energy, np.nan), label=name)
        ax.set_title(case_id); ax.set_xlabel("time"); ax.set_ylabel("field energy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.output_dir / "field_energy.png", dpi=150); plt.close(fig)
    print(f"Comparison ready: {args.output_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
