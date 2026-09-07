"""Completion-aware, paired-case reporting for the Round 11 closure experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/continuum_v1_closure_round11"


def median(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def interval_metrics(pred, truth, mask):
    p, t = pred[mask].astype(np.float64), truth[mask].astype(np.float64)
    if not len(p) or not np.isfinite(p).all():
        return None
    ep, et = .5 * np.mean(p[:, 3]**2, axis=-1), .5 * np.mean(t[:, 3]**2, axis=-1)
    floor = max(float(.5*np.mean(truth[0, 3]**2))*1e-8, 1e-30)
    phat, that = np.fft.rfft(p[:, 3])[:, 1], np.fft.rfft(t[:, 3])[:, 1]
    reference_amplitude = np.max(np.abs(np.fft.rfft(truth[:, 3])[:, 1]))
    signal = np.abs(that) > max(reference_amplitude*1e-4, 1e-12)
    # A vanishing prediction has no meaningful phase; count it explicitly.
    predicted_signal = np.abs(phat) > max(reference_amplitude*1e-6, 1e-14)
    phase = np.abs(np.angle(phat * np.conj(that)))
    phase = np.where(predicted_signal, phase, np.pi)
    equilibrium = np.array([1., 0., 1., 0.])[None, :, None]
    return {
        "field_log10_rmse": float(np.sqrt(np.mean((np.log10(np.maximum(ep, floor)) - np.log10(np.maximum(et, floor)))**2))),
        "electric_mode1_phase_mae": float(phase[signal].mean()) if signal.any() else None,
        "phase_valid_frames": int(signal.sum()),
        "predicted_phase_missing_frames": int((signal & ~predicted_signal).sum()),
        "perturbation_relative_l2": float(np.linalg.norm(p-t)/max(np.linalg.norm(t-equilibrium), 1e-12)),
        "electric_relative_l2": float(np.linalg.norm(p[:,3]-t[:,3])/max(np.linalg.norm(t[:,3]),1e-12)),
    }


def inspect_run(path):
    summary = json.loads(path.read_text())
    selected = Path(summary["selected_checkpoint"])
    # The selected checkpoint records the candidate, but the JSON has the same key.
    candidates = summary["candidate_evaluations"]
    chosen = min(candidates, key=lambda row: row["selection_key"])
    npz = path.parent / "candidates" / Path(chosen["checkpoint"]).stem / "rollout.npz"
    with np.load(npz) as data:
        times, prediction, truth = data["time"], data["prediction"], data["truth"]
        ids = data["case_ids"].tolist()
    metadata = {r["case_id"]: r for r in summary["full_validation"]["case_metrics"]}
    rows = []
    for i, case_id in enumerate(ids):
        row = dict(metadata[case_id])
        for name, lo, hi in (("early",0.,30.), ("late",30.,80.), ("full",0.,80.)):
            row[name] = interval_metrics(prediction[:,i],truth[:,i],(times>=lo)&(times<=hi)) if row["complete"] else None
        rows.append(row)
    completed = [row for row in rows if row["complete"]]
    late = {key: median([row["late"][key] for row in completed])
            for key in ("field_log10_rmse","electric_mode1_phase_mae","perturbation_relative_l2","electric_relative_l2")}
    regimes = {}
    for regime in sorted({row["regime"] for row in rows}):
        group = [row for row in rows if row["regime"] == regime]
        regimes[regime] = {"count":len(group), "complete":sum(row["complete"] for row in group),
                           "late_field_log10_rmse_median":median([row["late"]["field_log10_rmse"] for row in group if row["complete"]])}
    result = {"name":str(path.parent.relative_to(RESULT)), "arm":summary["arm"], "seed":summary["seed"],
              "history_span":summary["config"]["history_span"], "checkpoint":str(selected),
              "case_count":len(rows), "complete":len(completed), "late_medians_completed_only":late,
              "regimes":regimes, "rows":rows, "rollout_file":str(npz)}
    return result


def acceptance(run, config):
    reasons = []
    if run["complete"] / run["case_count"] < config["required_completed_fraction"]:
        reasons.append("incomplete_or_nonpositive_trajectory")
    for key, threshold in (("total_energy_max_relative_drift","maximum_total_energy_relative_drift"),
                           ("mass_max_absolute_drift","maximum_mass_absolute_drift"),
                           ("momentum_max_absolute_drift","maximum_momentum_absolute_drift")):
        if any(row.get(key,float("inf")) > config[threshold] for row in run["rows"]):
            reasons.append(key)
    for key, threshold in (("field_log10_rmse","maximum_median_late_field_log10_rmse"),
                           ("electric_mode1_phase_mae","maximum_median_late_electric_mode1_phase_mae"),
                           ("perturbation_relative_l2","maximum_median_late_perturbation_relative_l2")):
        value = run["late_medians_completed_only"][key]
        if value is None or value > config[threshold]:
            reasons.append("late_"+key)
    if any(group["late_field_log10_rmse_median"] is None or group["late_field_log10_rmse_median"] > config["maximum_regime_median_late_field_log10_rmse"] for group in run["regimes"].values()):
        reasons.append("regime_late_field_error")
    return {"passed":not reasons, "reasons":reasons}


def paired(a, b):
    ar, br = ({row["case_id"]:row for row in run["rows"]} for run in (a,b))
    common = sorted(case for case in ar.keys() & br.keys() if ar[case]["complete"] and br[case]["complete"])
    result = {"A":a["name"], "B":b["name"], "case_count":len(common), "case_ids":common,
              "warning":"Paired complete cases only; completion rates must be compared separately."}
    for key in ("field_log10_rmse","electric_mode1_phase_mae","perturbation_relative_l2"):
        av, bv = ([rows[case]["late"][key] for case in common] for rows in (ar,br))
        result[key] = {"A_median":median(av), "B_median":median(bv),
                       "median_paired_difference_B_minus_A":median([y-x for x,y in zip(av,bv) if x is not None and y is not None])}
    return result


def figures(runs, output):
    if not runs:
        return
    # Deterministic representative cases from the validation set, independent of error.
    rows = sorted(runs[0]["rows"],key=lambda row:(row["regime"],row["K"],row["alpha"]))
    selected = []
    for regime in sorted({r["regime"] for r in rows}):
        group = [r for r in rows if r["regime"] == regime]
        selected.extend([group[0]["case_id"], group[len(group)//2]["case_id"], group[-1]["case_id"]])
    fig, axes = plt.subplots(3,3,figsize=(13,9),sharex=True)
    for j,run in enumerate(runs):
        with np.load(run["rollout_file"]) as data:
            ids=data["case_ids"].tolist()
            for case,ax in zip(selected,axes.flat):
                i=ids.index(case)
                if j==0:
                    ax.semilogy(data["time"],np.maximum(.5*np.mean(data["truth"][:,i,3]**2,axis=-1),1e-16),"k--",label="Kinetic reference",lw=1.5)
                ax.semilogy(data["time"],np.maximum(.5*np.mean(data["prediction"][:,i,3]**2,axis=-1),1e-16),label=run["name"].replace("pilot/", ""),lw=1,alpha=.9)
                ax.set_title(case,fontsize=10);ax.grid(alpha=.2)
    for ax in axes[-1]: ax.set_xlabel("Time")
    for ax in axes[:,0]: ax.set_ylabel("Electric field energy")
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Round 11 validation: full free rollout; failed trajectories stop at first violation")
    fig.tight_layout()
    for ext in ("png","pdf","svg"):fig.savefig(output/f"field_energy.{ext}",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["pilot","formal"],default="pilot")
    args=parser.parse_args()
    output=RESULT/"reports"/args.stage;output.mkdir(parents=True,exist_ok=True)
    gate=json.loads((ROOT/"configs/training/continuum_history_closure_round11_quality.json").read_text())
    paths=sorted((RESULT/args.stage).glob("*/summary.json" if args.stage=="pilot" else "*/*/summary.json"))
    runs=[inspect_run(path) for path in paths]
    for run in runs:run["quality_gate"]=acceptance(run,gate)
    comparisons=[paired(a,b) for a in runs if a["arm"]=="A" for b in runs if b["arm"]!="A" and a["seed"]==b["seed"]]
    comparisons += [paired(a,b) for a in runs if a["arm"]=="B" for b in runs if b["arm"]=="C" and a["seed"]==b["seed"] and a["history_span"]==b["history_span"]]
    eligible=[]
    if args.stage=="formal":
        for arm in ("A","B","C"):
            group=[r for r in runs if r["arm"]==arm]
            if sorted(r["seed"] for r in group)==gate["required_formal_seeds"] and all(r["quality_gate"]["passed"] for r in group):eligible.append(arm)
    report={"stage":args.stage,"test_used":False,"quality_configuration":gate,"runs":runs,"paired_comparisons":comparisons,"eligible_arms_for_acceleration":eligible}
    (output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines=[f"# Round 11 {args.stage}","", "All error medians below use completed trajectories only. Assess completion first and use the paired tables in summary.json for model comparisons.", "",
           "| Run | Complete | Late log field RMSE | Late phase MAE (rad) | Late perturbation relative L2 | Quality gate |", "|---|---:|---:|---:|---:|---|"]
    def number(v):return "—" if v is None else f"{v:.4g}"
    for run in runs:
        m=run["late_medians_completed_only"]
        lines.append(f"| {run['name']} | {run['complete']}/{run['case_count']} | {number(m['field_log10_rmse'])} | {number(m['electric_mode1_phase_mae'])} | {number(m['perturbation_relative_l2'])} | {'PASS' if run['quality_gate']['passed'] else 'FAIL'} |")
    lines += ["", "Engineering tolerances were fixed before B/C pilot and formal results. Passing this validation gate is not a claim of unseen-parameter generalization.", "", f"Eligible formal arms for acceleration: {', '.join(eligible) or 'none yet'}."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    # Plot only seed zero in a combined figure to keep traces readable.
    figures([r for r in runs if r["seed"]==0],output)
    print(json.dumps({"runs":len(runs),"eligible_arms_for_acceleration":eligible,"report":str(output/"REPORT.md")}))


if __name__=="__main__":main()
