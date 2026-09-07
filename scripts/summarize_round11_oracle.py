"""Produce a reviewable numerical gate and oracle comparison figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/"results/continuum_v1_closure_round11"
paths=list((RESULT/"oracle").glob("mode*/summary.json"))+list((RESULT/"oracle_dt").glob("mode*/summary.json"))+list((RESULT/"oracle_fp64").glob("mode*/summary.json"))
rows=[]
for path in sorted(paths):
    value=json.loads(path.read_text())
    metrics=value["case_metrics"]
    rows.append({"path":str(path),"mode":value["mode"],"dt":value["dt"],
                 "dtype":value.get("dtype","torch.float32"),"complete":value["complete"],
                 "median_log_energy":float(np.median([r["full_t80_field_energy_log10_rmse"] for r in metrics])),
                 "max_log_energy":max(r["full_t80_field_energy_log10_rmse"] for r in metrics),
                 "max_perturbation_relative_l2":max(r["perturbation_relative_l2"] for r in metrics),
                 "max_energy_drift":max(r["total_energy_max_relative_drift"] for r in metrics),
                 "max_gradient_tail_fraction":max(r["filtered_truth_gradient_tail_fraction"] for r in metrics)})
selected=next(r for r in rows if "oracle_dt/" in r["path"] and r["dt"]==.02)
limits={"max_log_energy":.05,"max_perturbation_relative_l2":.005,"max_energy_drift":.001}
passed=selected["complete"]==9 and all(selected[k]<=v for k,v in limits.items())
report={"stage":"round11_oracle_gate","passed":passed,"test_used":False,
        "selected_maximum_mode":16,"selected_dt":.02,"limits":limits,"runs":rows,
        "interpretation":"Float64 runs nearly agree at dt 0.002/0.02. Much larger float32 error at dt 0.002 is consistent with accumulated roundoff. Oracle validity does not establish stability of a learned closure at dt 0.02.",
        "scope":"9 preselected train/validation cases, full t=0..80, no clipping"}
(RESULT/"oracle_gate.json").write_text(json.dumps(report,indent=2)+"\n")
figure_dir=RESULT/"figures";figure_dir.mkdir(exist_ok=True)
with np.load(RESULT/"oracle_dt/mode16_dt0.02/rollout.npz") as data:
    times=data["time"];truth=data["truth"];prediction=data["prediction"];ids=data["case_ids"]
fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True)
for i,ax in enumerate(axes.flat):
    true_energy=.5*np.mean(truth[:,i,3]**2,axis=-1)
    pred_energy=.5*np.mean(prediction[:,i,3]**2,axis=-1)
    ax.semilogy(times,np.maximum(true_energy/true_energy[0],1e-10),color="black",lw=1.4,label="Kinetic moments")
    ax.semilogy(times,np.maximum(pred_energy/true_energy[0],1e-10),color="#0072B2",lw=.9,ls="--",label="Fluid + true heat flux")
    ax.set_title(str(ids[i]));ax.grid(alpha=.2);ax.set_xlabel("Time (plasma units)");ax.set_ylabel("Electric energy / initial")
axes[0,0].legend(fontsize=8)
fig.suptitle("Round 11 oracle: mode 16, dt=0.02, no clipping; train/validation only")
fig.tight_layout()
for suffix in ("png","pdf","svg"):
    fig.savefig(figure_dir/f"oracle_field_energy.{suffix}",dpi=180)
plt.close(fig)
print(json.dumps(report,indent=2))
