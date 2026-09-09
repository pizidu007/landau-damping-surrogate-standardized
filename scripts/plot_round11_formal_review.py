"""Reproducible scientific figures from the nine frozen Round 11 results."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/continuum_v1_closure_round11/reports/formal"
OUT = REPORT / "visual_review"
COLORS = {"A": "#3978b6", "B": "#169b88", "C": "#db7943"}
LABELS = {"A": "A 当前状态 + K", "B": "B 历史状态 + K", "C": "C 历史状态 + K + α"}
REGIMES = {"weak": "弱扰动", "transition": "过渡", "strong_nonlinear": "强非线性"}
METRICS = [("field_log10_rmse", "场能 log10 RMSE", .3),
           ("electric_mode1_phase_mae", "电场一阶模态相位 MAE / rad", .35),
           ("perturbation_relative_l2", "扰动状态相对 L2", .25)]


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def overview(runs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    for panel, ax in enumerate(axes.flat):
        for ai, arm in enumerate("ABC"):
            for seed in range(3):
                r = runs[arm, seed]
                x = ai + (seed-1)*.23
                y = r["complete"] if panel == 0 else r["late_medians_completed_only"][METRICS[panel-1][0]]
                ax.bar(x, y, width=.20, color=COLORS[arm], alpha=(.55, .78, 1.)[seed])
                ax.annotate(str(y) if panel == 0 else f"{y:.2f}", (x,y), xytext=(0,5),
                            textcoords="offset points", ha="center", fontsize=9)
        threshold = 20 if panel == 0 else METRICS[panel-1][2]
        ax.axhline(threshold, color="#aa3653", ls="--", lw=1.3)
        ax.set_xticks(range(3), ["A 当前状态", "B + 历史", "C + 历史 + α"])
        ax.set_title("完成到 t=80 的轨迹数（每组 20 条）" if panel == 0 else METRICS[panel-1][1], loc="left", fontsize=12)
        ax.set_ylim(0, 23 if panel == 0 else ax.get_ylim()[1]*1.17)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        ax.text(.98,.95,f"目标 {'=' if panel == 0 else '≤'} {threshold:g}",transform=ax.transAxes,
                ha="right",va="top",color="#aa3653",fontsize=10)
    fig.suptitle("Round 11｜九组实验完成，但没有一组通过精度门", fontsize=18, y=.98)
    fig.text(.5,.927,"各模型内从左至右：seed 0 / 1 / 2；颜色由浅到深",ha="center",fontsize=11,color="#555555")
    fig.text(.06,.025,"误差统计区间 t=30–80；仅对各运行完成的轨迹取中位数，幸存案例集合不同。\n误差柱用于检查绝对精度；模型优劣请结合配对图。虚线为预设门限。",fontsize=10,color="#555555")
    fig.tight_layout(rect=(.015,.095,.995,.91))
    save(fig,"01_overview")


def paired(runs, report):
    fig, axes = plt.subplots(2,3,figsize=(14,8.3))
    for row,(a,b) in enumerate((("A","B"),("B","C"))):
        for col,(metric,title,_) in enumerate(METRICS):
            ax=axes[row,col]
            ax.axhline(0,color="#888888",lw=1,ls="--")
            for seed in range(3):
                left,right=({r["case_id"]:r for r in runs[arm,seed]["rows"]} for arm in (a,b))
                ids=sorted(c for c in left if left[c]["complete"] and right[c]["complete"])
                delta=np.array([right[c]["late"][metric]-left[c]["late"][metric] for c in ids])
                expected=next(p for p in report["paired_comparisons"] if p["A"]==runs[a,seed]["name"] and p["B"]==runs[b,seed]["name"])
                assert np.isclose(np.median(delta),expected[metric]["median_paired_difference_B_minus_A"])
                offsets=np.linspace(-.12,.12,len(ids))
                ax.scatter(seed+offsets,delta,s=24,color=COLORS[b],alpha=.6,edgecolors="none")
                med=float(np.median(delta)); ax.plot(seed,med,"_",ms=24,mew=3,color="#182735")
                ax.annotate(f"{med:+.3f}",(seed,med),xytext=(10,6),textcoords="offset points",fontsize=9,
                            bbox=dict(facecolor="white",alpha=.8,edgecolor="none",pad=1))
                ax.text(seed,.025,f"n={len(ids)}",transform=ax.get_xaxis_transform(),ha="center",fontsize=9)
            ax.set_xticks(range(3),["seed 0","seed 1","seed 2"])
            ax.set_xlim(-.45,2.55);ax.margins(y=.18);ax.grid(axis="y",alpha=.15)
            ax.set_title(title,fontsize=12)
            if col==0: ax.set_ylabel(f"{b} − {a}\n晚期误差差值",fontsize=12)
    fig.suptitle("配对比较｜负值表示后一个模型在同一案例上误差更小",fontsize=17,y=.99)
    fig.text(.06,.035,"每个点：同种子下两模型均完成的一个验证案例；深色横线及数字：逐案例误差差值的中位数。\n上排检验历史输入（B−A），下排检验显式 α（C−B）；失败案例不进入本图，请同时查看完成数。三个种子共享同一验证集。",fontsize=10,color="#555555")
    fig.tight_layout(rect=(.01,.11,.995,.95))
    save(fig,"02_paired_differences")


def load_fields(runs):
    data={}
    for key,run in runs.items():
        with np.load(run["rollout_file"]) as d:
            p=d["prediction"][:,:,3].astype(np.float64)
            truth=d["truth"][:,:,3].astype(np.float64)
            data[key]={"t":d["time"],"ids":d["case_ids"].tolist(),
                       "energy":.5*np.mean(p*p,axis=-1),"reference":.5*np.mean(truth*truth,axis=-1)}
    return data


def curve_panels(cases,data,runs):
    fig,axes=plt.subplots(len(cases),3,figsize=(14,3.25*len(cases)+1.25),squeeze=False,sharex=True)
    for row,case in enumerate(cases):
        cid=case["case_id"]; all_values=[]
        for seed in range(3):
            ax=axes[row,seed]; base=data["A",seed]; index=base["ids"].index(cid)
            truth=base["reference"][:,index]
            # Display raw positive energy; no metric floor or smoothing on curves.
            all_values.extend(truth[np.isfinite(truth)&(truth>0)])
            for arm in "ABC":
                d=data[arm,seed]; i=d["ids"].index(cid); y=d["energy"][:,i]
                assert np.allclose(truth,d["reference"][:,i],rtol=1e-5,atol=1e-20)
                ax.semilogy(d["t"],np.where(y>0,y,np.nan),color=COLORS[arm],lw=1.45)
                all_values.extend(y[np.isfinite(y)&(y>0)])
                meta=next(v for v in runs[arm,seed]["rows"] if v["case_id"]==cid)
                if not meta["complete"]:
                    valid=np.flatnonzero(np.isfinite(y)&(y>0))
                    if len(valid): ax.plot(d["t"][valid[-1]],y[valid[-1]],"x",color=COLORS[arm],ms=8,mew=2)
            ax.semilogy(base["t"],truth,"k--",lw=1.7)
            ax.axvspan(30,80,color="#52657c",alpha=.055)
            ax.set_title(f"{REGIMES[case['regime']]}  K={case['K']:g}, α={case['alpha']:g}  |  seed {seed}",fontsize=10)
            ax.set_xlim(0,80);ax.grid(alpha=.15,which="major")
            if seed==0: ax.set_ylabel("场能 〈E²〉 / 2")
            if row==len(cases)-1:ax.set_xlabel("时间 t")
        lower=max(min(all_values)*.5,1e-30); upper=max(all_values)*2
        for ax in axes[row]:ax.set_ylim(lower,upper)
    handles=[Line2D([],[],color="black",ls="--",label="动力学参考解")]
    handles += [Line2D([],[],color=COLORS[a],label=LABELS[a]) for a in "ABC"]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.954),ncol=4,frameon=False,fontsize=10)
    fig.suptitle("完整自由推进｜相同案例、相同纵轴，对照三个随机种子",fontsize=17,y=.993)
    fig.text(.055,.012,"浅灰区：t=30–80 晚期统计区间；×：失效前最后有效输出，失效后不补线。纵轴为对数，未平滑、未裁切低场能。",fontsize=10,color="#555555")
    fig.tight_layout(rect=(.01,.045,.995,.91 if len(cases)>1 else .82))
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"WenQuanYi Micro Hei","axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False,"font.size":11,
                         "pdf.fonttype":42,"svg.fonttype":"none"})
    source=REPORT/"summary.json"; report=json.loads(source.read_text())
    runs={(r["arm"],r["seed"]):r for r in report["runs"]}
    assert set(runs)=={(a,s) for a in "ABC" for s in range(3)}
    overview(runs); paired(runs,report)
    cases=[]
    all_cases=[]
    for regime in REGIMES:
        group=sorted([r for r in runs["A",0]["rows"] if r["regime"]==regime],key=lambda r:(r["K"],r["alpha"]))
        cases.append(group[len(group)//2]);all_cases.extend(group)
    data=load_fields(runs)
    save(curve_panels(cases,data,runs),"03_representative_field_energy")
    with PdfPages(OUT/"04_all_20_cases.pdf") as pdf:
        for case in all_cases:
            fig=curve_panels([case],data,runs);pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
    provenance={"report_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "representative_selection":"Within each regime sort by (K, alpha), choose index len(group)//2; independent of model error.",
                "representative_case_ids":[c["case_id"] for c in cases],"late_interval":[30,80],
                "paired_medians_verified_against_report":True,"test_used":False,
                "script":str(Path(__file__).resolve()),"run_count":len(runs)}
    (OUT/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(OUT),**provenance},ensure_ascii=False))


if __name__=="__main__":main()
