"""Resumable sequential oracle-gated pilot and three-seed closure experiments."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/"results/continuum_v1_closure_round11"


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def quality(summary):
    r=summary["full_validation"]
    return (r["case_count"]-r["complete"],
            r["full_t80_field_energy_log10_rmse_median"] if r["complete"] else 1e9,
            r["electric_mode1_phase_mae_median"] if r["complete"] else 1e9,
            r["perturbation_relative_l2_median"] if r["complete"] else 1e9)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--pilot-only",action="store_true")
    parser.add_argument("--formal-overrides",type=Path)
    args=parser.parse_args()
    control=RESULT/"control";control.mkdir(parents=True,exist_ok=True)
    lock=(control/"runner.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    status={"workflow":"round11_history_closure","pid":os.getpid(),"state":"starting",
            "device":args.device,"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started":datetime.now(timezone.utc).isoformat(),"tasks":{},"test_used":False}
    def save():
        status["updated"]=datetime.now(timezone.utc).isoformat();write(control/"status.json",status)
    save()
    gate=json.loads((RESULT/"oracle_gate.json").read_text())
    if not gate["passed"]: raise RuntimeError("Oracle gate has not passed")
    pilot=ROOT/"configs/training/continuum_history_closure_round11_pilot.json"
    config=json.loads(pilot.read_text())
    if (config["dt"],config["maximum_mode"])!=(gate["selected_dt"],gate["selected_maximum_mode"]):
        raise RuntimeError("Training configuration differs from numerical gate")
    def task(name,arm,span,seed,config_path):
        output=RESULT/name
        marker=output/"summary.json"
        if marker.exists():
            result=json.loads(marker.read_text())
            status["tasks"][name]={"state":"complete","summary":str(marker),"selection_key":quality(result)}
            save();return result
        output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"-u","-m","landau_surrogate.training.continuum_history_closure",
                 "--config",str(config_path),"--output-dir",str(output),"--arm",arm,
                 "--seed",str(seed),"--device",args.device]
        if span is not None: command += ["--history-span",str(span)]
        status.update(state="running",current_task=name)
        status["tasks"][name]={"state":"running","command":command,"log":str(output/"run.log")}
        save();print(json.dumps({"starting":name}),flush=True)
        beginning=time.monotonic()
        with (output/"run.log").open("a",buffering=1) as log:
            completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status["tasks"][name].update(returncode=completed.returncode,wall_seconds=time.monotonic()-beginning)
        if completed.returncode:
            status["tasks"][name]["state"]="failed";status["state"]="failed";save()
            raise RuntimeError(f"{name} failed; inspect its log before resuming")
        result=json.loads(marker.read_text())
        status["tasks"][name].update(state="complete",selection_key=quality(result),summary=str(marker))
        save();return result
    try:
        a=task("pilot/A_seed0","A",None,0,pilot)
        bs=[]
        for span in (.5,2.,5.):
            result=task(f"pilot/B_span{span:g}_seed0","B",span,0,pilot)
            bs.append((quality(result),span,result))
        selected=min(bs,key=lambda row:row[0])
        span=selected[1]
        c=task(f"pilot/C_span{span:g}_seed0","C",span,0,pilot)
        selection={"selected_history_span":span,"selection_split":"validation","A":quality(a),
                   "B_candidates":[{"span":s,"key":key} for key,s,_ in bs],"C":quality(c),"test_used":False}
        write(RESULT/"pilot_selection.json",selection)
        if args.pilot_only:
            status.update(state="pilot_complete",next_stage="formal_three_seed");save();return
        formal=dict(config)
        formal.update(history_span=span,supervised_epochs=20,samples_per_case=512,
                      curriculum=[{"horizon":.2,"epochs":3},{"horizon":1.,"epochs":4},{"horizon":2.,"epochs":4}],
                      protocol="Frozen after validation-only pilot; matched A/B/C three seeds")
        if args.formal_overrides:
            overrides=json.loads(args.formal_overrides.read_text())
            formal.update(overrides["training_overrides"])
            formal["override_provenance"]=str(args.formal_overrides.resolve())
            audit=Path(overrides["numerical_gate"])
            numerical=json.loads(audit.read_text())
            if not numerical["passed"] or (formal["dt"],formal["maximum_mode"])!=(numerical["selected_dt"],numerical["selected_maximum_mode"]):
                raise RuntimeError("Formal overrides have no matching successful oracle gate")
        formal_path=RESULT/"formal_config.json";write(formal_path,formal)
        summaries={}
        for arm in ("A","B","C"):
            summaries[arm]=[]
            for seed in (0,1,2):
                result=task(f"formal/{arm}/seed{seed}",arm,None if arm=="A" else span,seed,formal_path)
                summaries[arm].append(result)
            write(RESULT/"formal_summary.json",{"arms":summaries,"test_used":False})
        status.update(state="formal_complete",next_stage="quality_review_then_acceleration_and_frozen_holdout")
        subprocess.run([sys.executable,str(ROOT/"scripts/summarize_round11_history.py"),"--stage","formal"],cwd=ROOT,check=True)
        report=json.loads((RESULT/"reports/formal/summary.json").read_text())
        status["eligible_arms_for_acceleration"]=report["eligible_arms_for_acceleration"]
        if not report["eligible_arms_for_acceleration"]:
            status.update(state="formal_quality_gate_failed",next_stage="diagnose_remaining_three_moment_error_before_acceleration")
        save()
    except BaseException as error:
        status.update(state="failed",error=repr(error));save();raise


if __name__=="__main__":main()
