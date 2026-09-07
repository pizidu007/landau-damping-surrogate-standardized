"""Quality-gated acceleration, checkpoint freeze, independent holdout, and timing.

May wait for the sequential formal controller. Failed accuracy gates produce a
negative result and never trigger kinetic holdout generation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from summarize_round11_history import inspect_run,acceptance

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/"results/continuum_v1_closure_round11"


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with log.open("a",buffering=1) as stream:
        subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return time.perf_counter()-start


def train_candidate(name,config,arm,seed,device):
    output=RESULT/"acceleration"/name
    config_path=RESULT/"acceleration_configs"/(name+".json")
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise RuntimeError("Acceleration configuration changed; use a new candidate name")
    write(config_path,config)
    if not (output/"summary.json").exists():
        command([sys.executable,"-u","-m","landau_surrogate.training.continuum_history_closure",
                 "--config",str(config_path),"--output-dir",str(output),"--arm",arm,"--seed",str(seed),
                 "--device",device],output/"run.log")
    return inspect_run(output/"summary.json")


def no_material_degradation(candidate,baseline):
    violations=[]
    for key in ("field_log10_rmse","electric_mode1_phase_mae","perturbation_relative_l2"):
        before=baseline["late_medians_completed_only"][key]
        after=candidate["late_medians_completed_only"][key]
        if before is None or after is None or after>1.10*before+1e-4:
            violations.append(key)
    return {"passed":not violations,"violations":violations,"relative_tolerance":.10,"absolute_slack":1e-4}


def benchmark_before_freeze(candidates,device,gpu,output):
    """Benchmark accepted models on an existing validation case before any blind data."""
    import torch
    from landau_surrogate.data.continuum_history import load_history_cache,HistoryCache
    from landau_surrogate.data.continuum_v1 import load_continuum_case_index
    from landau_surrogate.fluid.history_closure_1d import advance_history
    from landau_surrogate.training.continuum_history_closure import construct
    from summarize_round11_history import interval_metrics

    identifier="K0p407_a0p105"
    generation=copy.deepcopy(json.loads((ROOT/"configs/gkeyll/continuum_round11_holdout.json").read_text()))
    profile=copy.deepcopy(generation["profiles"]["blind"])
    profile.update(purpose="Fresh same-GPU timing on an already-open validation case",
                   cases=[{"K":.407,"alpha":.105}])
    generation["profiles"]={"benchmark":profile}
    generation["dataset_root"]=f"/rydata/duxinxu/landau-damping-surrogate-standardized/continuum_round11_benchmark_gpu{gpu}"
    generation["description"]="Round 11 existing-validation-case timing; no independent holdout data"
    generation.pop("preregistration_note",None)
    path=output/"kinetic_config.json";write(path,generation)
    command([sys.executable,"-u","-m","landau_surrogate.tools.generate_gkeyll_continuum_v1",
             "--config",str(path),"--profile","benchmark","--gpu",str(gpu),"--wait-gpu-minutes","60"],output/"kinetic_generation.log")
    kinetic_path=Path(generation["dataset_root"])/"profiles/benchmark/cases"/identifier/"provenance/COMPLETE.json"
    kinetic=json.loads(kinetic_path.read_text())
    torch.cuda.set_per_process_memory_fraction(.15,torch.device(device));torch.set_num_threads(2)
    rows=[]
    for candidate in candidates:
        name=candidate["name"].replace("/","_")
        marker=output/(name+".json")
        if marker.exists():rows.append(json.loads(marker.read_text()));continue
        begin=time.perf_counter()
        checkpoint=torch.load(candidate["checkpoint"],map_location=device,weights_only=False)
        config=checkpoint["config"]
        model=construct(config,checkpoint["normalization"],device)
        model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
        torch.cuda.synchronize();load_seconds=time.perf_counter()-begin
        cases=[c for c in load_continuum_case_index(config["dataset_root"]) if c.case_id==identifier and c.split=="validation"]
        if len(cases)!=1:raise RuntimeError("Timing case must belong to validation")
        cache=load_history_cache(cases,config["maximum_mode"],torch.device(device),disk_cache=Path(config["cache_dir"]))
        initial=cache.state[:,:1].expand(-1,2,-1,-1).clone()
        deployment=HistoryCache(cases,initial,torch.zeros_like(initial[:,:,0]),.02,[])
        with np.load(candidate["rollout_file"]) as data:
            index=data["case_ids"].tolist().index(identifier)
            truth=data["truth"][:,index];times=data["time"]
        previous=next(row["late"] for row in candidate["rows"] if row["case_id"]==identifier)
        repeats=[]
        for repeat in range(3):
            bad=torch.zeros(1,dtype=torch.bool,device=device)
            def observe(_,state):
                bad.logical_or_(~torch.isfinite(state).all(dim=(1,2)) | (state[:,0].amin(-1)<=0) | (state[:,2].amin(-1)<=0))
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.no_grad():
                prediction=advance_history(model,deployment,torch.tensor([0],device=device),torch.tensor([0.],device=device),
                    horizon=80.,dt=config["dt"],use_checkpoint=False,step_observer=observe)
            torch.cuda.synchronize();compute=time.perf_counter()-begin
            array=prediction.cpu().numpy()[0]
            np.savez_compressed(output/f"{name}_repeat{repeat}.npz",time=times,state=array)
            wall=time.perf_counter()-begin
            actual=interval_metrics(array,truth,times>=30)
            matched=actual is not None and all(actual[key] is not None and previous[key] is not None and actual[key]<=1.1*previous[key]+1e-4
                for key in ("field_log10_rmse","electric_mode1_phase_mae","perturbation_relative_l2"))
            repeats.append({"integration_and_validity_checks_seconds":compute,"integration_transfer_write_seconds":wall,
                            "physical_complete":not bool(bad.item()),"batch1_accuracy_retained":matched,"late_metrics":actual})
        valid=all(row["physical_complete"] and row["batch1_accuracy_retained"] for row in repeats)
        row={"candidate":candidate["name"],"checkpoint":candidate["checkpoint"],"case_id":identifier,
             "model_load_seconds":load_seconds,"repeats":repeats,"eligible":valid,
             "median_integration_transfer_write_seconds":float(np.median([row["integration_transfer_write_seconds"] for row in repeats])),
             "kinetic_provenance":str(kinetic_path),"kinetic_solver_seconds":kinetic["gkeyll_stat"]["total_tm"],
             "scope":"Existing validation case, same physical A800 GPU, batch1, moment cadence0.1; shared-node timing. Model load and reference-data preparation are separated. No blind trajectories opened."}
        write(marker,row);rows.append(row)
        del model,checkpoint,cache,deployment,initial,prediction
        torch.cuda.empty_cache()
    valid=[row for row in rows if row["eligible"]]
    if not valid:raise RuntimeError("No candidate retained validation accuracy in the batch1 timing run")
    winner=min(valid,key=lambda row:row["median_integration_transfer_write_seconds"])
    write(output/"summary.json",{"candidates":rows,"selected":winner,"test_opened":False})
    return next(candidate for candidate in candidates if candidate["name"]==winner["candidate"])


def evaluate_holdout(frozen,config,root,device,output):
    import h5py
    import torch
    from landau_surrogate.data.continuum_history import load_history_cache,HistoryCache
    from landau_surrogate.data.continuum_v1 import ContinuumCase
    from landau_surrogate.fluid.history_closure_1d import advance_history
    from landau_surrogate.training.continuum_history_closure import construct,full_validation
    from landau_surrogate.tools.generate_gkeyll_continuum_v1 import case_id

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.15,torch.device(device))
    model_start=time.perf_counter()
    checkpoint=torch.load(frozen["checkpoint"],map_location=device,weights_only=False)
    model=construct(checkpoint["config"],checkpoint["normalization"],device)
    model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
    torch.cuda.synchronize()
    model_load_seconds=time.perf_counter()-model_start
    cases=[];qc=[]
    for item in config["profiles"]["blind"]["cases"]:
        identifier=case_id(item)
        path=root/"profiles/blind/cases"/identifier/"processed/trajectory.h5"
        with h5py.File(path) as h:
            central=h["diagnostics/central_moments"][:]
            moment=h["integrated/moments"][:]
            total=h["integrated/total_energy"][:]
            negative=h["kinetic/negative_sample_mass_fraction"][:]
            mass_drift=float(np.max(np.abs(moment[:,0]-moment[0,0]))/abs(moment[0,0]))
            energy_drift=float(np.max(np.abs(total-total[0]))/abs(total[0]))
            passed=bool(np.isfinite(central).all() and central[...,0].min()>0 and central[...,2].min()>0
                        and np.isfinite(total).all() and mass_drift<1e-5 and energy_drift<1e-3
                        and np.isfinite(negative).all() and negative.max()<1e-3)
        qc.append({"case_id":identifier,"passed":passed,"mass_relative_drift":mass_drift,
                   "energy_relative_drift":energy_drift,"endpoint_negative_sample_mass_fraction":float(negative.max())})
        cases.append(ContinuumCase(identifier,item["K"],item["alpha"],"independent_holdout",item["domain"],path))
    write(output/"reference_qc.json",{"cases":qc,"passed":all(row["passed"] for row in qc),
          "scope":"Moment positivity, integrated mass/energy, endpoint distribution validity; no new resolution-convergence claim."})
    if not all(row["passed"] for row in qc):raise RuntimeError("Independent reference QC failed; inspect preserved results")
    cache=load_history_cache(cases,checkpoint["config"]["maximum_mode"],torch.device(device),
                             disk_cache=RESULT/"holdout_cache",evaluation_only=True)
    evaluation=output/"full_validation"
    summary=json.loads((evaluation/"summary.json").read_text()) if (evaluation/"summary.json").exists() else full_validation(model,cache,checkpoint["config"],evaluation)
    complete={row["case_id"]:row["complete"] for row in summary["case_metrics"]}
    timing=[]
    # A deployment call sees only the initial state; future kinetic frames are not in this cache.
    for i,case in enumerate(cases):
        marker=output/"timing"/(case.case_id+".json")
        if marker.exists():timing.append(json.loads(marker.read_text()));continue
        initial=cache.state[i:i+1,:1].expand(-1,2,-1,-1).clone()
        initial_cache=HistoryCache([case],initial,torch.zeros_like(initial[:,:,0]),.02,[])
        ids=torch.tensor([0],device=device);starts=torch.tensor([0.],device=device)
        repeats=[]
        for repeat in range(3):
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.no_grad():
                prediction=advance_history(model,initial_cache,ids,starts,horizon=80.,dt=checkpoint["config"]["dt"],use_checkpoint=False)
            torch.cuda.synchronize();compute=time.perf_counter()-begin
            array=prediction.cpu().numpy()
            path=output/"timing"/f"{case.case_id}_repeat{repeat}.npz";path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,time=np.arange(801)*.1,state=array)
            repeats.append({"compute_seconds":compute,"compute_transfer_write_seconds":time.perf_counter()-begin})
        source=json.loads((case.path.parents[1]/"provenance/COMPLETE.json").read_text())
        row={"case_id":case.case_id,"physical_rollout_complete":complete[case.case_id],"repeats":repeats,
             "model_load_seconds":model_load_seconds,"gkeyll_stat":source["gkeyll_stat"],
             "GPU":source["gpu_before"],"batch_size":1,"output_dt":.1,
             "observed_compute_speed_ratio":source["gkeyll_stat"]["total_tm"]/float(np.median([r["compute_seconds"] for r in repeats])),
             "timing_scope":"Same A800 and moment cadence, shared node; solver statistic versus resident-model integration. Model load and transfer/write are separate. Not an exclusive-device or distribution-output equivalence claim."}
        write(marker,row);timing.append(row)
    write(output/"summary.json",{"frozen":frozen,"independent_holdout":summary,"timing":timing,"test_opened":True,
          "interpretation":"All holdout successes and failures are retained. No fitting or threshold changes after model freeze."})


def main():
    p=argparse.ArgumentParser();p.add_argument("--wait-for-formal",action="store_true")
    p.add_argument("--device",default="cuda:0");p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--processing-python",type=Path,default=Path("/wangx/home/duxinxu/software/micromamba-root-v1/envs/gkeyll-build/bin/python"))
    args=p.parse_args()
    directory=RESULT/"postquality";directory.mkdir(exist_ok=True)
    lock=(directory/"runner.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((directory/"status.json").read_text()) if (directory/"status.json").exists() else {}
    status={"pid":os.getpid(),"state":"waiting_for_formal","test_opened":previous.get("test_opened",False),"gpu":args.gpu}
    def save():status["updated"]=datetime.now(timezone.utc).isoformat();write(directory/"status.json",status)
    save()
    try:
        marker=RESULT/"reports/formal/summary.json"
        while not marker.exists():
            if not args.wait_for_formal:raise RuntimeError("Formal quality report is not ready")
            control=RESULT/"control/status.json"
            if control.exists() and json.loads(control.read_text()).get("state")=="failed":
                raise RuntimeError("Formal controller failed; continuation must wait for its repair")
            time.sleep(30)
        report=json.loads(marker.read_text())
        if len(report["runs"])!=9:raise RuntimeError("Formal report does not contain all nine matched runs")
        eligible=report["eligible_arms_for_acceleration"]
        if not eligible:
            status.update(state="complete_negative_accuracy_result",reason="No A/B/C arm passed the frozen quality gate across all three training seeds.",
                          next_research="Diagnose finite training horizon and closure structure before inferring that more evolved moments are necessary.")
            save();return
        best=min((run for run in report["runs"] if run["arm"] in eligible),key=lambda r:(r["late_medians_completed_only"]["field_log10_rmse"],r["late_medians_completed_only"]["electric_mode1_phase_mae"]))
        import torch
        parent=torch.load(best["checkpoint"],map_location="cpu",weights_only=False)
        base=copy.deepcopy(parent["config"])
        base.update(parent_checkpoint=best["checkpoint"],supervised_epochs=0,rollout_lr=5e-6,
                    curriculum=[{"horizon":.2,"epochs":1},{"horizon":1.,"epochs":1},{"horizon":2.,"epochs":2}])
        quality=report["quality_configuration"]
        candidates=[];accepted=[(0.,best)]
        for interval in (.02,.04,.1):
            status.update(state="acceleration_training",current_interval=interval);save()
            config={**base,"closure_interval":interval}
            candidate=train_candidate(f"cadence{interval:g}",config,best["arm"],best["seed"],args.device)
            candidate["absolute_gate"]=acceptance(candidate,quality)
            candidate["relative_gate"]=no_material_degradation(candidate,best)
            candidates.append(candidate)
            if candidate["absolute_gate"]["passed"] and candidate["relative_gate"]["passed"]:accepted.append((interval,candidate))
            write(directory/"acceleration_candidates.json",{"baseline":best,"candidates":candidates})
        cadence,winner=max(accepted,key=lambda item:item[0])
        compression_parent=torch.load(winner["checkpoint"],map_location="cpu",weights_only=False)
        config=copy.deepcopy(compression_parent["config"])
        config.update(parent_checkpoint=winner["checkpoint"],width=64,supervised_epochs=8,
                      rollout_lr=5e-6,curriculum=base["curriculum"],
                      compression_initialization="prefix-channel pruning followed by supervised retraining and full-gradient rollout")
        status.update(state="width64_compression_training");save()
        compressed=train_candidate("width64",config,best["arm"],best["seed"],args.device)
        compressed["absolute_gate"]=acceptance(compressed,quality)
        compressed["relative_gate"]=no_material_degradation(compressed,best)
        candidates.append(compressed)
        if compressed["absolute_gate"]["passed"] and compressed["relative_gate"]["passed"]:accepted.append((cadence,compressed))
        status.update(state="validation_timing_before_freeze");save()
        winner=benchmark_before_freeze([candidate for _,candidate in accepted],args.device,args.gpu,directory/"validation_timing")
        write(directory/"acceleration_candidates.json",{"baseline":best,"candidates":candidates,"selected":winner})
        holdout_config=ROOT/"configs/gkeyll/continuum_round11_holdout.json"
        frozen={"checkpoint":winner["checkpoint"],"checkpoint_sha256":sha(winner["checkpoint"]),
                "processing_python":str(args.processing_python),
                "quality_configuration":quality,"holdout_config":str(holdout_config),"holdout_config_sha256":sha(holdout_config),
                "evaluation_source_sha256":{str(path.relative_to(ROOT)):sha(path) for path in
                    [*sorted((ROOT/"scripts").glob("*round11*.py")),
                     *sorted((ROOT/"src/landau_surrogate").rglob("*.py"))]},
                "selection_split":"validation","frozen_at":datetime.now(timezone.utc).isoformat(),"test_opened_at_freeze":False}
        freeze_path=directory/"FROZEN.json"
        if freeze_path.exists():
            prior=json.loads(freeze_path.read_text())
            for key in ("checkpoint_sha256","holdout_config_sha256","quality_configuration","evaluation_source_sha256"):
                if prior[key]!=frozen[key]:raise RuntimeError("Frozen model/protocol changed; do not reopen this holdout for tuning")
            frozen=prior
        else:
            for relative in frozen["evaluation_source_sha256"]:
                snapshot=directory/"frozen_source"/relative;snapshot.parent.mkdir(parents=True,exist_ok=True)
                snapshot.write_bytes((ROOT/relative).read_bytes())
            write(freeze_path,frozen)
        config=json.loads(holdout_config.read_text());root=Path(config["dataset_root"])
        status.update(state="independent_holdout_generation",frozen_manifest=str(freeze_path));save()
        command([sys.executable,"-u","-m","landau_surrogate.tools.generate_gkeyll_continuum_v1",
                 "--config",str(holdout_config),"--profile","blind","--gpu",str(args.gpu),"--wait-gpu-minutes","60"],directory/"generation.log")
        status.update(state="independent_holdout_processing",test_opened=True);save()
        command([str(args.processing_python),"-u","-m","landau_surrogate.tools.process_gkeyll_continuum_v1","--dataset-root",str(root),"--profile","blind"],directory/"processing.log")
        status.update(state="independent_holdout_evaluation_and_timing");save()
        evaluate_holdout(frozen,config,root,args.device,directory/"holdout")
        status.update(state="complete",summary=str(directory/"holdout/summary.json"));save()
    except BaseException as error:
        status.update(state="failed",error=repr(error));save();raise


if __name__=="__main__":main()
