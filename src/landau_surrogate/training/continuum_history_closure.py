"""Round 11 supervised initialization and full-gradient history closure training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from landau_surrogate.data.continuum_history import load_history_cache, normalization, supervised_history
from landau_surrogate.data.continuum_v1 import load_continuum_case, load_continuum_case_index
from landau_surrogate.fluid.history_closure_1d import advance_history
from landau_surrogate.models.history_closure import HistoryClosureFNO
from landau_surrogate.tools.audit_continuum_v1_closure_oracle import atomic_json, metrics, interpolate, electric_numpy


def save_checkpoint(path, model, config, norm, **extra):
    payload = {"stage": "continuum_history_closure_round11", "config": config, "normalization": norm,
               "model_state_dict": model.state_dict(), "source_bundle":getattr(model,"source_bundle",None), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def construct(config, norm, device):
    model = HistoryClosureFNO(maximum_mode=config["maximum_mode"], width=config["width"],
                             layers=config["layers"], history_span=config["history_span"],
                             include_alpha=config["include_alpha"], normalization=norm,
                             enforce_reflection=config.get("enforce_reflection",False)).to(device)
    model.closure_interval=float(config.get("closure_interval",0.))
    return model


def initialize_history_parent(model, parent):
    """Exact transfer, or explicit prefix-channel pruning before supervised retraining."""
    target=model.state_dict();source=parent["model_state_dict"]
    for name,value in target.items():
        old=source[name]
        if old.shape==value.shape:
            value.copy_(old)
        elif name.startswith("network.") and old.ndim==value.ndim and all(a>=b for a,b in zip(old.shape,value.shape)):
            value.copy_(old[tuple(slice(0,size) for size in value.shape)])
        else:
            raise ValueError(f"Unsupported history-parent transfer: {name} {old.shape} -> {value.shape}")
    model.load_state_dict(target)


def parameters(cache, indices):
    selected = [cache.cases[int(i)] for i in indices.cpu()]
    return (torch.tensor([c.K for c in selected], device=cache.state.device),
            torch.tensor([c.alpha for c in selected], device=cache.state.device))


def closure_loss(model, cache, indices, starts):
    history, valid = supervised_history(cache, indices, starts, model.history_span)
    k, alpha = parameters(cache, indices)
    prediction = model(history, valid, k, alpha)
    target = cache.sample(indices, starts, "gradient")
    return ((prediction-target)/model.gradient_std).square().mean()


def trajectory_losses(prediction, target, k, alpha):
    amplitude = alpha.clamp_min(.005)
    channel_scale = torch.stack((amplitude, amplitude/k, amplitude, amplitude/k), dim=-1)
    error = (prediction-target)/channel_scale[:, None, :, None]
    state = error.square().mean()
    phat = torch.fft.rfft(prediction[:, :, 3], dim=-1, norm="forward")[:, :, 1:5]
    that = torch.fft.rfft(target[:, :, 3], dim=-1, norm="forward")[:, :, 1:5]
    complex_field = ((phat-that)/(amplitude/k)[:,None,None]).abs().square().mean()
    energy = .5*prediction[:, :, 3].square().mean(-1)
    truth_energy = .5*target[:, :, 3].square().mean(-1)
    floor = (.25*(amplitude/k).square()*1e-8)[:,None]
    log_energy = (torch.log10(torch.maximum(energy, floor)) - torch.log10(torch.maximum(truth_energy, floor))).square().mean()
    density, velocity, pressure, electric = prediction.unbind(dim=2)
    total = .5*(pressure+density*velocity.square()+electric.square()).mean(-1)
    n0, u0, p0, e0 = target[:,0].unbind(dim=1)
    total0 = .5*(p0+n0*u0.square()+e0.square()).mean(-1)
    conservation = ((total-total0[:,None])/total0[:,None]).square().mean()
    conservation = conservation + (density.mean(-1)-n0.mean(-1)[:,None]).square().mean()
    conservation = conservation + ((density*velocity).mean(-1)-(n0*u0).mean(-1)[:,None]).square().mean()
    positivity = torch.relu(.3-density).square().mean() + torch.relu(.15-pressure).square().mean()
    return {"state":state, "complex_field":complex_field, "log_energy":log_energy,
            "conservation":conservation, "positivity":positivity}


def uniform_samples(cache, rng, samples_per_case):
    indices = np.repeat(np.arange(len(cache.cases)), samples_per_case)
    starts = rng.uniform(0, cache.end, len(indices))
    # The t=0 boundary and its missing-history mask are always included.
    starts[::samples_per_case] = 0.
    order = rng.permutation(len(indices))
    return indices[order], starts[order]


def training_windows(cache, rng, epoch, horizon):
    indices = rng.permutation(len(cache.cases))
    starts = []
    for index in indices:
        if epoch % 5 == 0:
            starts.append(0.)
            continue
        lo, hi = cache.phases[int(index)][(epoch-1)%4]
        hi = min(hi, cache.end-horizon)
        lo = min(lo, hi)
        if hi-lo < .05:
            lo, hi = 0., cache.end-horizon
        starts.append(float(rng.uniform(lo, hi)))
    return indices, starts


def training_rollout(model, cache, index, start, horizon, config):
    """Refresh the memory with generated states, then backpropagate one full window."""
    burn = float(config.get("burn_in", 0.))
    memory = None
    if burn:
        with torch.no_grad():
            _, memory = advance_history(model,cache,index,start,horizon=burn,dt=config["dt"],
                                         use_checkpoint=False,return_memory=True)
        start = start + burn
    prediction = advance_history(model,cache,index,start,horizon=horizon,dt=config["dt"],
                                  use_checkpoint=True,memory=memory)
    return prediction, start


@torch.no_grad()
def supervised_validation(model, cache):
    indices = torch.arange(len(cache.cases), device=cache.state.device).repeat_interleave(80)
    starts = torch.linspace(0, cache.end, 80, device=cache.state.device).repeat(len(cache.cases))
    total, count = 0., 0
    for offset in range(0,len(indices),256):
        sub = indices[offset:offset+256]
        total += float(closure_loss(model,cache,sub,starts[offset:offset+256]))*len(sub)
        count += len(sub)
    return total/count


@torch.no_grad()
def short_validation(model, cache, config, horizon=2.):
    ids, starts = [], []
    for i, regions in enumerate(cache.phases):
        ids.extend([i]*4)
        starts.extend([0., min(sum(regions[1])/2,cache.end-horizon),
                       min(sum(regions[2])/2,cache.end-horizon),cache.end-horizon])
    totals, count, failed = {}, 0, 0
    for offset in range(0,len(ids),config["rollout_batch_size"]):
        indices = torch.tensor(ids[offset:offset+config["rollout_batch_size"]],device=cache.state.device)
        times = torch.tensor(starts[offset:offset+config["rollout_batch_size"]],device=cache.state.device)
        prediction = advance_history(model,cache,indices,times,horizon=horizon,dt=config["dt"],use_checkpoint=False)
        target = torch.stack([cache.sample(indices,times+j*.1) for j in range(prediction.shape[1])],dim=1)
        good = torch.isfinite(prediction).all(dim=(1,2,3)) & (prediction[:,:,0].amin(dim=(1,2))>0) & (prediction[:,:,2].amin(dim=(1,2))>0)
        failed += int((~good).sum())
        if bool(good.any()):
            k,alpha = parameters(cache,indices)
            components = trajectory_losses(prediction[good],target[good],k[good],alpha[good])
            number = int(good.sum())
            for name,value in components.items():
                totals[name] = totals.get(name,0.)+float(value)*number
            count += number
    result = {name:value/max(count,1) for name,value in totals.items()}
    result.update(failed_windows=failed,window_count=len(ids),horizon=horizon)
    # Each component remains visible. Long validation is the final selection criterion.
    result["score"] = result.get("state",1e6)+result.get("log_energy",1e6)+result.get("complex_field",1e6)
    return result


@torch.no_grad()
def full_validation(model, cache, config, output):
    model.eval()
    indices = torch.arange(len(cache.cases),device=cache.state.device)
    starts = torch.zeros(len(cache.cases),device=cache.state.device)
    torch.cuda.synchronize()
    beginning=time.perf_counter()
    failed_at = torch.full((len(cache.cases),),float("inf"),device=cache.state.device,dtype=torch.float64)
    closure_times, closure_values = [], []
    def observe_closure(now, value):
        if np.isclose(now/.1, round(now/.1), atol=1e-7):
            closure_times.append(now)
            closure_values.append(value.detach())
    def observe(now,state):
        bad = ~torch.isfinite(state).all(dim=(1,2)) | (state[:,0].amin(-1)<=0) | (state[:,2].amin(-1)<=0)
        failed_at.copy_(torch.where(bad & torch.isinf(failed_at),now,failed_at))
    def progress(now,state):
        if np.isclose(now%10,0,atol=1e-8):
            print(json.dumps({"validation_time":now,"failed":int(torch.isfinite(failed_at).sum()),
                              "seconds":time.perf_counter()-beginning}),flush=True)
    prediction=advance_history(model,cache,indices,starts,horizon=80.,dt=config["dt"],
                               use_checkpoint=False,progress=progress,step_observer=observe,
                               closure_observer=observe_closure)
    torch.cuda.synchronize()
    wall=time.perf_counter()-beginning
    times=np.arange(801)*.1
    pred=prediction.permute(1,0,2,3).cpu().numpy()
    filtered_truth=cache.state[:,::5].permute(1,0,2,3).cpu().numpy()
    truth=filtered_truth
    reference=config.get("validation_reference","filtered")
    if reference=="unfiltered":
        reference_states=[]
        for case in cache.cases:
            trajectory=load_continuum_case(case)
            moments=interpolate(trajectory.state,trajectory.time,times)
            electric=electric_numpy(moments[:,None],np.array([case.K]))[:,0]
            reference_states.append(np.concatenate((moments,electric[:,None]),axis=1))
        truth=np.stack(reference_states,axis=1).astype(np.float32)
    elif reference!="filtered":
        raise ValueError("Unknown validation reference")
    failure_times=[float(v) if np.isfinite(v) else None for v in failed_at.cpu().numpy()]
    rows=[]
    for i,case in enumerate(cache.cases):
        if failure_times[i] is not None:
            pred[times>=failure_times[i]-1e-8,i]=np.nan
        rows.append(metrics(pred[:,i],truth[:,i],case,times,failure_times[i]))
    completed=[r for r in rows if r["complete"]]
    splits=sorted({case.split for case in cache.cases})
    result={"split":splits[0] if len(splits)==1 else splits,"case_count":len(rows),"complete":len(completed),
            "wall_seconds":wall,"model_calls":round(80/config["closure_interval"]) if config.get("closure_interval",0.) else 4*round(80/config["dt"]),
            "network_batch_multiplier":2 if config.get("enforce_reflection",False) else 1,
            "validation_reference":reference,
            "validity_checks":"all filtered RK provisional states and accepted states; n>0, p>0, finite",
            "rhs_density_floor":None,"state_clipping":False,
            "initialization":"given initial state; no kinetic heat-flux or history warmup",
            "case_metrics":rows}
    for name in ("full_t80_field_energy_log10_rmse","perturbation_relative_l2","electric_mode1_phase_mae","total_energy_max_relative_drift"):
        result[name+"_median"] = float(np.median([r[name] for r in completed if r[name] is not None])) if completed else None
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"rollout.npz",time=times,prediction=pred,truth=truth,filtered_truth=filtered_truth,
                        case_ids=np.array([c.case_id for c in cache.cases]),
                        closure_time=np.asarray(closure_times),
                        closure_gradient=torch.stack(closure_values).cpu().numpy())
    atomic_json(output/"summary.json",result)
    return result


def train(args):
    config=json.loads(args.config.read_text())
    if args.arm:
        config["include_alpha"]=args.arm=="C"
        if args.arm=="A": config["history_span"]=0.
    if args.history_span is not None: config["history_span"]=args.history_span
    rng=np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("Formal training requires CUDA")
    torch.cuda.set_per_process_memory_fraction(float(config.get("cuda_memory_fraction",.15)),device)
    cases=load_continuum_case_index(config["dataset_root"])
    train_cases=[c for c in cases if c.split=="train"]
    val_cases=[c for c in cases if c.split=="validation"]
    if args.smoke:
        train_cases=train_cases[::max(1,len(train_cases)//6)][:6]
        val_cases=val_cases[::max(1,len(val_cases)//3)][:3]
        config.update(supervised_epochs=1,samples_per_case=16,supervised_batch_size=32,
                      rollout_batch_size=2,burn_in=.2,curriculum=[{"horizon":.2,"epochs":1}])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parents[1]
    source_manifest={}
    sources={str(p.relative_to(source_root)):p.read_bytes() for p in source_root.rglob("*.py")}
    bundle=hashlib.sha256(b"".join(name.encode()+content for name,content in sorted(sources.items()))).hexdigest()[:12]
    for relative,content in sources.items():
        source_manifest[relative]=hashlib.sha256(content).hexdigest()
        snapshot=args.output_dir/"source_snapshot"/bundle/relative
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
    atomic_json(args.output_dir/"source_manifest.json",{"bundle":bundle,"files":source_manifest})
    atomic_json(args.output_dir/"config.json",{**config,"seed":args.seed,"arm":args.arm,
                "train_ids":[c.case_id for c in train_cases],"validation_ids":[c.case_id for c in val_cases]})
    disk=Path(config["cache_dir"])
    training=load_history_cache(train_cases,config["maximum_mode"],device,disk_cache=disk)
    validation=load_history_cache(val_cases,config["maximum_mode"],device,disk_cache=disk)
    parent=torch.load(config["parent_checkpoint"],map_location=device,weights_only=False)
    norm=parent["normalization"] if parent.get("stage")=="continuum_history_closure_round11" else normalization(training)
    model=construct(config,norm,device)
    model.source_bundle=bundle
    if parent.get("stage")=="continuum_history_closure_round11":
        initialize_history_parent(model,parent)
    else:
        model.initialize_from_supervised(parent)
    if config["supervised_epochs"]==0:
        save_checkpoint(args.output_dir/"best_supervised.pt",model,config,norm,seed=args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["supervised_lr"],weight_decay=1e-6)
    resume_path=args.output_dir/"last_training.pt"
    resume=torch.load(resume_path,map_location=device,weights_only=False) if resume_path.exists() else None
    if resume is not None and resume["config"]!=config:
        raise ValueError("Interrupted run configuration changed; use a new output directory")
    history=[] if resume is None else resume["history"]
    best=float("inf") if resume is None else resume.get("supervised_best",float("inf"))
    best_state=copy.deepcopy(model.state_dict())
    started=time.perf_counter()
    supervised_start=0
    if resume is not None:
        rng.bit_generator.state=resume["rng_state"]
        model.load_state_dict(resume["model_state_dict"])
        if resume["phase"]=="supervised":
            supervised_start=resume["completed_epoch"]
            optimizer.load_state_dict(resume["optimizer"])
        else:
            supervised_start=config["supervised_epochs"]
    for epoch in range(supervised_start,config["supervised_epochs"]):
        model.train()
        ids,times=uniform_samples(training,rng,config["samples_per_case"])
        losses=[]
        for off in range(0,len(ids),config["supervised_batch_size"]):
            index=torch.tensor(ids[off:off+config["supervised_batch_size"]],device=device)
            start=torch.tensor(times[off:off+config["supervised_batch_size"]],dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=closure_loss(model,training,index,start)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        score=supervised_validation(model,validation)
        row={"phase":"supervised","epoch":epoch+1,"train_loss":float(np.mean(losses)),
             "validation_loss":score,"elapsed_seconds":time.perf_counter()-started}
        history.append(row); print(json.dumps(row),flush=True)
        if score<best:
            best=score; best_state=copy.deepcopy(model.state_dict())
            save_checkpoint(args.output_dir/"best_supervised.pt",model,config,norm,seed=args.seed)
        atomic_json(args.output_dir/"history.json",{"history":history})
        save_checkpoint(resume_path,model,config,norm,seed=args.seed,phase="supervised",
                        completed_epoch=epoch+1,optimizer=optimizer.state_dict(),history=history,
                        rng_state=rng.bit_generator.state,supervised_best=best)
    supervised=torch.load(args.output_dir/"best_supervised.pt",map_location=device,weights_only=False)
    model.load_state_dict(supervised["model_state_dict"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["rollout_lr"],weight_decay=1e-6)
    global_epoch=0
    validation_horizon=.2 if args.smoke else 2.
    resumed_epoch=0
    if resume is not None and resume["phase"]=="rollout":
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        resumed_epoch=resume["completed_epoch"]
        best_key=tuple(resume["best_key"])
    else:
        initial_score=short_validation(model,validation,config,validation_horizon)
        best_key=(initial_score["failed_windows"],initial_score["score"])
        save_checkpoint(args.output_dir/"best_rollout.pt",model,config,norm,seed=args.seed,validation=initial_score)
    candidate_paths=[args.output_dir/"best_supervised.pt"]
    for stage in config["curriculum"]:
        for _epoch in range(stage["epochs"]):
            global_epoch+=1
            if global_epoch<=resumed_epoch:
                continue
            ids,times=training_windows(training,rng,global_epoch,stage["horizon"]+config.get("burn_in",0.))
            model.train(); losses=[]; failures=0; details={}
            for off in range(0,len(ids),config["rollout_batch_size"]):
                index=torch.tensor(ids[off:off+config["rollout_batch_size"]],device=device)
                start=torch.tensor(times[off:off+config["rollout_batch_size"]],dtype=torch.float32,device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction, target_start=training_rollout(model,training,index,start,stage["horizon"],config)
                target=torch.stack([training.sample(index,target_start+j*.1) for j in range(prediction.shape[1])],dim=1)
                k,alpha=parameters(training,index)
                parts=trajectory_losses(prediction,target,k,alpha)
                parts["supervised"]=closure_loss(model,training,index,start)
                loss=sum(config["loss_weights"][name]*value for name,value in parts.items())
                if not bool(torch.isfinite(loss)):
                    failures+=len(index)
                    raise RuntimeError("Non-finite training rollout; preserve logs and revise numerical/training settings")
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                for name,value in parts.items(): details[name]=details.get(name,0.)+float(value.detach())
                print(json.dumps({"epoch":global_epoch,"batch":off//config["rollout_batch_size"]+1,
                                  "batches":math.ceil(len(ids)/config["rollout_batch_size"]),
                                  "horizon":stage["horizon"],"loss":float(loss.detach()),"grad_norm":float(grad_norm)}),flush=True)
            model.eval()
            score=short_validation(model,validation,config,validation_horizon)
            row={"phase":"rollout","epoch":global_epoch,"horizon":stage["horizon"],
                 "train_loss":float(np.mean(losses)),"components":{k:v/len(losses) for k,v in details.items()},
                 "validation":score,"covered_cases":len(set(map(int,ids))),"elapsed_seconds":time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
            key=(score["failed_windows"],score["score"])
            if key<best_key:
                best_key=key
                save_checkpoint(args.output_dir/"best_rollout.pt",model,config,norm,seed=args.seed,validation=score)
            save_checkpoint(args.output_dir/f"epoch{global_epoch}.pt",model,config,norm,seed=args.seed,validation=score)
            atomic_json(args.output_dir/"history.json",{"history":history})
            save_checkpoint(resume_path,model,config,norm,seed=args.seed,phase="rollout",
                            completed_epoch=global_epoch,optimizer=optimizer.state_dict(),history=history,
                            rng_state=rng.bit_generator.state,supervised_best=best,best_key=best_key)
        candidate_paths.append(args.output_dir/f"epoch{global_epoch}.pt")
    selected=torch.load(args.output_dir/"best_rollout.pt",map_location=device,weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    result={"config":config,"seed":args.seed,"arm":args.arm,"smoke":args.smoke,
            "train_cases":len(training.cases),"validation_cases":len(validation.cases),
            "test_used":False,"elapsed_seconds":time.perf_counter()-started,
            "short_validation":selected["validation"]}
    if not args.smoke:
        candidate_paths.append(args.output_dir/"best_rollout.pt")
        evaluations=[]
        chosen=None
        for path in candidate_paths:
            candidate=torch.load(path,map_location=device,weights_only=False)
            model.load_state_dict(candidate["model_state_dict"])
            evaluation_path=args.output_dir/"candidates"/path.stem
            marker=evaluation_path/"summary.json"
            report=json.loads(marker.read_text()) if marker.exists() else full_validation(model,validation,config,evaluation_path)
            key=(report["case_count"]-report["complete"],
                 report["full_t80_field_energy_log10_rmse_median"] if report["complete"] else 1e9,
                 report["electric_mode1_phase_mae_median"] if report["complete"] else 1e9,
                 report["perturbation_relative_l2_median"] if report["complete"] else 1e9)
            evaluations.append({"checkpoint":str(path),"selection_key":key,"summary":report})
            if chosen is None or key<chosen[0]:
                chosen=(key,path,report)
        selected=torch.load(chosen[1],map_location=device,weights_only=False)
        model.load_state_dict(selected["model_state_dict"])
        save_checkpoint(args.output_dir/"selected_full_validation.pt",model,config,norm,
                        seed=args.seed,selected_source=str(chosen[1]),validation=chosen[2])
        result["full_validation"]=chosen[2]
        result["selected_checkpoint"]=str(args.output_dir/"selected_full_validation.pt")
        result["candidate_evaluations"]=evaluations
    result["elapsed_seconds"]=time.perf_counter()-started
    atomic_json(args.output_dir/"summary.json",result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--arm",choices=["A","B","C"])
    p.add_argument("--history-span",type=float)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--smoke",action="store_true")
    train(p.parse_args())


if __name__=="__main__": main()
