"""Read-only validation of a frozen closure with explicitly recorded deployment overrides."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import torch
from landau_surrogate.data.continuum_v1 import load_continuum_case_index
from landau_surrogate.data.continuum_history import load_history_cache
from landau_surrogate.training.continuum_history_closure import construct,full_validation
from landau_surrogate.tools.audit_continuum_v1_closure_oracle import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--enforce-reflection",action="store_true")
    p.add_argument("--maximum-mode",type=int)
    p.add_argument("--dt",type=float)
    p.add_argument("--closure-interval",type=float)
    p.add_argument("--data-root",type=Path,help="Override the shared continuum dataset location.")
    p.add_argument("--cache-dir",type=Path,help="Use a personal derived-cache directory.")
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    if (args.output_dir/"summary.json").exists():
        raise RuntimeError("Evaluation already exists; use a new output directory")
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type!="cuda":raise RuntimeError("Production evaluation requires CUDA")
    torch.cuda.set_per_process_memory_fraction(.15,device)
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    config=dict(checkpoint["config"])
    if args.data_root is not None:config["dataset_root"]=str(args.data_root.resolve())
    if args.cache_dir is not None:config["cache_dir"]=str(args.cache_dir.resolve())
    if args.enforce_reflection:config["enforce_reflection"]=True
    if args.dt is not None:config["dt"]=args.dt
    if args.closure_interval is not None:config["closure_interval"]=args.closure_interval
    # Optional output/solver bandlimit probe retains the trained internal FNO weights.
    mode=config["maximum_mode"] if args.maximum_mode is None else args.maximum_mode
    model=construct(config,checkpoint["normalization"],device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.maximum_mode=mode
    cases=[c for c in load_continuum_case_index(config["dataset_root"]) if c.split=="validation"]
    cache=load_history_cache(cases,mode,device,disk_cache=Path(config["cache_dir"]))
    config["maximum_mode"]=mode
    atomic_json(args.output_dir/"provenance.json",{
        "checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_config":checkpoint["config"],"evaluation_config":config,
        "training_performed":False,"test_used":False,
        "probe_only":config!=checkpoint["config"]})
    full_validation(model,cache,config,args.output_dir)


if __name__=="__main__":main()
