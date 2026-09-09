"""CPU/CUDA evaluation of historical Round 11 or HP/zero on portable arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from landau_surrogate.data.portable_closure import load_manifest, load_index, load_case, history_cache, hp_gradient
from landau_surrogate.data.continuum_history import supervised_history
from landau_surrogate.fluid.history_closure_1d import advance_history
from landau_surrogate.training.continuum_history_closure import construct
from landau_surrogate.tools.audit_continuum_v1_closure_oracle import atomic_json, metrics
from summarize_round11_history import interval_metrics
from landau_surrogate.models.pc_fno_ssm import SmallFNOSSM
from landau_surrogate.fluid.pc_ssm import advance_ssm


class AnalyticClosure(torch.nn.Module):
    history_span = 0.
    maximum_mode = 16
    closure_interval = 0.

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, history, valid, k, alpha):
        return hp_gradient(history[:, -1], k, self.scale, self.maximum_mode)


class TrajectoryFailed(RuntimeError):
    pass


@torch.no_grad()
def rollout(model, cache, arrays, case, horizon, device):
    output_time = np.arange(round(horizon / .1) + 1, dtype=np.float64) * .1
    prediction = np.full((len(output_time), 4, arrays["state"].shape[-1]), np.nan, dtype=np.float32)
    prediction[0] = arrays["state"][0]
    failed_time = None

    def observe(now, state):
        nonlocal failed_time
        if (not bool(torch.isfinite(state).all()) or float(state[:, 0].min()) <= 0
                or float(state[:, 2].min()) <= 0):
            failed_time = float(now)
            raise TrajectoryFailed("First invalid RK state")

    def progress(now, state):
        prediction[round(now / .1)] = state[0].cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        observe(0., cache.state[:, 0])
        if isinstance(model, SmallFNOSSM):
            advance_ssm(model, cache.state[:, 0], torch.tensor([case.K], device=device),
                        horizon=horizon, dt=.02, observer=observe, progress=progress)
        else:
            advance_history(model, cache, torch.tensor([0], device=device), torch.zeros(1, device=device),
                            horizon=horizon, dt=.02, use_checkpoint=False, output_dt=.1,
                            step_observer=observe, progress=progress)
    except TrajectoryFailed:
        prediction[output_time >= failed_time - 1e-8] = np.nan
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    truth = arrays["reference_state"][:len(output_time)]
    assert np.array_equal(arrays["reference_time"][:len(output_time)], output_time)
    row = metrics(prediction, truth, case, output_time, failed_time)
    row["horizon"] = horizon
    if horizon != 80.:
        row["complete_to_requested_horizon"] = row.pop("complete")
        row["complete_t80"] = None
        row["requested_horizon_field_energy_log10_rmse"] = row.pop("full_t80_field_energy_log10_rmse")
    row["late"] = (interval_metrics(prediction, truth, output_time >= 30.)
                   if horizon == 80. and row["complete"] else None)
    row["wall_seconds"] = seconds
    return row, {"time": output_time, "prediction": prediction, "truth": truth}


@torch.no_grad()
def offline(model, cache, case, device):
    # Regular dt=0.1 evaluation samples with the SAME dt=0.02 causal interpolation
    # and nonuniform eight-slot stencils used by historical Round 11.
    if isinstance(model, SmallFNOSSM):
        numerator = denominator = 0.
        hidden = None
        for start in range(0, cache.state.shape[1], 64):
            state = cache.state[:, start:start + 64]
            pred, hidden = model(state, torch.tensor([case.K], device=device), hidden=hidden, dt=.02)
            mask = torch.arange(start, start + state.shape[1], device=device) % 5 == 0
            pred = pred[:, mask]
            truth = cache.gradient[:, start:start + 64][:, mask]
            numerator += float((pred.double() - truth.double()).square().sum())
            denominator += float(truth.double().square().sum())
        return {"case_id": case.case_id, "split": case.split, "regime": case.regime,
                "gradient_relative_l2": float(np.sqrt(numerator / max(denominator, 1e-30))),
                "teacher_forced_history": True, "samples": 801, "memory_replay_from_t0": True}
    times = torch.arange(801, device=device, dtype=torch.float32) * .1
    numerator = denominator = 0.
    for offset in range(0, len(times), 32):
        query = times[offset:offset + 32]
        indices = torch.zeros(len(query), device=device, dtype=torch.long)
        history, valid = supervised_history(cache, indices, query, model.history_span)
        k = torch.full((len(query),), case.K, device=device)
        alpha = torch.full((len(query),), case.alpha, device=device)
        pred = model(history, valid, k, alpha)
        truth = cache.sample(indices, query, "gradient")
        numerator += float((pred.double() - truth.double()).square().sum())
        denominator += float(truth.double().square().sum())
    return {"case_id": case.case_id, "split": case.split, "regime": case.regime,
            "gradient_relative_l2": float(np.sqrt(numerator / max(denominator, 1e-30))),
            "teacher_forced_history": True, "samples": len(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", choices=["round11", "ssm", "hp", "zero"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--subset", choices=["starter", "full"], default="full")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--mode", choices=["offline", "rollout"], default="rollout")
    parser.add_argument("--horizon", type=float, default=2.)
    parser.add_argument("--limit", type=int, default=1, help="0 evaluates every case in the requested subset")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory; previous comparisons are preserved")
    if args.limit < 0 or not 0 < args.horizon <= 80 or not np.isclose(args.horizon / .1, round(args.horizon / .1)):
        parser.error("Require nonnegative limit and a horizon in (0,80] divisible by 0.1")
    if (args.baseline in ("round11", "ssm")) != (args.checkpoint is not None):
        parser.error("Provide --checkpoint exactly for round11/ssm")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("This comparison supports CPU or CUDA")
    manifest = load_manifest(args.data_root)
    cases = load_index(args.data_root, args.split, args.subset)
    cases = cases[:args.limit] if args.limit else cases
    provenance = {"baseline": args.baseline, "mode": args.mode, "subset": args.subset,
                  "split": args.split, "device": str(device), "horizon": args.horizon if args.mode == "rollout" else 80.,
                  "solver_dt": .02, "output_dt": .1, "maximum_mode": 16,
                  "reference": "unfiltered" if args.mode == "rollout" else "filtered_gradient",
                  "state_clipping": False, "density_floor": None,
                  "failure_policy": "stop at first invalid provisional/accepted RK state; missing output stays NaN",
                  "full_formal_validation": args.mode == "rollout" and args.horizon == 80. and args.split == "validation" and len(cases) == 20,
                  "dataset_manifest_sha256": hashlib.sha256((args.data_root / "manifest.json").read_bytes()).hexdigest(),
                  "cases": [c.case_id for c in cases], "training_performed": False,
                  "new_independent_holdout_used": False}
    if args.baseline == "round11":
        data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if data.get("stage") != "continuum_history_closure_round11":
            parser.error("This adapter requires a historical Round 11 checkpoint")
        if data["config"]["maximum_mode"] != manifest["maximum_mode"] or data["config"]["dt"] != .02:
            parser.error("Checkpoint numerical contract differs from this dataset")
        model = construct(data["config"], data["normalization"], device)
        model.load_state_dict(data["model_state_dict"], strict=True)
        provenance.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                          checkpoint_training_config=data["config"], checkpoint_source_bundle=data["source_bundle"],
                          checkpoint_training_cases=139, embedded_normalization_preserved=True)
    elif args.baseline == "ssm":
        data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if data.get("stage") != "pc_fno_ssm_experimental_v1":
            parser.error("Expected a portable experimental SSM checkpoint")
        model = SmallFNOSSM(data["normalization"], **data["model_config"]).to(device)
        model.load_state_dict(data["model_state_dict"], strict=True)
        provenance.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                          checkpoint_training_config=data["config"],
                          memory_rule="zero at t0; frozen through RK stages; one exact diagonal update using U_t per accepted dt")
    else:
        stats = json.loads((args.data_root / f"statistics_{args.subset}.json").read_text())
        scale = stats["hp_scale"] if args.baseline == "hp" else 0.
        model = AnalyticClosure(scale).to(device)
        provenance.update(hp_scale=scale, hp_fit_cases=stats["case_count"], hp_fit_split="train")
    model.eval()
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "provenance.json", provenance)
    rows = []
    for case in cases:
        arrays = load_case(case, verify=True)
        cache = history_cache(case, arrays, device)
        if args.mode == "rollout":
            row, values = rollout(model, cache, arrays, case, args.horizon, device)
            np.savez_compressed(args.output_dir / f"{case.case_id}.npz", **values)
        else:
            row = offline(model, cache, case, device)
        rows.append(row)
        atomic_json(args.output_dir / "summary.json", {**provenance, "case_metrics": rows})
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
