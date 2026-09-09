"""Train a small portable closure with chronological, stateful truncated BPTT."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from landau_surrogate.data.portable_closure import load_index, load_case
from landau_surrogate.models.pc_fno_ssm import SmallFNOSSM
from landau_surrogate.tools.audit_continuum_v1_closure_oracle import atomic_json


def run_cases(model, cases, device, chunk, horizon, optimizer=None):
    total, count = 0., 0
    model.train(optimizer is not None)
    for case in cases:
        arrays = load_case(case)
        stop = min(len(arrays["time"]), round(horizon / .02) + 1)
        hidden = None  # Reset at every trajectory, never between its chronological chunks.
        k = torch.tensor([case.K], device=device)
        numerator = denominator = 0.
        for start in range(0, stop, chunk):
            state = torch.from_numpy(arrays["state"][start:min(start + chunk, stop)])[None].to(device)
            target = torch.from_numpy(arrays["gradient"][start:min(start + chunk, stop)])[None].to(device)
            with torch.set_grad_enabled(optimizer is not None):
                prediction, hidden = model(state, k, hidden=hidden, dt=.02)
                loss = ((prediction - target) / model.gradient_std).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite loss at {case.case_id}, frame {start}")
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            hidden = hidden.detach()
            numerator += float((prediction.detach().double() - target.double()).square().sum())
            denominator += float(target.double().square().sum())
        total += (numerator / max(denominator, 1e-30)) ** .5
        count += 1
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/ssm"))
    p.add_argument("--variant", choices=["fno", "fno_ssm", "fno_residual", "fno_ssm_residual"], default="fno_ssm_residual")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--memory-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset", choices=["starter", "full"], default="full")
    p.add_argument("--limit-train", type=int, default=0, help="Smoke checks only; 0 uses all train cases")
    p.add_argument("--limit-validation", type=int, default=0)
    p.add_argument("--horizon", type=float, default=80., help="Shorten only for interface smoke checks")
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    if min(args.epochs, args.chunk, args.width, args.memory_size) < 1 or not 0 < args.horizon <= 80:
        p.error("Invalid training sizes/horizon")
    if min(args.limit_train, args.limit_validation) < 0:
        p.error("Limits must be nonnegative")
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        p.error("Use CPU or CUDA")
    norm = json.loads((args.data_root / f"statistics_{args.subset}.json").read_text())
    train = load_index(args.data_root, "train", args.subset)
    val = load_index(args.data_root, "validation", args.subset)
    train = train[:args.limit_train] if args.limit_train else train
    val = val[:args.limit_validation] if args.limit_validation else val
    model_config = {"width": args.width, "layers": 2, "memory_size": args.memory_size, "maximum_mode": 16,
                    "use_memory": "ssm" in args.variant, "use_residual": "residual" in args.variant}
    model = SmallFNOSSM(norm, **model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    args.output_dir.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(train_case_ids=[c.case_id for c in train], validation_case_ids=[c.case_id for c in val],
                  device=str(device), train_only_normalization_subset=args.subset,
                  smoke_only=bool(args.limit_train or args.limit_validation or args.horizon != 80.),
                  dataset_manifest_sha256=hashlib.sha256((args.data_root / "manifest.json").read_bytes()).hexdigest(),
                  parameter_real_scalars=sum(x.numel() * (2 if x.is_complex() else 1) for x in model.parameters()),
                  test_used=False, dt=.02, state_clipping=False,
                  selection="minimum offline validation case-macro gradient relative L2; NOT full-rollout quality selection")
    atomic_json(args.output_dir / "config.json", config)
    best = float("inf"); records = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        order = rng.permutation(len(train))
        training_error = run_cases(model, [train[i] for i in order], device, args.chunk, args.horizon, optimizer)
        validation_error = run_cases(model, val, device, args.chunk, args.horizon)
        if not np.isfinite(validation_error):
            raise RuntimeError("Nonfinite validation score")
        record = {"epoch": epoch + 1, "train_online_gradient_relative_l2": training_error,
                  "validation_gradient_relative_l2": validation_error, "wall_seconds": time.perf_counter() - started}
        if device.type == "cuda":
            record["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        records.append(record)
        payload = {"stage": "pc_fno_ssm_experimental_v1", "model_config": model_config,
                   "normalization": norm, "model_state_dict": model.state_dict(), "config": config, "record": record}
        for name in (["last.pt", "best.pt"] if validation_error < best else ["last.pt"]):
            temporary = args.output_dir / (name + ".tmp")
            torch.save(payload, temporary); temporary.replace(args.output_dir / name)
        best = min(best, validation_error)
        atomic_json(args.output_dir / "history.json", {"epochs": records, "best_validation": best})
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
