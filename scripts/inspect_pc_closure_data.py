"""Inspect portable data and optionally check one small FNO backward pass."""
from __future__ import annotations
import argparse
import json
import torch
from torch.utils.data import DataLoader
from landau_surrogate.data.portable_closure import load_index, load_case, ClosureSequenceDataset
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subset", choices=["starter", "full"], default="full")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--smoke-backward", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for split in ("train", "validation"):
        cases = load_index(args.data_root, split, args.subset)
        if args.verify:
            for case in cases:
                load_case(case, verify=True)
        print(split, len(cases), "verified" if args.verify else "files present", flush=True)
    data = ClosureSequenceDataset(args.data_root, subset=args.subset, length=64, burn_in=64)
    batch = next(iter(DataLoader(data, batch_size=1, num_workers=0)))
    print(json.dumps({k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}))
    if args.smoke_backward:
        device = torch.device(args.device)
        model = ClosureFNO1d(state_channels=4, width=32, modes=17, layers=2).to(device)
        state = batch["state"][:, 64:].flatten(0, 1).to(device)
        target = batch["gradient"][:, 64:].flatten(0, 1).to(device)
        k = batch["K"].repeat_interleave(64).to(device)
        loss = (model(state, k) - target).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print("Small FNO forward/backward: OK; random initialization, no trained result or saved checkpoint")
        if device.type == "cuda":
            print("CUDA allocated peak MiB:", torch.cuda.max_memory_allocated(device) / 2**20)


if __name__ == "__main__":
    main()
