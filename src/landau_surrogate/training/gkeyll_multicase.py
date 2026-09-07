"""Trajectory-wise multi-case Gkeyll training for a band-consistent closure FNO."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from landau_surrogate.data.huang2025 import HuangTrajectory, load_huang_mat
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


def case_id(k: float, alpha: float) -> str:
    return f"k{k:.3f}_a{alpha:.3f}".replace(".", "p")


def lowpass_numpy(value: np.ndarray, maximum_mode: int) -> np.ndarray:
    transformed = np.fft.rfft(value, axis=-1)
    transformed[..., maximum_mode + 1 :] = 0.0
    return np.fft.irfft(transformed, n=value.shape[-1], axis=-1).astype(np.float32)


def lowpass_torch(value: torch.Tensor, maximum_mode: int) -> torch.Tensor:
    transformed = torch.fft.rfft(value.float(), dim=-1)
    transformed[..., maximum_mode + 1 :] = 0.0
    return torch.fft.irfft(transformed, n=value.shape[-1], dim=-1).to(value.dtype)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(target), 1.0e-12)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": float(np.corrcoef(prediction.ravel(), target.ravel())[0, 1]),
    }


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_cases(manifest: Path, dataset_root: Path) -> list[tuple[dict, HuangTrajectory]]:
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    cases = []
    for spec in plan["cases"]:
        identifier = case_id(float(spec["k"]), float(spec["alpha"]))
        path = dataset_root / "cases" / identifier / "processed" / "trajectory.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        trajectory = load_huang_mat(path)
        if not np.isclose(trajectory.k, float(spec["k"])) or not np.isclose(
            trajectory.alpha, float(spec["alpha"])
        ):
            raise ValueError(f"Metadata mismatch for {identifier}")
        cases.append(({**spec, "case_id": identifier, "path": str(path)}, trajectory))
    return cases


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    normalized_state: np.ndarray,
    normalized_k: float,
    batch_size: int,
    maximum_mode: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(normalized_state)), batch_size=batch_size)
    output = []
    model.eval()
    for (state,) in loader:
        state = state.to(device)
        k_value = torch.full((len(state),), normalized_k, device=device)
        output.append(lowpass_torch(model(state, k_value), maximum_mode).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--maximum-mode", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal Gkeyll multi-case training requires CUDA")

    cases = load_cases(args.manifest, args.dataset_root)
    prepared = []
    for spec, trajectory in cases:
        prepared.append(
            (
                spec,
                trajectory,
                lowpass_numpy(trajectory.state, args.maximum_mode),
                lowpass_numpy(trajectory.heat_flux_gradient, args.maximum_mode),
            )
        )
    train = [item for item in prepared if item[0]["split"] == "train"]
    validation = [item for item in prepared if item[0]["split"] == "validation"]
    test = [item for item in prepared if item[0]["split"] == "test"]
    if not train or not validation or not test:
        raise ValueError("Manifest must provide non-empty train/validation/test case splits")

    train_state = np.concatenate([item[2] for item in train])
    train_target = np.concatenate([item[3] for item in train])
    train_k = np.concatenate(
        [np.full(len(item[2]), item[1].k, dtype=np.float32) for item in train]
    )
    input_mean = train_state.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    input_std = train_state.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    target_mean = float(train_target.mean(dtype=np.float64))
    target_std = float(train_target.std(dtype=np.float64))
    k_mean = float(train_k.mean(dtype=np.float64))
    k_std = float(train_k.std(dtype=np.float64))
    if min(float(input_std.min()), target_std, k_std) <= 0.0:
        raise ValueError("Degenerate training normalization")

    normalized_state = ((train_state - input_mean[None, :, None]) / input_std[None, :, None]).astype(np.float32)
    normalized_target = ((train_target - target_mean) / target_std).astype(np.float32)
    normalized_k = ((train_k - k_mean) / k_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_state), torch.from_numpy(normalized_target),
            torch.from_numpy(normalized_k),
        ),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    model = ClosureFNO1d(3, True, args.width, args.modes, args.layers, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-5)
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    stale = 0

    def evaluate(items: list[tuple]) -> tuple[float, float, list[dict]]:
        predictions, targets, full_targets, rows = [], [], [], []
        for spec, trajectory, state, target in items:
            state_n = ((state - input_mean[None, :, None]) / input_std[None, :, None]).astype(np.float32)
            k_n = (trajectory.k - k_mean) / k_std
            prediction = predict(
                model, state_n, k_n, args.batch_size, args.maximum_mode, device
            ) * target_std + target_mean
            current = metrics(prediction, target)
            full = metrics(prediction, trajectory.heat_flux_gradient)
            discarded = trajectory.heat_flux_gradient - target
            rows.append({
                "case_id": spec["case_id"], "k": trajectory.k, "alpha": trajectory.alpha,
                **current,
                "relative_l2_to_unfiltered_target": full["relative_l2"],
                "target_fraction_above_maximum_mode": float(
                    np.linalg.norm(discarded)
                    / max(np.linalg.norm(trajectory.heat_flux_gradient), 1.0e-12)
                ),
            })
            predictions.append(prediction)
            targets.append(target)
            full_targets.append(trajectory.heat_flux_gradient)
        aggregate = metrics(np.concatenate(predictions), np.concatenate(targets))["relative_l2"]
        aggregate_full = metrics(
            np.concatenate(predictions), np.concatenate(full_targets)
        )["relative_l2"]
        return aggregate, aggregate_full, rows

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for state, target, k_value in loader:
            state = state.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            k_value = k_value.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = lowpass_torch(model(state, k_value), args.maximum_mode)
            pointwise = torch.mean((output - target) ** 2)
            output_hat = torch.fft.rfft(output.float(), dim=-1)
            target_hat = torch.fft.rfft(target.float(), dim=-1)
            spectral = torch.mean(torch.abs(output_hat - target_hat) ** 2) / torch.clamp(
                torch.mean(torch.abs(target_hat) ** 2), min=1.0e-8
            )
            loss = pointwise + 0.1 * spectral
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_value, _validation_full, _ = evaluate(validation)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_relative_l2": validation_value,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_value < best_value:
            best_value = validation_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    split_metrics = {}
    for name, items in (("train", train), ("validation", validation), ("test", test)):
        aggregate, aggregate_full, rows = evaluate(items)
        split_metrics[name] = {
            "relative_l2": aggregate,
            "relative_l2_to_unfiltered_target": aggregate_full,
            "cases": rows,
        }
        evaluation_dir = args.output_dir / "evaluation" / name
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        for spec, trajectory, state, target in items:
            state_n = ((state - input_mean[None, :, None]) / input_std[None, :, None]).astype(np.float32)
            prediction = predict(
                model, state_n, (trajectory.k - k_mean) / k_std,
                args.batch_size, args.maximum_mode, device,
            ) * target_std + target_mean
            np.savez_compressed(
                evaluation_dir / f"{spec['case_id']}.npz", time=trajectory.time,
                x_over_l=trajectory.x_over_l, target=target,
                unfiltered_target=trajectory.heat_flux_gradient, prediction=prediction,
            )

    normalization = {
        "input_mean": input_mean.tolist(), "input_std": input_std.tolist(),
        "target_mean": target_mean, "target_std": target_std,
        "k_mean": k_mean, "k_std": k_std,
    }
    summary = {
        "protocol": "trajectory_casewise",
        "source_manifest": str(args.manifest),
        "maximum_mode": args.maximum_mode,
        "split_case_ids": {
            name: [item[0]["case_id"] for item in items]
            for name, items in (("train", train), ("validation", validation), ("test", test))
        },
        "normalization": normalization,
        "best_validation_relative_l2": best_value,
        "metrics": split_metrics,
        "epochs_completed": len(history),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "huang2025_gkeyll_multicase_v1",
            "protocol": "trajectory_casewise",
            "source_kind": "gkeyll_hdf5_multicase",
            "moment_definition": "central",
            "model_arguments": {
                "state_channels": 3, "include_k": True, "width": args.width,
                "modes": args.modes, "layers": args.layers, "enforce_zero_mean": True,
            },
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            **summary,
        },
        args.output_dir / "best.pt",
    )
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "history.json", history)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
