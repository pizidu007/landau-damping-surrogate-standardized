"""Train a paper-like FNO closure on a Huang et al. trajectory."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from landau_surrogate.data.huang2025 import causal_indices, load_huang_mat, paper_like_indices
from landau_surrogate.fluid.multimoment_1d import spectral_filter
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(target), 1.0e-12)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "correlation": float(np.corrcoef(prediction.ravel(), target.ravel())[0, 1]),
    }


@torch.no_grad()
def predict(model: torch.nn.Module, state: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(state)), batch_size=batch_size)
    values = []
    model.eval()
    for (batch,) in loader:
        dummy_k = torch.zeros(len(batch), device=device)
        values.append(model(batch.to(device), dummy_k).cpu().numpy())
    return np.concatenate(values)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("paper_like", "causal", "paper_full"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--normalization", choices=("group", "none"), default="group")
    parser.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    parser.add_argument("--deployment-mode", type=int)
    parser.add_argument("--supervised-loss-weight", type=float, default=1.0)
    parser.add_argument("--deployment-loss-weight", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--closure-work-weight", type=float, default=0.0)
    parser.add_argument("--closure-rms-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric", choices=("full", "deployment"), default="full",
        help="Select the best checkpoint using full-spectrum or deployment-filtered validation.",
    )
    args = parser.parse_args()
    if args.selection_metric == "deployment" and args.deployment_mode is None:
        parser.error("--selection-metric deployment requires --deployment-mode")
    if (
        args.deployment_loss_weight > 0 or args.consistency_weight > 0
        or args.closure_work_weight > 0 or args.closure_rms_weight > 0
    ) and args.deployment_mode is None:
        parser.error("deployment or consistency loss requires --deployment-mode")
    if any(weight < 0 for weight in (
        args.supervised_loss_weight, args.deployment_loss_weight, args.consistency_weight,
        args.closure_work_weight, args.closure_rms_weight,
    )):
        parser.error("loss weights must be non-negative")
    if args.supervised_loss_weight == 0 and args.deployment_loss_weight == 0:
        parser.error("at least one supervised loss weight must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal Huang reproduction training requires CUDA")

    trajectory = load_huang_mat(args.mat)
    if args.protocol == "paper_full":
        train_index = np.arange(min(len(trajectory.time), 8_000), dtype=np.int64)
        validation_index = train_index
        test_index = train_index
    else:
        split = paper_like_indices if args.protocol == "paper_like" else causal_indices
        train_index, validation_index, test_index = split(len(trajectory.time))
    input_mean = trajectory.state[train_index].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    input_std = trajectory.state[train_index].std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    target_mean = float(trajectory.heat_flux_gradient[train_index].mean(dtype=np.float64))
    target_std = float(trajectory.heat_flux_gradient[train_index].std(dtype=np.float64))
    state = ((trajectory.state - input_mean[None, :, None]) / input_std[None, :, None]).astype(np.float32)
    target = ((trajectory.heat_flux_gradient - target_mean) / target_std).astype(np.float32)
    input_mean_tensor = torch.tensor(input_mean, device=device).reshape(1, 3, 1)
    input_std_tensor = torch.tensor(input_std, device=device).reshape(1, 3, 1)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(state[train_index]), torch.from_numpy(target[train_index])),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    model = ClosureFNO1d(
        3, False, args.width, args.modes, args.layers, True,
        args.normalization, args.activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-5)
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        supervised_losses = []
        deployment_losses = []
        consistency_losses = []
        closure_work_losses = []
        closure_rms_losses = []
        for batch_state, batch_target in train_loader:
            batch_state = batch_state.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            dummy_k = torch.zeros(len(batch_state), device=device)
            output = None
            supervised_loss = torch.zeros((), device=device)
            if args.supervised_loss_weight > 0 or args.consistency_weight > 0:
                output = model(batch_state, dummy_k)
            if args.supervised_loss_weight > 0:
                pointwise = torch.mean((output - batch_target) ** 2)
                output_hat = torch.fft.rfft(output.float(), dim=-1)
                target_hat = torch.fft.rfft(batch_target.float(), dim=-1)
                spectral = torch.mean(torch.abs(output_hat - target_hat) ** 2) / torch.clamp(
                    torch.mean(torch.abs(target_hat) ** 2), min=1.0e-8
                )
                supervised_loss = pointwise + 0.1 * spectral
            deployment_loss = torch.zeros((), device=device)
            consistency_loss = torch.zeros((), device=device)
            closure_work_loss = torch.zeros((), device=device)
            closure_rms_loss = torch.zeros((), device=device)
            if (
                args.deployment_loss_weight > 0 or args.consistency_weight > 0
                or args.closure_work_weight > 0 or args.closure_rms_weight > 0
            ):
                filtered_state = spectral_filter(batch_state, args.deployment_mode)
                filtered_target = spectral_filter(batch_target, args.deployment_mode)
                deployment_output = spectral_filter(
                    model(filtered_state, dummy_k), args.deployment_mode
                )
                if args.deployment_loss_weight > 0:
                    deployment_pointwise = torch.mean((deployment_output - filtered_target) ** 2)
                    deployment_output_hat = torch.fft.rfft(deployment_output.float(), dim=-1)
                    filtered_target_hat = torch.fft.rfft(filtered_target.float(), dim=-1)
                    deployment_spectral = torch.mean(
                        torch.abs(deployment_output_hat - filtered_target_hat) ** 2
                    ) / torch.clamp(
                        torch.mean(torch.abs(filtered_target_hat) ** 2), min=1.0e-8
                    )
                    deployment_loss = deployment_pointwise + 0.1 * deployment_spectral
                if args.consistency_weight > 0:
                    consistency_loss = torch.mean(
                        (spectral_filter(output, args.deployment_mode) - deployment_output) ** 2
                    )
                if args.closure_work_weight > 0 or args.closure_rms_weight > 0:
                    physical_state = filtered_state * input_std_tensor + input_mean_tensor
                    temperature_perturbation = physical_state[:, 2] - physical_state[:, 0]
                    physical_output = deployment_output * target_std + target_mean
                    physical_target = filtered_target * target_std + target_mean
                    if args.closure_work_weight > 0:
                        predicted_work = torch.mean(
                            temperature_perturbation * physical_output, dim=-1
                        )
                        target_work = torch.mean(
                            temperature_perturbation * physical_target, dim=-1
                        )
                        closure_work_loss = torch.mean(
                            (predicted_work - target_work) ** 2
                        ) / torch.clamp(torch.mean(target_work ** 2), min=1.0e-12)
                    if args.closure_rms_weight > 0:
                        predicted_power = torch.mean(physical_output ** 2, dim=-1)
                        target_power = torch.mean(physical_target ** 2, dim=-1)
                        closure_rms_loss = torch.mean(
                            (predicted_power - target_power) ** 2
                        ) / torch.clamp(torch.mean(target_power ** 2), min=1.0e-12)
            loss = (
                args.supervised_loss_weight * supervised_loss
                + args.deployment_loss_weight * deployment_loss
                + args.consistency_weight * consistency_loss
                + args.closure_work_weight * closure_work_loss
                + args.closure_rms_weight * closure_rms_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised_loss.detach()))
            deployment_losses.append(float(deployment_loss.detach()))
            consistency_losses.append(float(consistency_loss.detach()))
            closure_work_losses.append(float(closure_work_loss.detach()))
            closure_rms_losses.append(float(closure_rms_loss.detach()))
        scheduler.step()
        normalized_prediction = predict(model, state[validation_index], args.batch_size, device)
        physical_prediction = normalized_prediction * target_std + target_mean
        validation = metrics(physical_prediction, trajectory.heat_flux_gradient[validation_index])
        deployment_validation = None
        if args.deployment_mode is not None:
            filtered_state = spectral_filter(
                torch.from_numpy(state[validation_index]), args.deployment_mode
            ).numpy()
            normalized_deployment_prediction = predict(
                model, filtered_state, args.batch_size, device
            )
            physical_deployment_prediction = normalized_deployment_prediction * target_std + target_mean
            physical_deployment_prediction = spectral_filter(
                torch.from_numpy(physical_deployment_prediction), args.deployment_mode
            ).numpy()
            filtered_validation_target = spectral_filter(
                torch.from_numpy(trajectory.heat_flux_gradient[validation_index]),
                args.deployment_mode,
            ).numpy()
            deployment_validation = metrics(
                physical_deployment_prediction, filtered_validation_target
            )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "supervised_train_loss": float(np.mean(supervised_losses)),
            "deployment_train_loss": float(np.mean(deployment_losses)),
            "consistency_train_loss": float(np.mean(consistency_losses)),
            "closure_work_train_loss": float(np.mean(closure_work_losses)),
            "closure_rms_train_loss": float(np.mean(closure_rms_losses)),
            "validation_relative_l2": validation["relative_l2"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if deployment_validation is not None:
            row["deployment_validation_relative_l2"] = deployment_validation["relative_l2"]
        history.append(row)
        print(json.dumps(row), flush=True)
        selection_value = (
            deployment_validation["relative_l2"]
            if args.selection_metric == "deployment" else validation["relative_l2"]
        )
        if selection_value < best_value:
            best_value = selection_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    prediction = predict(model, state, args.batch_size, device) * target_std + target_mean
    deployment_evaluation = None
    deployment_prediction = None
    if args.deployment_mode is not None:
        filtered_state = spectral_filter(torch.from_numpy(state), args.deployment_mode).numpy()
        deployment_prediction = predict(model, filtered_state, args.batch_size, device)
        deployment_prediction = deployment_prediction * target_std + target_mean
        deployment_prediction = spectral_filter(
            torch.from_numpy(deployment_prediction), args.deployment_mode
        ).numpy()
        deployment_target = spectral_filter(
            torch.from_numpy(trajectory.heat_flux_gradient), args.deployment_mode
        ).numpy()
        deployment_evaluation = {
            "train": metrics(deployment_prediction[train_index], deployment_target[train_index]),
            "validation": metrics(
                deployment_prediction[validation_index], deployment_target[validation_index]
            ),
            "test": metrics(deployment_prediction[test_index], deployment_target[test_index]),
        }
    summary = {
        "protocol": args.protocol,
        "architecture": {
            "normalization": args.normalization,
            "activation": args.activation,
        },
        "stability_constraint": {
            "deployment_mode": args.deployment_mode,
            "supervised_loss_weight": args.supervised_loss_weight,
            "deployment_loss_weight": args.deployment_loss_weight,
            "consistency_weight": args.consistency_weight,
            "closure_work_weight": args.closure_work_weight,
            "closure_rms_weight": args.closure_rms_weight,
            "selection_metric": args.selection_metric,
        },
        "source": str(args.mat),
        "source_kind": trajectory.source_kind,
        "moment_definition": trajectory.moment_definition,
        "split_counts": {"train": len(train_index), "validation": len(validation_index), "test": len(test_index)},
        "normalization": {"input_mean": input_mean.tolist(), "input_std": input_std.tolist(), "target_mean": target_mean, "target_std": target_std},
        "best_validation_relative_l2": best_value,
        "best_selection_value": best_value,
        "train": metrics(prediction[train_index], trajectory.heat_flux_gradient[train_index]),
        "validation": metrics(prediction[validation_index], trajectory.heat_flux_gradient[validation_index]),
        "test": metrics(prediction[test_index], trajectory.heat_flux_gradient[test_index]),
        "epochs_completed": len(history),
    }
    if deployment_evaluation is not None:
        summary["deployment_evaluation"] = deployment_evaluation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": "huang2025_gkeyll_reproduction",
        "protocol": args.protocol,
        "model_arguments": {
            "state_channels": 3, "include_k": False, "width": args.width,
            "modes": args.modes, "layers": args.layers,
            "enforce_zero_mean": True, "normalization": args.normalization,
            "activation": args.activation,
        },
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        **summary,
    }, args.output_dir / "best.pt")
    np.savez_compressed(
        args.output_dir / "evaluation.npz", time=trajectory.time, x_over_l=trajectory.x_over_l,
        target=trajectory.heat_flux_gradient, prediction=prediction,
        **({"deployment_prediction": deployment_prediction} if deployment_prediction is not None else {}),
        train_index=train_index, validation_index=validation_index, test_index=test_index,
    )
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "history.json", history)
    figure_stop = min(len(trajectory.time), int(np.searchsorted(trajectory.time, 40.0, side="right")))
    extent = (trajectory.time[0], trajectory.time[figure_stop - 1], 0.0, 1.0)
    figure_truth = trajectory.heat_flux_gradient[:figure_stop]
    figure_prediction = prediction[:figure_stop]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    scale = float(np.quantile(np.abs(figure_truth), 0.995))
    for axis, value, title in zip(axes[:2], (figure_truth, figure_prediction), ("Gkeyll truth", "FNO")):
        image = axis.imshow(value.T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set_title(title); axis.set_xlabel("time")
    error = np.abs(figure_prediction - figure_truth)
    error_image = axes[2].imshow(error.T, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0, vmax=float(np.quantile(error, 0.995)))
    axes[2].set_title("absolute error"); axes[2].set_xlabel("time"); axes[0].set_ylabel("x/L")
    fig.colorbar(image, ax=axes[:2], shrink=0.8, label=r"$\partial_x q$")
    fig.colorbar(error_image, ax=axes[2], shrink=0.8, label=r"$|\Delta\partial_x q|$")
    fig.subplots_adjust(wspace=0.18, right=0.96)
    fig.savefig(args.output_dir / "closure_xt.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
