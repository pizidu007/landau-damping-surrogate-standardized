"""Visualize supervised closure learning and held-out continuum_v1 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from landau_surrogate.data.continuum_v1 import (
    ContinuumCase,
    load_continuum_case,
    load_continuum_case_index,
)
from landau_surrogate.models.closure_fno1d import ClosureFNO1d


SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")
REGIME_MARKERS = {"weak": "o", "transition": "s", "strong_nonlinear": "^"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_learning_curves(result_root: Path, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.35))
    for seed, color in enumerate(SEED_COLORS):
        history = read_json(result_root / f"seed{seed}" / "history.json")
        summary = read_json(result_root / f"seed{seed}" / "summary.json")
        epoch = np.asarray([row["epoch"] for row in history])
        pointwise = np.asarray([row["train_pointwise_mse"] for row in history])
        validation = 100.0 * np.asarray(
            [row["validation_relative_l2"] for row in history]
        )
        axes[0].semilogy(epoch, pointwise, color=color, lw=1.8, label=f"seed {seed}")
        axes[1].plot(epoch, validation, color=color, lw=1.8, label=f"seed {seed}")
        best_epoch = int(summary["best_epoch"])
        axes[1].scatter(
            best_epoch,
            100.0 * float(summary["best_validation_relative_l2"]),
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=42,
            zorder=3,
        )
    axes[0].set_title("Supervised training loss")
    axes[0].set_ylabel("normalized pointwise MSE")
    axes[1].set_title("Whole-trajectory validation")
    axes[1].set_ylabel("relative $L_2$ error (%)")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(True, which="both", alpha=0.24)
        axis.legend(frameon=False)
    figure.suptitle("Continuum-v1 pure FNO closure: width=128, modes=32")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves_3seed.png", dpi=210)
    plt.close(figure)


def casewise_seed_medians(result_root: Path) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for seed in range(3):
        cases = read_json(result_root / f"seed{seed}" / "summary.json")["metrics"][
            "test"
        ]["cases"]
        for case in cases:
            item = by_case.setdefault(
                case["case_id"],
                {
                    "case_id": case["case_id"],
                    "K": float(case["K"]),
                    "alpha": float(case["alpha"]),
                    "regime": case["regime"],
                    "relative_l2_by_seed": [],
                },
            )
            item["relative_l2_by_seed"].append(float(case["relative_l2"]))
    rows = []
    for item in by_case.values():
        values = item["relative_l2_by_seed"]
        if len(values) != 3:
            raise ValueError(f"Expected three seed values for {item['case_id']}")
        rows.append(
            {
                **item,
                "relative_l2_median": float(np.median(values)),
                "relative_l2_minimum": float(np.min(values)),
                "relative_l2_maximum": float(np.max(values)),
            }
        )
    return sorted(rows, key=lambda row: row["case_id"])


def plot_generalization(result_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    summaries = [
        read_json(result_root / f"seed{seed}" / "summary.json")
        for seed in range(3)
    ]
    case_rows = casewise_seed_medians(result_root)
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    split_names = ("train", "validation", "test")
    x = np.arange(len(split_names))
    for seed, (summary, color) in enumerate(zip(summaries, SEED_COLORS, strict=True)):
        values = [100.0 * summary["metrics"][name]["relative_l2"] for name in split_names]
        axes[0].plot(x, values, marker="o", lw=1.6, color=color, label=f"seed {seed}")
    axes[0].set_xticks(x, ("Train", "Validation", "Held-out test"))
    axes[0].set_ylabel("global relative $L_2$ error (%)")
    axes[0].set_title("Generalization gap")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].text(
        0.97,
        0.04,
        "Test trajectories were not used for model selection",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    values = np.asarray([100.0 * row["relative_l2_median"] for row in case_rows])
    color_norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    color_map = plt.get_cmap("viridis")
    for regime, marker in REGIME_MARKERS.items():
        selected = [row for row in case_rows if row["regime"] == regime]
        axes[1].scatter(
            [row["K"] for row in selected],
            [row["alpha"] for row in selected],
            c=[100.0 * row["relative_l2_median"] for row in selected],
            norm=color_norm,
            cmap=color_map,
            marker=marker,
            s=62,
            edgecolor="black",
            linewidth=0.45,
            label=regime.replace("_", " "),
        )
    for row in sorted(
        case_rows, key=lambda item: item["relative_l2_median"], reverse=True
    )[:3]:
        axes[1].annotate(
            f"{100.0 * row['relative_l2_median']:.1f}%",
            (row["K"], row["alpha"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("fundamental wavenumber $K$")
    axes[1].set_ylabel("initial amplitude $A$")
    axes[1].set_title("Held-out case error over parameter space")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map), ax=axes[1], pad=0.02
    )
    colorbar.set_label("casewise relative $L_2$ error: 3-seed median (%)")
    figure.suptitle("Cross-case generalization of the supervised closure")
    figure.tight_layout()
    figure.savefig(output_dir / "heldout_generalization.png", dpi=210)
    plt.close(figure)
    return case_rows


@torch.no_grad()
def predict_case(
    case: ContinuumCase,
    *,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | float | str]:
    data_config = checkpoint["config"]["data"]
    trajectory = load_continuum_case(
        case,
        input_maximum_mode=data_config.get("input_maximum_mode"),
        target_maximum_mode=data_config.get("target_maximum_mode"),
        temporal_stride=int(data_config.get("temporal_stride", 1)),
    )
    model = ClosureFNO1d(**checkpoint["model_arguments"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    normalization = checkpoint["normalization"]
    mean = np.asarray(normalization["input_mean"], dtype=np.float32)[None, :, None]
    std = np.asarray(normalization["input_std"], dtype=np.float32)[None, :, None]
    state = np.ascontiguousarray((trajectory.state - mean) / std, dtype=np.float32)
    normalized_k = (
        trajectory.K - float(normalization["k_mean"])
    ) / float(normalization["k_std"])
    predictions = []
    for start in range(0, len(state), batch_size):
        current = torch.from_numpy(state[start : start + batch_size]).to(device)
        k_value = torch.full(
            (len(current),), normalized_k, dtype=torch.float32, device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(current, k_value)
            maximum_mode = data_config.get("target_maximum_mode")
            if maximum_mode is not None:
                transformed = torch.fft.rfft(output.float(), dim=-1)
                transformed[..., int(maximum_mode) + 1 :] = 0.0
                output = torch.fft.irfft(
                    transformed, n=output.shape[-1], dim=-1
                )
        predictions.append(output.float().cpu().numpy())
    prediction = np.concatenate(predictions)
    prediction = (
        prediction * float(normalization["gradient_std"])
        + float(normalization["gradient_mean"])
    )
    target = trajectory.heat_flux_gradient
    error = prediction - target
    return {
        "case_id": case.case_id,
        "K": case.K,
        "alpha": case.alpha,
        "regime": case.regime,
        "time": trajectory.time,
        "x_over_l": trajectory.x_over_l,
        "target": target,
        "prediction": prediction,
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(target)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": float(np.corrcoef(target.ravel(), prediction.ravel())[0, 1]),
    }


def plot_field_comparison(
    predictions: list[dict[str, Any]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(
        len(predictions), 4, figsize=(16.0, 4.25 * len(predictions)), squeeze=False
    )
    for row_index, result in enumerate(predictions):
        time = result["time"]
        x = result["x_over_l"]
        target = result["target"]
        prediction = result["prediction"]
        error = prediction - target
        shared_scale = max(
            float(np.quantile(np.abs(np.concatenate((target.ravel(), prediction.ravel()))), 0.995)),
            1.0e-10,
        )
        error_scale = max(float(np.quantile(np.abs(error), 0.995)), 1.0e-10)
        for column, (value, title, scale) in enumerate(
            (
                (target, r"Gkeyll truth: $\partial_xq$", shared_scale),
                (prediction, r"FNO prediction: $\partial_xq$", shared_scale),
                (error, "FNO $-$ truth", error_scale),
            )
        ):
            image = axes[row_index, column].pcolormesh(
                x,
                time,
                value,
                shading="auto",
                cmap="RdBu_r",
                vmin=-scale,
                vmax=scale,
                rasterized=True,
            )
            if column == 0:
                title = (
                    f"K={result['K']:.3f}, A={result['alpha']:.3f}\n" + title
                )
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xlabel("x/L")
            figure.colorbar(image, ax=axes[row_index, column], pad=0.02)
        axes[row_index, 0].set_ylabel("time")

        spatial_target = np.sqrt(np.mean(target**2, axis=1))
        spatial_prediction = np.sqrt(np.mean(prediction**2, axis=1))
        spatial_error = np.sqrt(np.mean(error**2, axis=1))
        axes[row_index, 3].semilogy(
            time, spatial_target, color="black", lw=1.6, label="truth RMS"
        )
        axes[row_index, 3].semilogy(
            time, spatial_prediction, color="#D55E00", lw=1.35, label="FNO RMS"
        )
        axes[row_index, 3].semilogy(
            time, spatial_error, color="#0072B2", lw=1.1, label="error RMS"
        )
        axes[row_index, 3].set_xlabel("time")
        axes[row_index, 3].set_ylabel("spatial RMS")
        axes[row_index, 3].set_title(
            f"rel. $L_2$={100.0 * result['relative_l2']:.2f}%, "
            f"corr.={result['correlation']:.4f}"
        )
        upper_limit = 1.35 * max(
            float(spatial_target.max()),
            float(spatial_prediction.max()),
            float(spatial_error.max()),
        )
        axes[row_index, 3].set_ylim(upper_limit * 1.0e-5, upper_limit)
        axes[row_index, 3].grid(True, which="both", alpha=0.22)
        axes[row_index, 3].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Held-out trajectories: validation-selected seed 1 (common truth/FNO color scale)"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "heldout_closure_fields.png", dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=["K0p400_a0p084", "K0p280_a0p200"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(args.result_root, args.output_dir)
    case_rows = plot_generalization(args.result_root, args.output_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA visualization requested but CUDA is unavailable")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoint_path = args.result_root / "seed1" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cases = {case.case_id: case for case in load_continuum_case_index(args.dataset_root)}
    predictions = []
    for case_id in args.case_ids:
        if case_id not in cases or cases[case_id].split != "test":
            raise ValueError(f"{case_id} is not an eligible held-out test case")
        predictions.append(
            predict_case(
                cases[case_id],
                checkpoint=checkpoint,
                device=device,
                batch_size=args.batch_size,
            )
        )
    plot_field_comparison(predictions, args.output_dir)
    for result in predictions:
        np.savez_compressed(
            args.output_dir / f"{result['case_id']}_seed1_prediction.npz",
            time=result["time"],
            x_over_l=result["x_over_l"],
            target=result["target"],
            prediction=result["prediction"],
        )
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_by": "minimum validation relative_l2 across seeds",
        "casewise_test_metrics": case_rows,
        "visualized_cases": [
            {
                key: result[key]
                for key in (
                    "case_id",
                    "K",
                    "alpha",
                    "regime",
                    "relative_l2",
                    "rmse",
                    "correlation",
                )
            }
            for result in predictions
        ],
    }
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["visualized_cases"], indent=2))


if __name__ == "__main__":
    main()
