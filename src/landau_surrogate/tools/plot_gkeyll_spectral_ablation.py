"""Compare spectral filters for the strict single-case Gkeyll rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


def _through(values: dict[str, np.ndarray], maximum_time: float) -> dict[str, np.ndarray]:
    keep = values["time"] <= maximum_time + 1.0e-9
    return {name: array[keep] for name, array in values.items()}


def _metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    prediction = values["prediction"]
    truth = values["truth"]
    field = values["field_energy"]
    truth_field = values["truth_field_energy"]
    floor = max(float(truth_field[0]) * 1.0e-10, 1.0e-14)
    result = {
        f"{name}_relative_l2": float(
            np.linalg.norm(prediction[:, index] - truth[:, index])
            / np.linalg.norm(truth[:, index])
        )
        for index, name in enumerate(("density", "velocity", "pressure"))
    }
    result["field_energy_log10_rmse"] = float(
        np.sqrt(
            np.mean(
                (
                    np.log10(np.maximum(field, floor))
                    - np.log10(np.maximum(truth_field, floor))
                )
                ** 2
            )
        )
    )
    fluctuation = prediction - prediction.mean(axis=-1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(fluctuation, axis=-1)) ** 2
    result["state_power_fraction_modes_ge_17"] = float(
        spectrum[:, :, 17:].sum() / max(float(spectrum[:, :, 1:].sum()), 1.0e-30)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode24", type=Path, required=True)
    parser.add_argument("--mode16", type=Path, required=True)
    parser.add_argument("--mode8-short", type=Path, required=True)
    parser.add_argument("--mode8-long", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load(args.mode24)
    cases = {
        "mode-24": _through(baseline, 20.0),
        "mode-16": _load(args.mode16),
        "mode-8": _load(args.mode8_short),
    }
    colors = {"mode-24": "#ca6f1e", "mode-16": "#7d3c98", "mode-8": "#117a65"}
    truth = cases["mode-8"]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    axes[0].semilogy(
        truth["time"], truth["truth_field_energy"], color="black",
        linewidth=2.0, label="Gkeyll truth",
    )
    floor = max(float(truth["truth_field_energy"][0]) * 1.0e-10, 1.0e-14)
    for label, values in cases.items():
        axes[0].semilogy(
            values["time"], values["field_energy"], color=colors[label],
            linewidth=1.35, label=f"Fluid + FNO ({label})",
        )
        error = np.abs(
            np.log10(np.maximum(values["field_energy"], floor))
            - np.log10(np.maximum(values["truth_field_energy"], floor))
        )
        axes[1].plot(values["time"], error, color=colors[label], linewidth=1.2, label=label)
    axes[0].set_ylabel(r"$\frac{1}{2}\int |E_x|^2\,dx$")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].set(xlabel="time", ylabel=r"$|\Delta\log_{10} E|$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "spectral_ablation_t20.png", dpi=180)
    plt.close(fig)

    final = _load(args.mode8_long)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    axes[0].semilogy(
        final["time"], final["truth_field_energy"], color="black",
        linewidth=2.0, label="Gkeyll truth",
    )
    for label, values in (("mode-24", baseline), ("mode-8", final)):
        axes[0].semilogy(
            values["time"], values["field_energy"], color=colors[label],
            linewidth=1.35, label=f"Fluid + FNO ({label})",
        )
        axes[1].plot(
            values["time"], np.abs(values["field_energy"] - values["truth_field_energy"]),
            color=colors[label], linewidth=1.2, label=label,
        )
    axes[0].set_ylabel(r"$\frac{1}{2}\int |E_x|^2\,dx$")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].set(xlabel="time", ylabel="absolute field-energy error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "field_energy_mode24_vs_mode8_t40.png", dpi=180)
    plt.close(fig)

    time = final["time"]
    prediction = final["prediction"]
    target = final["truth"]
    extent = (float(time[0]), float(time[-1]), 0.0, 1.0)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
    for row, name in enumerate(("density", "velocity", "pressure")):
        offset = 1.0 if name in ("density", "pressure") else 0.0
        difference = prediction[:, row] - target[:, row]
        scale = float(np.quantile(np.abs(target[:, row] - offset), 0.995))
        error_scale = max(float(np.quantile(np.abs(difference), 0.995)), 1.0e-12)
        for column, value in enumerate(
            (target[:, row] - offset, prediction[:, row] - offset, difference)
        ):
            bound = scale if column < 2 else error_scale
            axes[row, column].imshow(
                value.T, origin="lower", extent=extent, aspect="auto",
                cmap="RdBu_r", vmin=-bound, vmax=bound,
            )
        axes[row, 0].set_ylabel(f"{name}\nx/L")
    for axis, title in zip(axes[0], ("Gkeyll truth", "Fluid + FNO (mode-8)", "prediction - truth")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("time")
    fig.tight_layout()
    fig.savefig(args.output_dir / "closed_loop_mode8_moments_xt.png", dpi=180)
    plt.close(fig)

    summary = {label: _metrics(values) for label, values in cases.items()}
    summary["mode-8-t40"] = _metrics(final)
    (args.output_dir / "spectral_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
