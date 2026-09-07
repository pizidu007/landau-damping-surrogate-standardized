"""Plot the casewise Gkeyll/FNO round-2 evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


CASES = ("k0p350_a0p075", "k0p400_a0p100")
LABELS = {
    "k0p350_a0p075": r"$k=0.35,\ A=0.075$",
    "k0p400_a0p100": r"$k=0.40,\ A=0.10$",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )


def _field_energy(rollout_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True, constrained_layout=True)
    for axis, identifier in zip(axes, CASES):
        supervised = np.load(rollout_root / "supervised" / identifier / "rollout.npz")
        finetuned = np.load(rollout_root / "finetuned" / identifier / "rollout.npz")
        stride = max(1, len(supervised["time"]) // 2000)
        selection = slice(None, None, stride)
        axis.semilogy(
            supervised["time"][selection],
            np.maximum(supervised["truth_field_energy"][selection], 1.0e-12),
            color="black", linewidth=1.7, label="Gkeyll truth",
        )
        axis.semilogy(
            supervised["time"][selection],
            np.maximum(supervised["field_energy"][selection], 1.0e-12),
            linewidth=1.15, label="supervised FNO",
        )
        axis.semilogy(
            finetuned["time"][selection],
            np.maximum(finetuned["field_energy"][selection], 1.0e-12),
            linewidth=1.15, label="50-step fine-tuned FNO",
        )
        axis.set_title(LABELS[identifier])
        axis.set_xlabel("physical time")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel(r"field energy $\int E_x^2\,dx$")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Held-out cases: long-time field-energy evolution")
    fig.savefig(output_dir / "heldout_long_time_field_energy.png", bbox_inches="tight")
    plt.close(fig)


def _offline_metrics(supervised_summary: Path, rollout_root: Path, output_dir: Path) -> None:
    summary = json.loads(supervised_summary.read_text(encoding="utf-8"))
    rows = summary["metrics"]["test"]["cases"]
    identifiers = [row["case_id"] for row in rows]
    offline = [row["relative_l2"] for row in rows]
    supervised_long = []
    finetuned_long = []
    for identifier in identifiers:
        supervised_long.append(
            json.loads(
                (rollout_root / "supervised" / identifier / "summary.json").read_text()
            )["field_energy_log10_rmse"]
        )
        finetuned_long.append(
            json.loads(
                (rollout_root / "finetuned" / identifier / "summary.json").read_text()
            )["field_energy_log10_rmse"]
        )
    x = np.arange(len(identifiers))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
    axes[0].bar(x, offline, width=0.55, color="tab:blue")
    axes[0].axhline(summary["metrics"]["test"]["relative_l2"], color="black", ls="--", lw=1)
    axes[0].set_title("Instantaneous closure")
    axes[0].set_ylabel("relative L2 error")
    axes[0].set_xticks(x, [LABELS[value] for value in identifiers])
    for index, value in enumerate(offline):
        axes[0].text(index, value, f"{value:.3f}", ha="center", va="bottom")
    width = 0.34
    axes[1].bar(x - width / 2, supervised_long, width, label="supervised")
    axes[1].bar(x + width / 2, finetuned_long, width, label="50-step fine-tuned")
    axes[1].set_title(r"Long rollout to $t=40$")
    axes[1].set_ylabel("field-energy log10 RMSE")
    axes[1].set_xticks(x, [LABELS[value] for value in identifiers])
    axes[1].legend(frameon=False)
    for offset, values in ((-width / 2, supervised_long), (width / 2, finetuned_long)):
        for index, value in enumerate(values):
            axes[1].text(index + offset, value, f"{value:.3f}", ha="center", va="bottom")
    fig.suptitle("Held-out-case performance: offline accuracy versus rollout fidelity")
    fig.savefig(output_dir / "heldout_metrics.png", bbox_inches="tight")
    plt.close(fig)


def _phase_space(dataset_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.1), sharex=True, sharey=True, constrained_layout=True)
    for row, identifier in enumerate(CASES):
        path = dataset_root / "cases" / identifier / "processed" / "trajectory.h5"
        with h5py.File(path, "r") as handle:
            x = np.asarray(handle["phase/x"])
            v = np.asarray(handle["phase/v"])
            times = np.asarray(handle["phase/time"])
            phase = np.asarray(handle["phase/f"])
        x_normalized = (x - x.min()) / (x.max() - x.min())
        velocity = (v >= -3.0) & (v <= 3.0)
        perturbation = phase - phase.mean(axis=1, keepdims=True)
        visible = perturbation[:, :, velocity]
        limit = float(np.quantile(np.abs(visible), 0.995))
        for column, axis in enumerate(axes[row]):
            image = axis.pcolormesh(
                x_normalized,
                v[velocity],
                visible[column].T,
                shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
            )
            axis.set_title(f"t={times[column]:.1f}")
            axis.set_ylim(-3.0, 3.0)
            if column == 0:
                axis.set_ylabel(f"{LABELS[identifier]}\nvelocity v")
            if row == 1:
                axis.set_xlabel("x/L")
        fig.colorbar(image, ax=axes[row], shrink=0.75, label=r"$f-\langle f\rangle_x$")
    fig.suptitle("Held-out Gkeyll phase-space perturbation (cropped to -3 ≤ v ≤ 3)")
    fig.savefig(output_dir / "heldout_phase_space.png", bbox_inches="tight")
    plt.close(fig)

    for identifier in CASES:
        path = dataset_root / "cases" / identifier / "processed" / "trajectory.h5"
        with h5py.File(path, "r") as handle:
            x = np.asarray(handle["phase/x"])
            v = np.asarray(handle["phase/v"])
            times = np.asarray(handle["phase/time"])
            phase = np.asarray(handle["phase/f"])
        x_normalized = (x - x.min()) / (x.max() - x.min())
        velocity = (v >= -3.0) & (v <= 3.0)
        perturbation = phase - phase.mean(axis=1, keepdims=True)
        visible = perturbation[:, :, velocity]
        limit = float(np.quantile(np.abs(visible), 0.995))
        figure, axis = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
        mesh = axis.pcolormesh(
            x_normalized, v[velocity], visible[0].T,
            shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
        )
        title = axis.set_title(f"{LABELS[identifier]}, t={times[0]:.1f}")
        axis.set(xlabel="x/L", ylabel="velocity v", ylim=(-3.0, 3.0))
        figure.colorbar(mesh, ax=axis, label=r"$f-\langle f\rangle_x$")

        def update(frame: int):
            mesh.set_array(visible[frame].T.ravel())
            title.set_text(f"{LABELS[identifier]}, t={times[frame]:.1f}")
            return mesh, title

        movie = animation.FuncAnimation(figure, update, frames=len(times), interval=650, blit=False)
        movie.save(output_dir / f"{identifier}_phase_space.gif", writer="pillow", fps=1.5)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--supervised-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    _field_energy(args.rollout_root, args.output_dir)
    _offline_metrics(args.supervised_summary, args.rollout_root, args.output_dir)
    _phase_space(args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
