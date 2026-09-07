"""Physical-time caches and causal history for continuum closure experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.data.continuum_v1 import load_continuum_case, spectral_lowpass
from landau_surrogate.tools.audit_continuum_v1_closure_oracle import interpolate, electric_numpy


@dataclass
class HistoryCache:
    cases: list
    state: torch.Tensor  # case,time,channel,x; dimensional fields
    gradient: torch.Tensor  # case,time,x
    dt: float
    phases: list[list[tuple[float, float]]]

    @property
    def end(self):
        return (self.state.shape[1] - 1) * self.dt

    def sample(self, indices: torch.Tensor, times: torch.Tensor, field="state"):
        """Per-case, physical-time interpolation; negative time pads the initial state."""
        values = getattr(self, field)
        clipped = times.to(dtype=values.dtype).clamp(0, self.end)
        position = clipped / self.dt
        lower = position.floor().long().clamp(0, values.shape[1] - 2)
        weight = position - lower
        # At a grid point, do not fetch a future frame with nominal zero weight.
        # In particular, a t=0 deployment must work when all later data are absent/NaN.
        upper = torch.where(weight > 0, lower + 1, lower)
        while weight.ndim < values.ndim - 1:
            weight = weight.unsqueeze(-1)
        return values[indices, lower] * (1 - weight) + values[indices, upper] * weight


def phase_windows(state: np.ndarray, time: np.ndarray):
    """Four envelope-relative sampling regions, including the initial-condition region."""
    energy = np.mean(state[:, 3].astype(np.float64) ** 2, axis=-1)
    # RMS-like envelope over two plasma oscillations, rather than individual field nulls.
    width = max(3, int(round(6 / (time[1] - time[0]))))
    padded = np.pad(energy, (width // 2, width - 1 - width // 2), mode="edge")
    envelope = np.convolve(padded, np.ones(width) / width, mode="valid")
    lo = np.searchsorted(time, 5.)
    hi = np.searchsorted(time, time[-1] - 5.)
    minimum = lo + int(np.argmin(envelope[lo:hi]))
    turning = float(time[minimum])
    return [(0., max(0.1, turning - 3)), (max(0., turning - 3), min(time[-1], turning + 3)),
            (min(time[-1], turning + 3), min(time[-1], turning + 15)),
            (min(time[-1], turning + 15), float(time[-1]))]


def load_history_cache(cases, maximum_mode, device, dt=.02, disk_cache: Path | None = None,
                       evaluation_only=False):
    if not evaluation_only and any(c.split not in ("train","validation") for c in cases):
        raise ValueError("Training cache can include only train/validation trajectories")
    frames, gradients, phases = [], [], []
    grid = np.arange(round(80 / dt) + 1, dtype=np.float64) * dt
    for index, case in enumerate(cases):
        target = None if disk_cache is None else disk_cache / f"mode{maximum_mode}_{case.case_id}.npz"
        if target is not None and target.exists():
            with np.load(target) as saved:
                if not np.isclose(float(saved["dt"]), dt):
                    raise ValueError("Cache time spacing differs")
                state, gradient = saved["state"], saved["gradient"]
        else:
            trajectory = load_continuum_case(case)
            moments = interpolate(spectral_lowpass(trajectory.state, maximum_mode), trajectory.time, grid)
            electric = electric_numpy(moments[:, None], np.array([case.K]))[:, 0]
            state = np.concatenate((moments, electric[:, None]), axis=1).astype(np.float32)
            gradient = interpolate(spectral_lowpass(trajectory.heat_flux_gradient, maximum_mode), trajectory.time, grid)
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, state=state, gradient=gradient, dt=np.array(dt))
        frames.append(state)
        gradients.append(gradient)
        phases.append(phase_windows(state, grid))
        if (index + 1) % 20 == 0 or index + 1 == len(cases):
            print(f"history cache: {index+1}/{len(cases)}", flush=True)
    return HistoryCache(cases, torch.as_tensor(np.stack(frames), device=device),
                        torch.as_tensor(np.stack(gradients), device=device), dt, phases)


def normalization(cache: HistoryCache):
    if any(case.split != "train" for case in cache.cases):
        raise ValueError("Normalization must use train trajectories only")
    mean = cache.state.mean(dim=(0, 1, 3))
    std = cache.state.std(dim=(0, 1, 3), correction=0).clamp_min(1e-6)
    k = torch.tensor([c.K for c in cache.cases], dtype=torch.float64)
    alpha = torch.tensor([c.alpha for c in cache.cases], dtype=torch.float64)
    return {"input_mean": mean.cpu().tolist(), "input_std": std.cpu().tolist(),
            "gradient_std": float(cache.gradient.std(correction=0)), "gradient_mean": 0.,
            "k_mean": float(k.mean()), "k_std": float(k.std(correction=0)),
            "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std(correction=0)),
            "source_split": "train", "case_count": len(cache.cases)}


def history_offsets(span: float):
    if span < 0:
        raise ValueError("history span must be nonnegative")
    return np.array([1., .75, .5, .25, .125, .0625, .03125, 0.]) * span


def supervised_history(cache: HistoryCache, indices, times, span):
    fields, valid = [], []
    for age in history_offsets(span):
        query = times - float(age)
        fields.append(cache.sample(indices, query))
        valid.append(query >= -1e-7)
    return torch.stack(fields, dim=1), torch.stack(valid, dim=1)


__all__ = ["HistoryCache", "load_history_cache", "normalization", "history_offsets", "supervised_history"]
