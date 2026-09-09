"""Lossless, per-trajectory closure arrays for machines without the kinetic dataset."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PortableCase:
    case_id: str
    K: float
    alpha: float
    split: str
    regime: str
    path: Path
    sha256: str
    starter: bool


def load_manifest(root):
    result = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    if result["schema"] != "landau_pc_closure_v1":
        raise ValueError("Unsupported portable dataset")
    return result


def load_index(root, split="train", subset="starter"):
    if split not in ("train", "validation", "test") or subset not in ("starter", "full"):
        raise ValueError("Choose an existing split and starter/full subset")
    root = Path(root).resolve()
    cases = []
    for item in load_manifest(root)["cases"]:
        if item["split"] != split or (subset == "starter" and not item["starter"]):
            continue
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Case path escapes dataset root")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {item['case_id']}; extract its extension archive: {path}")
        cases.append(PortableCase(**{k: item[k] for k in
                     ("case_id", "K", "alpha", "split", "regime", "sha256", "starter")}, path=path))
    if not cases:
        raise ValueError(f"No cases for {split}/{subset}")
    return cases


def load_case(case, *, verify=False):
    if verify and hashlib.sha256(case.path.read_bytes()).hexdigest() != case.sha256:
        raise ValueError(f"SHA256 mismatch: {case.case_id}")
    with np.load(case.path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    state, gradient, time = arrays["state"], arrays["gradient"], arrays["time"]
    if (state.ndim != 3 or state.shape[1] != 4 or gradient.shape != (len(state), state.shape[-1])
            or time.shape != (len(state),) or state.dtype != np.float32 or gradient.dtype != np.float32):
        raise ValueError("Invalid state/gradient contract")
    if (not np.isfinite(state).all() or not np.isfinite(gradient).all()
            or not np.allclose(np.diff(time), float(arrays["dt"]), atol=1e-10, rtol=0)):
        raise ValueError("Nonfinite arrays or irregular exported time grid")
    return arrays


def history_cache(case, arrays, device="cpu"):
    from landau_surrogate.data.continuum_history import HistoryCache, phase_windows
    return HistoryCache([case], torch.as_tensor(arrays["state"][None], device=device),
                        torch.as_tensor(arrays["gradient"][None], device=device),
                        float(arrays["dt"]), [phase_windows(arrays["state"], arrays["time"])])


def hp_gradient(state, K, scale, maximum_mode=16):
    """Train-calibrated linear |d/dx|(p-n) baseline, NOT a nonlinear kinetic oracle.

    This matches the linear-temperature convention used by the historical PIC HP
    baseline, with its coefficient re-fitted on the chosen continuum train set.
    """
    if state.shape[-2] != 4:
        raise ValueError("Expected (...,4,x) physical states")
    modes = torch.arange(state.shape[-1] // 2 + 1, device=state.device)
    wave = torch.as_tensor(K, device=state.device, dtype=state.dtype)[..., None] * modes
    transformed = torch.fft.rfft((state[..., 2, :] - state[..., 0, :]).float(), dim=-1)
    return (float(scale) * torch.fft.irfft(transformed * wave * (modes <= maximum_mode),
                                        n=state.shape[-1], dim=-1)).to(state.dtype)


class ClosureSequenceDataset(Dataset):
    """Ordered windows from one trajectory, loaded on CPU with one-case caching.

    No sample crosses a trajectory or split. ``loss_mask`` excludes the causal
    warmup prefix. A cropped window does not recreate the exact hidden state of
    a model recurrently deployed from t=0: full causal replay is still required
    for final validation. Use num_workers=0 on a first Windows/laptop run.
    """
    def __init__(self, root, *, split="train", subset="starter", length=64,
                 burn_in=64, stride=64):
        if split not in ("train", "validation"):
            raise ValueError("Diagnostic test is not a training/validation window dataset")
        if length < 1 or burn_in < 0 or stride < 1:
            raise ValueError("Invalid window sizes")
        self.cases = load_index(root, split, subset)
        self.length, self.burn_in = length, burn_in
        self._cached_index, self._arrays = None, None
        self.windows = []
        entries = {c["case_id"]: c for c in load_manifest(root)["cases"]}
        for index, case in enumerate(self.cases):
            count = entries[case.case_id]["shape"][0]
            self.windows.extend((index, start) for start in range(0, count - length + 1, stride))
        if not self.windows:
            raise ValueError("Requested windows exceed trajectory length")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        case_index, start = self.windows[index]
        case = self.cases[case_index]
        if case_index != self._cached_index:
            self._arrays = load_case(case)
            self._cached_index = case_index
        positions = np.arange(start - self.burn_in, start + self.length)
        valid = positions >= 0
        selected = np.maximum(positions, 0)
        # Copies keep a batch independent of the bounded CPU cache.
        state = torch.from_numpy(self._arrays["state"][selected].copy())
        gradient = torch.from_numpy(self._arrays["gradient"][selected].copy())
        mask = np.arange(len(positions)) >= self.burn_in
        return {"state": state, "gradient": gradient,
                "valid": torch.from_numpy(valid), "loss_mask": torch.from_numpy(mask),
                "time": torch.from_numpy(positions * float(self._arrays["dt"])),
                "dt": torch.tensor(float(self._arrays["dt"]), dtype=torch.float32),
                "K": torch.tensor(case.K, dtype=torch.float32),
                "alpha": torch.tensor(case.alpha, dtype=torch.float32),
                "case_id": case.case_id, "window_start_index": start}


__all__ = ["PortableCase", "load_manifest", "load_index", "load_case", "history_cache",
           "hp_gradient", "ClosureSequenceDataset"]
