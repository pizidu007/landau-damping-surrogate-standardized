"""
Metrics for surrogate curves vs PIC / theory (Landau damping).
"""
import numpy as np
import torch


def gamma_eff_from_log10_curve(log_e, t):
    """
    Fit log10 E vs t to a line. For E ∝ exp(-2*γ t), log10 E = C - (2γ/ln 10) t.

    Parameters
    ----------
    log_e, t : 1D arrays of equal length (e.g. length 1000 after preprocessing).

    Returns
    -------
    gamma_eff : float
        Decay rate γ from the slope. NaN if fit fails or too few points.
    """
    log_e = np.asarray(log_e, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    if log_e.size < 2 or t.size != log_e.size:
        return float("nan")
    try:
        slope, _ = np.polyfit(t, log_e, 1)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    return float(-slope * np.log(10.0) / 2.0)


def batch_gamma_metrics(gamma_pred, gamma_ref, eps=1e-30):
    """
    Mean absolute relative error |γ_pred - γ_ref| / |γ_ref| on finite entries.
    Both torch tensors or numpy arrays, shape [N].
    """
    gp = torch.as_tensor(gamma_pred, dtype=torch.float64).flatten()
    gr = torch.as_tensor(gamma_ref, dtype=torch.float64).flatten()
    mask = torch.isfinite(gp) & torch.isfinite(gr) & (torch.abs(gr) > eps)
    if mask.sum() == 0:
        return float("nan"), 0
    rel = torch.abs(gp[mask] - gr[mask]) / torch.abs(gr[mask])
    return float(rel.mean().item()), int(mask.sum().item())
