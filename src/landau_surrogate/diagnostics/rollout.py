#!/usr/bin/env python3
"""
Physics and stability diagnostics for Stage 8D-2C.

All physical diagnostics are derived directly from normalized delta_f0 on the
frozen 128 x 193 grid. No electric-field reconstruction or Poisson-sign
convention is assumed.

The module provides:
- deterministic stepper checkpoint interpolation;
- pure-stepper and snapshot-mean-anchored rollout;
- density mode-1 amplitude/phase diagnostics;
- effective damping/frequency fits;
- velocity-moment diagnostics;
- full-distribution positivity diagnostics;
- high-x spectral-energy diagnostics;
- perturbation amplification and out-of-domain long-rollout stability checks.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from landau_surrogate.models.snapshot_fno import ConditionalFNO2d
from landau_surrogate.models.stepper_fno import ConditionalStepperFNO2d


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    stepper_source: str
    average_weight_uniform: float | None = None
    mean_anchor_beta: float = 0.0
    complexity_rank: int = 0

    @property
    def uses_snapshot_anchor(self) -> bool:
        return self.mean_anchor_beta > 0.0


def model_config_from_checkpoint(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    config = checkpoint["model_config"]
    return {
        "variant": str(config["variant"]),
        "width": int(config["width"]),
        "layers": int(config["layers"]),
        "modes_x": int(config["modes_x"]),
        "modes_v": int(config["modes_v"]),
        "v_padding": int(config["v_padding"]),
        "condition_channels": int(config["condition_channels"]),
        "condition_hidden_dim": int(
            config["condition_hidden_dim"]
        ),
        "condition_hidden_layers": int(
            config["condition_hidden_layers"]
        ),
        "time_harmonics": int(config["time_harmonics"]),
        "phase_harmonics": int(config["phase_harmonics"]),
        "x_coordinate_harmonics": int(
            config["x_coordinate_harmonics"]
        ),
    }


def snapshot_config_from_checkpoint(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return dict(checkpoint["model_config"])


def build_stepper(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> ConditionalStepperFNO2d:
    contract = checkpoint["cache_contract"]
    model = ConditionalStepperFNO2d(
        normalized_x=contract["normalized_x"].cpu().numpy(),
        velocity=contract["velocity"].cpu().numpy(),
        **model_config_from_checkpoint(checkpoint),
    )
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    return model.to(device).eval()


def build_snapshot_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> ConditionalFNO2d:
    contract = checkpoint["cache_contract"]
    model = ConditionalFNO2d(
        normalized_x=contract["normalized_x"].cpu().numpy(),
        velocity=contract["velocity"].cpu().numpy(),
        **snapshot_config_from_checkpoint(checkpoint),
    )
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    return model.to(device).eval()


def average_state_dicts(
    uniform_state: dict[str, torch.Tensor],
    late_state: dict[str, torch.Tensor],
    uniform_weight: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= uniform_weight <= 1.0:
        raise ValueError(uniform_weight)
    if set(uniform_state) != set(late_state):
        missing = sorted(set(uniform_state) ^ set(late_state))
        raise RuntimeError(
            f"State-dict keys differ: {missing[:10]}"
        )

    output: dict[str, torch.Tensor] = {}
    for key in uniform_state:
        left = uniform_state[key]
        right = late_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                f"State mismatch for {key}: "
                f"{left.shape}/{left.dtype} vs "
                f"{right.shape}/{right.dtype}"
            )
        if torch.is_floating_point(left) or torch.is_complex(left):
            output[key] = (
                uniform_weight * left.float()
                + (1.0 - uniform_weight) * right.float()
            ).to(dtype=left.dtype)
        else:
            if not torch.equal(left, right):
                raise RuntimeError(
                    f"Non-floating state differs for {key}."
                )
            output[key] = left.clone()
    return output


def averaged_checkpoint(
    uniform_checkpoint: dict[str, Any],
    late_checkpoint: dict[str, Any],
    uniform_weight: float,
) -> dict[str, Any]:
    if (
        uniform_checkpoint["model_config"]
        != late_checkpoint["model_config"]
    ):
        raise RuntimeError(
            "Cannot average checkpoints with different model configs."
        )
    checkpoint = copy.deepcopy(uniform_checkpoint)
    checkpoint["stage"] = "stage8d2c_averaged_candidate"
    checkpoint["candidate"] = (
        f"average_u{uniform_weight:.2f}"
    )
    checkpoint["model_state_dict"] = average_state_dicts(
        uniform_checkpoint["model_state_dict"],
        late_checkpoint["model_state_dict"],
        uniform_weight,
    )
    checkpoint["average_contract"] = {
        "uniform_weight": float(uniform_weight),
        "late_weighted_weight": float(
            1.0 - uniform_weight
        ),
    }
    return checkpoint


def normalize_condition(
    k_value: np.ndarray,
    alpha_value: np.ndarray,
    time_value: float,
    bounds: dict[str, float],
) -> np.ndarray:
    def normalize(
        values: np.ndarray | float,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        return (
            2.0
            * (np.asarray(values, dtype=np.float32) - lower)
            / (upper - lower)
            - 1.0
        ).astype(np.float32)

    return np.stack(
        [
            normalize(
                k_value,
                bounds["k_min"],
                bounds["k_max"],
            ),
            normalize(
                alpha_value,
                bounds["alpha_min"],
                bounds["alpha_max"],
            ),
            normalize(
                np.full_like(k_value, time_value),
                bounds["time_min"],
                bounds["time_max"],
            ),
        ],
        axis=1,
    ).astype(np.float32)


def rollout_candidate(
    stepper: ConditionalStepperFNO2d,
    initial_field: np.ndarray,
    k_values: np.ndarray,
    alpha_values: np.ndarray,
    start_time: float,
    steps: int,
    dt: float,
    bounds: dict[str, float],
    device: torch.device,
    snapshot_model: ConditionalFNO2d | None = None,
    mean_anchor_beta: float = 0.0,
    amp: str = "bf16",
) -> np.ndarray:
    """
    Roll out a batch of fields.

    For mean_anchor_beta > 0, the stepper's predicted nonzero component is
    preserved, while only mean_delta is blended with the direct-snapshot mean
    at the next requested time.
    """
    if steps < 1:
        raise ValueError(steps)
    if not 0.0 <= mean_anchor_beta <= 1.0:
        raise ValueError(mean_anchor_beta)
    if mean_anchor_beta > 0.0 and snapshot_model is None:
        raise ValueError(
            "A snapshot model is required for mean anchoring."
        )

    initial = np.asarray(initial_field, dtype=np.float32)
    if initial.ndim == 2:
        initial = initial[None]
    batch = int(initial.shape[0])
    k_array = np.asarray(k_values, dtype=np.float32).reshape(-1)
    alpha_array = np.asarray(
        alpha_values, dtype=np.float32
    ).reshape(-1)
    if k_array.size != batch or alpha_array.size != batch:
        raise ValueError("Condition batch does not match fields.")

    current = torch.from_numpy(initial).to(device)
    outputs = [initial.copy()]
    autocast_enabled = device.type == "cuda" and amp != "none"
    autocast_dtype = (
        torch.bfloat16
        if amp == "bf16"
        else torch.float16
        if amp == "fp16"
        else torch.float32
    )

    stepper.eval()
    if snapshot_model is not None:
        snapshot_model.eval()

    with torch.no_grad():
        for step_index in range(steps):
            current_time = start_time + step_index * dt
            next_time = current_time + dt
            normalized = normalize_condition(
                k_array,
                alpha_array,
                current_time,
                bounds,
            )
            physical = np.stack(
                [
                    k_array,
                    alpha_array,
                    np.full(
                        batch,
                        current_time,
                        dtype=np.float32,
                    ),
                ],
                axis=1,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                step_output = stepper(
                    current,
                    torch.from_numpy(normalized).to(device),
                    torch.from_numpy(physical).to(device),
                )
            next_field = step_output["field"].float()

            if mean_anchor_beta > 0.0:
                next_normalized = normalize_condition(
                    k_array,
                    alpha_array,
                    next_time,
                    bounds,
                )
                next_physical = np.stack(
                    [
                        k_array,
                        alpha_array,
                        np.full(
                            batch,
                            next_time,
                            dtype=np.float32,
                        ),
                    ],
                    axis=1,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    snapshot_output = snapshot_model(
                        torch.from_numpy(
                            next_normalized
                        ).to(device),
                        torch.from_numpy(
                            next_physical
                        ).to(device),
                    )
                snapshot_mean = torch.mean(
                    snapshot_output["field"].float(),
                    dim=1,
                )
                step_mean = step_output[
                    "mean_delta"
                ].float()
                corrected_mean = (
                    (1.0 - mean_anchor_beta) * step_mean
                    + mean_anchor_beta * snapshot_mean
                )
                next_field = (
                    step_output["nonzero"].float()
                    + corrected_mean[:, None, :]
                )

            if not torch.isfinite(next_field).all():
                raise RuntimeError(
                    f"Non-finite rollout at step {step_index + 1}."
                )
            current = next_field
            outputs.append(
                current.cpu().numpy().astype(np.float32)
            )

    return np.stack(outputs, axis=1)


def trapezoidal_integral(
    values: np.ndarray,
    coordinates: np.ndarray,
    axis: int,
) -> np.ndarray:
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(values, x=coordinates, axis=axis)


def relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    floor: float = 1.0e-30,
) -> float:
    """
    Relative L2 for both real and complex arrays.

    Complex diagnostics must use |z|^2. Casting a complex mode to float would
    silently discard its imaginary component and under-report the error.
    """
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    dtype = (
        np.complex128
        if (
            np.iscomplexobj(prediction_array)
            or np.iscomplexobj(target_array)
        )
        else np.float64
    )
    prediction_typed = prediction_array.astype(
        dtype, copy=False
    )
    truth = target_array.astype(dtype, copy=False)
    error = prediction_typed - truth
    numerator = float(np.sum(np.abs(error) ** 2))
    denominator = float(np.sum(np.abs(truth) ** 2))
    return math.sqrt(
        numerator / max(denominator, floor)
    )


def effective_complex_mode_fit(
    complex_mode: np.ndarray,
    phase_time: np.ndarray,
    start_time: float,
    end_time: float,
) -> dict[str, float]:
    mode = np.asarray(complex_mode, dtype=np.complex128)
    time = np.asarray(phase_time, dtype=np.float64)
    mask = (
        np.isfinite(mode.real)
        & np.isfinite(mode.imag)
        & (time >= start_time)
        & (time <= end_time)
    )
    if np.sum(mask) < 3:
        raise RuntimeError(
            f"Insufficient fit points in [{start_time},{end_time}]."
        )

    selected_mode = mode[mask]
    selected_time = time[mask]
    amplitude = np.abs(selected_mode)
    amplitude_floor = max(
        float(np.max(amplitude)) * 1.0e-8,
        1.0e-30,
    )
    log_amplitude = np.log(
        np.maximum(amplitude, amplitude_floor)
    )
    phase = np.unwrap(np.angle(selected_mode))

    design = np.stack(
        [selected_time, np.ones_like(selected_time)],
        axis=1,
    )
    gamma, amplitude_intercept = np.linalg.lstsq(
        design, log_amplitude, rcond=None
    )[0]
    omega, phase_intercept = np.linalg.lstsq(
        design, phase, rcond=None
    )[0]

    amplitude_fit = (
        gamma * selected_time + amplitude_intercept
    )
    phase_fit = omega * selected_time + phase_intercept

    def r_squared(
        observed: np.ndarray,
        fitted: np.ndarray,
    ) -> float:
        residual = float(
            np.sum((observed - fitted) ** 2)
        )
        centered = float(
            np.sum(
                (observed - np.mean(observed)) ** 2
            )
        )
        if centered <= 1.0e-30:
            return 1.0 if residual <= 1.0e-30 else 0.0
        return 1.0 - residual / centered

    return {
        "gamma": float(gamma),
        "omega": float(omega),
        "log_amplitude_r2": r_squared(
            log_amplitude, amplitude_fit
        ),
        "phase_r2": r_squared(phase, phase_fit),
        "point_count": int(np.sum(mask)),
    }


def density_mode1(
    field: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    density = trapezoidal_integral(
        np.asarray(field, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
        axis=-1,
    )
    transformed = np.fft.rfft(
        density, axis=-1, norm="forward"
    )
    return transformed[..., 1]


def velocity_moments(
    field: np.ndarray,
    velocity: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(field, dtype=np.float64)
    velocity64 = np.asarray(velocity, dtype=np.float64)
    mean_x = np.mean(values, axis=-2)
    return {
        "mass": trapezoidal_integral(
            mean_x, velocity64, axis=-1
        ),
        "momentum": trapezoidal_integral(
            mean_x * velocity64[None, :],
            velocity64,
            axis=-1,
        ),
        "kinetic": trapezoidal_integral(
            mean_x
            * (0.5 * velocity64[None, :] ** 2),
            velocity64,
            axis=-1,
        ),
    }


def high_x_energy_ratio(
    field: np.ndarray,
    cutoff_mode: int = 8,
) -> np.ndarray:
    transformed = np.fft.rfft(
        np.asarray(field, dtype=np.float64),
        axis=-2,
        norm="forward",
    )
    energy = np.sum(
        np.abs(transformed) ** 2,
        axis=-1,
    )
    total_nonzero = np.sum(energy[..., 1:], axis=-1)
    high = np.sum(
        energy[..., cutoff_mode:],
        axis=-1,
    )
    return high / np.maximum(total_nonzero, 1.0e-30)


def physics_case_metrics(
    model_name: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    case_ids: Iterable[str],
    k_values: np.ndarray,
    alpha_values: np.ndarray,
    group_codes: np.ndarray,
    phase_time: np.ndarray,
    velocity: np.ndarray,
    delta_global_rms: float,
    f0_train: np.ndarray,
) -> list[dict[str, Any]]:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction {pred.shape} != truth {target.shape}"
        )

    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        pred_case = pred[index]
        truth_case = target[index]
        pred_mode = density_mode1(
            pred_case, velocity
        )
        truth_mode = density_mode1(
            truth_case, velocity
        )

        pred_early = effective_complex_mode_fit(
            pred_mode,
            phase_time,
            start_time=0.5,
            end_time=min(5.0, float(phase_time[-1])),
        )
        truth_early = effective_complex_mode_fit(
            truth_mode,
            phase_time,
            start_time=0.5,
            end_time=min(5.0, float(phase_time[-1])),
        )
        pred_full = effective_complex_mode_fit(
            pred_mode,
            phase_time,
            start_time=0.5,
            end_time=float(phase_time[-1]),
        )
        truth_full = effective_complex_mode_fit(
            truth_mode,
            phase_time,
            start_time=0.5,
            end_time=float(phase_time[-1]),
        )

        pred_moments = velocity_moments(
            pred_case, velocity
        )
        truth_moments = velocity_moments(
            truth_case, velocity
        )

        physical_pred = (
            pred_case * float(delta_global_rms)
        )
        physical_truth = (
            truth_case * float(delta_global_rms)
        )
        equilibrium = np.asarray(
            f0_train, dtype=np.float64
        )[None, None, :]
        full_distribution = physical_pred + equilibrium
        truth_full_distribution = (
            physical_truth + equilibrium
        )
        negative_fraction_by_time = np.mean(
            full_distribution < 0.0,
            axis=(1, 2),
        )
        truth_negative_fraction_by_time = np.mean(
            truth_full_distribution < 0.0,
            axis=(1, 2),
        )

        pred_high = high_x_energy_ratio(pred_case)
        truth_high = high_x_energy_ratio(truth_case)

        phase_difference = np.angle(
            pred_mode * np.conj(truth_mode)
        )
        amplitude_mask = np.abs(truth_mode) > max(
            float(np.max(np.abs(truth_mode))) * 1.0e-3,
            1.0e-12,
        )

        row: dict[str, Any] = {
            "model": model_name,
            "case_id": str(case_id),
            "k": float(k_values[index]),
            "alpha": float(alpha_values[index]),
            "group_code": int(group_codes[index]),
            "density_mode1_amplitude_relative_l2": (
                relative_l2(
                    np.abs(pred_mode),
                    np.abs(truth_mode),
                )
            ),
            "density_mode1_complex_relative_l2": (
                relative_l2(pred_mode, truth_mode)
            ),
            "density_mode1_phase_mae_rad": float(
                np.mean(
                    np.abs(
                        phase_difference[amplitude_mask]
                    )
                )
            )
            if np.any(amplitude_mask)
            else 0.0,
            "early_gamma_true": truth_early["gamma"],
            "early_gamma_pred": pred_early["gamma"],
            "early_gamma_abs_error": abs(
                pred_early["gamma"]
                - truth_early["gamma"]
            ),
            "early_omega_true": truth_early["omega"],
            "early_omega_pred": pred_early["omega"],
            "early_omega_abs_error": abs(
                pred_early["omega"]
                - truth_early["omega"]
            ),
            "early_log_amplitude_r2_true": truth_early[
                "log_amplitude_r2"
            ],
            "early_log_amplitude_r2_pred": pred_early[
                "log_amplitude_r2"
            ],
            "early_phase_r2_true": truth_early["phase_r2"],
            "early_phase_r2_pred": pred_early["phase_r2"],
            "full_gamma_true": truth_full["gamma"],
            "full_gamma_pred": pred_full["gamma"],
            "full_gamma_abs_error": abs(
                pred_full["gamma"]
                - truth_full["gamma"]
            ),
            "full_omega_true": truth_full["omega"],
            "full_omega_pred": pred_full["omega"],
            "full_omega_abs_error": abs(
                pred_full["omega"]
                - truth_full["omega"]
            ),
            "mass_moment_relative_l2": relative_l2(
                pred_moments["mass"],
                truth_moments["mass"],
                floor=1.0e-24,
            ),
            "momentum_moment_relative_l2": relative_l2(
                pred_moments["momentum"],
                truth_moments["momentum"],
                floor=1.0e-24,
            ),
            "kinetic_moment_relative_l2": relative_l2(
                pred_moments["kinetic"],
                truth_moments["kinetic"],
                floor=1.0e-24,
            ),
            "mass_drift_relative_l2": relative_l2(
                pred_moments["mass"]
                - pred_moments["mass"][0],
                truth_moments["mass"]
                - truth_moments["mass"][0],
                floor=1.0e-24,
            ),
            "momentum_drift_relative_l2": relative_l2(
                pred_moments["momentum"]
                - pred_moments["momentum"][0],
                truth_moments["momentum"]
                - truth_moments["momentum"][0],
                floor=1.0e-24,
            ),
            "kinetic_drift_relative_l2": relative_l2(
                pred_moments["kinetic"]
                - pred_moments["kinetic"][0],
                truth_moments["kinetic"]
                - truth_moments["kinetic"][0],
                floor=1.0e-24,
            ),
            "negative_fraction_mean": float(
                np.mean(negative_fraction_by_time)
            ),
            "negative_fraction_max": float(
                np.max(negative_fraction_by_time)
            ),
            "minimum_full_distribution": float(
                np.min(full_distribution)
            ),
            "truth_negative_fraction_mean": float(
                np.mean(truth_negative_fraction_by_time)
            ),
            "truth_negative_fraction_max": float(
                np.max(truth_negative_fraction_by_time)
            ),
            "truth_minimum_full_distribution": float(
                np.min(truth_full_distribution)
            ),
            "negative_fraction_excess_mean": float(
                np.mean(
                    negative_fraction_by_time
                    - truth_negative_fraction_by_time
                )
            ),
            "negative_fraction_excess_max": float(
                np.max(
                    negative_fraction_by_time
                    - truth_negative_fraction_by_time
                )
            ),
            "high_x_energy_ratio_relative_l2": relative_l2(
                pred_high,
                truth_high,
                floor=1.0e-24,
            ),
            "high_x_energy_ratio_final_true": float(
                truth_high[-1]
            ),
            "high_x_energy_ratio_final_pred": float(
                pred_high[-1]
            ),
        }
        if not all(
            np.isfinite(float(value))
            for key, value in row.items()
            if key not in {"model", "case_id"}
        ):
            raise RuntimeError(
                f"Non-finite physics metric for {case_id}."
            )
        rows.append(row)
    return rows


def aggregate_physics_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No physics rows.")
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key
        not in {
            "model",
            "case_id",
            "k",
            "alpha",
            "group_code",
        }
        and isinstance(value, (int, float, np.number))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        keys = [
            (row["model"], "all_test"),
            (
                row["model"],
                f"alpha_{float(row['alpha']):.3f}",
            ),
            (
                row["model"],
                f"group_{int(row['group_code'])}",
            ),
        ]
        for key in keys:
            grouped.setdefault(key, []).append(row)

    output = []
    for (model_name, group_name), selected in sorted(
        grouped.items()
    ):
        record: dict[str, Any] = {
            "model": model_name,
            "diagnostic_group": group_name,
            "case_count": len(selected),
        }
        for key in numeric_keys:
            values = np.asarray(
                [float(row[key]) for row in selected],
                dtype=np.float64,
            )
            record[f"{key}_mean"] = float(
                np.mean(values)
            )
            record[f"{key}_max"] = float(
                np.max(values)
            )
        output.append(record)
    return output


def perturbation_amplification(
    baseline_rollout: np.ndarray,
    perturbed_rollout: np.ndarray,
) -> dict[str, float]:
    baseline = np.asarray(
        baseline_rollout, dtype=np.float64
    )
    perturbed = np.asarray(
        perturbed_rollout, dtype=np.float64
    )
    if baseline.shape != perturbed.shape:
        raise ValueError("Perturbed rollout shape mismatch.")
    difference = perturbed - baseline
    initial_norm = np.sqrt(
        np.sum(difference[:, 0] ** 2, axis=(1, 2))
    )
    difference_norm = np.sqrt(
        np.sum(difference**2, axis=(2, 3))
    )
    amplification = difference_norm / np.maximum(
        initial_norm[:, None], 1.0e-30
    )
    return {
        "amplification_case_macro_final": float(
            np.mean(amplification[:, -1])
        ),
        "amplification_case_macro_max": float(
            np.mean(np.max(amplification, axis=1))
        ),
        "amplification_global_max": float(
            np.max(amplification)
        ),
    }


def norm_stability_summary(
    rollout: np.ndarray,
) -> dict[str, float | bool]:
    values = np.asarray(rollout, dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    initial_norm = np.sqrt(
        np.sum(values[:, 0] ** 2, axis=(1, 2))
    )
    norms = np.sqrt(
        np.sum(values**2, axis=(2, 3))
    )
    ratio = norms / np.maximum(
        initial_norm[:, None], 1.0e-30
    )
    return {
        "finite": finite,
        "case_macro_final_norm_ratio": float(
            np.mean(ratio[:, -1])
        ),
        "case_macro_max_norm_ratio": float(
            np.mean(np.max(ratio, axis=1))
        ),
        "global_max_norm_ratio": float(np.max(ratio)),
        "global_max_abs_value": float(
            np.max(np.abs(values))
        ),
    }


__all__ = [
    "CandidateSpec",
    "aggregate_physics_rows",
    "average_state_dicts",
    "averaged_checkpoint",
    "build_snapshot_model",
    "build_stepper",
    "density_mode1",
    "effective_complex_mode_fit",
    "high_x_energy_ratio",
    "model_config_from_checkpoint",
    "norm_stability_summary",
    "normalize_condition",
    "perturbation_amplification",
    "physics_case_metrics",
    "relative_l2",
    "rollout_candidate",
    "snapshot_config_from_checkpoint",
    "trapezoidal_integral",
    "velocity_moments",
]
