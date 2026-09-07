from pathlib import Path
import random

import numpy as np

from landau_surrogate.data.continuum_v1 import ContinuumCase
from landau_surrogate.training.continuum_v1_rollout_stability import (
    CachedTrajectory,
    interpolate,
    regime_representatives,
    sample_windows,
)


def make_case(identifier: str, regime: str) -> ContinuumCase:
    return ContinuumCase(
        case_id=identifier,
        K=0.35,
        alpha=0.1,
        split="train",
        regime=regime,
        path=Path("unused.h5"),
    )


def test_interpolate_is_linear() -> None:
    time = np.array([0.0, 0.2, 0.4])
    value = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    np.testing.assert_allclose(interpolate(value, time, 0.3), [3.0])


def test_balanced_windows_stay_inside_trajectory() -> None:
    cache = []
    for regime in ("weak", "transition", "strong_nonlinear"):
        cache.append(
            CachedTrajectory(
                case=make_case(regime, regime),
                state=np.zeros((2, 3, 4), dtype=np.float32),
                gradient=np.zeros((2, 4), dtype=np.float32),
                time=np.array([0.0, 80.0]),
            )
        )
    trajectories, starts = sample_windows(
        cache,
        batch_size=6,
        horizon=1.0,
        batch_index=0,
        rng=random.Random(0),
    )
    assert [item.case.regime for item in trajectories] == [
        "weak",
        "transition",
        "strong_nonlinear",
        "weak",
        "transition",
        "strong_nonlinear",
    ]
    assert all(0.0 <= value <= 79.0 for value in starts)


def test_smoke_representatives_include_every_regime() -> None:
    cases = [
        make_case("strong", "strong_nonlinear"),
        make_case("weak", "weak"),
        make_case("transition", "transition"),
    ]
    assert [case.regime for case in regime_representatives(cases)] == [
        "weak",
        "transition",
        "strong_nonlinear",
    ]
