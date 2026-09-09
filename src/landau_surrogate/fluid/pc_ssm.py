"""RK4 with explicit SSM memory committed once per accepted physical step."""
from __future__ import annotations
import numpy as np
import torch
from landau_surrogate.fluid.multimoment_1d import spectral_filter, fluid_rhs_ampere


def advance_ssm(model, initial, K, *, horizon, dt=.02, output_dt=.1,
                observer=None, progress=None):
    steps, stride = round(horizon / dt), round(output_dt / dt)
    if dt <= 0 or stride < 1 or not np.isclose(steps * dt, horizon) or not np.isclose(stride * dt, output_dt):
        raise ValueError("dt must divide horizon and output interval")
    state = initial
    hidden = model.initial_memory(initial)
    outputs = [initial]
    for step in range(steps):
        first = spectral_filter(state, model.maximum_mode)
        features = model.encode(first, K)

        def rhs(value, fraction, encoded=None):
            filtered = spectral_filter(value, model.maximum_mode)
            if observer is not None:
                observer((step + fraction) * dt, filtered)
            z = model.encode(filtered, K) if encoded is None else encoded
            gradient = model.decode(filtered, K, z, model.read_memory(hidden))
            return fluid_rhs_ampere(filtered, K, lambda _: gradient, density_floor=None,
                                    maximum_mode=model.maximum_mode)

        a = rhs(state, 0., features)
        b = rhs(state + .5 * dt * a, .5)
        c = rhs(state + .5 * dt * b, .5)
        d = rhs(state + dt * c, 1.)
        accepted = spectral_filter(state + dt / 6 * (a + 2 * b + 2 * c + d), model.maximum_mode)
        if observer is not None:
            observer((step + 1) * dt, accepted)
        # The observer can reject a provisional/accepted state by raising.
        # Until acceptance, hidden stays unchanged through all four RHS calls.
        hidden = model.advance_memory(hidden, features, dt)
        state = accepted
        if (step + 1) % stride == 0:
            outputs.append(state)
            if progress is not None:
                progress((step + 1) * dt, state)
    return torch.stack(outputs, dim=1), hidden


__all__ = ["advance_ssm"]
