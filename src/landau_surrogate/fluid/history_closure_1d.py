"""Causal history-aware RK4 with differentiable accepted-state memory."""
from __future__ import annotations

from bisect import bisect_right

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from landau_surrogate.data.continuum_history import history_offsets
from landau_surrogate.fluid.multimoment_1d import fluid_rhs_ampere, spectral_filter


def initial_memory(cache, indices, starts, span):
    count = int(np.ceil(span / cache.dt))
    times = [float(index * cache.dt) for index in range(-count, 1)]
    states = [cache.sample(indices, starts + value) for value in times]
    return times, states


def stencil(times, now, dt, span):
    """Three RK abscissae, eight ages each; None denotes the provisional state."""
    stages = []
    needed = set()
    for stage_time in (now, now + .5*dt, now + dt):
        slots = []
        for age in history_offsets(span):
            query = stage_time - float(age)
            if age == 0:
                slots.append((None, None, 0., query))
            elif query > times[-1] + 1e-10:
                lower = len(times)-1
                weight = (query-times[-1]) / (stage_time-times[-1])
                slots.append((lower, None, weight, query))
                needed.add(lower)
            else:
                upper = min(max(bisect_right(times, query), 1), len(times)-1)
                lower = max(upper-1, 0)
                if len(times) == 1:
                    upper = lower = 0
                    weight = 0.
                else:
                    weight = (query-times[lower]) / (times[upper]-times[lower])
                slots.append((lower, upper, weight, query))
                needed.update((lower, upper))
        stages.append(slots)
    lookup = {original: local for local, original in enumerate(sorted(needed))}
    return [[(lookup.get(lo), lookup.get(hi), w, query) for lo, hi, w, query in stage]
            for stage in stages], sorted(needed)


def predict_gradient(current, memory, slots_spec, starts, k, alpha, model):
    slots, valid = [], []
    for lower, upper, weight, query in slots_spec:
        if lower is None:
            value = current
        else:
            next_value = current if upper is None else memory[upper]
            value = memory[lower] * (1-weight) + next_value * weight
        slots.append(value)
        valid.append(starts + query >= -1e-7)
    return model(torch.stack(slots, dim=1), torch.stack(valid, dim=1), k, alpha)


def rk4_history_step(state, memory, specifications, starts, k, alpha, model, dt, frozen_gradient=None,
                     closure_observer=None,state_observer=None):
    mode = model.maximum_mode

    def rhs(current, stage):
        current = spectral_filter(current, mode)
        if state_observer is not None:
            state_observer(stage,current)
        gradient = predict_gradient(current,memory,specifications[stage],starts,k,alpha,model) if frozen_gradient is None else frozen_gradient
        if stage == 0 and closure_observer is not None:
            closure_observer(gradient)
        return fluid_rhs_ampere(current, k, lambda _: gradient, density_floor=None,maximum_mode=mode)

    a = rhs(state, 0)
    b = rhs(state + .5*dt*a, 1)
    c = rhs(state + .5*dt*b, 1)
    d = rhs(state + dt*c, 2)
    return spectral_filter(state + dt/6*(a+2*b+2*c+d), mode)


def advance_history(model, cache, indices, starts, *, horizon, dt, use_checkpoint=True,
                    output_dt=.1, initial=None, progress=None, step_observer=None,
                    memory=None, return_memory=False, closure_observer=None):
    steps = round(horizon/dt)
    stride = max(1, round(output_dt/dt))
    if not np.isclose(steps*dt, horizon) or not np.isclose(stride*dt, output_dt):
        raise ValueError("dt must divide the rollout and output intervals")
    times, states = initial_memory(cache, indices, starts, model.history_span) if memory is None else memory
    times, states = list(times), list(states)
    state = states[-1] if initial is None else initial
    states[-1] = state
    k = torch.tensor([cache.cases[int(i)].K for i in indices.cpu()], device=state.device)
    alpha = torch.tensor([cache.cases[int(i)].alpha for i in indices.cpu()], device=state.device)
    outputs = [state]
    cadence = float(getattr(model,"closure_interval",0.))
    interval_steps = round(cadence/dt) if cadence else 0
    if cadence and (interval_steps<1 or not np.isclose(interval_steps*dt,cadence)):
        raise ValueError("Closure interval must be an integer multiple of the fluid dt")
    frozen = None
    for step in range(steps):
        now = step * dt
        specs, needed = stencil(times, now, dt, model.history_span)
        memory = tuple(states[index] for index in needed)

        if interval_steps and step%interval_steps==0:
            def evaluate(current,*old,slot_spec=specs[0]):
                return predict_gradient(current,old,slot_spec,starts,k,alpha,model)
            frozen = checkpoint(evaluate,state,*memory,use_reentrant=False) if use_checkpoint and torch.is_grad_enabled() else evaluate(state,*memory)

        # Capture immutable stencils by value: checkpoint recomputes after the forward loop.
        def update(current, gradient, *old, stage_specs=specs, closure_time=now):
            observer = None if closure_observer is None else lambda value: closure_observer(closure_time, value)
            state_observer = None if step_observer is None else lambda stage,value: step_observer(closure_time+(0.,.5*dt,dt)[stage],value)
            return rk4_history_step(current, old, stage_specs, starts, k, alpha, model, dt,gradient,observer,state_observer)

        state = checkpoint(update, state, frozen, *memory, use_reentrant=False) if use_checkpoint and torch.is_grad_enabled() else update(state, frozen, *memory)
        if step_observer is not None:
            step_observer((step+1)*dt, state)
        times.append((step+1)*dt)
        states.append(state)
        if (step+1) % stride == 0:
            outputs.append(state)
            if progress is not None:
                progress((step+1)*dt, state)
        # Old accepted states are unnecessary once all requested lags lie to their right.
        cutoff = bisect_right(times, (step+1)*dt-model.history_span-dt)-1
        if cutoff > 0:
            times, states = times[cutoff:], states[cutoff:]
    result = torch.stack(outputs, dim=1)
    if return_memory:
        return result, ([t-horizon for t in times], states)
    return result


__all__ = ["initial_memory", "stencil", "rk4_history_step", "advance_history"]
