"""Small experimental FNO + diagonal SSM + optional linear-closure residual.

This is a portable starting implementation, not an S4/Mamba reproduction or a
validated kinetic closure. Memory is explicit; forward calls never mutate it.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from landau_surrogate.models.closure_fno1d import FNOBlock1d
from landau_surrogate.data.portable_closure import hp_gradient


class SmallFNOSSM(nn.Module):
    def __init__(self, normalization, width=24, layers=2, memory_size=8,
                 maximum_mode=16, use_memory=True, use_residual=True):
        super().__init__()
        self.width, self.memory_size = int(width), int(memory_size)
        self.maximum_mode = int(maximum_mode)
        self.use_memory, self.use_residual = bool(use_memory), bool(use_residual)
        self.register_buffer("input_mean", torch.tensor(normalization["input_mean"])[None, :, None])
        self.register_buffer("input_std", torch.tensor(normalization["input_std"])[None, :, None])
        for key in ("gradient_std", "k_mean", "k_std"):
            self.register_buffer(key, torch.tensor(float(normalization[key])))
        self.hp_scale = float(normalization["hp_scale"]) if use_residual else 0.
        self.lift = nn.Conv1d(5, width, 1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, maximum_mode + 1, normalization="none") for _ in range(layers)])
        if use_memory:
            rates = torch.logspace(-2, 1, memory_size).expand(width, -1).clone()
            self.raw_rate = nn.Parameter(torch.log(torch.expm1(rates)))
            self.drive = nn.Parameter(torch.ones(width, memory_size) / math.sqrt(memory_size))
            self.readout = nn.Parameter(torch.randn(width, memory_size) / math.sqrt(memory_size))
        self.head = nn.Sequential(nn.Conv1d(2 * width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))
        # Start from the explicit linear baseline (or zero for direct variants).
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def initial_memory(self, state):
        return state.new_zeros(state.shape[0], self.width, self.memory_size, state.shape[-1])

    def encode(self, state, K):
        scalar = ((K - self.k_mean) / self.k_std)[:, None, None].expand(-1, 1, state.shape[-1])
        z = self.lift(torch.cat(((state - self.input_mean) / self.input_std, scalar), dim=1))
        for block in self.blocks:
            z = block(z)
        return z

    def read_memory(self, hidden):
        if not self.use_memory:
            return hidden.new_zeros(hidden.shape[0], self.width, hidden.shape[-1])
        return (hidden * self.readout[None, :, :, None]).sum(dim=2)

    def advance_memory(self, hidden, features, dt, valid=None):
        if not self.use_memory:
            return hidden
        rate = F.softplus(self.raw_rate)[None, :, :, None]
        decay = torch.exp(-rate * dt)
        integral = -torch.expm1(-rate * dt) / rate.clamp_min(1e-12)
        updated = decay * hidden + integral * self.drive[None, :, :, None] * features[:, :, None]
        if valid is not None:
            updated = torch.where(valid[:, None, None, None], updated, hidden)
        return updated

    def decode(self, state, K, features, memory_features):
        residual = self.head(torch.cat((features, memory_features), dim=1))[:, 0] * self.gradient_std
        value = residual + hp_gradient(state, K, self.hp_scale, self.maximum_mode)
        transformed = torch.fft.rfft(value.float(), dim=-1)
        modes = torch.arange(transformed.shape[-1], device=value.device)
        return torch.fft.irfft(transformed * ((modes > 0) & (modes <= self.maximum_mode)), n=state.shape[-1])

    def gradient(self, state, K, hidden):
        return self.decode(state, K, self.encode(state, K), self.read_memory(hidden))

    def forward(self, states, K, hidden=None, dt=.02, valid=None):
        """Predict each frame using its incoming memory, then advance memory once."""
        if states.ndim != 4 or states.shape[2] != 4 or dt <= 0:
            raise ValueError("Expected [batch,time,4,x] and positive dt")
        batch, steps, _, nx = states.shape
        hidden = self.initial_memory(states[:, 0]) if hidden is None else hidden
        features = self.encode(states.flatten(0, 1), K.repeat_interleave(steps)).reshape(batch, steps, self.width, nx)
        memories = []
        for step in range(steps):
            memories.append(self.read_memory(hidden))
            hidden = self.advance_memory(hidden, features[:, step], dt, None if valid is None else valid[:, step])
        memory_features = torch.stack(memories, dim=1)
        gradients = self.decode(states.flatten(0, 1), K.repeat_interleave(steps),
                                features.flatten(0, 1), memory_features.flatten(0, 1)).reshape(batch, steps, nx)
        return gradients, hidden


__all__ = ["SmallFNOSSM"]
