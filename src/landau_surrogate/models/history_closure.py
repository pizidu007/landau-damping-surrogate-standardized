"""Matched A/B/C FNO closures with physical-time history and explicit scale."""
from __future__ import annotations

import torch
from torch import nn

from landau_surrogate.models.closure_fno1d import ClosureFNO1d


class HistoryClosureFNO(nn.Module):
    def __init__(self, *, maximum_mode=16, width=128, layers=4, include_alpha=False,
                 history_span=2., normalization, enforce_reflection=False):
        super().__init__()
        self.include_alpha = bool(include_alpha)
        self.history_span = float(history_span)
        self.maximum_mode = int(maximum_mode)
        self.enforce_reflection = bool(enforce_reflection)
        # Same input tensor size in all arms; A repeats the current state in eight slots.
        # C uses the last scalar channel; A/B receive zero in this channel.
        self.network = ClosureFNO1d(state_channels=8*5+1, include_k=True, width=width,
                                  modes=maximum_mode+1, layers=layers, enforce_zero_mean=True,
                                  normalization="none", activation="relu")
        self.register_buffer("input_mean", torch.tensor(normalization["input_mean"]).reshape(1, 1, 4, 1))
        self.register_buffer("input_std", torch.tensor(normalization["input_std"]).reshape(1, 1, 4, 1))
        for name in ("gradient_std", "k_mean", "k_std", "alpha_mean", "alpha_std"):
            self.register_buffer(name, torch.tensor(float(normalization[name])))

    def forward(self, history, valid, k, alpha):
        if history.ndim != 4 or history.shape[1:3] != (8, 4):
            raise ValueError("Expected history [batch,8,4,x]")
        if self.enforce_reflection:
            # n,p are scalars and u,E are vectors. dq/dx is a scalar under x,v -> -x,-v.
            parity = history.new_tensor([1.,-1.,1.,-1.])[None,None,:,None]
            reflected = history.flip(-1)*parity
            values = self._predict(torch.cat((history,reflected)), torch.cat((valid,valid)),
                                   torch.cat((k,k)), torch.cat((alpha,alpha)))
            direct, reverse = values.chunk(2)
            return .5*(direct+reverse.flip(-1))
        return self._predict(history, valid, k, alpha)

    def _predict(self, history, valid, k, alpha):
        normalized = (history - self.input_mean) / self.input_std
        mask = valid.to(history.dtype)[:, :, None, None].expand(-1, -1, 1, history.shape[-1])
        features = torch.cat((normalized, mask), dim=2).flatten(1, 2)
        amplitude = (alpha - self.alpha_mean) / self.alpha_std if self.include_alpha else torch.zeros_like(alpha)
        features = torch.cat((features, amplitude[:, None, None].expand(-1, 1, history.shape[-1])), dim=1)
        output = self.network(features, (k-self.k_mean)/self.k_std) * self.gradient_std
        transformed = torch.fft.rfft(output.float(), dim=-1)
        modes = torch.arange(transformed.shape[-1], device=output.device)
        transformed = transformed * ((modes > 0) & (modes <= self.maximum_mode))
        return torch.fft.irfft(transformed, n=output.shape[-1], dim=-1)

    @torch.no_grad()
    def initialize_from_supervised(self, parent):
        """Transfer the instantaneous closure, preserving normalization and output scale."""
        state = parent["model_state_dict"]
        old = parent["normalization"]
        own = self.network.state_dict()
        for name, value in state.items():
            if name.startswith("lift."):
                continue
            if name not in own:
                continue
            if own[name].shape == value.shape:
                own[name].copy_(value)
            elif name.endswith("spectral.weight") and own[name].shape[:2] == value.shape[:2]:
                own[name].zero_()
                count = min(own[name].shape[-1], value.shape[-1])
                own[name][..., :count].copy_(value[..., :count])
            else:
                raise ValueError(f"Incompatible warm-start parameter {name}")
        self.network.load_state_dict(own)
        self.network.lift.weight.zero_()
        self.network.lift.bias.copy_(state["lift.bias"])
        old_std = torch.tensor(old["input_std"], device=self.input_std.device)
        old_mean = torch.tensor(old["input_mean"], device=self.input_mean.device)
        old_weight = state["lift.weight"].to(self.input_mean.device)
        for channel in range(3):
            self.network.lift.weight[:, 7*5+channel] = old_weight[:, channel] * self.input_std[0, 0, channel] / old_std[channel]
            self.network.lift.bias.add_(old_weight[:, channel, 0] * (self.input_mean[0, 0, channel, 0]-old_mean[channel]) / old_std[channel])
        self.network.lift.weight[:, -1] = old_weight[:, 3] * self.k_std / float(old["k_std"])
        self.network.lift.bias.add_(old_weight[:, 3, 0] * (self.k_mean-float(old["k_mean"])) / float(old["k_std"]))
        scale = float(old["gradient_std"]) / self.gradient_std
        self.network.project[2].weight.mul_(scale)
        self.network.project[2].bias.mul_(scale)


__all__ = ["HistoryClosureFNO"]
