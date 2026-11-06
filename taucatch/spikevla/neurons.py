"""Spiking neurons, normalisations and spike routing of SpikeVLA (inference).

A neuron's membrane is passed in and returned by the caller, one state per observation; nothing is kept between calls.
"""
import math

import torch
from torch import nn


def spike(voltage):
    """A unit event wherever the voltage reaches zero."""
    return (voltage >= 0).to(voltage.dtype)


class PLIF(nn.Module):
    """Parametric leaky integrate-and-fire: u <- u + sigmoid(w) (I - u); an event at u >= threshold resets u to 0."""

    def __init__(self, threshold=0.5):
        super().__init__()
        self.decay_logit = nn.Parameter(torch.zeros(()))
        self.log_threshold = nn.Parameter(torch.tensor(math.log(threshold)))

    def forward(self, current, membrane=None, threshold_scale=1.0, valid=None):
        if membrane is None:
            membrane = torch.zeros_like(current)
        voltage = membrane + self.decay_logit.sigmoid() * (current - membrane)
        events = spike(voltage / (self.log_threshold.exp() * threshold_scale) - 1.0)
        if valid is not None:
            events = events.masked_fill(~valid, 0.0)
            voltage = voltage.masked_fill(~valid, 0.0)
        return events, voltage * (1.0 - events)


class BipolarPLIF(PLIF):
    """PLIF with symmetric thresholds: events in {-1, 0, +1}, either sign resets the membrane."""

    def forward(self, current, membrane=None, threshold_scale=1.0, valid=None):
        if membrane is None:
            membrane = torch.zeros_like(current)
        voltage = membrane + self.decay_logit.sigmoid() * (current - membrane)
        normalized = voltage / (self.log_threshold.exp() * threshold_scale)
        events = spike(normalized - 1.0) - spike(-normalized - 1.0)
        if valid is not None:
            events = events.masked_fill(~valid, 0.0)
            voltage = voltage.masked_fill(~valid, 0.0)
        return events, voltage * (1.0 - events.abs())


def fire(nodes, state, name, current, valid=None):
    """Drive neuron `name` of `nodes` with its membrane in `state`; the events."""
    events, state[name] = nodes[name](current, state.get(name), valid=valid)
    return events


class TemporalNorm(nn.Module):
    """Channel normalisation with one running mean and variance per spiking timestep (from training) and a shared
    affine; masked tokens stay zero."""

    calibrating = False

    def __init__(self, channels, timesteps, eps=1e-5):
        super().__init__()
        self.channels, self.eps = channels, float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(timesteps, channels))
        self.register_buffer("running_var", torch.ones(timesteps, channels))

    def forward(self, current, valid=None, time_index=0):
        x = current.float()
        if valid is None:
            valid = torch.ones((*x.shape[:-1], 1), device=x.device, dtype=torch.bool)
        valid = valid.expand(*x.shape[:-1], 1)
        x = x.masked_fill(~valid, 0)
        normalized = (x - self.running_mean[time_index]) * torch.rsqrt(self.running_var[time_index] + self.eps)
        return (normalized * self.weight + self.bias).masked_fill(~valid, 0)


class TokenNorm(nn.LayerNorm):
    """LayerNorm over each token's channels, with the TemporalNorm call signature."""

    def forward(self, current, valid=None, time_index=0):
        output = super().forward(current)
        return output if valid is None else output.masked_fill(~valid, 0.0)


def multi_winner(scores, allowed, topk):
    """Binary routing: 1 on the top-k allowed keys of each row (ties by key order), else 0."""
    ranked = scores.masked_fill(~allowed, -torch.inf)
    indices = torch.argsort(ranked, dim=-1, descending=True, stable=True)[..., :topk]
    winners = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
    return winners.masked_fill_(~allowed, 0.0)


def shiftmax(scores, allowed, bits=0):
    """Base-2 attention weights 2^(x - ceil(log2 sum 2^x)), the exponent rounded up to 1/2^bits; masked keys get 0.
    scores [B, H, Nq, Nk], allowed bool [B, Nq, Nk] or [B, H, Nq, Nk]."""
    if allowed.ndim == 3:
        allowed = allowed[:, None]
    allowed = allowed.expand_as(scores)
    live = allowed.any(-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    maximum = masked.amax(-1, keepdim=True)
    maximum = torch.where(live, maximum, torch.zeros_like(maximum))
    numerator = torch.exp2(masked - maximum)
    step = float(2 ** bits)
    exponent = (torch.log2(numerator.sum(-1, keepdim=True).clamp_min(1)) * step).ceil() / step
    return numerator / torch.exp2(exponent)
