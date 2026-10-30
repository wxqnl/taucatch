"""Masked hindsight supervision. Labels never enter the online observation."""
import torch


def trajectory_loss(mean, target, mask):
    if mean.shape != target.shape or tuple(mean.shape[-2:]) != (12, 3):
        raise ValueError('Expected matching12x3 future trajectories')
    if mask.shape != mean.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError('Expected a boolean per-knot validity mask')
    if not torch.isfinite(mean).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite trajectory')
    if not bool(mask.any()) or bool((target.norm(dim=-1) > 1.000001).any()):
        raise ValueError('Missing support or unbounded normalized target')
    return (mean-target).square().mean(-1)[mask].mean()
