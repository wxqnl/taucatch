"""Neutral XYZ policy using frozen rich native features and a new22D context.

The rejected absolute readout is never an executed action. Its upstream learned
features initialize this separate policy; all new behavior probabilities and
executor contracts must be recorded afresh.
"""
import math
import torch
from torch import nn

from .chooser import _paths
from .native_servo_rich_v109 import require_rich_checkpoint
from .xyz_context_v134 import CONTEXT_CONTRACT, CONTEXT_DIM
from .action_rl_v1 import sample_action, temporal_cholesky

KIND = 'rich_native_xyz_residual_context_actor_v139'
STD = (.5, .5, .5)
TRAINABLE = ('readout.layers.2.weight', 'readout.layers.2.bias', 'context.weight')
# Kept pure so CTS never imports the physical executor/ruckig dependency.
# The initializer independently compares this value with the actual V134 lane.
XYZ_CONTRACT = dict(name='final_fast_relative_xyz_only_v134', horizon=12, dt=.016,
                    action_dim=3, translation_m=.015, translation_rate_mps=.2,
                    normal='fast_unchanged', closure='fast_timer_unchanged',
                    expiry='observation_plus_192ms_then_slew_to_zero')


class XYZActor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        require_rich_checkpoint(checkpoint)
        _paths()
        from spikevla.config import ActionConfig
        from spikevla.modules.spike_act import SpikeActionReadout
        prefix = 'vla.action_head.decoder.action_projection.'
        state = {k[len(prefix):]: v.detach().cpu().clone() for k, v in checkpoint['state'].items() if k.startswith(prefix)}
        assert state['layers.2.weight'].shape == (7, 512)
        state['layers.2.weight'] = torch.zeros((3, 512))
        state['layers.2.bias'] = torch.zeros(3)
        self.readout = SpikeActionReadout(ActionConfig(action_dim=3, horizon=12, readout_dim=512),
                                          int(state['layers.0.weight'].shape[1]), 1e-5)
        self.readout.load_state_dict(state, strict=True)
        self.readout.requires_grad_(False)
        self.readout.layers[-1].requires_grad_(True)
        self.context = nn.Linear(CONTEXT_DIM, 3, bias=False)
        nn.init.zeros_(self.context.weight)
        self.eval()
        assert tuple(k for k, v in self.named_parameters() if v.requires_grad) == TRAINABLE

    def forward(self, features, context):
        if features.ndim != 4 or tuple(features.shape[2:]) != (12, 256) or context.shape != (features.shape[1], CONTEXT_DIM):
            raise ValueError('Expected native[S,B,12,256] features and new context[B,22]')
        if not torch.isfinite(features).all() or not torch.isfinite(context).all():
            raise ValueError('Nonfinite XYZ policy input')
        if features.shape[1] > 1:
            return torch.cat([self(features[:, i:i+1], context[i:i+1]) for i in range(features.shape[1])])
        state = {}
        logits = torch.stack([self.readout.forward_step(features[t], state, t)
                              for t in range(features.shape[0])]).mean(0)
        return torch.tanh(logits + self.context(context)[:, None, :])


def build_critic():
    critic = nn.Sequential(nn.LayerNorm(256 + CONTEXT_DIM), nn.Linear(256 + CONTEXT_DIM, 128),
                           nn.Tanh(), nn.Linear(128, 1))
    nn.init.zeros_(critic[-1].weight)
    nn.init.zeros_(critic[-1].bias)
    return critic


def value(critic, features, context):
    return critic(torch.cat((features.mean((0, 2)), context), -1)).squeeze(-1)


def draw(mean, generator):
    latent = sample_action(mean, temporal_cholesky(mean.device, mean.dtype), mean.new_tensor(STD), generator)
    return latent, latent.clamp(-1, 1), log_prob(latent, mean)


def log_prob(latent, mean):
    if latent.shape != mean.shape or tuple(mean.shape[-2:]) != (12, 3):
        raise ValueError('Expected XYZ latent/mean[...,12,3]')
    std = mean.new_tensor(STD)
    chol = temporal_cholesky(mean.device, mean.dtype)
    white = torch.linalg.solve_triangular(chol, (latent-mean)/std, upper=False)
    # The legacy helper hardcodes12x7. This new36D density must use three channels.
    normalizer = 36*.5*math.log(2*math.pi) + 12*std.log().sum() + 3*chol.diag().log().sum()
    return -.5*white.square().sum((-2, -1))-normalizer


def require_artifact(artifact, base_sha):
    if artifact.get('kind') != KIND or artifact.get('base_checkpoint_sha256') != base_sha:
        raise ValueError('Wrong XYZ artifact/base identity')
    if artifact.get('context_contract') != CONTEXT_CONTRACT or artifact.get('xyz_contract') != XYZ_CONTRACT:
        raise ValueError('Wrong new XYZ context/executor contract')
    if artifact.get('trainable_keys') != list(TRAINABLE) or artifact.get('std') != list(STD):
        raise ValueError('Wrong XYZ trainable scope or behavior distribution')
