"""Stage 2 of selective catching (plan §5.10): the spike-driven VLA as the chooser.

Runs in the `cts` conda environment (PyTorch 2.7), not in `taucatch`. The VLA is the pretrained
spike-driven-vla base (148.8M parameters, `spike_driven_vla_pretrain_60000.pth`, EMA weights).
Its two camera views take the overview with numbered marks and the crop mosaic (taucatch/choice.py),
its 8 state coordinates take speed and time to the catch window per candidate, and the
instruction goes in as text. The action readout is replaced by a tap that keeps the action
decoder's output per spiking time step; a new readout turns its mean into five scores: marks 1 to
4 and "none".
"""
import os
import sys

import torch
import torch.nn as nn

CTS_ROOT = "/home/user01/Minko/calude/cts-v1-pretrain"
SPIKEVLA_ROOT = "/home/user01/Minko/calude/Spike-driven-VLA"
ASSET_ROOT = "/home/user01/Minko/spike-new-vla-icml"
PRETRAINED = "/home/user01/Minko/calude/runs/v1_pretrain_full/checkpoints/spike_driven_vla_pretrain_60000.pth"
TEXT_PROJECTION = "/home/user01/Minko/calude/runs/v1_pretrain_full_text/text_projection.pth"
N_CHOICES = 5          # marks 1-4 and "none"


def _paths():
    for p in (SPIKEVLA_ROOT, CTS_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("SPIKE_WORK_ROOT", ASSET_ROOT)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ChoiceTap(nn.Module):
    """Stands in for the action readout: keeps the decoder output of every spiking time step and
    returns zero actions (the action output is not used)."""

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.steps = []

    def forward_step(self, x, state, t):
        if t == 0:
            self.steps = []
        self.steps.append(x)
        return x.new_zeros(*x.shape[:-1], self.action_dim)


class Chooser(nn.Module):
    def __init__(self, vla, hidden_dim):
        super().__init__()
        self.vla = vla
        dec = vla.action_head.decoder
        self.tap = ChoiceTap(vla.action_dim)
        dec.action_projection = self.tap
        self.readout = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, N_CHOICES))

    # The existing inference runtime inspects these parts before preparing its kernels.
    @property
    def config(self):
        return self.vla.config

    @property
    def spike_projectors(self):
        return self.vla.spike_projectors

    @property
    def text_encoder(self):
        return self.vla.text_encoder

    @property
    def vision_encoder(self):
        return self.vla.vision_encoder

    @property
    def vision_projection(self):
        return self.vla.vision_projection

    @property
    def action_head(self):
        return self.vla.action_head

    def _normalize_samples(self, samples):
        return self.vla._normalize_samples(samples)

    def forward(self, instructions, pixels, state, frozen_text_features=None):
        """instructions: list of B strings; pixels: normalized float [B, 2, 3, 224, 224]; state [B, 8]."""
        samples = pixels if isinstance(pixels, dict) else {"pixel_values": pixels}
        self.vla(instructions, samples, state, frozen_text_features=frozen_text_features)
        hidden = torch.stack(self.tap.steps).mean(0)          # [B, queries, hidden]
        return self.readout(hidden.mean(1))


def chooser_runtime(model, backend="cached", spike_mode="none"):
    """Reuse the user's runtime without editing its source or quantizing the new choice readout.

    `cached` works on CPU too. `compile_graph` plus `mixed_fp16_ext` is an explicit CUDA
    candidate; latency and choice agreement must be measured before using it for results.
    """
    _paths()
    from turbovla.evaluation.runtime import InferenceRuntime

    optimized = backend != "cached"
    return InferenceRuntime(model, backend=backend, cache_size=128, batch_views=optimized, cache_neurons=optimized,
                            hoist_action_memory=optimized, gpu_preprocess=optimized, spike_mode=spike_mode)


def build_chooser(checkpoint=PRETRAINED, run_dir="/tmp/chooser_run", ema=True):
    """The pretrained VLA (strict load) wrapped as a Chooser; returns (chooser, prepare_images)."""
    _paths()
    from experiments.pretrain.common import build_initialized_model, load_recipe, pretraining_args
    from spikevla.data import prepare_images

    args = pretraining_args(load_recipe(), run_dir, TEXT_PROJECTION)
    vla = build_initialized_model(args)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck["ema_model_state_dict"] if ema and ck.get("ema_model_state_dict") else ck["model_state_dict"]
    vla.load_state_dict(state, strict=True)
    hidden = vla.action_head.decoder.action_queries.weight.shape[1]
    return Chooser(vla, hidden), prepare_images


def to_pixels(over, mosaic, prepare_images, device, augment=False, generator=None):
    """uint8 [B, 224, 224, 3] x 2 -> normalized [B, 2, 3, 224, 224] (view 0 overview, view 1 mosaic)."""
    x = torch.stack([torch.as_tensor(over), torch.as_tensor(mosaic)], 1).permute(0, 1, 4, 2, 3).contiguous()
    return prepare_images(x.to(device), None, augment, generator)
