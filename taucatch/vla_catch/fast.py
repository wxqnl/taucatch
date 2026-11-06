"""Fast fixed-weight inference for the catch policy, after the SDVLA v1 runtime (cts-v1-pretrain,
turbovla/evaluation/runtime.py): running statistics checked once instead of on every call, the fixed
instruction encoded once, optional torch.compile, and one CUDA Graph per batch size. Options as the v1 runtime's
fastest mode: both views in one backbone pass, FP16 convolutions where the input is a MultiSpike output (k/4, exact
in FP16; weights rounded, FP32 accumulation), kernel autotuning; and the spiking timesteps folded into the batch
(fold_time).

In the eager ICML forward, 212 of the 213 host synchronisations per observation are the running-statistics
checks in MaskedTemporalBatchNorm; once verify_running_norms has passed they are dropped and the whole forward
(pixel normalisation to denormalised chunk) replays as one graph.

    CUDA_VISIBLE_DEVICES=0 python -m taucatch.vla_catch.fast runs_v2/trace_v1_f7/ema_8000.pt --data data_v2/bc_v1
"""
import argparse
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import IMAGE_MEAN, IMAGE_STD, INSTRUCTION

FASTEST = {"compile": True, "batch_views": True, "half": True, "tf32": True, "fold": True}   # GraphPolicy options


def _eval_norm(self, current, valid=None, time_index=0):
    """MaskedTemporalBatchNorm.forward in eval mode, without the per-call statistics check (same arithmetic)."""
    x = current.float()
    if valid is not None:
        valid = valid.expand(*x.shape[:-1], 1)
        x = x.masked_fill(~valid, 0)
    normalized = (x - self.running_mean[time_index]) * torch.rsqrt(self.running_var[time_index] + self.eps)
    normalized = normalized * self.weight + self.bias
    return normalized if valid is None else normalized.masked_fill(~valid, 0)


def prepare(model, batch=1, selective=False):
    """Freeze an eval model for graph replay: verified statistics, no host syncs, instruction features fixed;
    selective: instruction features read from buffers [batch, ...] (returned), one row per toss.
    In eval the VLA encodes one observation at a time; its row is told by the placeholder instruction ROWS[i].
    A released model (taucatch.spikevla) is already in that form."""
    if model.training:
        raise ValueError("fast inference needs an eval model")
    if getattr(model, "release", False):
        if selective:
            raise ValueError("released models hold one instruction")
        return None
    from turbovla.models.components.wta_operators import MaskedTemporalBatchNorm, verify_running_norms
    verify_running_norms(model)
    for m in model.modules():
        if isinstance(m, MaskedTemporalBatchNorm):
            m.forward = types.MethodType(_eval_norm, m)
    device = next(model.parameters()).device
    encoder = model.text_encoder
    if not hasattr(encoder, "cached_encode"):
        encoder.cached_encode = encoder.encode_bert_hidden          # policy.cache_text
    if not selective:
        text = encoder.cached_encode([INSTRUCTION], device)
        encoder.encode_bert_hidden = lambda instructions, device: text
        return None
    buffers = tuple(x.clone() for x in encoder.cached_encode([INSTRUCTION] * batch, device))
    rows = {name: i for i, name in enumerate(ROWS[:batch])}

    def lookup(instructions, device):
        if len(instructions) == batch:
            return buffers
        i = rows[instructions[0]]
        return tuple(b[i:i + 1] for b in buffers)
    encoder.encode_bert_hidden = lookup
    return buffers


ROWS = tuple(f"<toss {i}>" for i in range(256))


class HalfSpikeConv(nn.Module):
    """FP16 convolution for a MultiSpike input; FP32 out."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.register_buffer("weight", conv.weight.detach().half(), persistent=False)
        self.register_buffer("bias", None if conv.bias is None else conv.bias.detach().half(), persistent=False)
        self.fn = F.conv1d if conv.weight.ndim == 3 else F.conv2d

    def forward(self, x):
        c = self.conv
        return self.fn(x.half(), self.weight, self.bias, c.stride, c.padding, c.dilation, c.groups).float()


SPIKE_INPUT_CONVS = {          # SDT-V3 convolutions whose input is a MultiSpike output (v1 runtime, mixed_fp16_ext)
    "MS_MLP": ("fc1_conv", "fc2_conv"),
    "MS_ConvBlock": ("conv1", "conv2"),
    "MS_ConvBlock_spike_SepConv": ("conv1", "conv2"),
    "MS_DownSampling": ("encode_conv",),
    "MS_Attention_linear": ("q_conv.0", "k_conv.0", "v_conv.0", "proj_conv.0"),
}


def half_spike_convs(vla):
    """Swap the spike-input convolutions of the backbone for FP16 ones -> number swapped."""
    n = 0
    for parent in list(vla.vision_encoder.modules()):
        names = SPIKE_INPUT_CONVS.get(type(parent).__name__, ())
        if type(parent).__name__ == "MS_DownSampling" and not hasattr(parent, "encode_spike"):
            names = ()          # the stem sees continuous pixels
        for name in names:
            owner, _, leaf = name.rpartition(".")
            owner = parent.get_submodule(owner) if owner else parent
            conv = getattr(owner, leaf)
            if isinstance(conv, (nn.Conv1d, nn.Conv2d)) and conv.groups == 1:
                setattr(owner, leaf, HalfSpikeConv(conv))
                n += 1
    return n


def check_spike_inputs(vla, *inputs, call):
    """One forward with hooks: every HalfSpikeConv input must be k/4 in [0, 1]."""
    bad = []

    def hook(mod, args):
        x = args[0].float()
        if not (((x * 4).round() == x * 4).all() and x.min() >= 0 and x.max() <= 1):
            bad.append(mod)
    handles = [m.register_forward_pre_hook(hook) for m in vla.modules() if isinstance(m, HalfSpikeConv)]
    try:
        call(*inputs)
    finally:
        for h in handles:
            h.remove()
    if bad:
        raise RuntimeError(f"{len(bad)} FP16 convolutions saw non-spike inputs")


def register_sources(model):
    """The vendored SDT-V3 source is loaded by file path; torch.compile looks module globals up in sys.modules."""
    for m in model.modules():
        name = type(m).__module__
        fn = next((v for v in vars(type(m)).values() if isinstance(v, types.FunctionType)), None)
        if name not in sys.modules and fn is not None:
            module = types.ModuleType(name)
            module.__dict__.update(fn.__globals__)
            sys.modules[name] = module


# ----- time folded into the batch ---------------------------------------------------------------------------------
# The fusion, the projections and the action head run T = 4 spiking timesteps per observation, each a chain of small
# GEMMs: 568 of the 638 GEMMs of a call, latency-bound. Across timesteps a layer depends on itself only through its
# neurons' membranes (elementwise) and, in the action head, the integrated QK match (a running mean). fold_time runs
# each layer once on the T timesteps stacked into the batch: every Linear and QK product is one GEMM over all T, the
# neurons step through T inside, the per-timestep BN reads its own statistics per row block. Elementwise arithmetic is
# that of the timestep loop; GEMMs over stacked rows may round differently.

def _stack_time(x, T):
    """[B, ...] -> [T * B, ...], the same input at every timestep."""
    return x.unsqueeze(0).expand(T, *x.shape).flatten(0, 1)


def _fold_node(node, T):
    step = type(node).forward

    def forward(current, membrane=None, threshold_scale=1.0, valid=None):
        def split(a):
            return a.unflatten(0, (T, -1)) if torch.is_tensor(a) and a.ndim and a.shape[0] == current.shape[0] else None
        x, ok, scale = current.unflatten(0, (T, -1)), split(valid), split(threshold_scale)
        spikes, m = [], None
        for t in range(T):
            s, m = step(node, x[t], m, threshold_scale if scale is None else scale[t], valid if ok is None else ok[t])
            spikes.append(s)
        return torch.stack(spikes).flatten(0, 1), None
    return forward


def _fold_norm(self, current, valid=None, time_index=0):
    """_eval_norm with timestep t's statistics on row block t of [T * B, ...]."""
    T = self.running_mean.shape[0]
    x = current.float().unflatten(0, (T, -1))
    stats = (T,) + (1,) * (x.ndim - 2) + (self.channels,)
    if valid is not None:
        valid = valid.unflatten(0, (T, -1)) if valid.shape[0] == current.shape[0] else valid
        valid = valid.expand(*x.shape[:-1], 1)
        x = x.masked_fill(~valid, 0)
    normalized = (x - self.running_mean.view(stats)) * torch.rsqrt(self.running_var.view(stats) + self.eps)
    normalized = normalized * self.weight + self.bias
    return (normalized if valid is None else normalized.masked_fill(~valid, 0)).flatten(0, 1)


def _sdsa_folded(self, v, l, attention_mask_v=None, attention_mask_l=None):
    """SpikeBiMultiHeadAttention._forward_spike with the T timesteps in the batch."""
    T, bsz, nv = self.timesteps, v.shape[0], v.shape[1]
    nl = l.shape[1]
    valid_v, valid_l = self._valid(v, attention_mask_v), self._valid(l, attention_mask_l)
    v, l = v.masked_fill(~valid_v, 0.0), l.masked_fill(~valid_l, 0.0)
    source_count_v = valid_v.sum(1, keepdim=True).clamp_min(1).to(v.dtype)
    source_count_l = valid_l.sum(1, keepdim=True).clamp_min(1).to(l.dtype)
    fv, fl = _stack_time(valid_v, T), _stack_time(valid_l, T)
    nodes = self.spike_nodes

    def fire(name, current, valid, scale=1.0):
        return nodes[name](current, None, scale, valid)[0]
    input_v = fire("input_v", _stack_time(v, T), fv)
    input_l = fire("input_l", _stack_time(l, T), fl)
    q = fire("query_v", nn.functional.layer_norm(self.v_proj(input_v), (self.embed_dim,)), fv)
    k = fire("key_l", nn.functional.layer_norm(self.l_proj(input_l), (self.embed_dim,)), fl)
    vv = fire("value_v", nn.functional.layer_norm(self.values_v_proj(input_v), (self.embed_dim,)), fv)
    vl = fire("value_l", nn.functional.layer_norm(self.values_l_proj(input_l), (self.embed_dim,)), fl)
    q, k = self._shape(q, nv, T * bsz), self._shape(k, nl, T * bsz)
    vv, vl = self._shape(vv, nv, T * bsz), self._shape(vl, nl, T * bsz)
    counts = q @ k.transpose(-1, -2)
    context_v = (counts @ vl).transpose(1, 2).reshape(T * bsz, nv, self.embed_dim)
    context_l = (counts.transpose(-1, -2) @ vv).transpose(1, 2).reshape(T * bsz, nl, self.embed_dim)
    sv = fire("context_v", context_v, fv, _stack_time(source_count_l, T))
    sl = fire("context_l", context_l, fl, _stack_time(source_count_v, T))
    dv, dl = self.out_v_proj(sv).unflatten(0, (T, -1)), self.out_l_proj(sl).unflatten(0, (T, -1))
    output_v, output_l = dv[0], dl[0]
    for t in range(1, T):                      # the loop's summation order
        output_v, output_l = output_v + dv[t], output_l + dl[t]
    keep_v = valid_v & valid_l.any(1, keepdim=True)
    keep_l = valid_l & valid_v.any(1, keepdim=True)
    return (output_v / T).masked_fill(~keep_v, 0.0), (output_l / T).masked_fill(~keep_l, 0.0)


def _text_layer_folded(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
    """SpikeTextEncoderLayer.forward (eval) with the T timesteps in the batch."""
    T = self.config.cross_timesteps
    current = src.transpose(0, 1)
    batch, length = current.shape[:2]
    if src_key_padding_mask is None:
        valid = torch.ones((batch, length, 1), device=src.device, dtype=torch.bool)
    else:
        valid = ~src_key_padding_mask[..., None]
    allowed = valid & valid.transpose(-1, -2)
    if src_mask is not None:
        allowed = allowed & ~src_mask
    out = self.forward_step(_stack_time(current, T), _stack_time(valid, T), _stack_time(allowed, T), {}, 0)
    return out.unflatten(0, (T, -1)).mean(0).transpose(0, 1)


def _ifr_folded(T):
    def forward_step(self, query, memory, memory_valid, state, t):
        """IFRCrossAttention.forward_step on [T * B, ...]: the integrated match is the running mean over timesteps."""
        query_valid = torch.ones((*query.shape[:2], 1), dtype=torch.bool, device=query.device)

        def fire(name, current, valid):
            return self.spike_nodes[name](self.norms[name](current, valid, t), None, valid=valid)[0]
        if self.continuous_inputs:
            query_input = self.norms["input_query"](query, query_valid, t)
            memory_input = self.norms["input_memory"](memory, memory_valid, t)
        else:
            query_input = fire("input_query", query, query_valid)
            memory_input = fire("input_memory", memory, memory_valid)
        q = fire("query", self.query(query_input), query_valid)
        k = fire("key", self.key(memory_input), memory_valid)
        v = self.norms["value"](self.value(memory_input), memory_valid, t)
        batch, nq, channels = q.shape
        nk = k.shape[1]
        q = q.reshape(batch, nq, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        v = self.read_values(v.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2), None)
        allowed = query_valid & memory_valid.transpose(-1, -2)
        scores = q @ k.transpose(-1, -2)
        if self.integrate:
            per_step = scores.unflatten(0, (T, -1))
            total, steps = per_step[0], [per_step[0] / 1]
            for i in range(1, T):
                total = total + per_step[i]
                steps.append(total / (i + 1))
            scores = torch.stack(steps).flatten(0, 1)
        if self.shift:
            scores = scores * 2.0 ** -self.shift
        scores = self.spatial_scores(scores)
        if self.state_tokens:
            s = self.state_tokens
            route = torch.cat((self.routing(scores[..., :-s], allowed[..., :-s]),
                               self.routing(scores[..., -s:], allowed[..., -s:])), dim=-1)
        else:
            route = self.routing(scores, allowed)
        context = self.read(route, v).transpose(1, 2).reshape(batch, nq, channels)
        live = query_valid & memory_valid.any(dim=1, keepdim=True)
        return self.output(fire("context", context, live)).masked_fill(~live, 0.0)
    return forward_step


def _decoder_folded(self, memories, memory_valid, anchor=None):
    """SpikeACTDecoder.forward with the T timesteps in the batch (no hand anchor)."""
    T = memories.shape[0]
    queries = self.action_queries.weight[None].expand(memories.shape[1], -1, -1)
    sequence = _stack_time(queries, T)
    memory, valid = memories.flatten(0, 1), _stack_time(memory_valid, T)
    for layer in self.layers:
        sequence = layer.forward_step(sequence, memory, valid, {}, 0)
    currents = self.action_projection.forward_step(sequence, {}, 0)
    return torch.tanh(currents.unflatten(0, (T, -1)).mean(0))


def _cross_folded(self, v, l, padding_l):
    """taucatch.spikevla SpikeCrossAttention.forward with the T timesteps in the batch (as _sdsa_folded)."""
    T, bsz, nv = self.timesteps, v.shape[0], v.shape[1]
    nl = l.shape[1]
    valid_v = torch.ones((bsz, nv, 1), device=v.device, dtype=torch.bool)
    valid_l = ~padding_l.unsqueeze(-1)
    v, l = v.masked_fill(~valid_v, 0.0), l.masked_fill(~valid_l, 0.0)
    count_v = valid_v.sum(1, keepdim=True).clamp_min(1).to(v.dtype)
    count_l = valid_l.sum(1, keepdim=True).clamp_min(1).to(l.dtype)
    fv, fl = _stack_time(valid_v, T), _stack_time(valid_l, T)
    nodes = self.spike_nodes

    def fire(name, current, valid, scale=1.0):
        return nodes[name](current, None, scale, valid)[0]
    input_v = fire("input_v", _stack_time(v, T), fv)
    input_l = fire("input_l", _stack_time(l, T), fl)
    q = fire("query_v", F.layer_norm(self.v_proj(input_v), (self.embed_dim,)), fv)
    k = fire("key_l", F.layer_norm(self.l_proj(input_l), (self.embed_dim,)), fl)
    vv = fire("value_v", F.layer_norm(self.values_v_proj(input_v), (self.embed_dim,)), fv)
    vl = fire("value_l", F.layer_norm(self.values_l_proj(input_l), (self.embed_dim,)), fl)
    q, k = self._shape(q, nv, T * bsz), self._shape(k, nl, T * bsz)
    vv, vl = self._shape(vv, nv, T * bsz), self._shape(vl, nl, T * bsz)
    counts = q @ k.transpose(-1, -2)
    context_v = (counts @ vl).transpose(1, 2).reshape(T * bsz, nv, self.embed_dim)
    context_l = (counts.transpose(-1, -2) @ vv).transpose(1, 2).reshape(T * bsz, nl, self.embed_dim)
    dv = self.out_v_proj(fire("context_v", context_v, fv, _stack_time(count_l, T))).unflatten(0, (T, -1))
    dl = self.out_l_proj(fire("context_l", context_l, fl, _stack_time(count_v, T))).unflatten(0, (T, -1))
    out_v, out_l = dv[0], dl[0]
    for t in range(1, T):                      # the loop's summation order
        out_v, out_l = out_v + dv[t], out_l + dl[t]
    keep_v = valid_v & valid_l.any(1, keepdim=True)
    keep_l = valid_l & valid_v.any(1, keepdim=True)
    return (out_v / T).masked_fill(~keep_v, 0.0), (out_l / T).masked_fill(~keep_l, 0.0)


def _text_folded(self, text, padding, blocked):
    """taucatch.spikevla TextLayer.forward with the T timesteps in the batch."""
    T = self.timesteps
    valid = ~padding[..., None]
    allowed = valid & valid.transpose(-1, -2) & ~blocked
    out = self.forward_step(_stack_time(text, T), _stack_time(valid, T), _stack_time(allowed, T), {}, 0)
    return out.unflatten(0, (T, -1)).mean(0)


def _sgsa_scores_folded(T):
    def scores(self, q, k, state):
        """SGSA.scores on [T * B, ...]: the running mean of the spike match over the timestep blocks."""
        per_step = (q @ k.transpose(-1, -2)).unflatten(0, (T, -1))
        total, steps = per_step[0], [per_step[0] / 1]
        for i in range(1, T):
            total = total + per_step[i]
            steps.append(total / (i + 1))
        return torch.stack(steps).flatten(0, 1) * 2.0 ** -self.shift
    return scores


def fold_release(vla):
    """fold_time for a released model (taucatch.spikevla) -> number of patched modules."""
    from taucatch.spikevla.action_head import SGSA
    from taucatch.spikevla.fusion import SpikeCrossAttention, TextLayer
    from taucatch.spikevla.neurons import PLIF, TemporalNorm
    T = vla.timesteps
    if vla.action_head.timesteps != T:
        raise ValueError("fold_time covers equal fusion / head timesteps")
    decoder = vla.action_head.decoder
    vision, text = vla.vision_projection, vla.text_encoder.text_projection
    n = 0
    for scope in (vision, text, vla.vision_language_interaction, decoder):
        for m in scope.modules():
            if isinstance(m, PLIF):                     # BipolarPLIF derives from it
                m.forward = _fold_node(m, T)
            elif isinstance(m, TemporalNorm):
                m.forward = types.MethodType(_fold_norm, m)
            elif isinstance(m, SpikeCrossAttention):
                m.forward = types.MethodType(_cross_folded, m)
            elif isinstance(m, TextLayer):
                m.forward = types.MethodType(_text_folded, m)
            elif isinstance(m, SGSA):
                m.scores = types.MethodType(_sgsa_scores_folded(T), m)
            else:
                continue
            n += 1
    decoder.forward = types.MethodType(_decoder_folded, decoder)
    # The projections see the same input at every timestep: all T are computed at t = 0, then read back.
    vision_step, text_step = type(vision).forward_step, type(text).forward_step

    def vision_projection_step(tokens, state, time_index=0):
        if time_index == 0:
            state["folded"] = vision_step(vision, _stack_time(tokens, T), {}, 0).unflatten(0, (T, -1))
        return state["folded"][time_index]

    def text_projection_step(hidden, state, valid, time_index=0):
        if time_index == 0:
            state["folded"] = text_step(text, _stack_time(hidden, T), {}, _stack_time(valid, T), 0).unflatten(0, (T, -1))
        return state["folded"][time_index]
    vision.forward_step, text.forward_step = vision_projection_step, text_projection_step
    return n + 3


def fold_time(vla):
    """Patch an eval TurboVLA (after prepare) to run its T-step spiking stages folded -> number of patched modules."""
    if getattr(vla, "release", False):
        return fold_release(vla)
    from turbovla.models.components.spike_fusion import FunctionalPLIF, SpikeBiMultiHeadAttention
    from turbovla.models.components.spike_text_only import SpikeTextEncoderLayer
    from turbovla.models.components.wta_operators import MaskedTemporalBatchNorm
    from turbovla.models.shiftmax_action_head import IFRCrossAttention
    T = vla.config.interaction.cross_timesteps
    if vla.config.action.spike_timesteps != T or vla.hand_locator is not None or vla.fine_merge is not None:
        raise ValueError("fold_time covers equal fusion / head timesteps without hand anchor or fine merge")
    decoder = vla.action_head.decoder
    scopes = (vla.vision_projection, vla.text_encoder.text_projection, vla.vision_language_interaction, decoder)
    n = 0
    for scope in scopes:
        for m in scope.modules():
            if isinstance(m, FunctionalPLIF):           # BipolarPLIF, IntegerPLIF derive from it
                m.forward = _fold_node(m, T)
            elif isinstance(m, MaskedTemporalBatchNorm):
                m.forward = types.MethodType(_fold_norm, m)
            elif isinstance(m, SpikeBiMultiHeadAttention):
                m._forward_spike = types.MethodType(_sdsa_folded, m)
            elif isinstance(m, SpikeTextEncoderLayer):
                m.forward = types.MethodType(_text_layer_folded, m)
            elif isinstance(m, IFRCrossAttention):
                m.forward_step = types.MethodType(_ifr_folded(T), m)
            else:
                continue
            n += 1
    decoder.forward = types.MethodType(_decoder_folded, decoder)
    # The projections see the same input at every timestep: all T are computed at t = 0, then read back.
    visual_step, projection = type(vla)._project_visual_step, vla.text_encoder.text_projection
    text_step = type(projection).forward_step

    def project_visual_step(tokens, blocks, state, t):
        if t == 0:
            state["folded"] = visual_step(vla, _stack_time(tokens, T), None, {}, 0).unflatten(0, (T, -1))
        return state["folded"][t]

    def text_projection_step(hidden, state, valid, time_index=0):
        if time_index == 0:
            state["folded"] = text_step(projection, _stack_time(hidden, T), {}, _stack_time(valid, T),
                                        0).unflatten(0, (T, -1))
        return state["folded"][time_index]
    vla._project_visual_step = project_visual_step
    projection.forward_step = text_projection_step
    return n + 3


class GraphPolicy:
    """uint8 frames [B, 2, H, W, 3] and normalised state [B, S] (CUDA) -> normalised chunk [B, 12, D];
    with the ball locator (locator.CatchPolicy) also memory [B, M, 3] and valid [B, M] in and out."""

    def __init__(self, model, batch, shape=(2, 224, 224, 3), compile=False, batch_views=False, half=False,
                 autotune=False, tf32=False, fold=False):
        vla = getattr(model, "vla", model)
        self.selective = getattr(model, "selective", False)
        self.text = prepare(vla, batch, self.selective)
        self.encode = getattr(vla.text_encoder, "cached_encode", None)
        if batch_views and getattr(vla, "release", False):
            vla.vision_encoder.batch_views = True
        elif batch_views:
            vla.vision_encoder.config.encode_views_separately = False
        self.swapped = half_spike_convs(vla) if half else 0
        self.folded = fold_time(vla) if fold else 0
        if tf32:                  # the remaining FP32 GEMMs (fusion, action head) on TF32 tensor cores
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        self.model, self.batch = model, batch
        dev = next(model.parameters()).device
        mean = torch.tensor(IMAGE_MEAN, device=dev)[:, None, None]
        std = torch.tensor(IMAGE_STD, device=dev)[:, None, None]

        def forward(frames, *rest):
            x = (frames.permute(0, 1, 4, 2, 3).float() / 255.0 - mean) / std
            out = model(list(ROWS[:batch]) if self.selective else [INSTRUCTION] * batch, x, *rest)
            return out.float() if torch.is_tensor(out) else (out[0].float(), out[2], out[3])
        if compile:
            register_sources(model)
            options = {"triton.cudagraphs": False, "freezing": False}
            if autotune:
                options.update({"max_autotune": True, "coordinate_descent_tuning": True})
            forward = torch.compile(forward, fullgraph=True, dynamic=False, options=options)
        self.forward = forward      # the graph reads mean/std from this closure: keep them allocated
        self.inputs = [torch.zeros((batch, *shape), dtype=torch.uint8, device=dev)]
        if vla is model:
            self.inputs.append(torch.zeros((batch, vla.config.action.state_dim), device=dev))
        else:
            m = model.locator.memory
            self.inputs += [torch.zeros((batch, 8), device=dev), torch.zeros((batch, m, 3), device=dev),
                            torch.zeros((batch, m), dtype=torch.bool, device=dev)]
        start = time.perf_counter()
        stream = torch.cuda.Stream(dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                forward(*self.inputs)
        torch.cuda.current_stream(dev).wait_stream(stream)
        torch.cuda.synchronize(dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.out = forward(*self.inputs)
        self.startup_s = time.perf_counter() - start

    def __call__(self, *inputs, instructions=None):
        for buf, x in zip(self.inputs, inputs):
            buf.copy_(x, non_blocking=True)
        if self.selective:
            for buf, x in zip(self.text, self.encode(list(instructions), self.inputs[0].device)):
                buf.copy_(x)
        self.graph.replay()
        return self.out.clone() if torch.is_tensor(self.out) else tuple(x.clone() for x in self.out)


def main():
    from .ball import track_memory
    from .data import load_tosses
    from .evaluate import Policy
    from .policy import pixel_values, set_numerics
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="data_v2/bc_v1")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--batch-views", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--fold", action="store_true")
    args = ap.parse_args()
    set_numerics()
    pol = Policy(args.ckpt)
    s = pol.stats
    tosses = load_tosses(args.data, lambda m: m["caught"] and m["seed"] % 20 == 0)[:args.samples]
    picks = [(t, len(t.times) // 2) for t in tosses]
    rng = np.random.default_rng(0)
    state = np.stack([t.state8[i] for t, i in picks])
    if pol.no_velocity:
        state[:, 3:6] = 0.0
    host = [torch.from_numpy(np.stack([t.frames(i, mode=pol.frames) for t, i in picks])),
            torch.from_numpy((state - s["state_mean"][:8]) / s["state_std"][:8]).float()]
    if pol.locator:
        memory, valid = zip(*(track_memory(t, i, rng) for t, i in picks))
        host += [torch.from_numpy(np.stack(memory)), torch.from_numpy(np.stack(valid))]
    host = [x.pin_memory() for x in host]
    sample = lambda i: [x[i:i + 1] for x in host]

    def eager(*inputs):
        frames, *rest = (x.cuda() for x in inputs)
        out = pol.model([INSTRUCTION], pixel_values(frames), *rest)
        return out[0].float() if pol.locator else out.float()
    with torch.no_grad():
        ref = torch.cat([eager(*sample(i)) for i in range(len(picks))])

    def timed(fn):
        times = []
        for k in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(*sample(k % len(picks))).cpu()            # observation on the host -> chunk on the host
            times.append(time.perf_counter() - t0)
        return 1000 * np.median(times[5:]), 1000 * np.percentile(times[5:], 95)
    with torch.no_grad():
        med, p95 = timed(eager)
    print(f"eager          median {med:6.2f} ms  p95 {p95:6.2f} ms", flush=True)
    fast = GraphPolicy(pol.model, 1, tuple(host[0].shape[1:]), compile=args.compile, batch_views=args.batch_views,
                       half=args.half, autotune=args.autotune, tf32=args.tf32, fold=args.fold)
    if args.half:
        with torch.no_grad():
            check_spike_inputs(pol.model, *sample(0), call=eager)

    def run(*inputs):
        out = fast(*(x.cuda(non_blocking=True) for x in inputs))
        return out[0] if pol.locator else out
    with torch.inference_mode():
        out = torch.cat([run(*sample(i)) for i in range(len(picks))])
        med, p95 = timed(run)
    a, b = denorm(ref, s), denorm(out, s)
    name = "+".join(k for k in ("compile", "batch_views", "half", "autotune", "tf32", "fold") if getattr(args, k)) or "graph"
    print(f"{name:14s} median {med:6.2f} ms  p95 {p95:6.2f} ms  startup {fast.startup_s:.1f} s  "
          f"FP16 convs {fast.swapped}", flush=True)
    print(f"vs eager: max |normalised diff| {(ref - out).abs().max().item():.4g}, "
          f"max position diff {100 * np.abs(a[..., :3] - b[..., :3]).max():.3f} cm, "
          f"median {100 * np.median(np.abs(a[..., :3] - b[..., :3])):.4f} cm, "
          f"gripper flips {int(((a[..., 6] > 0.5) != (b[..., 6] > 0.5)).sum())}", flush=True)


def denorm(a, s):
    """As Policy.__call__: physical units, gripper command in [0, 1]."""
    a = a.cpu().numpy()
    out = a * s["action_std"] + s["action_mean"]
    out[..., 6] = (a[..., 6] + 1) / 2
    return out


if __name__ == "__main__":
    main()
