"""Spiking projections of the visual and text tokens, and the six vision-language fusion layers.

Each fusion layer is a bidirectional spike cross-attention between the visual and text tokens (the layout of the
Grounding DINO feature enhancer, https://github.com/IDEA-Research/GroundingDINO, Copyright (c) 2023 IDEA, Apache
License 2.0) followed by a spiking self-attention over the text with binary top-k routing.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .neurons import PLIF, BipolarPLIF, TemporalNorm, fire, multi_winner


class TextProjection(nn.Linear):
    """Text encoder states -> fusion width: input events, one synapse, a normalised current."""

    def __init__(self, in_features, out_features, timesteps, eps):
        super().__init__(in_features, out_features)
        self.input_norm = TemporalNorm(in_features, timesteps, eps)
        self.output_norm = TemporalNorm(out_features, timesteps, eps)
        self.spike_nodes = nn.ModuleDict({"input": BipolarPLIF()})

    def forward_step(self, hidden, state, valid, time_index=0):
        events = fire(self.spike_nodes, state, "input", self.input_norm(hidden, valid, time_index), valid)
        return self.output_norm(F.linear(events, self.weight, self.bias), valid, time_index).masked_fill(~valid, 0)


class VisionProjection(nn.Module):
    """Backbone tokens -> fusion width: input events, a hidden spiking layer and a direct synapse."""

    def __init__(self, in_dim, out_dim, hidden_dim, timesteps, eps):
        super().__init__()
        self.input_norm = TemporalNorm(in_dim, timesteps, eps)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.hidden_norm = TemporalNorm(hidden_dim, timesteps, eps)
        self.output_norm = TemporalNorm(out_dim, timesteps, eps)
        self.spike_nodes = nn.ModuleDict({"input": BipolarPLIF(), "hidden": BipolarPLIF()})

    def forward_step(self, tokens, state, time_index=0):
        events = fire(self.spike_nodes, state, "input", self.input_norm(tokens, time_index=time_index))
        hidden = fire(self.spike_nodes, state, "hidden", self.hidden_norm(self.linear1(events), time_index=time_index))
        return self.output_norm(self.skip(events) + self.linear2(hidden), time_index=time_index)


class TextSelfAttention(nn.Module):
    """Spiking self-attention over the text tokens: event Q/K/V, each query reads its top-k keys (binary routing)."""

    NAMES = ("input", "q", "k", "v", "context")

    def __init__(self, dim, heads, topk, timesteps, eps):
        super().__init__()
        self.heads, self.head_dim, self.topk = heads, dim // heads, topk
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norms = nn.ModuleDict({name: TemporalNorm(dim, timesteps, eps) for name in self.NAMES})
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in self.NAMES})

    def forward_step(self, src, valid, allowed, state, time_index=0):
        def encode(name, current, mask=valid):
            return fire(self.spike_nodes, state, name, self.norms[name](current, mask, time_index), mask)

        def heads(current):
            return current.reshape(current.shape[0], current.shape[1], self.heads, self.head_dim).transpose(1, 2)

        q, k, v = self.qkv(encode("input", src)).chunk(3, dim=-1)
        q, k, v = heads(encode("q", q)), heads(encode("k", k)), heads(encode("v", v))
        # a query or key without any event takes no part
        allowed = allowed[:, None].expand(-1, self.heads, -1, -1)
        allowed = allowed & q.ne(0).any(-1, keepdim=True) & k.ne(0).any(-1).unsqueeze(-2)
        scores = q @ k.transpose(-1, -2)
        route = multi_winner(scores, allowed, min(self.topk, scores.shape[-1]))
        context = (route @ v).transpose(1, 2).reshape(src.shape)
        keep = valid & route.ne(0).any(-1).any(1).unsqueeze(-1)
        return self.out_proj(encode("context", context, keep)).masked_fill(~keep, 0)


class TextLayer(nn.Module):
    """Text self-attention and a spiking FFN, a constant input current integrated over the timesteps."""

    def __init__(self, dim, inner_dim, heads, topk, timesteps, eps):
        super().__init__()
        self.timesteps = timesteps
        self.self_attn = TextSelfAttention(dim, heads, topk, timesteps, eps)
        self.linear1 = nn.Linear(dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, dim)
        self.norms = nn.ModuleDict({"input": TemporalNorm(dim, timesteps, eps),
                                    "hidden": TemporalNorm(inner_dim, timesteps, eps),
                                    "output": TemporalNorm(dim, timesteps, eps)})
        self.spike_nodes = nn.ModuleDict({"input": BipolarPLIF(), "hidden": BipolarPLIF()})

    def forward_step(self, src, valid, allowed, state, time_index):
        src = src + self.self_attn.forward_step(src, valid, allowed, state.setdefault("attention", {}), time_index)
        ffn = state.setdefault("ffn", {})
        events = fire(self.spike_nodes, ffn, "input", self.norms["input"](src, valid, time_index), valid)
        hidden = fire(self.spike_nodes, ffn, "hidden", self.norms["hidden"](self.linear1(events), valid, time_index),
                      valid)
        return (src + self.norms["output"](self.linear2(hidden), valid, time_index)).masked_fill(~valid, 0)

    def forward(self, text, padding, blocked):
        """text [B, N, C], padding [B, N] (True: no token), blocked [B, N, N] (True: other sub-sentence)."""
        valid = ~padding[..., None]
        allowed = valid & valid.transpose(-1, -2) & ~blocked
        state = {}
        steps = [self.forward_step(text, valid, allowed, state, t) for t in range(self.timesteps)]
        return torch.stack(steps).mean(0)


class SpikeCrossAttention(nn.Module):
    """Bidirectional cross-attention on events: one Q/K spike match drives both the visual and the text read."""

    def __init__(self, v_dim, l_dim, embed_dim, num_heads, timesteps):
        super().__init__()
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.timesteps = timesteps
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)
        self.spike_nodes = nn.ModuleDict({name: PLIF() for name in
                                          ("input_v", "input_l", "query_v", "key_l", "value_v", "value_l")})
        self.spike_nodes["context_v"] = PLIF(self.head_dim / 64.0)
        self.spike_nodes["context_l"] = PLIF(self.head_dim / 64.0)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, v, l, padding_l):
        bsz, nv, _ = v.shape
        nl = l.shape[1]
        valid_v = torch.ones((bsz, nv, 1), device=v.device, dtype=torch.bool)
        valid_l = ~padding_l.unsqueeze(-1)
        v, l = v.masked_fill(~valid_v, 0.0), l.masked_fill(~valid_l, 0.0)
        count_v = valid_v.sum(1, keepdim=True).clamp_min(1).to(v.dtype)
        count_l = valid_l.sum(1, keepdim=True).clamp_min(1).to(l.dtype)
        state = {name: None for name in self.spike_nodes}

        def fire_(name, current, valid, scale=1.0):
            events, state[name] = self.spike_nodes[name](current, state[name], scale, valid)
            return events

        out_v = out_l = None
        for _ in range(self.timesteps):
            input_v, input_l = fire_("input_v", v, valid_v), fire_("input_l", l, valid_l)
            q = fire_("query_v", F.layer_norm(self.v_proj(input_v), (self.embed_dim,)), valid_v)
            k = fire_("key_l", F.layer_norm(self.l_proj(input_l), (self.embed_dim,)), valid_l)
            vv = fire_("value_v", F.layer_norm(self.values_v_proj(input_v), (self.embed_dim,)), valid_v)
            vl = fire_("value_l", F.layer_norm(self.values_l_proj(input_l), (self.embed_dim,)), valid_l)
            q, k = self._shape(q, nv, bsz), self._shape(k, nl, bsz)
            vv, vl = self._shape(vv, nv, bsz), self._shape(vl, nl, bsz)
            counts = q @ k.transpose(-1, -2)
            context_v = (counts @ vl).transpose(1, 2).reshape(bsz, nv, self.embed_dim)
            context_l = (counts.transpose(-1, -2) @ vv).transpose(1, 2).reshape(bsz, nl, self.embed_dim)
            dv = self.out_v_proj(fire_("context_v", context_v, valid_v, count_l))
            dl = self.out_l_proj(fire_("context_l", context_l, valid_l, count_v))
            out_v = dv if out_v is None else out_v + dv
            out_l = dl if out_l is None else out_l + dl
        keep_v = valid_v & valid_l.any(1, keepdim=True)
        keep_l = valid_l & valid_v.any(1, keepdim=True)
        return (out_v / self.timesteps).masked_fill(~keep_v, 0.0), (out_l / self.timesteps).masked_fill(~keep_l, 0.0)


class FusionLayer(nn.Module):
    """Pre-norm bidirectional cross-attention with layer scale, the residual from the normalised tokens."""

    def __init__(self, dim, embed_dim, num_heads, timesteps):
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(dim)
        self.layer_norm_l = nn.LayerNorm(dim)
        self.attn = SpikeCrossAttention(dim, dim, embed_dim, num_heads, timesteps)
        self.gamma_v = nn.Parameter(torch.ones(dim))
        self.gamma_l = nn.Parameter(torch.ones(dim))

    def forward(self, v, l, padding_l):
        v, l = self.layer_norm_v(v), self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, padding_l)
        return v + self.gamma_v * delta_v, l + self.gamma_l * delta_l


class VisionLanguageInteraction(nn.Module):
    def __init__(self, dim, inner_dim, nheads, layers, topk, timesteps, eps):
        super().__init__()
        heads = max(1, nheads // 2)
        self.text_layers = nn.ModuleList(TextLayer(dim, inner_dim, heads, topk, timesteps, eps) for _ in range(layers))
        self.fusion_layers = nn.ModuleList(FusionLayer(dim, inner_dim, heads, timesteps) for _ in range(layers))

    def forward(self, visual, text, padding, blocked):
        for fusion, text_layer in zip(self.fusion_layers, self.text_layers):
            visual, text = fusion(visual, text, padding)
            text = text_layer(text, padding, blocked)
        return visual, text
