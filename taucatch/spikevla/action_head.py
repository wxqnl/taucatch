"""Spiking action head of SpikeVLA (inference): continuous state tokens, three spiking decoder layers over T = 4
timesteps and a spiking read-out of the 12-step action chunk.

Self-attention routes with Shiftmax (Bipolar Self-attention for Spiking Transformers, NeurIPS 2025: Q/K are signed
events, the weights powers of two). The cross-attention is Spatially Grounded Spike Attention (SGSA): each query-key
pair integrates its spike match over the timesteps, the scores are shifted by 2 bits, the normaliser is rounded to
1/16 of a power of two, and the visual values carry a fixed sign field and a learned position value of their patch.
"""
import math

import torch
from torch import nn

from .neurons import BipolarPLIF, TokenNorm, fire, shiftmax


class CoordinateStateProjection(nn.Module):
    """One token per state coordinate: a per-coordinate linear map, GELU, a shared linear map and a position."""

    def __init__(self, state_dim, hidden_dim, state_hidden_dim, eps):
        super().__init__()
        self.input_projection = nn.Module()
        self.input_projection.weight = nn.Parameter(torch.zeros(state_dim, state_hidden_dim, 1))
        self.input_projection.bias = nn.Parameter(torch.zeros(state_dim, state_hidden_dim))
        self.output_projection = nn.Linear(state_hidden_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, state_dim, hidden_dim))
        self.hidden_norm = TokenNorm(state_hidden_dim, eps=eps)
        self.output_norm = TokenNorm(hidden_dim, eps=eps)

    def forward(self, state):
        p = self.input_projection
        hidden = torch.einsum("bnk,ndk->bnd", state[..., None], p.weight) + p.bias
        hidden = nn.functional.gelu(self.hidden_norm(hidden))
        return self.output_norm(self.output_projection(hidden) + self.position)


class ShiftmaxAttention(nn.Module):
    """Continuous inputs, event Q/K, a normalised current as V, Shiftmax routing, a spiking context."""

    NORMS = ("input_query", "input_memory", "query", "key", "value", "context")

    def __init__(self, dim, heads, eps, bits=0):
        super().__init__()
        self.heads, self.head_dim, self.bits = heads, dim // heads, int(bits)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.norms = nn.ModuleDict({name: TokenNorm(dim, eps=eps) for name in self.NORMS})
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in ("query", "key", "context")})

    def encode(self, name, current, valid, state, t):
        return fire(self.spike_nodes, state, name, self.norms[name](current, valid, t), valid)

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def read_values(self, v):
        return v

    def scores(self, q, k, state):
        return q @ k.transpose(-1, -2)

    def forward_step(self, query, memory, memory_valid, state, t):
        query_valid = torch.ones((*query.shape[:2], 1), dtype=torch.bool, device=query.device)
        query_input = self.norms["input_query"](query, query_valid, t)
        memory_input = self.norms["input_memory"](memory, memory_valid, t)
        q = self.encode("query", self.query(query_input), query_valid, state, t)
        k = self.encode("key", self.key(memory_input), memory_valid, state, t)
        v = self.norms["value"](self.value(memory_input), memory_valid, t)
        batch, nq, channels = q.shape
        q, k, v = self.split(q), self.split(k), self.read_values(self.split(v))
        allowed = query_valid & memory_valid.transpose(-1, -2)
        route = shiftmax(self.scores(q, k, state), allowed, self.bits)
        context = (route @ v).transpose(1, 2).reshape(batch, nq, channels)
        live = query_valid & memory_valid.any(dim=1, keepdim=True)
        return self.output(self.encode("context", context, live, state, t)).masked_fill(~live, 0.0)


class SGSA(ShiftmaxAttention):
    """Spatially Grounded Spike Attention: the cross-attention of the action head. Memory: views * grid^2 visual
    tokens first, then text and state tokens."""

    LOCATE_FREQUENCIES = (1, 2, 4, 8)
    SIGN_SCALES = (1, 2, 4, 8)

    def __init__(self, dim, heads, eps, views, grid, shift=2, bits=4):
        super().__init__(dim, heads, eps, bits)
        self.shift = int(shift)
        axis = torch.arange(grid, dtype=torch.float64) / max(grid - 1, 1)
        rows, cols = torch.meshgrid(axis, axis, indexing="ij")
        pattern = self.sign_codes(rows.flatten(), cols.flatten(), self.head_dim).repeat(views, 1).float()
        self.register_buffer("sign_pattern", pattern, persistent=False)
        axis = torch.arange(grid, dtype=torch.float32) / max(grid - 1, 1)
        rows, cols = torch.meshgrid(axis, axis, indexing="ij")
        features = torch.block_diag(*[self.locate_codes(rows.flatten(), cols.flatten())] * views)
        self.register_buffer("locate_features", features, persistent=False)
        self.locate_value = nn.Parameter(torch.zeros(features.shape[1], dim))

    @classmethod
    def sign_codes(cls, rows, cols, head_dim):
        """[N, head_dim] in {-1, +1}: the first half +1, the second half square waves sign(cos(pi s x + phase)) along
        rows and columns, scales SIGN_SCALES, phases pi/4 and 3pi/4, cycling."""
        waves = []
        for s in cls.SIGN_SCALES:
            for phase in (math.pi / 4, 3 * math.pi / 4):
                for coord in (rows, cols):
                    waves.append(torch.cos(math.pi * s * coord + phase))
        waves = torch.stack(waves, -1).sign()
        waves[waves == 0] = 1
        half = head_dim // 2
        modulated = waves.repeat(1, half // waves.shape[1] + 1)[:, :half]
        return torch.cat((torch.ones(len(rows), head_dim - half, dtype=torch.float64), modulated), -1)

    @classmethod
    def locate_codes(cls, rows, cols):
        """[N, 19]: 1, row, column, sin and cos of pi f row / column for f in LOCATE_FREQUENCIES."""
        coords = torch.stack((rows, cols), -1)
        angles = math.pi * coords[..., None] * torch.tensor(cls.LOCATE_FREQUENCIES, dtype=torch.float32)
        return torch.cat((torch.ones(len(coords), 1), coords, angles.sin().flatten(1), angles.cos().flatten(1)), -1)

    def read_values(self, v):
        """[B, H, Nk, head_dim]: the sign field, then each visual token's position value."""
        nvis = self.sign_pattern.shape[0]
        v = torch.cat((v[:, :, :nvis] * self.sign_pattern.to(v.dtype), v[:, :, nvis:]), dim=2)
        position = (self.locate_features @ self.locate_value).to(v.dtype)
        position = position.view(nvis, self.heads, self.head_dim).transpose(0, 1)
        return torch.cat((v[:, :, :nvis] + position, v[:, :, nvis:]), dim=2)

    def scores(self, q, k, state):
        """The running mean over timesteps of the spike match q_t . k_t, shifted right by `shift` bits."""
        total = q @ k.transpose(-1, -2)
        if state.get("match") is not None:
            total = state["match"] + total
        state["match"], state["steps"] = total, state.get("steps", 0) + 1
        return total / state["steps"] * 2.0 ** -self.shift


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim, eps, views, grid):
        super().__init__()
        self.self_attn = ShiftmaxAttention(dim, heads, eps)
        self.cross_attn = SGSA(dim, heads, eps, views, grid)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.norms = nn.ModuleDict({"input": TokenNorm(dim, eps=eps), "hidden": TokenNorm(ffn_dim, eps=eps)})
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in ("input", "hidden")})

    def forward_step(self, x, memory, memory_valid, state, t):
        query_valid = torch.ones((*x.shape[:2], 1), device=x.device, dtype=torch.bool)
        x = x + self.self_attn.forward_step(x, x, query_valid, state.setdefault("self", {}), t)
        x = x + self.cross_attn.forward_step(x, memory, memory_valid, state.setdefault("cross", {}), t)
        residual, ffn = x, state.setdefault("ffn", {})
        for name, linear in (("input", self.linear1), ("hidden", self.linear2)):
            x = linear(fire(self.spike_nodes, ffn, name, self.norms[name](x, time_index=t)))
        return residual + x


class Readout(nn.Module):
    """Three spiking layers; the chunk is tanh of the mean over timesteps of the last layer's current."""

    def __init__(self, dim, mlp_dim, action_dim, eps):
        super().__init__()
        dims = (dim, mlp_dim, mlp_dim, action_dim)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(TokenNorm(d, eps=eps) for d in dims[:-1])
        self.spike_nodes = nn.ModuleList(BipolarPLIF() for _ in dims[:-1])

    def forward_step(self, x, state, t):
        for i, (norm, node, linear) in enumerate(zip(self.norms, self.spike_nodes, self.layers)):
            events, state[i] = node(norm(x, time_index=t), state.get(i))
            x = linear(events)
        return x


class Decoder(nn.Module):
    def __init__(self, horizon, dim, heads, ffn_dim, layers, mlp_dim, action_dim, eps, views, grid):
        super().__init__()
        self.action_queries = nn.Embedding(horizon, dim)
        self.layers = nn.ModuleList(DecoderLayer(dim, heads, ffn_dim, eps, views, grid) for _ in range(layers))
        self.action_projection = Readout(dim, mlp_dim, action_dim, eps)

    def forward(self, memories, memory_valid):
        """memories [T, B, N, C], memory_valid [B, N, 1] -> chunk [B, horizon, action_dim] in [-1, 1]."""
        timesteps = memories.shape[0]
        queries = self.action_queries.weight[None].expand(memories.shape[1], -1, -1)
        sequence = queries[None].expand(timesteps, -1, -1, -1)
        for layer in self.layers:
            state = {}
            sequence = torch.stack([layer.forward_step(sequence[t], memories[t], memory_valid, state, t)
                                    for t in range(timesteps)])
        state = {}
        currents = [self.action_projection.forward_step(sequence[t], state, t) for t in range(timesteps)]
        return torch.tanh(torch.stack(currents).mean(0))


class ActionHead(nn.Module):
    def __init__(self, cfg, dim, heads, ffn_dim, eps, views, grid):
        super().__init__()
        self.state_dim, self.timesteps = cfg["state_dim"], cfg["spike_timesteps"]
        self.state_projection = CoordinateStateProjection(cfg["state_dim"], dim, cfg["state_hidden_dim"], eps)
        self.decoder = Decoder(cfg["horizon"], dim, heads, ffn_dim, cfg["num_layers"], cfg["mlp_hidden_dim"],
                               cfg["action_dim"], eps, views, grid)

    def forward(self, memory, state, padding):
        """memory [B, N, C] (fusion output), state [B, state_dim] (normalised), padding [B, N] (True: no token)."""
        tokens = self.state_projection(state)
        memories = torch.stack([torch.cat((memory, tokens), dim=1) for _ in range(self.timesteps)])
        state_valid = torch.ones(tokens.shape[:2], device=memory.device, dtype=torch.bool)
        valid = torch.cat((~padding, state_valid), dim=1)[..., None]
        return self.decoder(memories, valid)
