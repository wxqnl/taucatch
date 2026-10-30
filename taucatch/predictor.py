"""TauCatch core and baselines (plan §3.4, §3.5).

Every 1 ms the core reads the finder output of both cameras and gives the predicted path:
ball positions 0 to 400 ms ahead (every 10 ms) with an uncertainty per point. The path is
written as a gravity arc from the estimated position and velocity plus a learned
correction, which carries drag and the bend from spin.

Cores, all with the same input and readout:
- "spread": leaky units with their own time constants, log-uniform over 1 to 300 ms, learned (TauCatch)
- "single": the same units with one shared, fixed time constant (the single-timescale network)
- "lstm":   an LSTM
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .ball import GRAVITY

DT_MS = 1.0
HORIZONS = np.arange(41) * 0.01                     # s, 0 to 400 ms
P_CENTER = np.array([2.0, 0.0, 1.2])
P_SCALE, V_SCALE, C_SCALE = 2.0, 10.0, 0.5
UV_SCALE = 64.0                                    # px per input unit
P_IN_SCALE = np.array([1.0, 0.3, 0.3])             # m per input unit of the triangulated position
N_IN = 11


def triangulate(geoms, uv_l, uv_r):
    """Midpoint of the closest approach of the two camera rays; arrays (..., 2) -> (..., 3)."""
    rays = []
    for g, uv in zip(geoms, (uv_l, uv_r)):
        dc = np.stack([(uv[..., 0] - g.cx) / g.f, -(uv[..., 1] - g.cy) / g.f, -np.ones(uv.shape[:-1])], -1)
        d = dc @ g.rot.T
        rays.append((g.pos, d / np.linalg.norm(d, axis=-1, keepdims=True)))
    (o1, d1), (o2, d2) = rays
    w = o1 - o2
    b = np.sum(d1 * d2, -1)
    d, e = np.sum(d1 * w, -1), np.sum(d2 * w, -1)
    den = np.maximum(1 - b * b, 1e-9)
    s = (b * e - d) / den
    t = (e - b * d) / den
    return 0.5 * ((o1 + s[..., None] * d1) + (o2 + t[..., None] * d2))


def features(geoms, det, score_min=0.5):
    """Core input per step from the finder output det (T, 2, 3) = [u, v, score] per camera."""
    T = det.shape[0]
    x = np.zeros((T, N_IN), np.float32)
    seen = det[:, :, 2] > score_min
    for c in range(2):
        s = det[:, c, 2] * seen[:, c]
        x[:, 3 * c] = s
        x[:, 3 * c + 1] = s * (det[:, c, 0] - 320) / UV_SCALE
        x[:, 3 * c + 2] = s * (det[:, c, 1] - 240) / UV_SCALE
    both = seen.all(1)
    if both.any():
        p = triangulate(geoms, det[both, 0, :2].astype(float), det[both, 1, :2].astype(float))
        x[both, 6] = 1.0
        x[both, 7:10] = np.clip((p - P_CENTER) / P_IN_SCALE, -6, 6)
    x[:, 10] = seen.any(1)
    return x


class LeakyCore(nn.Module):
    """h <- h + (dt / tau) * (-h + tanh(W x + U h + b)), one tau per unit."""

    def __init__(self, n_in, n=512, tau_range=(1.0, 300.0), tau=None):
        super().__init__()
        self.n = n
        self.inp = nn.Linear(n_in, n)
        self.rec = nn.Linear(n, n, bias=False)
        nn.init.orthogonal_(self.rec.weight, gain=0.9)
        self.log_min, self.log_max = math.log(tau_range[0]), math.log(tau_range[1])
        if tau is None:
            u = torch.rand(n) * (self.log_max - self.log_min) + self.log_min
            self.log_tau = nn.Parameter(u)
        else:
            self.register_buffer("log_tau", torch.full((n,), math.log(tau)))

    def tau(self):
        return self.log_tau.clamp(self.log_min, self.log_max).exp()

    def forward(self, x, h=None, mask=None):
        """x (B, T, n_in) -> states (B, T, n). `mask` (n,) silences units (E3)."""
        alpha = DT_MS / self.tau()
        drive = self.inp(x).unbind(1)       # unbind: slicing inside the loop would make backward allocate T full-size grads
        h = x.new_zeros(x.shape[0], self.n) if h is None else h
        out = []
        for t in range(x.shape[1]):
            h = h + alpha * (-h + torch.tanh(drive[t] + self.rec(h)))
            if mask is not None:
                h = h * mask
            out.append(h)
        return torch.stack(out, 1), h


class PathPredictor(nn.Module):
    def __init__(self, core="spread", n=512, tau=None):
        super().__init__()
        self.kind = core
        if core == "lstm":
            self.core = nn.LSTM(N_IN, n, batch_first=True)
        else:
            self.core = LeakyCore(N_IN, n, tau=tau if core == "single" else None)
        k = len(HORIZONS)
        self.head = nn.Linear(n, 6 + 3 * (k - 1) + k + 3)     # p, v, correction (h > 0), log sigma, spin (aux)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("h", torch.tensor(HORIZONS, dtype=torch.float32))
        self.register_buffer("center", torch.tensor(P_CENTER, dtype=torch.float32))

    def forward(self, x, state=None, mask=None):
        if self.kind == "lstm":
            hs, state = self.core(x, state)
        else:
            hs, state = self.core(x, state, mask)
        return self.decode(self.head(hs)), state

    def decode(self, y):
        k = len(HORIZONS)
        p = self.center + P_SCALE * y[..., 0:3]
        v = V_SCALE * y[..., 3:6]
        corr = C_SCALE * y[..., 6:6 + 3 * (k - 1)].unflatten(-1, (k - 1, 3))
        corr = torch.cat([torch.zeros_like(corr[..., :1, :]), corr], -2)
        log_sigma = y[..., 6 + 3 * (k - 1):6 + 3 * (k - 1) + k] - 2.0
        spin = 100.0 * y[..., -3:]
        h = self.h[:, None]
        g = torch.tensor([0.0, 0.0, -GRAVITY], device=y.device)
        path = p[..., None, :] + v[..., None, :] * h + 0.5 * g * h ** 2 + corr
        return dict(p=p, v=v, path=path, log_sigma=log_sigma, spin=spin)


def path_loss(out, target, valid):
    """Gaussian NLL of the predicted path (isotropic sigma per point) over valid (step, horizon)."""
    err2 = ((out["path"] - target) ** 2).sum(-1)
    ls = out["log_sigma"]
    nll = 0.5 * err2 * torch.exp(-2 * ls) + 3 * ls
    return (nll * valid).sum() / valid.sum().clamp(min=1.0)


class GroupedPredictor(nn.Module):
    """Several single-timescale networks trained side by side (the tau sweep of D1): group g has
    its own weights, readout and fixed time constant taus[g]; groups share only the GPU kernels."""

    def __init__(self, taus, n=512):
        super().__init__()
        G, k = len(taus), len(HORIZONS)
        self.taus, self.n = list(taus), n
        self.w_in = nn.Parameter(torch.randn(G, N_IN, n) / math.sqrt(N_IN))
        self.b_in = nn.Parameter(torch.zeros(G, n))
        rec = torch.empty(G, n, n)
        for g in range(G):
            nn.init.orthogonal_(rec[g], gain=0.9)
        self.w_rec = nn.Parameter(rec)
        self.register_buffer("alpha", torch.tensor([DT_MS / t for t in taus], dtype=torch.float32)[:, None])
        n_out = 6 + 3 * (k - 1) + k + 3
        self.w_out = nn.Parameter(torch.zeros(G, n, n_out))
        self.b_out = nn.Parameter(torch.zeros(G, n_out))
        self.decoder = PathPredictor("single", 8)       # only its decode() is used

    def forward(self, x, h=None):
        """x (B, T, n_in) -> list over groups of decoded outputs."""
        B, T, _ = x.shape
        drive = (torch.einsum("btk,gkn->gbtn", x, self.w_in) + self.b_in[:, None, None]).unbind(2)
        h = x.new_zeros(len(self.taus), B, self.n) if h is None else h
        hs = []
        for t in range(T):
            h = h + self.alpha[:, None] * (-h + torch.tanh(drive[t] + torch.bmm(h, self.w_rec)))
            hs.append(h)
        hs = torch.stack(hs, 2)                                           # G, B, T, n
        y = torch.einsum("gbtn,gno->gbto", hs, self.w_out) + self.b_out[:, None, None]
        return [self.decoder.decode(y[g]) for g in range(len(self.taus))], h
