"""Ball finder (plan §3.3): fast neurons on the event stream of one camera.

Input: a leaky trace of the events with tau = 2 ms (the fast neurons), kept per 4x4 pixel
cell and polarity as (log count, mean x offset, mean y offset) inside the cell, so the
sub-pixel position survives the pooling. A small CNN outputs a ball heat map at stride 8
and the sub-pixel offset of the ball centre in each cell (CenterNet style).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit

CELL = 4
STRIDE = 8
TAU_US = 2000.0
WINDOW_US = 10000          # events older than 5 tau are dropped
WIDTH, HEIGHT = 640, 480
GRID_W, GRID_H = WIDTH // CELL, HEIGHT // CELL
OUT_W, OUT_H = WIDTH // STRIDE, HEIGHT // STRIDE


@njit(cache=True)
def event_grid(x, y, t, p, t_end, tau=TAU_US):
    """Leaky trace of events (t <= t_end, in us) per cell: channels [ON, OFF] x [log count, mean dx, mean dy]."""
    acc = np.zeros((6, GRID_H, GRID_W), np.float32)
    half = (CELL - 1) / 2.0
    for k in range(x.size):
        dt = t_end - t[k]
        if dt < 0 or dt > WINDOW_US:
            continue
        w = np.exp(-dt / tau)
        cx, cy = x[k] // CELL, y[k] // CELL
        c = 0 if p[k] > 0 else 3
        acc[c, cy, cx] += w
        acc[c + 1, cy, cx] += w * (x[k] - cx * CELL - half)
        acc[c + 2, cy, cx] += w * (y[k] - cy * CELL - half)
    for c in (0, 3):
        for i in range(GRID_H):
            for j in range(GRID_W):
                n = acc[c, i, j]
                if n > 0:
                    acc[c + 1, i, j] /= n * half
                    acc[c + 2, i, j] /= n * half
                    acc[c, i, j] = np.log1p(n)
    return acc


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BallFinder(nn.Module):
    """6 x 120 x 160 event grid -> heat logits (1 x 60 x 80) and offsets (2 x 60 x 80, in output cells)."""

    def __init__(self, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(conv(6, w // 2 + 8), conv(w // 2 + 8, w, 2), conv(w, w))        # 60 x 80
        self.down = nn.Sequential(conv(w, w + 16, 2), conv(w + 16, w + 16), conv(w + 16, w + 16))  # 30 x 40
        self.fuse = nn.Sequential(nn.Conv2d(2 * w + 16, w, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                                  conv(w, w))
        self.heat = nn.Conv2d(w, 1, 1)
        self.offset = nn.Conv2d(w, 2, 1)
        nn.init.constant_(self.heat.bias, -4.0)

    def forward(self, grid):
        a = self.stem(grid)
        b = F.interpolate(self.down(a), size=a.shape[-2:], mode="nearest")
        h = self.fuse(torch.cat([a, b], 1))
        return self.heat(h), self.offset(h)


def decode(heat, offset):
    """Best ball position per image: u, v in pixels and the peak score in [0, 1]."""
    n = heat.shape[0]
    prob = torch.sigmoid(heat.reshape(n, -1))
    score, idx = prob.max(1)
    gy, gx = idx // OUT_W, idx % OUT_W
    off = offset.reshape(n, 2, -1)[torch.arange(n), :, idx]
    u = (gx.float() + off[:, 0]) * STRIDE
    v = (gy.float() + off[:, 1]) * STRIDE
    return u, v, score


def decode_peaks(heat, offset, k=4, score_min=0.3):
    """Up to k local maxima of the heat map per image, each [u, v, score]: list of (n_i, 3) arrays."""
    prob = torch.sigmoid(heat)
    peak = (prob == F.max_pool2d(prob, 3, 1, 1)) & (prob > score_min)
    out = []
    for i in range(heat.shape[0]):
        flat = (prob[i, 0] * peak[i, 0]).reshape(-1)
        score, idx = flat.topk(k)
        keep = score > score_min
        score, idx = score[keep], idx[keep]
        gy, gx = idx // OUT_W, idx % OUT_W
        off = offset[i].reshape(2, -1)[:, idx]
        out.append(torch.stack([(gx + off[0]) * STRIDE, (gy + off[1]) * STRIDE, score], 1).numpy().astype(float))
    return out


def targets(uv, visible, sigma=0.8):
    """Gaussian heat targets and offset targets (3 x 3 cells around each centre) for a batch.

    uv: (n, 2) ball centre in pixels and visible: (n,) bool; or, for several balls per image,
    lists of (m_i, 2) and (m_i,) arrays. Where two balls' cells overlap, the nearer centre sets the offset."""
    if isinstance(uv, np.ndarray):
        uv, visible = uv[:, None], np.asarray(visible)[:, None]
    n = len(uv)
    heat = np.zeros((n, 1, OUT_H, OUT_W), np.float32)
    off = np.zeros((n, 2, OUT_H, OUT_W), np.float32)
    mask = np.zeros((n, 1, OUT_H, OUT_W), np.float32)
    gy, gx = np.mgrid[0:OUT_H, 0:OUT_W]
    for i in range(n):
        near = np.full((OUT_H, OUT_W), np.inf)
        for b in np.nonzero(visible[i])[0]:
            cu, cv = uv[i][b] / STRIDE
            heat[i, 0] = np.maximum(heat[i, 0], np.exp(-((gx + 0.5 - cu) ** 2 + (gy + 0.5 - cv) ** 2) / (2 * sigma ** 2)))
            jx, jy = int(np.floor(cu)), int(np.floor(cv))
            heat[i, 0, min(max(jy, 0), OUT_H - 1), min(max(jx, 0), OUT_W - 1)] = 1.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = jy + dy, jx + dx
                    d = np.hypot(cu - x - 0.5, cv - y - 0.5)
                    if 0 <= y < OUT_H and 0 <= x < OUT_W and d < near[y, x]:
                        near[y, x] = d
                        off[i, :, y, x] = (cu - x, cv - y)
                        mask[i, 0, y, x] = 1.0
    return heat, off, mask


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    p = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    pos = target.eq(1).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)
