"""FP32 finder plus GPU peaks; ambiguous ordering/thresholds use original CPU.

Only measured event grids are accepted. One small packet is returned normally;
raw outputs are transferred for the first32 audit batches or CPU fallback.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .finder import decode_peaks


def peak_packet(heat, offset, k=4, score_min=.3):
    prob = torch.sigmoid(heat)
    pooled = F.max_pool2d(prob, 3, 1, 1)
    maxima = prob == pooled
    flat = (prob * (maxima & (prob > score_min))).flatten(1)
    scores, idx = flat.topk(k + 1, dim=1)
    near_threshold = (maxima & ((prob - score_min).abs() <= 1e-6)).flatten(1).any(1)
    near_local_tie = ((pooled != prob) & ((pooled - prob).abs() <= 1e-6)
                      & (prob > score_min - 1e-6)).flatten(1).any(1)
    gaps = scores[:, :-1] - scores[:, 1:]
    near_order_tie = ((gaps <= 1e-6) & (scores[:, :-1] > score_min)).any(1)
    fallback = near_threshold | near_local_tie | near_order_tie
    score, idx = scores[:, :k], idx[:, :k]
    off = offset.flatten(2).gather(2, idx[:, None, :].expand(-1, 2, -1))
    rows = torch.stack(((idx % 80 + off[:, 0]) * 8,
                        (idx // 80 + off[:, 1]) * 8, score), dim=2)
    marker = fallback[:, None, None].expand(-1, 1, 3).float()
    return torch.cat((rows, marker), dim=1)


class FusedFinderGraph:
    def __init__(self, model, k=4, score_min=.3):
        if k != 4 or score_min != .3:
            raise ValueError('Only original stereo tracker k4/threshold.3 is gated')
        self.k, self.score_min = k, score_min
        self.model = model
        self.input = torch.zeros((2,6,120,160), device='cuda', dtype=torch.float32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                raw = model(self.input)
                peak_packet(*raw, k, score_min)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.raw = model(self.input)
            self.packet = peak_packet(*self.raw, k, score_min)

    @torch.inference_mode()
    def __call__(self, grid):
        if grid.shape != self.input.shape or grid.dtype != torch.float32 or not grid.is_cuda:
            raise ValueError('Expected original stereo CUDA float32 grids')
        self.input.copy_(grid)
        self.graph.replay()
        return self.raw, self.packet

    def unpack(self, raw, packet, save_raw):
        values = packet.cpu().numpy()
        flags = values[:, self.k, 0] > .5
        cpu_raw = tuple(v.cpu() for v in raw) if save_raw or flags.any() else None
        reference = decode_peaks(*cpu_raw, self.k, self.score_min) if flags.any() else None
        peaks = []
        for camera, rows in enumerate(values[:, :self.k]):
            keep = rows[:, 2] > self.score_min
            peaks.append(reference[camera] if flags[camera] else np.asarray(rows[keep], dtype=float))
        audit_raw = None if not save_raw else tuple(v.numpy().copy() for v in cpu_raw)
        return peaks, audit_raw, flags.tolist()
