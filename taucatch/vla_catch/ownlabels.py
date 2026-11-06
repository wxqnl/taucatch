"""Arm-position labels from the expert planner on the policy's own track, for every recorded step.

The true-state expert knows where the ball will be; early in the flight the policy cannot, so the best it can do
against those labels is to stop short (on r1's own states the head's height target follows the planner's with slope
0.37 at 0.05-0.15 s after release, 0.8-1.06 later). The planner on the policy's own track commits to what that track
says, and in closed loop it catches 91% to the head's 85%. Here the planner reads the track closed loop gives
(slicemem.py: points on the SliceTrack fit, once ready), fits it with gravity and picks the intercept as
expert.Expert.intercept does; the position channels of every chunk entry before the true-state expert closes the hand
become that point minus the grasp. Rotation, fingers and the entries from closure on keep the true-state labels.

    python -m taucatch.vla_catch.ownlabels DATA_DIR...      # caches ownlab_<shard>.npz beside each slicemem cache
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .collect import HORIZON
from .sim import GRAVITY, ballistic
from .slicemem import AGES, _fit

TAUS = np.arange(0.0, 1.2, 0.005)          # as expert.Expert
READY = np.array([0.413, -0.219, 0.957])   # B601 ready grasp (median first state of the expert tosses)


def geometry():
    from .b601 import B601Sim
    return B601Sim.SHOULDER, B601Sim.REACH


def intercept(p, v, grasp, shoulder, reach):
    """expert.Expert.intercept with hand_speed 1: point the hand reaches with the lowest cost, or None."""
    pts = ballistic(p, v, TAUS)
    ok = (pts[:, 2] > 0.45) & (np.linalg.norm(pts - shoulder, axis=1) < reach) & (pts[:, 0] > 0.12)
    if not ok.any():
        return None
    dist = np.linalg.norm(pts - grasp, axis=1)
    cost = np.linalg.norm(pts - READY, axis=1) + 5.0 * np.maximum(dist - np.maximum(TAUS - 0.05, 0.0), 0.0)
    cost[~ok] = np.inf
    return pts[int(np.argmin(cost))]


def own_positions(state8, labels, mask, memory, valid, ready, shoulder, reach):
    """Per step: position labels [12, 3] and where they replace the true ones [12] (bool)."""
    n = len(state8)
    pos = np.zeros((n, HORIZON, 3), np.float32)
    use = np.zeros((n, HORIZON), bool)
    for s in range(n):
        if not ready[s] or valid[s].sum() < 3:
            continue
        p, v = _fit(memory[s][valid[s]], AGES[valid[s]])
        g = state8[s, :3]
        point = intercept(p, v, g, shoulder, reach)
        if point is None:
            continue
        pos[s] = point - g
        use[s] = mask[s] & (labels[s, :, 6] < 0.5)
    return pos, use


def cache_shard(path):
    out = path.parent / f"ownlab_{path.name.removeprefix('shard_')}"
    if out.exists():
        return str(out)
    cached = path.parent / f"slicemem_{path.name.removeprefix('shard_')}"
    z, c = np.load(path), np.load(cached)
    labels = z["labels_true"] if "labels_true" in z else z["labels"]
    shoulder, reach = geometry()
    pos, use = own_positions(z["state8"], labels, z["mask"], c["memory"], c["valid"], c["ready"], shoulder, reach)
    np.savez(out, pos=pos, use=use)
    return str(out)


if __name__ == "__main__":
    shards = [p for root in sys.argv[1:] for p in sorted(Path(root).glob("shard_*.npz"))]
    with Pool(32) as pool:
        for i, out in enumerate(pool.imap_unordered(cache_shard, shards)):
            print(i + 1, len(shards), out, flush=True)
