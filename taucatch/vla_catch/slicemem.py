"""The locator memory a recorded toss would get in closed loop with --slice-track, for training on it.

In closed loop (evaluate.rollout) a SliceTrack measures the ball at every 4 ms event slice; it starts once the
policy's own track holds (three measurements) and, once its gravity fit spans 40 ms, every policy call reads points
on that fit at t - ages, all valid. Training memory (ball.track_memory) holds one raw measurement per 20 ms step
instead. slice_memories replays the closed loop on the recorded events: the same SliceTrack on the same fresh pixels,
started from track_memory's fit where the policy's locator would start it, and returns per step the memory the head
reads (the slice fit once ready, the raw memory before).

    python -m taucatch.vla_catch.slicemem DATA_DIR...      # caches slicemem_<shard>.npz beside each shard
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import ball as bm
from .collect import EV_DT, STEP
from .events import CAMERAS, RES
from .locator import SliceTrack

AGES = EV_DT + STEP * np.arange(1, bm.MEMORY + 1)       # = BallLocator.ages[1:]


def slice_memories(toss, rng):
    """Per policy step s: (memory [M, 3], valid [M], ready) as the closed loop with --slice-track feeds the model."""
    track = SliceTrack()
    n = len(toss.times)
    out_mem = np.zeros((n, bm.MEMORY, 3), np.float32)
    out_valid = np.zeros((n, bm.MEMORY), bool)
    out_ready = np.zeros(n, bool)
    last = np.full((len(CAMERAS), RES * RES), -10 ** 6, np.int64)
    events = [toss.events[cam] for cam in CAMERAS]
    lo = [0] * len(CAMERAS)
    j_steps = [int(round(t / EV_DT)) for t in toss.times]
    j = min(int(e[0][0]) for e in events if len(e[0])) if any(len(e[0]) for e in events) else j_steps[0]
    memory, valid = bm.track_memory(toss, 0, rng)
    for s in range(n):                                          # memory, valid: the policy's own memory before call s
        while j <= j_steps[s]:                                  # slices up to this call, as rollout adds them
            for c, (sl, pix, *_) in enumerate(events):
                hi = np.searchsorted(sl, j + 1)
                last[c, pix[lo[c]:hi]] = j
                lo[c] = hi
            track.add(j * EV_DT - EV_DT, ((j - last) < 2).reshape(len(CAMERAS), RES, RES))
            j += 1
        held = track.memory(toss.times[s], AGES)
        if held is not None:
            out_mem[s], out_valid[s], out_ready[s] = held[0], held[1], True
        else:
            out_mem[s], out_valid[s] = memory, valid
        memory, valid = bm.track_memory(toss, s + 1, rng)      # after call s: its own measurement in slot 0
        if not track.active and valid.sum() >= 3:              # the policy's track holds: start the slice track
            p, v = _fit(memory[valid], AGES[valid] - STEP)      # ages at call s
            track.start(toss.times[s], p, v)
    return out_mem, out_valid, out_ready


def _fit(points, ages):
    """Gravity least squares at time 0 = now (as BallLocator.fit): points [K, 3] at ages [K] -> position, velocity."""
    dt = -ages
    A = np.stack([np.ones_like(dt), dt], 1)
    c = np.linalg.lstsq(A, points - 0.5 * bm.GRAVITY * dt[:, None] ** 2, rcond=None)[0]
    return c[0], c[1]


def cache_shard(path):
    import json
    from .data import Toss
    out = path.parent / f"slicemem_{path.name.removeprefix('shard_')}"
    if out.exists():
        return str(out)
    z = dict(np.load(path))
    tosses = [Toss(z, i, m) for i, m in enumerate(json.loads(str(z["meta"])))]
    mem, valid, ready = [], [], []
    for t in tosses:
        m, v, r = slice_memories(t, np.random.default_rng(t.meta["seed"]))
        mem.append(m); valid.append(v); ready.append(r)
    np.savez(out, memory=np.concatenate(mem), valid=np.concatenate(valid), ready=np.concatenate(ready))
    return str(out)


if __name__ == "__main__":
    shards = [p for root in sys.argv[1:] for p in sorted(Path(root).glob("shard_*.npz"))]
    with Pool(32) as pool:
        for i, out in enumerate(pool.imap_unordered(cache_shard, shards)):
            print(i + 1, len(shards), out, flush=True)
