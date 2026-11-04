"""Shards from collect.py as a torch dataset: two event frames, state8 and the 12 x 7 label chunk per policy step.
With ball=True the state also carries the ball measurement of ball.py (state8 + BALL_DIM); with locator=True the
sample instead carries what the in-model locator needs: its track memory, the teacher gate and the scorer target.
Selective shards (collect.py --distractors) add the two RGB views (the newest 30 Hz frame RGB_DELAY old) after the
event views, and the sample ends with the named ball's stage-2 cell in each RGB view (Toss.ground) and its instruction."""
import json
from pathlib import Path

import numpy as np
import torch

from . import ball as ball_meter
from .collect import EV_DT, HORIZON, STEP
from .policy import INSTRUCTION
from .events import CAMERAS, RES, event_frame, trace_frame
from .sim import GRAVITY

WINDOW = int(round(STEP / EV_DT))          # 4 ms slices per event frame
NOISE_SPAN = int(round(0.3 / EV_DT))       # slices of background noise behind a trace frame
RGB_DELAY = 0.010                          # s from an RGB frame's exposure to its use
CELLS = 28                                 # SDT-V3 stage-2 grid (8 px cells); CELLS ** 2 marks "not in view"


class Toss:
    """One toss of a loaded shard: step arrays and per-camera events sorted by slice."""

    def __init__(self, z, i, meta):
        a, b = z["offset_steps"][i], z["offset_steps"][i + 1]
        self.meta = meta
        self.times = z["step_times"][a:b]
        self.state8 = z["state8"][a:b]
        self.labels = z["labels"][a:b]
        self.mask = z["mask"][a:b]
        self.events = {}
        for cam in CAMERAS:
            c, d = z[f"offset_{cam}"][i], z[f"offset_{cam}"][i + 1]
            sl = z[f"{cam}_slice"][c:d]
            self.events[cam] = (sl, z[f"{cam}_pix"][c:d].astype(np.int64), z[f"{cam}_pos"][c:d], z[f"{cam}_neg"][c:d])
        self.rgb = None
        if "rgb" in z:
            c, d = z["offset_rgb"][i], z["offset_rgb"][i + 1]
            self.rgb, self.rgb_times = z["rgb"][c:d], z["rgb_times"][c:d]
        self.instruction = meta.get("instruction", INSTRUCTION)

    def rgb_frames(self, s):
        """Newest RGB frame pair at least RGB_DELAY old at step s: [2, H, W, 3] (black before the first)."""
        k = int(np.searchsorted(self.rgb_times, self.times[s] - RGB_DELAY + 1e-6)) - 1
        return self.rgb[k] if k >= 0 else np.zeros_like(self.rgb[0])

    def ground(self, s):
        """Stage-2 cell (row-major) of the named ball in each RGB view at the exposure used at step s: [2] int64,
        CELLS ** 2 outside the image. Before release the ball waits at its launch point."""
        k = int(np.searchsorted(self.rgb_times, self.times[s] - RGB_DELAY + 1e-6)) - 1
        out = np.full(len(CAMERAS), CELLS ** 2, np.int64)
        if k >= 0:
            p = self.ball_at(self.rgb_times[k])
            for v, cam in enumerate(CAMERAS):
                col, row = (ball_meter.project_rgb(cam, p) // (RES // CELLS)).astype(int)
                if 0 <= col < CELLS and 0 <= row < CELLS:
                    out[v] = row * CELLS + col
        return out

    def forecast(self, s):
        """Ball position at each chunk time relative to the grasp centre now (held at launch before release,
        at the aim point after arrival): [HORIZON, 3]."""
        m = self.meta
        tb = np.clip(self.times[s] + STEP * np.arange(HORIZON) - m["release"], 0.0, m["flight"])[:, None]
        ball = np.asarray(m["launch"]) + np.asarray(m["velocity"]) * tb + 0.5 * GRAVITY * tb ** 2
        return (ball - self.state8[s, :3]).astype(np.float32)

    def ball_at(self, t):
        m = self.meta
        tb = np.clip(t - m["release"], 0.0, m["flight"])
        return np.asarray(m["launch"]) + np.asarray(m["velocity"]) * tb + 0.5 * GRAVITY * tb ** 2

    def ball_state(self, s, rng):
        """Ball measurement at step s, from the two newest event slices of every step so far (ball.py)."""
        track = []
        for k in range(s + 1):
            j = int(round(self.times[k] / EV_DT))
            t_obs = (j - 1) * EV_DT                   # mid-time of slices j-1 and j
            events = []
            for cam in CAMERAS:
                sl, pix, pos, neg = self.events[cam]
                lo, hi = np.searchsorted(sl, j - 1), np.searchsorted(sl, j + 1)
                events.append((pix[lo:hi], pos[lo:hi].astype(np.float64) + neg[lo:hi]))
            p = ball_meter.measure(events, ball_meter.gate_centres(self.ball_at(t_obs), rng))
            if p is not None:
                track.append((t_obs, p))
        return ball_meter.features(track, self.times[s], self.state8[s, :3])

    def frames(self, s, rng=None, noise=0.0, drop=0.0, mode="window"):
        """mode window: events of the last 20 ms (event_frame); trace: time surfaces of all events so far."""
        j = int(round(self.times[s] / EV_DT))
        span = WINDOW if mode == "window" else NOISE_SPAN
        out = []
        for cam in CAMERAS:
            sl, pix, pos, neg = self.events[cam]
            lo = np.searchsorted(sl, j - WINDOW + 1) if mode == "window" else 0
            hi = np.searchsorted(sl, j + 1)
            sl, pix, pos, neg = sl[lo:hi], pix[lo:hi], pos[lo:hi], neg[lo:hi]
            if rng is not None and drop > 0:
                keep = rng.random(pix.size) >= drop
                sl, pix, pos, neg = sl[keep], pix[keep], pos[keep], neg[keep]
            if rng is not None and noise > 0:      # background activity: random single events per slice
                k = rng.binomial(RES * RES * span, noise)
                sign = rng.random(k) < 0.5
                pix = np.concatenate([pix, rng.integers(0, RES * RES, k)])
                pos = np.concatenate([pos, sign.astype(np.uint8)])
                neg = np.concatenate([neg, (~sign).astype(np.uint8)])
                sl = np.concatenate([sl, rng.integers(j - span + 1, j + 1, k).astype(sl.dtype)])
            if mode == "window":
                out.append(event_frame(pix, pos, neg, sl == j))
            else:
                last = np.full(RES * RES, -np.inf)
                np.maximum.at(last, pix, sl.astype(np.float64))
                out.append(trace_frame(((j - last) * EV_DT).astype(np.float32)))
        if self.rgb is not None:
            out += list(self.rgb_frames(s))
        return np.stack(out)


def load_tosses(root, keep=lambda meta: True):
    tosses = []
    for path in sorted(Path(root).glob("shard_*.npz")):
        z = dict(np.load(path))
        meta = json.loads(str(z["meta"]))
        tosses += [Toss(z, i, m) for i, m in enumerate(meta) if keep(m)]
    return tosses


def label_stats(tosses, forecast=False, ball=False):
    state = np.concatenate([t.state8 for t in tosses])
    if ball:                 # measured on a fixed subset of tosses
        rng = np.random.default_rng(0)
        sub = [tosses[i] for i in rng.choice(len(tosses), min(300, len(tosses)), replace=False)]
        state = np.concatenate([np.concatenate([t.state8[s], t.ball_state(s, rng)])[None]
                                for t in sub for s in range(len(t.times))])
    labels = np.concatenate([t.labels[t.mask] for t in tosses])
    if forecast:
        ball = np.concatenate([np.stack([t.forecast(s) for s in range(len(t.times))])[t.mask] for t in tosses])
        labels = np.concatenate([labels, ball], axis=1)
    # The action head ends in tanh. Arm channels are scaled so that 99.5% fall inside +-0.9: as z-scores a third
    # of the +220 ms targets lie beyond 1, where the head cannot reach and tanh passes no gradient.
    mean = labels.mean(0)
    scale = np.percentile(np.abs(labels - mean), 99.5, axis=0) / 0.9
    return {"state_mean": state.mean(0), "state_std": state.std(0) + 1e-3,
            "action_mean": mean, "action_std": scale + 1e-3}


def normalize_actions(a, stats):
    """Arm channels into [-1, 1] (label_stats scale); the finger ramp from [0, 1] to [-1, 1]."""
    out = (a - stats["action_mean"]) / stats["action_std"]
    out[..., 6] = 2 * a[..., 6] - 1
    return out


def denormalize_actions(a, stats):
    out = a * stats["action_std"] + stats["action_mean"]
    out[..., 6] = (a[..., 6] + 1) / 2
    return out


class CatchDataset(torch.utils.data.Dataset):
    def __init__(self, tosses, stats, augment=False, noise=(0.0, 1e-3), drop=(0.0, 0.2), mode="window",
                 forecast=False, ball=False, locator=False, no_velocity=False):
        self.tosses, self.stats, self.augment, self.mode, self.forecast = tosses, stats, augment, mode, forecast
        self.ball, self.locator, self.no_velocity = ball, locator, no_velocity
        self.noise, self.drop = noise, drop
        self.index = [(i, s) for i, t in enumerate(tosses) for s in range(len(t.times))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        t = self.tosses[i]
        if self.augment:
            rng = np.random.default_rng()
            frames = t.frames(s, rng, rng.uniform(*self.noise), rng.uniform(*self.drop), self.mode)
        else:
            frames = t.frames(s, mode=self.mode)
        state = t.state8[s]
        if self.no_velocity:     # the hand's own velocity would let the policy copy the expert's ongoing motion
            state = np.concatenate([state[:3], np.zeros(3, state.dtype), state[6:]])
        rng = np.random.default_rng() if self.augment else np.random.default_rng((t.meta["seed"], s))
        if self.ball:
            state = np.concatenate([state, t.ball_state(s, rng)])
        n = len(state)
        state = (state - self.stats["state_mean"][:n]) / self.stats["state_std"][:n]
        labels = t.labels[s]
        if self.forecast:            # the chunk also predicts where the ball will be
            labels = np.concatenate([labels, t.forecast(s)], axis=1)
        labels = np.clip(normalize_actions(labels, self.stats), -1.0, 1.0)   # DAgger labels can exceed the scale
        out = (torch.from_numpy(frames), torch.from_numpy(state.astype(np.float32)),
               torch.from_numpy(labels.astype(np.float32)), torch.from_numpy(t.mask[s]))
        if self.locator:
            memory, valid = ball_meter.track_memory(t, s, rng)
            gate, target = ball_meter.acquisition(t, s, rng)
            out += tuple(torch.from_numpy(x) for x in (memory, valid, gate, target))
        if t.rgb is not None:
            out += (torch.from_numpy(t.ground(s)), t.instruction)
        return out
