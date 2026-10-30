"""Event tracking of every ball in a volley (plan §5.10).

Every 1 ms the finder reports up to a few balls per event camera. Detections are matched to the
existing tracks by where each track's physics filter expects its ball in the image; unmatched
detections seen by both cameras (rays meeting within a few cm) start new tracks. Each track has
its own physics filter and its own TauCatch state. The catcher follows the track bound to the
named ball. Binding here is by the simulator ("perfect choice", stage 1): at the choice time the
track nearest to the named ball is bound; stage 2 replaces this with the VLA's pointing.
"""
import collections

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment

from .ball import predict_path
from .ekf import BallEKF, BallEKFBank
from .events import EventCamConfig, EventRig
from .finder import WINDOW_US, BallFinder, decode_peaks, event_grid
from .predictor import HORIZONS, features, triangulate
from .tracker import fuse_path, load_core


class Track:
    def __init__(self, tid, ekf, t, delay):
        self.id, self.ekf = tid, ekf
        self.born = self.last_seen = t
        self.hits = 1
        self.state = None                       # TauCatch state
        self.outs = collections.deque(maxlen=delay + 1)


class VolleyTracker:
    """`core_path`: a trained path predictor, "ekf", or "fused:<predictor>" (as in EventTracker)."""

    def __init__(self, sim, finder_path, core_path, rig_cfg=EventCamConfig(), delay=0.001, score_min=0.5,
                 peak_min=0.3, max_balls=4, pair_gap=0.03, gate_px=6.0, max_miss=0.015, ekf_scale=1.0,
                 oracle=True):
        self.sim = sim
        self.rig = EventRig(sim, rig_cfg)
        self.geoms = self.rig.geoms
        self.finder = BallFinder().eval()
        self.finder.load_state_dict(torch.load(finder_path, map_location="cpu"))
        core_path = str(core_path)
        self.mode = "ekf" if core_path == "ekf" else "fused" if core_path.startswith("fused:") else "core"
        self.core = None if self.mode == "ekf" else load_core(core_path.removeprefix("fused:"))
        self.ekf_scale = ekf_scale
        self.filter_balls = list({b.name: b for b in sim.ball_types}.values())
        self.delay = round(delay / 0.001)
        self.score_min, self.peak_min, self.max_balls = score_min, peak_min, max_balls
        self.pair_gap, self.gate_px, self.max_miss = pair_gap, gate_px, max_miss
        self.fine = np.arange(0.0, HORIZONS[-1] + 1e-9, 0.001)
        self.stop_after_contact = True
        self.oracle = oracle            # False: binding comes from outside (the VLA), through bind()

    def reset(self, t_rec=-0.01):
        self.t_rec = t_rec
        self.on = False
        self.hist = [collections.deque(maxlen=WINDOW_US // 1000 + 1) for _ in self.geoms]
        self.tracks, self.next_id = [], 0
        self.bound, self.n_binds = None, 0
        self.first_det_t = None
        self.log = []

    # ---------- per step ----------
    @torch.no_grad()
    def step(self, sim, t):
        if not self.on:
            if t >= self.t_rec - 1e-9:
                self.rig.reset(t)
                self.on = True
            return
        if self.stop_after_contact and sim.catcher.contact_t is not None and t > sim.catcher.contact_t + 0.005:
            return
        evs = self.rig.step(t)
        t_us = int(round(t * 1e6))
        grids = []
        for c, e in enumerate(evs):
            self.hist[c].append((e["x"], e["y"], np.round(e["t"] * 1e6).astype(np.int32), e["p"]))
            x, y, ts, p = (np.concatenate(a) for a in zip(*self.hist[c]))
            grids.append(event_grid(x, y, ts, p, t_us))
        peaks = decode_peaks(*self.finder(torch.from_numpy(np.stack(grids))), self.max_balls, self.peak_min)

        for tr in self.tracks:
            tr.ekf.predict(t)
        dets = self._associate(peaks, t)
        self._spawn(peaks, t)
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_miss]
        for tr in self.tracks:
            if tr.id in dets:
                tr.ekf.update(dets[tr.id], self.score_min)
        if self.first_det_t is None and t >= 0 and self.tracks:
            self.first_det_t = t
        self._predict(dets, t)
        self._bind(sim, t)
        self.log.append(dict(t=t, peaks=peaks, events=evs, bound=self.bound,
                             tracks={tr.id: tr.ekf.x[:3].copy() for tr in self.tracks},
                             candidates={tr.id: dict(hits=tr.hits, speed=float(np.linalg.norm(tr.ekf.x[3:6])),
                                                     path=tr.outs[-1]["path"])
                                         for tr in self.tracks if tr.outs}))

    def _project_gate(self, tr, g):
        """Expected pixel position of the track in camera g and its gate radius (px)."""
        f = tr.ekf
        pc = (f.x[:3] - g.pos) @ g.rot
        z = -pc[2]
        if z < 0.1:
            return None
        u, v = g.cx + g.f * pc[0] / z, g.cy - g.f * pc[1] / z
        sigma = np.sqrt(np.max(np.linalg.eigvalsh(f.P[:3, :3]))) * g.f / z
        return np.array([u, v]), self.gate_px + 3 * sigma

    def _associate(self, peaks, t):
        """Match peaks to tracks per camera; returns {track id: det (2, 3)} for matched tracks."""
        dets = {tr.id: np.zeros((2, 3)) for tr in self.tracks}
        self.used = [np.zeros(len(p), bool) for p in peaks]
        for c, (g, pk) in enumerate(zip(self.geoms, peaks)):
            proj = [self._project_gate(tr, g) for tr in self.tracks]
            rows = [i for i, pr in enumerate(proj) if pr is not None]
            if not rows or not len(pk):
                continue
            cost = np.array([[np.linalg.norm(pk[j, :2] - proj[i][0]) for j in range(len(pk))] for i in rows])
            big = 1e6
            gated = np.where(cost <= np.array([proj[i][1] for i in rows])[:, None], cost, big)
            ri, ci = linear_sum_assignment(gated)
            for a, b in zip(ri, ci):
                if gated[a, b] < big:
                    tr = self.tracks[rows[a]]
                    dets[tr.id][c] = pk[b]
                    self.used[c][b] = True
        for tr in self.tracks:
            if dets[tr.id][:, 2].max() > 0:
                tr.last_seen, tr.hits = t, tr.hits + 1
        return {k: v for k, v in dets.items() if v[:, 2].max() > 0}

    def _spawn(self, peaks, t):
        """New tracks from unmatched peaks seen by both cameras whose rays meet."""
        left = [j for j in range(len(peaks[0])) if not self.used[0][j] and peaks[0][j, 2] > self.score_min]
        right = [j for j in range(len(peaks[1])) if not self.used[1][j] and peaks[1][j, 2] > self.score_min]
        pairs = []
        for i in left:
            for j in right:
                p = triangulate(self.geoms, peaks[0][i, :2], peaks[1][j, :2])
                gap = self._ray_gap(peaks[0][i, :2], peaks[1][j, :2], p)
                if gap < self.pair_gap and -0.5 < p[0] < 6 and 0.2 < p[2] < 3:
                    pairs.append((gap, i, j, p))
        taken_l, taken_r = set(), set()
        for gap, i, j, p in sorted(pairs, key=lambda x: x[0]):
            if i in taken_l or j in taken_r:
                continue
            if any(np.linalg.norm(tr.ekf.x[:3] - p) < 0.1 for tr in self.tracks):
                continue
            taken_l.add(i)
            taken_r.add(j)
            ekf = (BallEKF(self.filter_balls[0], self.geoms, self.ekf_scale) if len(self.filter_balls) == 1 else
                   BallEKFBank(self.filter_balls, self.geoms, self.ekf_scale))
            ekf.start(p, t)
            self.tracks.append(Track(self.next_id, ekf, t, self.delay))
            self.next_id += 1

    def _ray_gap(self, uv_l, uv_r, p):
        """Distance from the triangulated point to each camera ray, the larger one (m)."""
        gap = 0.0
        for g, uv in zip(self.geoms, (uv_l, uv_r)):
            d = g.ray(*uv)
            r = p - g.pos
            gap = max(gap, np.linalg.norm(r - (r @ d) * d))
        return 2 * gap

    def _predict(self, dets, t):
        """One TauCatch step for every track (batched), then the path each track gives the catcher."""
        if not self.tracks:
            return
        det = np.stack([dets.get(tr.id, np.zeros((2, 3))) for tr in self.tracks])
        if self.core is not None:
            x = torch.from_numpy(features(self.geoms, det, self.score_min))[:, None]
            n = self.core.head.in_features
            h = torch.stack([tr.state if tr.state is not None else torch.zeros(n) for tr in self.tracks])
            out, h_new = self.core(x, h)
            for k, tr in enumerate(self.tracks):
                tr.state = h_new[k]
        for k, tr in enumerate(self.tracks):
            if self.mode == "ekf":
                f = tr.ekf
                path = predict_path(f.x[:3], f.x[3:6], f.x[6:9] * f.lift_scale, f.params, 0.001, 401)[0][::10]
                h = HORIZONS[:, None, None]
                cov = f.P[:3, :3] + h * (f.P[:3, 3:6] + f.P[3:6, :3]) + h * h * f.P[3:6, 3:6]
                sigma = np.sqrt(np.trace(cov, axis1=1, axis2=2) / 3)
            else:
                path = out["path"][k, 0].numpy().astype(float)
                sigma = np.exp(out["log_sigma"][k, 0].numpy().astype(float))
                if self.mode == "fused":
                    path, sigma = fuse_path(tr.ekf, path, sigma, return_sigma=True)
            tr.outs.append(dict(t=t, path=path, sigma=sigma))

    def bind(self, track_id):
        """Follow this track (the chooser's answer)."""
        if track_id != self.bound:
            self.bound = track_id
            self.n_binds += 1

    def _bind(self, sim, t):
        """Perfect choice: from the choice time, bind the track nearest the named ball (again if it is lost)."""
        if not self.oracle:
            return
        if self.bound is not None and any(tr.id == self.bound for tr in self.tracks):
            return
        if t < sim.t_choose - 1e-9:
            return
        target = sim.ball_pos(sim.volley.target)
        cand = [(np.linalg.norm(tr.ekf.x[:3] - target), tr.id) for tr in self.tracks if tr.hits >= 3]
        if cand and min(cand)[0] < 0.15:
            self.bound = min(cand)[1]
            self.n_binds += 1

    # ---------- catcher input ----------
    def path_state(self, sim, t):
        """Predicted path of the bound track for the catcher, delayed by the compute delay."""
        tr = next((tr for tr in self.tracks if tr.id == self.bound), None)
        if tr is None or len(tr.outs) <= self.delay or t < 0:
            return None
        o = tr.outs[0]
        spline = CubicSpline(HORIZONS, o["path"], axis=0)
        h = self.fine[self.fine >= t - o["t"] - 1e-9]
        return dict(pos=spline(h), vel=spline(h, 1), sigma=np.interp(h, HORIZONS, o["sigma"]))

    def bound_on_target(self, sim):
        """Whether the bound track is now nearer the named ball than any other ball."""
        tr = next((tr for tr in self.tracks if tr.id == self.bound), None)
        if tr is None:
            return None
        d = [np.linalg.norm(tr.ekf.x[:3] - sim.ball_pos(i)) for i in range(sim.n)]
        return int(np.argmin(d)) == sim.volley.target
