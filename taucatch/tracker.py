"""Closed loop from events (plan §3.8, Stage C): event cameras -> ball finder -> core -> predicted path.

`EventTracker.step(t)` runs after every control step (as the episode's frame callback);
`EventTracker.path_state(sim, t)` gives the catcher the predicted path (as its state_fn),
delayed by the compute delay.
"""
import collections

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from .ball import predict_path
from .ekf import BallEKF
from .events import EventCamConfig, EventRig
from .finder import WINDOW_US, BallFinder, decode, event_grid
from .predictor import HORIZONS, PathPredictor, features, triangulate

REC_START = {"human": -0.12, "launcher": -0.01, "none": -0.01}


def fuse_path(f, path, sigma, return_sigma=False):
    """Inverse-variance mix of a learned path (sigma per point) and the path of physics filter f, per point
    (and, with return_sigma, the sigma of the mix)."""
    pos, _ = predict_path(f.x[:3], f.x[3:6], f.x[6:9] * f.lift_scale, f.params, 0.001, 401)
    h = HORIZONS[:, None, None]
    P = f.P
    cov = P[:3, :3] + h * (P[:3, 3:6] + P[3:6, :3]) + h * h * P[3:6, 3:6]   # position covariance along the path
    var_ekf = np.trace(cov, axis1=1, axis2=2) / 3
    w = (1 / sigma ** 2) / (1 / sigma ** 2 + 1 / var_ekf)
    mixed = w[:, None] * path + (1 - w[:, None]) * pos[::10]
    if return_sigma:
        return mixed, np.sqrt(1 / (1 / sigma ** 2 + 1 / var_ekf))
    return mixed


def load_core(path, device="cpu"):
    """A trained path predictor. "<sweep file>:<tau>" picks one network of the single-timescale sweep."""
    path, _, tau = str(path).partition(":")
    ck = torch.load(path, map_location=device)
    if ck["core"] != "sweep":
        core = PathPredictor(ck["core"], ck["units"], ck["tau"])
        core.load_state_dict(ck["state"])
        return core.to(device).eval()
    g = ck["taus"].index(float(tau))
    st = ck["state"]
    core = PathPredictor("single", ck["units"], float(tau))
    core.load_state_dict({
        "core.inp.weight": st["w_in"][g].T, "core.inp.bias": st["b_in"][g], "core.rec.weight": st["w_rec"][g].T,
        "core.log_tau": core.core.log_tau, "head.weight": st["w_out"][g].T, "head.bias": st["b_out"][g],
        "h": core.h, "center": core.center})
    return core.to(device).eval()


class EventTracker:
    """`core_path`: a trained path predictor; "ekf" for the physics filter baseline (`ekf_scale`
    scales its drag and lift coefficients); or "fused:<predictor>" for both, combined per path
    point by their uncertainties."""

    def __init__(self, sim, finder_path, core_path, rig_cfg=EventCamConfig(), delay=0.001, unit_mask=None,
                 score_min=0.5, ekf_scale=1.0):
        self.sim = sim
        self.rig = EventRig(sim, rig_cfg)
        self.geoms = self.rig.geoms
        self.finder = BallFinder().eval()
        self.finder.load_state_dict(torch.load(finder_path, map_location="cpu"))
        core_path = str(core_path)
        self.fused = core_path.startswith("fused:")
        self.ekf = BallEKF(sim.ball, self.geoms, ekf_scale) if core_path == "ekf" or self.fused else None
        self.core = None if core_path == "ekf" else load_core(core_path.removeprefix("fused:"))
        self.delay = round(delay / 0.001)
        self.unit_mask = None if unit_mask is None else torch.as_tensor(unit_mask, dtype=torch.float32)
        self.score_min = score_min
        self.fine = np.arange(0.0, HORIZONS[-1] + 1e-9, 0.001)
        self.stop_after_contact = True      # nothing after contact matters for the catch; videos keep going

    def reset(self, thrower):
        self.t_rec = REC_START[thrower]
        self.on = False
        self.hist = [collections.deque(maxlen=WINDOW_US // 1000 + 1) for _ in self.geoms]
        self.state = None
        self.outs = collections.deque(maxlen=self.delay + 1)
        self.first_det_t = None
        self.log = []
        if self.ekf is not None:
            self.ekf.x = None

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
        u, v, s = decode(*self.finder(torch.from_numpy(np.stack(grids))))
        det = torch.stack([u, v, s], 1).numpy()[None]
        if self.first_det_t is None and t >= 0 and det[0, :, 2].max() > self.score_min:
            self.first_det_t = t
        if self.core is None:
            self._ekf_step(t, det[0], evs)
            return
        if self.fused:
            self._ekf_step(t, det[0], evs, log=False)
        x = torch.from_numpy(features(self.geoms, det, self.score_min))[None]
        out, self.state = self.core(x, self.state, self.unit_mask)
        path = out["path"][0, 0].numpy().astype(float)
        sigma = np.exp(out["log_sigma"][0, 0].numpy().astype(float))
        if self.fused and self.ekf.x is not None:
            path = self._fuse(path, sigma)
        self.outs.append(dict(t=t, path=path, sigma=sigma))
        h = self.state[0] if isinstance(self.state, tuple) else self.state
        self.log.append(dict(t=t, det=det[0], path=path, h=h[0].numpy().copy(), events=evs))

    def _fuse(self, path, sigma):
        return fuse_path(self.ekf, path, sigma)

    def _ekf_step(self, t, det, evs, log=True):
        f = self.ekf
        if f.x is None:
            if t < 0 or (det[:, 2] <= self.score_min).any():
                return
            f.start(triangulate(self.geoms, det[0, :2].astype(float), det[1, :2].astype(float)), t)
        else:
            f.predict(t)
            f.update(det, self.score_min)
        if not log:
            return
        x = f.x.copy()
        pos, _ = predict_path(x[:3], x[3:6], x[6:9] * f.lift_scale, f.params, 0.001, 401)
        self.outs.append(dict(t=t, state=x))
        self.log.append(dict(t=t, det=det, path=pos[::10], events=evs))

    def path_state(self, sim, t):
        """Predicted path for the catcher at time t: positions and velocities every 1 ms from t."""
        if len(self.outs) <= self.delay or self.first_det_t is None or t < 0:
            return None
        o = self.outs[0] if len(self.outs) > self.delay else None
        if "state" in o:                               # physics filter: its state, brought forward to t
            x = o["state"]
            for _ in range(int(round((t - o["t"]) / 0.001))):
                x = self.ekf._f(x, 0.001)
            return x[:3], x[3:6], x[6:9] * self.ekf.lift_scale
        spline = CubicSpline(HORIZONS, o["path"], axis=0)
        h = self.fine[self.fine >= t - o["t"] - 1e-9]
        return dict(pos=spline(h), vel=spline(h, 1))
