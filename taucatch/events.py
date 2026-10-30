"""Simulated event cameras (plan §3.2, §5.2).

Each step, MuJoCo renders the scene without the ball (robot as its collision meshes, no
anti-aliasing: geometry, not pixel count, sets the render cost on the CPU renderer). The
ball is then drawn analytically with exact sub-pixel coverage and depth test, since it is
only 2 to 3 px wide at release. Changes in log intensity become events: ESIM-style
threshold crossings with timing interpolated between frames, per-pixel threshold
mismatch, a refractory period, and v2e-style shot noise and leak events.

Image coordinates: u to the right, v down, pixel (row i, col j) covers [j, j+1) x [i, i+1).
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import gl_context
from numba import njit

EVENT_CAMS = ("ev_left", "ev_right")


@dataclass
class EventCamConfig:
    width: int = 640
    height: int = 480
    threshold: float = 0.2          # log-intensity contrast threshold (mean)
    threshold_sigma: float = 0.03   # pixel-to-pixel threshold mismatch
    refractory: float = 1e-4        # s
    shot_rate: float = 0.2          # Hz per pixel, random polarity
    leak_rate: float = 0.1          # Hz per pixel, ON events
    log_eps: float = 0.01           # L = log(I + eps), I linear intensity in [0, 1]
    supersample: int = 6            # ball coverage samples per pixel side
    ball_gain: float = 1.0          # ball brightness relative to its color
    seed: int = 0


class CamGeom:
    """Pinhole model of a fixed MuJoCo camera (camera looks along its -z axis)."""

    def __init__(self, model, data, name, width, height):
        cid = model.camera(name).id
        self.name, self.width, self.height = name, width, height
        self.f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
        self.cx, self.cy = width / 2, height / 2
        self.pos = data.cam_xpos[cid].copy()
        self.rot = data.cam_xmat[cid].reshape(3, 3).copy()   # columns: camera x, y, z in world

    def project(self, p):
        """World points (..., 3) -> u, v (px) and depth along the optical axis (m)."""
        pc = (np.asarray(p) - self.pos) @ self.rot
        z = -pc[..., 2]
        return self.cx + self.f * pc[..., 0] / z, self.cy - self.f * pc[..., 1] / z, z

    def ray(self, u, v):
        """World direction (unit) through image point (u, v)."""
        d = self.rot @ np.array([(u - self.cx) / self.f, -(v - self.cy) / self.f, -1.0])
        return d / np.linalg.norm(d)


@njit(cache=True)
def _log_image(rgb, lut):
    """Flip a bottom-up RGB readout and map its gray level to log intensity."""
    h, w = rgb.shape[0], rgb.shape[1]
    out = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            g = (77 * np.int32(rgb[h - 1 - i, j, 0]) + 150 * np.int32(rgb[h - 1 - i, j, 1])
                 + 29 * np.int32(rgb[h - 1 - i, j, 2])) >> 8
            out[i, j] = lut[g]
    return out


@njit(cache=True)
def _ball_layer(L, depth, depth_c, depth_d, eps, cam_pos, rot, f, cx, cy, center, radius, shade, ss):
    """Draw a shaded sphere into the log image `L` with coverage and depth test.

    `depth` is the raw bottom-up reversed-Z buffer of the background render.
    Returns (u, v, radius_px, depth, visible samples, ball samples)."""
    h, w = L.shape
    rel = center - cam_pos
    pc = np.array([rel[0] * rot[0, k] + rel[1] * rot[1, k] + rel[2] * rot[2, k] for k in range(3)])
    z = -pc[2]
    if z <= radius:
        return np.nan, np.nan, np.nan, z, 0, 0
    u0 = cx + f * pc[0] / z
    v0 = cy - f * pc[1] / z
    rpx = f * radius / np.sqrt(max(z * z - radius * radius, 1e-12)) * (1.0 + radius / z) + 1.0
    j0, j1 = max(int(np.floor(u0 - rpx)), 0), min(int(np.ceil(u0 + rpx)), w)
    i0, i1 = max(int(np.floor(v0 - rpx)), 0), min(int(np.ceil(v0 + rpx)), h)
    oc = cam_pos - center
    c = oc @ oc - radius * radius
    n_vis, n_ball = 0, 0
    for i in range(i0, i1):
        for j in range(j0, j1):
            zb = depth_d / (depth[h - 1 - i, j] + depth_c)
            acc, hits = 0.0, 0
            for a in range(ss):
                for b in range(ss):
                    dc0 = (j + (b + 0.5) / ss - cx) / f
                    dc1 = -(i + (a + 0.5) / ss - cy) / f
                    norm = np.sqrt(dc0 * dc0 + dc1 * dc1 + 1.0)
                    d = (rot[:, 0] * dc0 + rot[:, 1] * dc1 - rot[:, 2]) / norm
                    bq = oc @ d
                    disc = bq * bq - c
                    if disc < 0.0:
                        continue
                    t = -bq - np.sqrt(disc)
                    n_ball += 1
                    if t / norm >= zb:
                        continue
                    n_vis += 1
                    hits += 1
                    nrm = (oc + t * d) / radius
                    acc += shade * (0.45 + 0.55 * max(0.0, -(nrm @ d)))
            if hits > 0:
                bg = np.exp(L[i, j]) - eps
                L[i, j] = np.log((acc + (ss * ss - hits) * bg) / (ss * ss) + eps)
    return u0, v0, f * radius / z, z, n_vis, n_ball


@njit(cache=True)
def _crossings(idx, L1, L0, Lref, last_t, con, coff, t0, t1, refr, w):
    """Threshold crossings of the pixels in `idx` between frames at t0 (L0) and t1 (L1)."""
    n = 0
    for k in range(idx.size):
        p = idx[k]
        if L1[p] - Lref[p] >= con[p]:
            n += int((L1[p] - Lref[p]) / con[p])
        elif Lref[p] - L1[p] >= coff[p]:
            n += int((Lref[p] - L1[p]) / coff[p])
    xs = np.empty(n, np.uint16)
    ys = np.empty(n, np.uint16)
    ts = np.empty(n, np.float64)
    ps = np.empty(n, np.int8)
    m = 0
    for k in range(idx.size):
        p = idx[k]
        d = L1[p] - L0[p]
        up = L1[p] - Lref[p] >= con[p]
        down = Lref[p] - L1[p] >= coff[p]
        if not (up or down):
            continue
        c = con[p] if up else -coff[p]
        cnt = int((L1[p] - Lref[p]) / c)
        for q in range(1, cnt + 1):
            level = Lref[p] + q * c
            frac = (level - L0[p]) / d if d != 0.0 else 1.0
            frac = min(max(frac, 0.0), 1.0)
            te = t0 + frac * (t1 - t0)
            if te - last_t[p] < refr:
                continue
            last_t[p] = te
            xs[m] = p % w
            ys[m] = p // w
            ts[m] = te
            ps[m] = 1 if up else -1
            m += 1
        Lref[p] += cnt * c
    return xs[:m], ys[:m], ts[:m], ps[:m]


class EventRig:
    """Two (or more) event cameras on one simulation, stepped after every control step.

    `reset(t)` takes the reference frame; `step(t)` renders the frame at time t and returns,
    per camera, the events in (t_prev, t] and the ball labels of the frame ("label": the first
    ball, "labels": every ball)."""

    def __init__(self, sim, cfg=EventCamConfig(), cams=EVENT_CAMS):
        self.sim, self.cfg = sim, cfg
        m, d = sim.model, sim.data
        W, H = cfg.width, cfg.height
        mujoco.mj_forward(m, d)
        self.geoms = [CamGeom(m, d, name, W, H) for name in cams]
        self.cam_ids = [m.camera(name).id for name in cams]

        # render context: small offscreen buffer, no multisampling (restore model settings after)
        saved = (m.vis.global_.offwidth, m.vis.global_.offheight, m.vis.quality.offsamples)
        m.vis.global_.offwidth, m.vis.global_.offheight, m.vis.quality.offsamples = W, H, 0
        self.gl = gl_context.GLContext(W, H)
        self.gl.make_current()
        self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.con.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        m.vis.global_.offwidth, m.vis.global_.offheight, m.vis.quality.offsamples = saved
        self.rect = mujoco.MjrRect(0, 0, W, H)
        self.scn = mujoco.MjvScene(m, maxgeom=2000)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SKYBOX):
            self.scn.flags[flag] = False      # shadows alone cost 6x the render time
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[0] = 1          # world, thrower
        self.opt.geomgroup[3] = 1          # robot collision meshes (the ball sits in group 1)
        self.opt.sitegroup[:] = 0
        self.pert = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        extent = m.stat.extent
        near, far = m.vis.map.znear * extent, m.vis.map.zfar * extent
        c_coef = -(far + near) / (far - near)
        d_coef = -(2 * far * near) / (far - near)
        self.depth_c, self.depth_d = -0.5 * c_coef - 0.5, -0.5 * d_coef

        self.log_lut = np.log((np.arange(256) / 255.0) ** 2.2 + cfg.log_eps)
        mask = np.ones(m.nq, bool)
        for b in sim.ids.balls:
            mask[b.qpos:b.qpos + 7] = False
        self.key_idx = np.nonzero(mask)[0]
        self.set_config(cfg)

    def set_config(self, cfg):
        """New sensor settings (same size): thresholds are drawn again per pixel."""
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        npix = cfg.width * cfg.height
        n = len(self.geoms)
        self.con_thr = [np.clip(self.rng.normal(cfg.threshold, cfg.threshold_sigma, npix), 0.05, None) for _ in range(n)]
        self.coff_thr = [np.clip(self.rng.normal(cfg.threshold, cfg.threshold_sigma, npix), 0.05, None) for _ in range(n)]

    # ---------- rendering ----------
    def _scene_key(self):
        d = self.sim.data
        return np.concatenate([d.qpos[self.key_idx], d.mocap_pos.ravel(), d.mocap_quat.ravel()])

    def _render_background(self, k):
        """Log intensity and raw depth buffer of camera k without the ball."""
        m, d = self.sim.model, self.sim.data
        W, H = self.cfg.width, self.cfg.height
        self.gl.make_current()
        self.cam.fixedcamid = self.cam_ids[k]
        mujoco.mjv_updateScene(m, d, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(self.rect, self.scn, self.con)
        rgb = np.empty((H, W, 3), np.uint8)
        depth = np.empty((H, W), np.float32)
        mujoco.mjr_readPixels(rgb, depth, self.rect, self.con)
        return _log_image(rgb, self.log_lut), depth

    def frame(self, k):
        """Log intensity image of camera k with the balls, plus one label per ball (n_balls, 5):
        u, v, radius (px), depth (m), visible fraction. The balls' brightness follows their current color."""
        key = self._scene_key()
        if self._bg[k] is None or np.max(np.abs(key - self._bg_key[k])) > 2e-5:
            self._bg[k] = self._render_background(k)
            self._bg_key[k] = key
        L, depth = self._bg[k]
        L = L.copy()
        g = self.geoms[k]
        m, d = self.sim.model, self.sim.data
        balls = self.sim.ids.balls
        centers = [d.xpos[b.body].copy() for b in balls]
        labels = np.zeros((len(balls), 5))
        for i in np.argsort([-np.linalg.norm(c - g.pos) for c in centers]):     # far balls first
            rgba = m.geom_rgba[balls[i].geom]
            shade = (self.cfg.ball_gain * (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2])) ** 2.2
            u, v, r, z, n_vis, n_ball = _ball_layer(L, depth, self.depth_c, self.depth_d, self.cfg.log_eps, g.pos,
                                                    g.rot, g.f, g.cx, g.cy, centers[i], m.geom_size[balls[i].geom, 0],
                                                    shade, self.cfg.supersample)
            labels[i] = u, v, r, z, n_vis / n_ball if n_ball else 0.0
        return L.ravel(), labels

    # ---------- events ----------
    def reset(self, t):
        cfg = self.cfg
        n = len(self.geoms)
        self._bg, self._bg_key = [None] * n, [None] * n
        self.t = t
        self.L, self.Lref, self.last_t = [], [], []
        labels = []
        for k in range(n):
            L, lab = self.frame(k)
            self.L.append(L)
            self.Lref.append(L.copy())
            self.last_t.append(np.full(L.size, -np.inf))
            labels.append(lab[0])
        return labels

    def step(self, t):
        cfg = self.cfg
        out = []
        for k in range(len(self.geoms)):
            L1, lab = self.frame(k)
            idx = np.flatnonzero(L1 != self.L[k])
            xs, ys, ts, ps = _crossings(idx, L1, self.L[k], self.Lref[k], self.last_t[k], self.con_thr[k],
                                        self.coff_thr[k], self.t, t, cfg.refractory, cfg.width)
            self.L[k] = L1
            xs, ys, ts, ps = self._add_noise(xs, ys, ts, ps, t)
            out.append(dict(x=xs, y=ys, t=ts, p=ps, label=lab[0], labels=lab))
        self.t = t
        return out

    def _add_noise(self, xs, ys, ts, ps, t):
        cfg = self.cfg
        dt = t - self.t
        npix = cfg.width * cfg.height
        n_shot = self.rng.poisson(cfg.shot_rate * npix * dt)
        n_leak = self.rng.poisson(cfg.leak_rate * npix * dt)
        n = n_shot + n_leak
        if n == 0:
            return xs, ys, ts, ps
        pix = self.rng.integers(0, npix, n)
        pol = np.concatenate([self.rng.choice(np.array([-1, 1], np.int8), n_shot), np.ones(n_leak, np.int8)])
        return (np.concatenate([xs, (pix % cfg.width).astype(np.uint16)]),
                np.concatenate([ys, (pix // cfg.width).astype(np.uint16)]),
                np.concatenate([ts, self.t + dt * self.rng.random(n)]),
                np.concatenate([ps, pol]))


def event_image(ev, width=640, height=480):
    """RGB picture of a batch of events: ON blue, OFF red, on white."""
    img = np.full((height, width, 3), 255, np.uint8)
    on = ev["p"] > 0
    img[ev["y"][on], ev["x"][on]] = (40, 90, 255)
    img[ev["y"][~on], ev["x"][~on]] = (230, 40, 40)
    return img
