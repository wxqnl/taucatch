"""Event cameras from rendered frames: per-pixel log-intensity change crossing a contrast threshold.

Each 4 ms slice is a sparse list of (pixel index, positive count, negative count). The contrast
threshold of every pixel is drawn once per toss, so pixels fire unevenly as on a real sensor.
"""
import mujoco
import numpy as np

CAMERAS = ("ev_left", "ev_right")
RES = 224
TRACE_TAUS = (0.01, 0.05, 0.25)            # s: decay of the R, G, B time surfaces
SLICE_S = 0.004
HOT_RATE = 200.0                           # events/s of a stuck pixel


class EventStream:
    def __init__(self, sim, threshold=0.15, spread=0.02, noise=0.0, hot=0.0):
        """noise: background events per pixel per second. hot: share of pixels stuck firing."""
        self.sim, self.threshold, self.spread, self.noise, self.hot = sim, threshold, spread, noise, hot
        self.r = mujoco.Renderer(sim.m, RES, RES)
        self.w = np.array([0.299, 0.587, 0.114], np.float32) / 255.0
        self.r.update_scene(sim.d, camera=CAMERAS[0])
        self.r.render()                  # the first frame after context creation differs
        self.c = self.ref = self.stuck = self.rng = None

    def _levels(self):
        mujoco.mj_kinematics(self.sim.m, self.sim.d)
        mujoco.mj_camlight(self.sim.m, self.sim.d)
        out = []
        for cam in CAMERAS:
            self.r.update_scene(self.sim.d, camera=cam)
            out.append(np.log(self.r.render().astype(np.float32) @ self.w + 0.02))
        return out

    def reset(self, seed):
        """Reference image from the current state; thresholds drawn for this toss."""
        rng = np.random.default_rng(seed + 7919)
        self.ref = self._levels()
        self.c = [np.clip(rng.normal(self.threshold, self.spread, (RES, RES)), 0.05, None).astype(np.float32)
                  for _ in CAMERAS]
        self.rng = rng
        self.stuck = [rng.choice(RES * RES, int(self.hot * RES * RES), replace=False).astype(np.uint16) for _ in CAMERAS]

    def _spurious(self, stuck):
        """Events no light caused this slice: background over the sensor and the stuck pixels, one each of random sign."""
        k = self.rng.poisson(RES * RES * self.noise * SLICE_S)
        pix = np.concatenate([self.rng.integers(0, RES * RES, k, dtype=np.uint16),
                              stuck[self.rng.random(stuck.size) < HOT_RATE * SLICE_S]])
        sign = self.rng.random(pix.size) < 0.5
        return pix, sign.astype(np.uint8), (~sign).astype(np.uint8)

    def slice(self):
        """Events since the previous call, per camera: (pixel index, positive, negative)."""
        out = []
        for level, ref, c, stuck in zip(self._levels(), self.ref, self.c, self.stuck):
            n = np.trunc((level - ref) / c)
            ref += n * c
            idx = np.flatnonzero(n)
            v = n.ravel()[idx]
            events = idx.astype(np.uint16), np.clip(v, 0, 255).astype(np.uint8), np.clip(-v, 0, 255).astype(np.uint8)
            if self.noise or self.hot:
                events = tuple(np.concatenate(p) for p in zip(events, self._spurious(stuck)))
            out.append(events)
        return out


def event_frame(pix, pos, neg, newest):
    """Policy image from the events of one window.
    Red: positive events; green: negative; blue: all events in the newest slice."""
    n = RES * RES
    pix = pix.astype(np.int64)
    p = np.bincount(pix, weights=pos, minlength=n)
    q = np.bincount(pix, weights=neg, minlength=n)
    r = np.bincount(pix[newest], weights=pos[newest].astype(np.float64) + neg[newest], minlength=n)
    frame = np.stack([p * 80, q * 80, r * 160], axis=-1).reshape(RES, RES, 3)
    return np.clip(frame, 0, 255).astype(np.uint8)


def trace_frame(age):
    """Policy image from the age (s) of each pixel's latest event: three time surfaces, so one frame
    shows the ball's track over the last few hundred ms. A pixel that has not fired has age inf."""
    frame = 255 * np.exp(-age.reshape(RES, RES, 1) / np.array(TRACE_TAUS, np.float32))
    return np.round(frame).astype(np.uint8)
