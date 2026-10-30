"""RGB camera for the VLA (plan §5.10): frames at a fixed rate with motion blur and readout delay.

A frame taken at time t integrates the light over [t - exposure, t]: it is the mean of renders
1 ms apart over the exposure (the control step is 1 ms). It becomes available `latency` s later.
"""
import collections
from dataclasses import dataclass

import mujoco
import numpy as np

from .events import CamGeom


@dataclass
class RGBCamConfig:
    name: str = "rgb_head"
    width: int = 640
    height: int = 480
    fps: float = 240.0
    exposure: float = 0.002    # s
    latency: float = 0.003     # s, readout and transfer after the exposure ends


class RGBCam:
    """Call `step(sim, t)` after every control step; `latest(t)` gives the newest frame available at t."""

    def __init__(self, sim, cfg=RGBCamConfig()):
        self.sim, self.cfg = sim, cfg
        self.renderer = mujoco.Renderer(sim.model, cfg.height, cfg.width)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
            self.renderer.scene.flags[flag] = False       # shadows dominate the CPU render time
        mujoco.mj_forward(sim.model, sim.data)          # camera poses
        self.geom = CamGeom(sim.model, sim.data, cfg.name, cfg.width, cfg.height)
        self.n_sub = max(1, round(cfg.exposure / 0.001) + 1)
        self.reset()

    def reset(self):
        self.subs = collections.deque(maxlen=self.n_sub)
        self.frames = collections.deque(maxlen=8)      # (time taken, image)
        self.next_t = None

    def step(self, sim, t):
        cfg = self.cfg
        if self.next_t is None:
            self.next_t = np.ceil(t * cfg.fps) / cfg.fps
        if t < self.next_t - cfg.exposure - 1e-9:
            return
        self.renderer.update_scene(sim.data, camera=cfg.name)
        self.subs.append((t, self.renderer.render().astype(np.float32)))
        if t >= self.next_t - 1e-9:
            # Rendering occurs on the 1 ms simulation grid, after the nominal frame time.
            # Timestamp the actual last sample, so readout and track snapshots cannot see ahead.
            imgs = [im for ts, im in self.subs if ts >= t - cfg.exposure - 1e-9]
            self.frames.append((t, np.mean(imgs, 0).astype(np.uint8)))
            self.next_t += 1.0 / cfg.fps

    def latest(self, t):
        """Newest frame whose readout has finished by time t: (time taken, image) or None."""
        ok = [f for f in self.frames if f[0] + self.cfg.latency <= t + 1e-9]
        return ok[-1] if ok else None
