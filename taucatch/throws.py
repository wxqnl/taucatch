"""Throw generator: the thrower aims (spin included) at a point in the catch window."""
from dataclasses import dataclass

import mujoco
import numpy as np

from .ball import BALLS, predict_path
from .scene import BASE_Z, WINDOW_X

# Default ranges per ball (plan §5.3).
SPEED_RANGE = {"pingpong": (8.0, 20.0), "tennis": (10.0, 25.0)}
SPIN_RANGE = {"pingpong": (0.0, 40.0), "tennis": (0.0, 30.0)}  # rev/s
RELEASE_X, RELEASE_Y, RELEASE_Z = (3.0, 4.0), (-0.5, 0.5), (1.0, 1.6)
WINDOW_CENTER = np.array([0.0, BASE_Z + 0.5])   # (y, z) of the aimed point; the palm waits here
WINDOW_HALF = 0.3                              # m, aimed points are uniform in a square of this half size


@dataclass
class Throw:
    ball: str
    p0: np.ndarray
    v0: np.ndarray
    w0: np.ndarray       # rad/s, world frame
    target: np.ndarray   # aimed point on the plane x = WINDOW_X
    speed: float
    spin_rps: float


def window_crossing(p0, v0, w0, params, dt=0.001, n=1500):
    """Point and time where the flight crosses x = WINDOW_X (None if it never does)."""
    pos, _ = predict_path(p0, v0, w0, params, dt, n)
    idx = np.nonzero(pos[:, 0] <= WINDOW_X)[0]
    if len(idx) == 0 or idx[0] == 0:
        return None, None
    i = idx[0]
    a = (pos[i - 1, 0] - WINDOW_X) / (pos[i - 1, 0] - pos[i, 0])
    return pos[i - 1] + a * (pos[i] - pos[i - 1]), (i - 1 + a) * dt


def _velocity(speed, az, el):
    return speed * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def solve_launch(p0, target, speed, w0, params, iters=12):
    """Find the launch direction that hits `target` (y, z on the window plane) with spin."""
    d = np.array([WINDOW_X, target[0], target[1]]) - p0
    horiz = np.hypot(d[0], d[1])
    az = np.arctan2(d[1], d[0])
    el = np.arctan2(d[2], horiz) + 0.5 * 9.81 * (horiz / speed) / speed
    x = np.array([az, el])
    for _ in range(iters):
        c, _ = window_crossing(p0, _velocity(speed, *x), w0, params)
        if c is None:
            x[1] += 0.05
            continue
        err = c[1:] - target
        if np.linalg.norm(err) < 1e-4:
            break
        jac = np.zeros((2, 2))
        for k in range(2):
            dx = np.zeros(2)
            dx[k] = 1e-4
            ck, _ = window_crossing(p0, _velocity(speed, *(x + dx)), w0, params)
            jac[:, k] = (ck[1:] - c[1:]) / 1e-4
        x = x - np.linalg.solve(jac, err)
    return _velocity(speed, *x)


def sample_throw(rng, ball="pingpong", speed=None, spin_rps=None, target=None, window_half=WINDOW_HALF, ring=None,
                 p0=None):
    """`ring` (m): aim at a point this far from the window center (where the palm waits), in a random
    direction, instead of a uniform point in the square. `p0`: release point (default: sampled)."""
    params = BALLS[ball].params
    if p0 is None:
        p0 = np.array([rng.uniform(*RELEASE_X), rng.uniform(*RELEASE_Y), rng.uniform(*RELEASE_Z)])
    speed = rng.uniform(*SPEED_RANGE[ball]) if speed is None else speed
    spin_rps = rng.uniform(*SPIN_RANGE[ball]) if spin_rps is None else spin_rps
    axis = rng.normal(size=3)
    w0 = 2 * np.pi * spin_rps * axis / np.linalg.norm(axis)
    if target is None and ring is not None:
        phi = rng.uniform(0, 2 * np.pi)
        target = WINDOW_CENTER + ring * np.array([np.cos(phi), np.sin(phi)])
    elif target is None:
        target = WINDOW_CENTER + rng.uniform(-window_half, window_half, size=2)
    v0 = solve_launch(p0, np.asarray(target, float), speed, w0, params)
    return Throw(ball, p0, v0, w0, np.asarray(target, float), speed, spin_rps)


# ---------- thrower figure (plan §5.3) ----------
ARM_UPPER, ARM_FORE = 0.30, 0.35      # m, shoulder to elbow, elbow to ball centre
REACH = 0.62                          # m, shoulder to ball centre during the swing (elbow slightly bent)
SHOULDER_DY, SHOULDER_Z = -0.20, 1.45   # nominal right shoulder: 20 cm left of the release point, 1.45 m high


def _z_to(vec):
    """Quaternion that turns the body z axis onto `vec`."""
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, np.asarray(vec, float))
    return q


class ThrowerMotion:
    """Right-handed windmill swing: the ball travels on a circle around the shoulder, speeding
    up from rest, and leaves the hand at p0 with velocity v0 at t = 0."""

    def __init__(self, throw, swing=2.6, cock=0.1, follow=0.15):
        p0, v0 = throw.p0, throw.v0
        speed = np.linalg.norm(v0)
        self.vhat = v0 / speed
        nominal = np.array([p0[0], p0[1] + SHOULDER_DY, SHOULDER_Z])
        n = p0 - nominal
        n -= n.dot(self.vhat) * self.vhat
        self.n = n / np.linalg.norm(n)
        self.c = p0 - REACH * self.n                  # shoulder
        self.omega = speed / REACH
        self.T = 2 * swing / self.omega               # swing time, constant angular acceleration
        self.swing, self.follow = swing, follow
        self.t_start = -round(self.T + cock, 3)

    def angle(self, t):
        w, T = self.omega, self.T
        if t <= -T:
            return -self.swing, 0.0
        if t <= 0:
            return w * t + w * t * t / (2 * T), w + w * t / T
        tt = min(t, self.follow)
        return w * tt - w * tt * tt / (2 * self.follow), max(w - w * tt / self.follow, 0.0)

    def hand(self, t):
        """Ball position and velocity while it is held (t < 0)."""
        th, thd = self.angle(t)
        radial = np.cos(th) * self.n + np.sin(th) * self.vhat
        tangent = -np.sin(th) * self.n + np.cos(th) * self.vhat
        return self.c + REACH * radial, REACH * thd * tangent

    def parts(self, t):
        """Mocap poses {name: (pos, quat)} of the thrower figure at time t."""
        th, _ = self.angle(t)
        radial = np.cos(th) * self.n + np.sin(th) * self.vhat
        tangent = -np.sin(th) * self.n + np.cos(th) * self.vhat
        h = self.c + REACH * radial
        cb = (ARM_UPPER ** 2 + REACH ** 2 - ARM_FORE ** 2) / (2 * ARM_UPPER * REACH)
        elbow = self.c + ARM_UPPER * (cb * radial - np.sqrt(1 - cb * cb) * tangent)   # elbow trails the hand
        fdir = (h - elbow) / np.linalg.norm(h - elbow)
        wrist = h - 0.09 * fdir
        body_c = self.c + np.array([0.0, -0.2, 0.0])
        hip = np.array([body_c[0] + 0.05, body_c[1], 0.95])
        neck = np.array([body_c[0], body_c[1], self.c[2] + 0.05])
        return {
            "thr_upper": ((self.c + elbow) / 2, _z_to(elbow - self.c)),
            "thr_fore": ((elbow + wrist) / 2, _z_to(wrist - elbow)),
            "thr_hand": (h - 0.055 * fdir, np.array([1.0, 0, 0, 0])),
            "thr_torso": ((hip + neck) / 2, _z_to(neck - hip)),
            "thr_head": (neck + np.array([0, 0, 0.2]), np.array([1.0, 0, 0, 0])),
            "thr_leg_l": (np.array([hip[0], hip[1] - 0.1, 0.5]), np.array([1.0, 0, 0, 0])),
            "thr_leg_r": (np.array([hip[0], hip[1] + 0.1, 0.5]), np.array([1.0, 0, 0, 0])),
        }


def launcher_parts(throw):
    """Static launcher: a barrel ending at the release point, on a stand."""
    vhat = throw.v0 / np.linalg.norm(throw.v0)
    muzzle = throw.p0 - 0.04 * vhat
    stand_top = muzzle - 0.4 * vhat
    return {
        "thr_barrel": (muzzle - 0.2 * vhat, _z_to(vhat)),
        "thr_stand": (np.array([stand_top[0], stand_top[1], stand_top[2] - 0.85]), np.array([1.0, 0, 0, 0])),
    }
