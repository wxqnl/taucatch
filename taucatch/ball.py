"""Ball properties and flight model (gravity, air drag, spin lift).

The same model drives the simulator (as an external force on the ball) and the
path prediction used by the privileged catcher, so the two always agree.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

RHO_AIR = 1.2
GRAVITY = 9.81


@dataclass(frozen=True)
class Ball:
    name: str
    radius: float      # m
    mass: float        # kg
    cd: float          # drag coefficient
    inertia_k: float   # I = k m r^2 (2/3 for a thin shell)
    spin_decay: float  # s, time constant of spin decay
    rgba: tuple

    @property
    def inertia(self):
        return self.inertia_k * self.mass * self.radius ** 2

    @property
    def params(self):
        """Packed parameters for the numba kernels."""
        return np.array([self.radius, self.mass, self.cd, self.spin_decay])


BALLS = {
    "pingpong": Ball("pingpong", 0.020, 0.0027, 0.50, 2 / 3, 1.5, (1.0, 0.55, 0.1, 1.0)),
    "tennis": Ball("tennis", 0.0335, 0.058, 0.55, 0.55, 3.0, (0.8, 0.9, 0.15, 1.0)),
}


@njit(cache=True)
def aero_accel(v, w, params):
    """Acceleration from drag and spin lift (gravity excluded)."""
    r, m, cd = params[0], params[1], params[2]
    area = np.pi * r * r
    speed = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    a = -0.5 * RHO_AIR * cd * area * speed * v / m
    wn = np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    if wn > 1e-9 and speed > 1e-6:
        s = r * wn / speed
        cl = s / (2.0 * s + 1.0)
        wv = np.cross(w / wn, v / speed)
        a = a + 0.5 * RHO_AIR * cl * area * speed * speed * wv / m
    return a


@njit(cache=True)
def _deriv(v, w, params):
    a = aero_accel(v, w, params)
    a[2] -= GRAVITY
    return a


@njit(cache=True)
def predict_path(p0, v0, w0, params, dt, n):
    """RK4 integration of the flight. Returns positions (n+1, 3) and velocities (n+1, 3)."""
    pos = np.empty((n + 1, 3))
    vel = np.empty((n + 1, 3))
    p = p0.copy()
    v = v0.copy()
    decay = np.exp(-dt / params[3])
    w = w0.copy()
    pos[0] = p
    vel[0] = v
    for i in range(n):
        k1v = _deriv(v, w, params)
        k1p = v
        k2v = _deriv(v + 0.5 * dt * k1v, w, params)
        k2p = v + 0.5 * dt * k1v
        k3v = _deriv(v + 0.5 * dt * k2v, w, params)
        k3p = v + 0.5 * dt * k2v
        k4v = _deriv(v + dt * k3v, w, params)
        k4p = v + dt * k3v
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        p = p + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        w = w * decay
        pos[i + 1] = p
        vel[i + 1] = v
    return pos, vel
