"""Selective catching (plan §5.10): several balls in the air at once, one of them named.

A volley is a set of throws, each from its own launcher or thrower, with its own release time
and color. The catcher learns which ball is named only at the choice time (release of the named
ball plus the choice delay) and waits until then.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from .ball import BALLS, aero_accel
from .catcher import HAND_CUP, CatcherConfig
from .scene import PARKED, TIMESTEP, part_name
from .sim import CONTROL_DT, CatchSim
from .throws import RELEASE_X, RELEASE_Z, WINDOW_CENTER, ThrowerMotion, launcher_parts, sample_throw

COLORS = {
    "orange": (1.0, 0.55, 0.1, 1.0),
    "yellow": (0.85, 0.9, 0.15, 1.0),
    "red": (0.85, 0.12, 0.1, 1.0),
    "blue": (0.15, 0.35, 0.9, 1.0),
    "green": (0.15, 0.7, 0.25, 1.0),
    "white": (0.92, 0.92, 0.9, 1.0),
    "purple": (0.55, 0.2, 0.75, 1.0),
    "pink": (1.0, 0.45, 0.7, 1.0),
}
VOLLEY_SPEED = {"pingpong": (10.0, 16.0), "tennis": (12.0, 18.0)}
VOLLEY_RELEASE_Y = (-0.9, 0.9)


@dataclass
class Volley:
    throws: list          # Throw per ball
    release: np.ndarray   # s, release time of each ball (the first at 0, whole ms)
    colors: list          # color name per ball
    target: int           # index of the named ball


def _spread(rng, sample, n, min_dist, tries=1000):
    """n samples whose pairwise distances are at least min_dist (rejection sampling)."""
    for _ in range(tries):
        pts = [sample() for _ in range(n)]
        if all(np.linalg.norm(pts[i] - pts[j]) >= min_dist for i in range(n) for j in range(i)):
            return pts
    raise RuntimeError("could not place the volley")


def sample_volley(rng, ball="tennis", n=2, speed=None, spin_rps=None, max_offset=0.08, window_half=0.3,
                  min_sep=0.2, launcher_sep=0.5, types=None):
    """n throws from launchers at least `launcher_sep` apart (sideways), aimed at points at least
    `min_sep` apart, released within `max_offset` s of each other. `types`: ball type per throw
    (default: all `ball`)."""
    types = [ball] * n if types is None else list(types)
    p0s = _spread(rng, lambda: np.array([rng.uniform(*RELEASE_X), rng.uniform(*VOLLEY_RELEASE_Y),
                                         rng.uniform(*RELEASE_Z)]), n, launcher_sep)
    aims = _spread(rng, lambda: WINDOW_CENTER + rng.uniform(-window_half, window_half, 2), n, min_sep)
    release = np.round(rng.uniform(0, max_offset, n), 3)
    release -= release.min()
    speeds = [None] * n if speed is None else np.broadcast_to(speed, (n,))
    spins = [None] * n if spin_rps is None else np.broadcast_to(spin_rps, (n,))
    throws = [sample_throw(rng, b, speed=rng.uniform(*VOLLEY_SPEED[b]) if sp is None else float(sp),
                           spin_rps=None if spin is None else float(spin), target=aim, p0=p0)
              for b, p0, aim, sp, spin in zip(types, p0s, aims, speeds, spins)]
    colors = list(rng.choice(list(COLORS), n, replace=False))
    return Volley(throws, release, colors, int(rng.integers(n)))


class VolleySim(CatchSim):
    """`thrower`: "launcher" (one static barrel per ball) or "human" (one thrower figure per ball)."""

    def __init__(self, ball="tennis", n_balls=2, cfg=CatcherConfig(), arm="agile", thrower="launcher", types=None):
        types = [ball] * n_balls if types is None else list(types)
        super().__init__(types[0], cfg, arm, thrower, balls=types)
        self.n = len(types)
        self.ball_types = [BALLS[b] for b in types]

    # ---------- state ----------
    def ball_state(self, i=0):
        d, b = self.data, self.ids.balls[i]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, d.qpos[b.qpos + 3:b.qpos + 7])
        w_world = rot.reshape(3, 3) @ d.qvel[b.dof + 3:b.dof + 6]
        return d.qpos[b.qpos:b.qpos + 3].copy(), d.qvel[b.dof:b.dof + 3].copy(), w_world

    def ball_pos(self, i):
        return self.data.xpos[self.ids.balls[i].body].copy()

    def reset(self, volley):
        m, d, ids = self.model, self.data, self.ids
        mujoco.mj_resetData(m, d)
        d.qpos[ids.arm_qpos] = self.home_q
        d.qpos[ids.hand_qpos] = HAND_CUP
        d.mocap_pos[:] = PARKED
        d.mocap_quat[:] = [1, 0, 0, 0]
        self.volley = volley
        self.motions = [ThrowerMotion(th) if self.thrower == "human" else None for th in volley.throws]
        self.rel_sub = [round(r / TIMESTEP) for r in volley.release]      # substep of each release
        starts = [0] + [round((r + mo.t_start) / CONTROL_DT) for r, mo in zip(volley.release, self.motions) if mo]
        self.k_start = min(starts)
        for i, (b, th) in enumerate(zip(ids.balls, volley.throws)):
            m.geom_rgba[b.geom] = COLORS[volley.colors[i]]
            if self.thrower == "launcher":
                self._place(launcher_parts(th), i)
        t0 = self.k_start * CONTROL_DT
        self._place_throwers(t0)
        for i in range(self.n):
            self._hold(i, t0)
        mujoco.mj_forward(m, d)
        self.throw, self.motion = volley.throws[volley.target], self.motions[volley.target]
        self.catcher.ball = self.ball_types[volley.target]
        self.catcher.reset(d)

    def dress(self, colors):
        """Thrower k's shirt (or launcher k's barrel) in colors[k] (rgb); kept until changed."""
        part = "thr_torso" if self.thrower == "human" else "thr_barrel"
        for k, rgb in enumerate(colors):
            self.model.geom_rgba[self.model.geom(part_name(part, k) + "_geom").id][:3] = rgb

    def _place(self, parts, k=0):
        for name, (pos, quat) in parts.items():
            j = self.ids.mocap[part_name(name, k)]
            self.data.mocap_pos[j] = pos
            self.data.mocap_quat[j] = quat

    def _place_throwers(self, t):
        for i, mo in enumerate(self.motions):
            if mo is not None:
                self._place(mo.parts(t - self.volley.release[i]), i)

    def _hold(self, i, t):
        """Ball i before its release: in the thrower's hand, or resting in the barrel."""
        d, b = self.data, self.ids.balls[i]
        mo = self.motions[i]
        if mo is not None:
            pos, vel = mo.hand(t - self.volley.release[i])
        else:
            pos, vel = self.volley.throws[i].p0, np.zeros(3)
        d.qpos[b.qpos:b.qpos + 3] = pos
        d.qpos[b.qpos + 3:b.qpos + 7] = [1, 0, 0, 0]
        d.qvel[b.dof:b.dof + 3] = vel
        d.qvel[b.dof + 3:b.dof + 6] = 0
        d.xfrc_applied[b.body] = 0

    def _release_ball(self, i):
        d, b, th = self.data, self.ids.balls[i], self.volley.throws[i]
        d.qpos[b.qpos:b.qpos + 3] = th.p0
        d.qpos[b.qpos + 3:b.qpos + 7] = [1, 0, 0, 0]
        d.qvel[b.dof:b.dof + 3] = th.v0
        d.qvel[b.dof + 3:b.dof + 6] = th.w0

    def _aero(self, i):
        p, v, w = self.ball_state(i)
        ball, body = self.ball_types[i], self.ids.balls[i].body
        self.data.xfrc_applied[body, :3] = ball.mass * aero_accel(v, w, ball.params)
        self.data.xfrc_applied[body, 3:] = -ball.inertia * w / ball.spin_decay

    def _hand_touches(self):
        """Indices of the balls touching the hand now."""
        geoms = {b.geom: i for i, b in enumerate(self.ids.balls)}
        out = set()
        for c in self.data.contact[: self.data.ncon]:
            if c.geom1 in geoms and c.geom2 in self.hand_geoms:
                out.add(geoms[c.geom1])
            elif c.geom2 in geoms and c.geom1 in self.hand_geoms:
                out.add(geoms[c.geom2])
        return out

    # ---------- episode ----------
    def run(self, volley, choose_delay=0.0, t_max=1.2, hold=0.5, frame_cb=None, state_fn=None):
        """One volley. The catcher sees the named ball's true state from the choice time on (or
        whatever `state_fn(sim, t)` gives). Outcome for the named ball, plus wrong touches/catches."""
        self.reset(volley)
        m, d, cat, ids = self.model, self.data, self.catcher, self.ids
        tgt = volley.target
        self.t_choose = volley.release[tgt] + choose_delay
        palm0 = d.site_xpos[ids.palm_site].copy()
        min_dist, touched, wrong_touch, palm_contact = np.inf, False, False, None
        k = self.k_start
        t = k * CONTROL_DT
        while t < t_max:
            if state_fn:
                state = state_fn(self, t)
            elif t >= max(self.t_choose, volley.release[tgt]) - 1e-9:
                state = self.ball_state(tgt)
            else:
                state = None
            cat.step(d, t, state)
            for s in range(self.substeps):
                g = k * self.substeps + s
                ts = g * TIMESTEP
                self._place_throwers(ts)
                for i in range(self.n):
                    if g < self.rel_sub[i]:
                        self._hold(i, ts)
                    else:
                        if g == self.rel_sub[i]:
                            self._release_ball(i)
                        self._aero(i)
                mujoco.mj_step(m, d)
            k += 1
            t = k * CONTROL_DT
            if frame_cb:
                frame_cb(self, t)
            bp = self.ball_pos(tgt)
            min_dist = min(min_dist, np.linalg.norm(bp - d.site_xpos[ids.palm_site]))
            touching = self._hand_touches()
            wrong_touch |= bool(touching - {tgt})
            if not touched and tgt in touching:
                touched = True
                cat.contact_t = t
                palm_contact = d.site_xpos[ids.palm_site].copy()
            if bp[2] < 0.1 or (cat.contact_t is not None and t > cat.contact_t + hold):
                break
            if cat.contact_t is None and bp[0] < -0.5:
                break
        palm = d.site_xpos[ids.palm_site]

        def held(i):
            p = self.ball_pos(i)
            return p[2] > 0.3 and np.linalg.norm(p - palm) < self.ball_types[i].radius + 0.05

        caught = touched and held(tgt) and t > (cat.contact_t or 0) + hold - 1e-6
        return dict(
            caught=bool(caught), touched=touched, wrong_touch=wrong_touch,
            wrong_held=any(held(i) for i in range(self.n) if i != tgt), min_dist=float(min_dist),
            contact_t=cat.contact_t, first_cmd_t=cat.first_cmd_t, t_choose=float(self.t_choose),
            reach=None if palm_contact is None else float(np.linalg.norm(palm_contact - palm0)),
            speed=volley.throws[tgt].speed,
        )
