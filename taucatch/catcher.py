"""Reach controller and hand closure (plan §3.6, §3.7).

`Catcher` turns a predicted ball path into arm torques and finger targets. The
privileged catcher feeds it the true ball state; the learned predictors feed it
their own predicted path, so every method shares this controller.

Reach: every few ms, candidate catch points along the predicted path are solved
by IK and timed with a jerk-limited joint trajectory (Ruckig) from the current
reference state; the point with the most time to spare (near the window plane)
wins. Every 1 ms the reference moves toward it, and computed torque tracks it.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from .arm import ArmIK, JointPlanner
from .ball import predict_path
from .scene import BASE_Z, WINDOW_X

# Hand poses as 24 joint angles, in model joint order:
# WRJ2 WRJ1 | FFJ4..1 | MFJ4..1 | RFJ4..1 | LFJ5..1 | THJ5..1
HAND_CUP = np.array([0, 0,
                     -0.1, 0.25, 0.3, 0.1,
                     0.0, 0.25, 0.3, 0.1,
                     0.1, 0.25, 0.3, 0.1,
                     0.1, 0.2, 0.25, 0.3, 0.1,
                     -0.3, 0.2, 0.0, 0.0, 0.0])


def closed_pose(mcp, pipdip, th5, th4, th2, th1):
    """Closing pose: all fingers flex at the knuckle (mcp) and curl (pipdip split 60/40), thumb opposes."""
    fingers = []
    for spread in (-0.1, 0.0, 0.1):                       # FF, MF, RF: J4 J3 J2 J1
        fingers += [spread, mcp, 0.6 * pipdip, 0.4 * pipdip]
    lf = [0.4, 0.15, mcp, 0.6 * pipdip, 0.4 * pipdip]       # LF: J5 J4 J3 J2 J1
    return np.array([0, 0] + fingers + lf + [th5, th4, 0.1, th2, th1])


# Found by scripts/grasp_refine.py: holds a ping-pong ball landing within 1 cm of the
# palm target at 3 and 8 m/s (10/10 trials).
HAND_CLOSED = closed_pose(1.3, 2.2, 0.0, 0.8, 0.6, 0.3)


@dataclass
class CatcherConfig:
    plan_every: float = 0.004       # s, replanning period
    horizon: float = 0.6            # s, predicted path length
    candidate_every: float = 0.005  # s, spacing of candidate catch points on the path
    slack_cap: float = 0.05         # s, extra time beyond this does not raise the score
    x_weight: float = 0.3           # s/m, preference for catch points near the window plane
    close_lead: float = 0.004       # s, start closing the fingers this long before contact
    start_delay: float = 0.0        # s, no arm command before this time (perception delay)
    kp: float = 1600.0              # joint tracking gains of the computed-torque loop
    kd: float = 80.0
    region_x: tuple = (0.45, 0.85)
    region_y: tuple = (-0.45, 0.45)
    region_z: tuple = (BASE_Z + 0.1, BASE_Z + 0.9)
    free_roll: bool = True          # leave the palm roll about its normal free (IK on position + normal)
    sigma_lo: float = None          # m: with a predicted path's sigma at the catch point above sigma_lo, the
    sigma_hi: float = None          # reach goes only part of the way from home (none at sigma_hi); None: off

    def __post_init__(self):
        if (self.sigma_lo is None) != (self.sigma_hi is None):
            raise ValueError("sigma_lo and sigma_hi must be set together")
        if self.sigma_lo is not None and not (0 <= self.sigma_lo < self.sigma_hi < np.inf):
            raise ValueError("sigma thresholds must satisfy 0 <= LO < HI < infinity")


def palm_rotation(normal, up=np.array([0.0, 0.0, 1.0])):
    """Palm frame whose -y axis is `normal` and whose fingers (+z) point as close to `up` as possible."""
    n = normal / np.linalg.norm(normal)
    f = up - up.dot(n) * n
    f /= np.linalg.norm(f)
    y = -n
    x = np.cross(y, f)
    return np.column_stack([x, y, f])


def cradle_rotation(vhat):
    """Palm facing up, fingers pointing back toward the thrower."""
    f = -np.array([vhat[0], vhat[1], 0.0])
    return palm_rotation(np.array([0.0, 0.0, 1.0]), up=f / np.linalg.norm(f))


def hand_ctrl(model, ids, pose):
    """Actuator targets for a 24-joint hand pose (tendon actuators get the summed joint angles)."""
    d = mujoco.MjData(model)
    d.qpos[ids.hand_qpos] = pose
    mujoco.mj_forward(model, d)
    return d.actuator_length[ids.hand_act].copy()


class Catcher:
    def __init__(self, model, ids, ball, q_home, cfg=CatcherConfig(), arm="fast"):
        self.m, self.ids, self.ball, self.cfg = model, ids, ball, cfg
        self.dt = 0.001
        self.q_home = q_home.copy()
        self.ik = ArmIK(model, ids, q_home)
        self.planner = JointPlanner(self.dt, arm)
        self.cup = hand_ctrl(model, ids, HAND_CUP)
        self.closed = hand_ctrl(model, ids, HAND_CLOSED)
        self.mfull = np.zeros((model.nv, model.nv))

    def reset(self, data):
        q = data.qpos[self.ids.arm_qpos].copy()
        self.ref = (q, np.zeros(7), np.zeros(7))
        self.target = None
        self.next_plan = 0.0
        self.contact_t = None
        self.closing = False
        self.first_cmd_t = None
        self.saturated = False

    # ---------- planning ----------
    def plan(self, t, p, v, w):
        """Choose the catch point on the path predicted from the ball state (p, v, w) at time t."""
        pos, vel = predict_path(p, v, w, self.ball.params, self.dt, int(self.cfg.horizon / self.dt))
        self.plan_path(t, pos, vel)

    def plan_path(self, t, pos, vel, sigma=None):
        """Choose the catch point on a predicted path: positions and velocities every 1 ms from time t
        (and the path's uncertainty per point, sigma)."""
        cfg = self.cfg
        ok = ((pos[:, 0] > cfg.region_x[0]) & (pos[:, 0] < cfg.region_x[1]) & (pos[:, 1] > cfg.region_y[0])
              & (pos[:, 1] < cfg.region_y[1]) & (pos[:, 2] > cfg.region_z[0]) & (pos[:, 2] < cfg.region_z[1]))
        idx = np.nonzero(ok)[0][:: max(1, round(cfg.candidate_every / self.dt))]
        if len(idx) == 0:
            return
        q_seed = self.target["q"] if self.target is not None else self.ref[0]
        best = None
        for k in idx:
            normal = -vel[k] / np.linalg.norm(vel[k])
            if cfg.free_roll:
                q, perr, rerr, _ = self.ik.solve_normal(q_seed, pos[k], normal)
            else:
                q, perr, rerr, _ = self.ik.solve(q_seed, pos[k], palm_rotation(normal))
            if perr > 3e-3 or rerr > 0.05:
                continue
            q_seed = q
            slack = k * self.dt - self.planner.duration(self.ref, q)
            score = min(slack, cfg.slack_cap) - cfg.x_weight * abs(pos[k, 0] - WINDOW_X)
            if best is None or score > best[0]:
                best = (score, k, q)
        if best is None:
            return
        _, k, q = best
        self.target = dict(p=pos[k].copy(), t=t + k * self.dt, v=vel[k].copy(), q=q, path=pos, path_t0=t,
                           sigma=None if sigma is None else float(sigma[k]))

    # ---------- control step ----------
    def step(self, data, t, ball_state):
        cfg, ids, m = self.cfg, self.ids, self.m
        if ball_state is not None and t >= cfg.start_delay and t >= self.next_plan and self.contact_t is None:
            if isinstance(ball_state, dict):          # a predicted path (learned predictors)
                self.plan_path(t, ball_state["pos"], ball_state["vel"], ball_state.get("sigma"))
            else:                                     # a ball state (privileged catcher, physics filter)
                self.plan(t, *ball_state)
            self.next_plan = t + cfg.plan_every
        if self.target is not None and self.first_cmd_t is None:
            self.first_cmd_t = t
        target_q = self.target["q"] if self.target is not None else self.q_home
        if self.target is not None and cfg.sigma_hi is not None and self.target["sigma"] is not None:
            a = np.clip((cfg.sigma_hi - self.target["sigma"]) / (cfg.sigma_hi - cfg.sigma_lo), 0.0, 1.0)
            target_q = self.q_home + a * (target_q - self.q_home)      # unsure: go only part of the way
        self.ref = self.planner.step(self.ref, target_q)
        q_ref, qd_ref, qdd_ref = self.ref

        # computed torque, clipped to the joint torque limits
        q, qd = data.qpos[ids.arm_qpos], data.qvel[ids.arm_dof]
        mujoco.mj_fullM(m, data, self.mfull)
        mass = self.mfull[np.ix_(ids.arm_dof, ids.arm_dof)]
        tau = mass @ (qdd_ref + cfg.kp * (q_ref - q) + cfg.kd * (qd_ref - qd)) + data.qfrc_bias[ids.arm_dof]
        lo, hi = m.actuator_ctrlrange[ids.arm_act].T
        self.saturated = bool(np.any((tau < lo) | (tau > hi)))
        data.ctrl[ids.arm_act] = np.clip(tau, lo, hi)

        # hand closure on time to contact
        if self.target is not None and not self.closing:
            if self.target["t"] - t <= cfg.close_lead or self.contact_t is not None:
                self.closing = True
        data.ctrl[ids.hand_act] = self.closed if self.closing else self.cup
