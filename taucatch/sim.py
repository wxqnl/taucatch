"""Catching episodes: physics loop with ball aerodynamics, contacts and outcome."""
import mujoco
import numpy as np

from .ball import BALLS, aero_accel
from .arm import rot_error
from .catcher import Catcher, CatcherConfig, HAND_CUP, palm_rotation
from .scene import ARM_HOME, BASE_Z, PARKED, WINDOW_X, build_model
from .throws import ThrowerMotion, launcher_parts

CONTROL_DT = 0.001
HOME_POS = np.array([WINDOW_X, 0.0, BASE_Z + 0.5])


class CatchSim:
    """`thrower`: "human" (animated swing, the ball rides in the hand until t = 0), "launcher"
    (static barrel) or "none". Throws are released at t = 0 in every mode."""

    def __init__(self, ball="pingpong", cfg=CatcherConfig(), arm="fast", thrower="none", balls=None):
        self.ball = BALLS[ball]
        self.arm = arm
        self.thrower = thrower
        self.model, self.ids = build_model(ball, arm, balls)
        self.data = mujoco.MjData(self.model)
        self.substeps = round(CONTROL_DT / self.model.opt.timestep)
        self.home_rot = palm_rotation(np.array([1.0, 0.0, 0.0]))
        self.home_q = self._solve_home()
        self.catcher = Catcher(self.model, self.ids, self.ball, self.home_q, cfg, arm)
        self.hand_geoms = set(self.ids.hand_geoms.tolist())

    def _solve_home(self):
        m, d, ids = self.model, self.data, self.ids
        d.qpos[ids.arm_qpos] = ARM_HOME
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(300):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, ids.palm_site)
            err = np.concatenate([HOME_POS - d.site_xpos[ids.palm_site],
                                  rot_error(self.home_rot, d.site_xmat[ids.palm_site].reshape(3, 3))])
            if np.linalg.norm(err) < 1e-5:
                break
            jac = np.vstack([jacp[:, ids.arm_dof], jacr[:, ids.arm_dof]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            q = d.qpos[ids.arm_qpos] + 0.5 * dq
            lo, hi = m.jnt_range[[m.joint(f"fr3_joint{i}").id for i in range(1, 8)]].T
            d.qpos[ids.arm_qpos] = np.clip(q, lo + 1e-3, hi - 1e-3)
        return d.qpos[ids.arm_qpos].copy()

    # ---------- state ----------
    def ball_state(self):
        """Ball position, velocity and spin (world frame) read from the free joint."""
        d, ids = self.data, self.ids
        b, v = ids.ball_qpos, ids.ball_dof
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, d.qpos[b + 3:b + 7])
        w_world = rot.reshape(3, 3) @ d.qvel[v + 3:v + 6]      # free joint angular velocity is local
        return d.qpos[b:b + 3].copy(), d.qvel[v:v + 3].copy(), w_world

    def reset(self, throw):
        m, d, ids = self.model, self.data, self.ids
        mujoco.mj_resetData(m, d)
        d.qpos[ids.arm_qpos] = self.home_q
        d.qpos[ids.hand_qpos] = HAND_CUP
        b = ids.ball_qpos
        d.qpos[b:b + 3] = throw.p0
        d.qpos[b + 3:b + 7] = [1, 0, 0, 0]
        v = ids.ball_dof
        d.qvel[v:v + 3] = throw.v0
        d.qvel[v + 3:v + 6] = throw.w0     # free joint angular velocity is local; identity orientation
        d.mocap_pos[:] = PARKED
        d.mocap_quat[:] = [1, 0, 0, 0]
        self.throw, self.motion = throw, None
        if self.thrower == "human":
            self.motion = ThrowerMotion(throw)
            self._place(self.motion.parts(self.motion.t_start))
            self._hold_ball(self.motion.t_start)
        elif self.thrower == "launcher":
            self._place(launcher_parts(throw))
        self.k_start = round(self.motion.t_start / CONTROL_DT) if self.motion else 0
        mujoco.mj_forward(m, d)
        self.catcher.reset(d)

    def _place(self, parts):
        for name, (pos, quat) in parts.items():
            k = self.ids.mocap[name]
            self.data.mocap_pos[k] = pos
            self.data.mocap_quat[k] = quat

    def _hold_ball(self, t):
        """Ball in the thrower's hand: kinematic, no spin, no air forces."""
        d, b, v = self.data, self.ids.ball_qpos, self.ids.ball_dof
        pos, vel = self.motion.hand(t)
        d.qpos[b:b + 3] = pos
        d.qpos[b + 3:b + 7] = [1, 0, 0, 0]
        d.qvel[v:v + 3] = vel
        d.qvel[v + 3:v + 6] = 0
        d.xfrc_applied[self.ids.ball_body] = 0

    def _release(self):
        d, b, v = self.data, self.ids.ball_qpos, self.ids.ball_dof
        d.qpos[b:b + 3] = self.throw.p0
        d.qpos[b + 3:b + 7] = [1, 0, 0, 0]
        d.qvel[v:v + 3] = self.throw.v0
        d.qvel[v + 3:v + 6] = self.throw.w0

    def _apply_aero(self):
        p, v, w = self.ball_state()
        ball, ids = self.ball, self.ids
        self.data.xfrc_applied[ids.ball_body, :3] = ball.mass * aero_accel(v, w, ball.params)
        self.data.xfrc_applied[ids.ball_body, 3:] = -ball.inertia * w / ball.spin_decay

    def _ball_hand_contact(self):
        g = self.ids.ball_geom
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 == g and c.geom2 in self.hand_geoms) or (c.geom2 == g and c.geom1 in self.hand_geoms):
                return True
        return False

    # ---------- episode ----------
    def run(self, throw, t_max=1.2, hold=0.5, frame_cb=None, state_fn=None):
        """Run one throw. `state_fn(sim, t)` gives the ball state the catcher sees (default: true state
        after release). With the human thrower the episode starts during the swing, at t < 0."""
        self.reset(throw)
        m, d, cat, ids = self.model, self.data, self.catcher, self.ids
        min_dist, touched = np.inf, False
        palm0, palm_contact = d.site_xpos[ids.palm_site].copy(), None
        k = self.k_start
        t = k * CONTROL_DT
        while t < t_max:
            if state_fn:
                state = state_fn(self, t)
            else:
                state = self.ball_state() if k >= 0 else None
            cat.step(d, t, state)
            for i in range(self.substeps):
                ts = (k + i / self.substeps) * CONTROL_DT
                if self.motion is not None:
                    self._place(self.motion.parts(ts))
                if self.motion is not None and k * self.substeps + i < 0:
                    self._hold_ball(ts)
                else:
                    if self.motion is not None and k * self.substeps + i == 0:
                        self._release()
                    self._apply_aero()
                mujoco.mj_step(m, d)
            k += 1
            t = k * CONTROL_DT
            if frame_cb:
                frame_cb(self, t)
            bp = d.xpos[ids.ball_body]
            min_dist = min(min_dist, np.linalg.norm(bp - d.site_xpos[ids.palm_site]))
            if not touched and self._ball_hand_contact():
                touched = True
                cat.contact_t = t
                palm_contact = d.site_xpos[ids.palm_site].copy()
            if bp[2] < 0.1 or (cat.contact_t is not None and t > cat.contact_t + hold):
                break
            if cat.contact_t is None and bp[0] < -0.5:
                break
        bp = d.xpos[ids.ball_body]
        held_dist = np.linalg.norm(bp - d.site_xpos[ids.palm_site])
        caught = touched and bp[2] > 0.3 and held_dist < self.ball.radius + 0.05 and t > (cat.contact_t or 0) + hold - 1e-6
        return dict(
            caught=bool(caught), touched=touched, min_dist=float(min_dist), held_dist=float(held_dist),
            contact_t=cat.contact_t, first_cmd_t=cat.first_cmd_t,
            target_t=None if cat.target is None else float(cat.target["t"]),
            speed=throw.speed, spin_rps=throw.spin_rps,
            reach=None if palm_contact is None else float(np.linalg.norm(palm_contact - palm0)),   # palm travel to contact
        )
