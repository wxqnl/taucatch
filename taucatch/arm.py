"""Arm kinematics and online joint trajectories (Ruckig) within the FR3/Panda joint limits."""
import mujoco
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig, RuckigError, Trajectory

from .scene import ARM_JOINTS, ARM_PROFILES, ARM_QD_MAX

# Panda joint limits (libfranka "Robot and interface specifications").
ARM_QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
ARM_QDDD_MAX = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])


def rot_error(r_des, r_cur):
    return 0.5 * sum(np.cross(r_cur[:, i], r_des[:, i]) for i in range(3))


class ArmIK:
    """Damped least-squares IK for the palm site on a private MjData (does not touch the sim)."""

    def __init__(self, model, ids, q_rest):
        self.m, self.ids = model, ids
        self.d = mujoco.MjData(model)
        self.jr = model.jnt_range[[model.joint(n).id for n in ARM_JOINTS]]
        self.q_rest = q_rest
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def _fk(self, q):
        self.d.qpos[self.ids.arm_qpos] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.ids.palm_site)
        return (self.d.site_xpos[self.ids.palm_site].copy(), self.d.site_xmat[self.ids.palm_site].reshape(3, 3).copy(),
                np.vstack([self.jacp[:, self.ids.arm_dof], self.jacr[:, self.ids.arm_dof]]))

    def solve(self, q0, pos, rot, iters=10):
        q = q0.copy()
        for _ in range(iters):
            p, r, jac = self._fk(q)
            err = np.concatenate([pos - p, rot_error(rot, r)])
            if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < 2e-3:
                break
            jpinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-4 * np.eye(6))
            dq = jpinv @ err + (np.eye(7) - jpinv @ jac) @ (0.05 * (self.q_rest - q))
            q = np.clip(q + dq, self.jr[:, 0] + 1e-3, self.jr[:, 1] - 1e-3)
        p, r, jac = self._fk(q)
        return q, np.linalg.norm(pos - p), np.linalg.norm(rot_error(rot, r)), jac

    def solve_normal(self, q0, pos, normal, iters=10):
        """IK for the palm position and palm normal only; the roll about the normal is left free
        (it does not matter for the catch), which shortens the reach. The palm normal is -y of the site."""
        q = q0.copy()
        n_des = normal / np.linalg.norm(normal)
        for _ in range(iters):
            p, r, jac = self._fk(q)
            n_cur = -r[:, 1]
            e_rot = np.cross(n_cur, n_des)
            if np.linalg.norm(pos - p) < 2e-4 and np.linalg.norm(e_rot) < 2e-3:
                break
            proj = np.eye(3) - np.outer(n_cur, n_cur)
            j = np.vstack([jac[:3], proj @ jac[3:]])
            jpinv = j.T @ np.linalg.inv(j @ j.T + 1e-4 * np.eye(6))
            dq = jpinv @ np.concatenate([pos - p, e_rot]) + (np.eye(7) - jpinv @ j) @ (0.05 * (self.q_rest - q))
            q = np.clip(q + dq, self.jr[:, 0] + 1e-3, self.jr[:, 1] - 1e-3)
        p, r, jac = self._fk(q)
        return q, np.linalg.norm(pos - p), np.linalg.norm(np.cross(-r[:, 1], n_des)), jac


class JointPlanner:
    """Time-optimal, jerk-limited joint trajectories from the current reference state."""

    def __init__(self, dt, arm="fast"):
        prof = ARM_PROFILES[arm]
        self.otg = Ruckig(7, dt)
        self.inp = InputParameter(7)
        self.out = OutputParameter(7)
        self.traj = Trajectory(7)
        self.qd_max = prof["vel"] * ARM_QD_MAX
        self.inp.max_velocity = self.qd_max.tolist()
        self.inp.max_acceleration = (prof["acc"] * ARM_QDD_MAX).tolist()
        self.inp.max_jerk = (prof["acc"] * ARM_QDDD_MAX).tolist()

    def _set(self, state, target_q, target_qd):
        q, qd, qdd = state
        self.inp.current_position, self.inp.current_velocity, self.inp.current_acceleration = q.tolist(), qd.tolist(), qdd.tolist()
        self.inp.target_position = target_q.tolist()
        self.inp.target_velocity = np.clip(target_qd, -0.95 * self.qd_max, 0.95 * self.qd_max).tolist()
        self.inp.target_acceleration = [0.0] * 7

    def duration(self, state, target_q, target_qd=np.zeros(7)):
        self._set(state, target_q, target_qd)
        try:
            res = self.otg.calculate(self.inp, self.traj)
        except RuckigError:
            return np.inf
        return self.traj.duration if res in (Result.Working, Result.Finished) else np.inf

    def step(self, state, target_q, target_qd=np.zeros(7), min_duration=None):
        self._set(state, target_q, target_qd)
        self.inp.minimum_duration = min_duration
        try:
            res = self.otg.update(self.inp, self.out)
        except RuckigError:
            res = None
        if res not in (Result.Working, Result.Finished):
            # infeasible input (e.g. the state left the limits): hold position, brake the reference
            return state[0], 0.9 * state[1], np.zeros(7)
        return np.array(self.out.new_position), np.array(self.out.new_velocity), np.array(self.out.new_acceleration)
