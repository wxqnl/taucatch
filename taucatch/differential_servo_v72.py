"""Bound-constrained local IK around the existing torque executor.

Independent mechanism candidate. No ball state, policy, or physics oracle input.
The solve is limited at the joint step itself, before task-space direction is
lost through independent clipping of a distant nonlinear IK solution.
"""
from dataclasses import asdict, dataclass
import time

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .action_catcher import ActionCatcher


@dataclass(frozen=True)
class ServoSpec:
    name: str = 'bounded_local_position_normal_IK_v72'
    damping: float = .01
    solver_tol: float = 1e-8
    max_iterations: int = 30

    def contract(self):
        return asdict(self)


SERVO_SPEC = ServoSpec()


def local_step(ik, q, position, normal, max_step, spec=SERVO_SPEC):
    q = np.asarray(q, float)
    position = np.asarray(position, float)
    normal = np.asarray(normal, float)
    max_step = np.asarray(max_step, float)
    if q.shape != (7,) or position.shape != (3,) or normal.shape != (3,) or max_step.shape != (7,):
        raise ValueError('Invalid local IK dimensions')
    if not np.isfinite(np.r_[q, position, normal, max_step]).all() or np.any(max_step <= 0) or np.linalg.norm(normal) < 1e-6:
        raise ValueError('Invalid local IK values')
    normal = normal / np.linalg.norm(normal)
    p, rotation, jac = ik._fk(q)
    current_normal = -rotation[:, 1]
    if current_normal @ normal < 0:
        raise ValueError('Opposite normal is outside this local IK contract')
    projector = np.eye(3) - np.outer(current_normal, current_normal)
    j = np.vstack([jac[:3], projector @ jac[3:]])
    error = np.r_[position - p, np.cross(current_normal, normal)]
    lower = np.maximum(-max_step, ik.jr[:, 0] + .001 - q)
    upper = np.minimum(max_step, ik.jr[:, 1] - .001 - q)
    if np.any(lower >= upper):
        raise ValueError('Reference outside the supported joint limits')
    if np.max(np.abs(error)) < 1e-12:
        delta = np.zeros(7)
    else:
        result = lsq_linear(np.vstack([j, spec.damping * np.eye(7)]), np.r_[error, np.zeros(7)],
                            bounds=(lower, upper), method='bvls', tol=spec.solver_tol, max_iter=spec.max_iterations)
        if not result.success:
            raise RuntimeError('Bounded IK did not converge')
        delta = result.x
    target = q + delta
    p1, r1, _ = ik._fk(target)
    after = np.r_[position - p1, np.cross(-r1[:, 1], normal)]
    return target, dict(position_error_before=float(np.linalg.norm(position-p)),
                        position_error_after=float(np.linalg.norm(position-p1)),
                        task_error_before=float(np.linalg.norm(error)), task_error_after=float(np.linalg.norm(after)),
                        max_velocity_fraction=float(np.max(np.abs(delta)/max_step)))


class DifferentialServoMixin:
    """Retain the configured command buffer, torque law and physical limits."""
    def reset(self, data):
        super().reset(data)
        self.servo_evidence = []

    def step(self, data, t, unused_ball_state=None):
        ids, m, cfg = self.ids, self.m, self.cfg
        action = self.buffer.sample(t+self.dt)
        closure = float(self.closing)
        q0, v0, _ = self.ref
        detail = None
        if action is not None:
            p, normal, closure = self.spec.decode(action, self.normal)
            start = time.perf_counter()
            qref, detail = local_step(self.ik, q0, p, normal, self.planner.qd_max*self.dt)
            detail['solver_ms'] = 1000*(time.perf_counter()-start)
            self.normal = normal
            if self.first_cmd_t is None:
                self.first_cmd_t = t
        else:
            qref = q0.copy()
        vref = (qref-q0)/self.dt
        acceleration = np.clip((vref-v0)/self.dt, -self.acc_max, self.acc_max)
        self.ref = qref, vref, acceleration
        mujoco.mj_fullM(m, data, self.mfull)
        mass = self.mfull[np.ix_(ids.arm_dof, ids.arm_dof)]
        q, qd = data.qpos[ids.arm_qpos], data.qvel[ids.arm_dof]
        tau = mass@(acceleration+cfg.kp*(qref-q)+cfg.kd*(vref-qd))+data.qfrc_bias[ids.arm_dof]
        lo, hi = m.actuator_ctrlrange[ids.arm_act].T
        self.saturated = bool(np.any((tau < lo) | (tau > hi)))
        data.ctrl[ids.arm_act] = np.clip(tau, lo, hi)
        data.ctrl[ids.hand_act] = self.cup+closure*(self.closed-self.cup)
        self.closing = closure >= .5
        if detail is not None:
            self.servo_evidence.append(dict(t=t, **detail, torque_saturated=self.saturated,
                q0=q0.tolist(), qref=qref.tolist(), command=p.tolist(), normal=normal.tolist(),
                closure=closure, hand_mapping_error=float(np.max(np.abs(data.ctrl[ids.hand_act]-(self.cup+closure*(self.closed-self.cup))))),
                arm_ctrl=data.ctrl[ids.arm_act].tolist()))


class DifferentialActionCatcher(DifferentialServoMixin, ActionCatcher):
    pass
