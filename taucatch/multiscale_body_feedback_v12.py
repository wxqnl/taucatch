"""Bounded measured-palm tracking compensation around the frozen V10 nominal loop.

Only causal DVS forecasts, delivered native chunks and robot proprioception enter
this controller. Physical IK/torque limits stay in the existing 1ms inner servo.
This is a tracking-compensation hypothesis, not an inverse-dynamics/reachability
solver. Nominal acceleration/jerk bounds do not bound total corrected-command jerk.
"""
from dataclasses import dataclass
import numpy as np
from .multiscale_bootstrap_v10 import ContinuousReferenceController
from .multiscale_v9 import FastConfig, limit_norm


@dataclass(frozen=True)
class BodyConfig:
    position_gain: float = .5
    velocity_gain_s: float = .008
    velocity_filter_s: float = .008
    max_compensation_m: float = .04
    max_compensation_rate: float = .75


class BodyFeedbackController:
    def __init__(self, spec, cfg=FastConfig(), mode='hybrid', body=BodyConfig()):
        self.spec, self.cfg, self.mode, self.body = spec, cfg, mode, body
        self.nominal = ContinuousReferenceController(spec, cfg, mode)
        self.last_t = None
        self.last_measured = None
        self.last_nominal = None
        self.last_action = None
        self.measured_velocity = np.zeros(3)
        self.compensation = np.zeros(3)
        self.hold_action = None

    def update(self, t, forecast, buffer, measured_action):
        dt = self.cfg.period if self.last_t is None else t-self.last_t
        if not 0 < dt <= .1:
            raise ValueError('Invalid body feedback interval')
        measured, _, _ = self.spec.decode(measured_action)
        if self.last_measured is not None:
            raw_velocity = (measured-self.last_measured)/dt
            alpha = 1-np.exp(-dt/self.body.velocity_filter_s)
            self.measured_velocity += alpha*(raw_velocity-self.measured_velocity)
        self.last_t, self.last_measured = t, measured.copy()
        if self.hold_action is not None:
            return self.hold_action.copy(), dict(reason='holding', phase='body_hold',
                native_reference_used=self.nominal.started, body_feedback_active=False,
                compensation_m=self.compensation.tolist())
        action, detail = self.nominal.update(t, forecast, buffer, measured_action)
        if action is None:
            self.compensation *= np.exp(-dt/.02)
            return None, dict(detail, body_feedback_active=False)
        point, normal, closure = self.spec.decode(action)
        nominal_velocity = np.zeros(3) if self.last_nominal is None else limit_norm(
            (point-self.last_nominal)/dt, self.cfg.max_velocity)
        confidence = detail.get('confidence', 0.) if detail['reason']=='correcting' else 0.
        error = point-measured
        desired = confidence*(self.body.position_gain*error +
            self.body.velocity_gain_s*(nominal_velocity-self.measured_velocity))
        desired = limit_norm(desired, self.body.max_compensation_m)
        self.compensation += limit_norm(desired-self.compensation,
            self.body.max_compensation_rate*dt)
        self.compensation = limit_norm(self.compensation, self.body.max_compensation_m)
        corrected = np.clip(point+self.compensation,
            np.asarray(self.spec.center)-self.spec.scale, np.asarray(self.spec.center)+self.spec.scale)
        # Bound every command transition, including a newly delivered native chunk.
        previous = measured if self.last_action is None else self.spec.decode(self.last_action)[0]
        proposed_step = corrected-previous
        corrected = previous+limit_norm(proposed_step, self.cfg.max_velocity*dt)
        result = self.spec.encode(corrected, normal, closure)
        self.last_nominal, self.last_action = point.copy(), result.copy()
        holding = self.nominal.hybrid.holding if self.nominal.started else self.nominal.fast.holding
        if holding:
            self.hold_action = result.copy()
            self.hold_action[6] = 1.
        return result, dict(detail, body_feedback_active=confidence>0,
            nominal_point=point.tolist(), measured_point=measured.tolist(),
            measured_velocity=self.measured_velocity.tolist(), nominal_velocity=nominal_velocity.tolist(),
            tracking_error_m=error.tolist(), tracking_error_norm_m=float(np.linalg.norm(error)),
            compensation_m=self.compensation.tolist(), compensation_norm_m=float(np.linalg.norm(self.compensation)),
            corrected_command_step_m=float(np.linalg.norm(corrected-previous)),
            command_transition_limited=bool(np.linalg.norm(proposed_step)>self.cfg.max_velocity*dt+1e-9))
