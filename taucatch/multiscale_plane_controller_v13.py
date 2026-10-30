"""Candidate V13: separate transverse interception and arrival-time uncertainty.

Independent of frozen V9-V12. Retains V12 measured-state compensation, bounds,
native pose nominal and motion lead. Covariance linearization is uncalibrated.
"""
from dataclasses import dataclass
import numpy as np
from .multiscale_v9 import FastConfig, limit_norm
from .multiscale_plane_geometry_v13 import crossing_covariance
from .multiscale_bootstrap_v10 import ContinuousReferenceController
from .multiscale_body_feedback_v12 import BodyFeedbackController


@dataclass(frozen=True)
class PlaneConfig(FastConfig):
    max_closing_sigma_s: float = .008


def interception(forecast, t, cfg):
    forecast.validate()
    if forecast.available_t > t + 1e-9:return None, 'not_delivered'
    if forecast.hits < cfg.min_hits or forecast.observed_t-forecast.born_t < cfg.min_track_age:
        return None, 'warming_track'
    if t-forecast.measured_t > cfg.max_measurement_age:return None, 'stale_measurement'
    p=forecast.positions
    crosses=np.flatnonzero((p[:-1,0]>=cfg.plane_x)&(p[1:,0]<cfg.plane_x))
    if not len(crosses):return None, 'no_crossing'
    k=int(crosses[0]);alpha=(p[k,0]-cfg.plane_x)/(p[k,0]-p[k+1,0])
    h=(1-alpha)*forecast.offsets[k]+alpha*forecast.offsets[k+1]
    at=forecast.observed_t+h
    if not -cfg.period <= at-t <= .35:return None, 'crossing_outside_horizon'
    point=(1-alpha)*p[k]+alpha*p[k+1]
    if not cfg.region_y[0]<point[1]<cfg.region_y[1] or not cfg.region_z[0]<point[2]<cfg.region_z[1]:
        return None, 'outside_workspace'
    velocity=(p[k+1]-p[k])/(forecast.offsets[k+1]-forecast.offsets[k])
    cov=forecast.covariance
    endpoint=cov[:3,:3]+h*(cov[:3,3:]+cov[3:,:3])+h*h*cov[3:,3:]
    geometry=crossing_covariance(cov,float(h),velocity)
    sigma=geometry['sigma_plane_m']
    confidence=float(np.clip((cfg.sigma_hi-sigma)/(cfg.sigma_hi-cfg.sigma_lo),0.,1.))
    if confidence<=0:return None, 'uncertain'
    return dict(t=at,point=point,velocity=velocity,sigma=sigma,confidence=confidence,
        sigma_xyz=float(np.sqrt(max(0.,np.trace(endpoint)/3))),sigma_time=geometry['sigma_time_s'],
        estimate_age=t-forecast.observed_t,measurement_age=t-forecast.measured_t), 'valid'


class PlaneResidualController:
    """Receding-horizon, acceleration/jerk-bounded task-space correction.

    Hybrid mode preserves the native palm normal and warps the native closure
    curve within a bounded time interval. It never uses simulator contact_t.
    """
    def __init__(self, spec, cfg=FastConfig(), mode='hybrid'):
        if mode not in ('hybrid', 'fast'):
            raise ValueError(mode)
        self.spec, self.cfg, self.mode = spec, cfg, mode
        self.delta = np.zeros(3); self.velocity = np.zeros(3); self.acceleration = np.zeros(3)
        self.last_t = None; self.home = None; self.holding = False; self.hold_action = None

    def update(self, t, forecast, buffer, measured_action):
        cfg = self.cfg
        dt = cfg.period if self.last_t is None else t-self.last_t
        if dt <= 0 or dt > .1:
            raise ValueError('Invalid feedback update interval')
        self.last_t = t
        if self.home is None:
            self.home = np.asarray(measured_action, np.float32).copy()
        if self.holding:
            return self.hold_action.copy(), dict(reason='holding', delta_norm=float(np.linalg.norm(self.delta)))
        native = buffer.sample(t) if self.mode == 'hybrid' else self.home
        if native is None:
            return None, dict(reason='waiting_native')
        if self.mode == 'hybrid' and t > buffer.anchor+self.spec.horizon*self.spec.dt+1e-9:
            return None, dict(reason='expired_native')
        target, reason = (None, 'no_estimate') if forecast is None else interception(forecast, t, cfg)
        p, normal, closure = self.spec.decode(native)
        if target is None:
            # Smoothly return toward the native reference after losing visual confidence.
            self.delta *= np.exp(-dt/.02); self.velocity *= np.exp(-dt/.02)
            if self.mode == 'fast':
                return None, dict(reason=reason)
            point = np.clip(p+self.delta, np.asarray(self.spec.center)-self.spec.scale,
                            np.asarray(self.spec.center)+self.spec.scale)
            action = self.spec.encode(point, normal, closure)
            return action, dict(reason=reason, delta_norm=float(np.linalg.norm(self.delta)))
        remaining = target['t']-t
        # Advance motion alone by the TRAIN-fitted actuator lag; do not move
        # perception timestamps or the physical closure/contact estimate.
        motion_remaining = max(cfg.motion_horizon_floor, remaining-cfg.actuation_lead)
        reference_at_contact = buffer.sample(target['t']) if self.mode == 'hybrid' else self.home
        reference_point, _, _ = self.spec.decode(reference_at_contact)
        desired = limit_norm(target['point']-reference_point, cfg.max_correction)
        acceleration = target['confidence']*(6*(desired-self.delta)/motion_remaining**2-4*self.velocity/motion_remaining)
        acceleration = limit_norm(acceleration, cfg.max_acceleration)
        self.acceleration += limit_norm(acceleration-self.acceleration, cfg.max_jerk*dt)
        self.acceleration = limit_norm(self.acceleration, cfg.max_acceleration)
        self.velocity = limit_norm(self.velocity+self.acceleration*dt, cfg.max_velocity)
        self.delta = limit_norm(self.delta+self.velocity*dt, cfg.max_correction)
        shift = 0.;closure_reflex=False
        if self.mode == 'hybrid':
            closed = np.flatnonzero(buffer.actions[:, 6] >= 0)
            if len(closed):
                native_close = buffer.anchor+int(closed[0])*self.spec.dt
                shift = float(np.clip(native_close-(target['t']-cfg.close_lead),
                                      -cfg.max_time_shift, cfg.max_time_shift))
                warped = buffer.sample(t+shift)
                if warped is not None:
                    _, _, closure = self.spec.decode(warped)
        else:
            normal = -target['velocity']/np.linalg.norm(target['velocity'])
            closure = float(remaining <= cfg.close_lead+cfg.period)
        # A native all-open chunk has no closing knot to time-warp. The fast
        # loop must still issue an imminent-contact grip from delivered DVS.
        if self.mode=='hybrid' and cfg.allow_reflex_closure and remaining<=cfg.close_lead+cfg.period and closure<.5:
            closure=1.;closure_reflex=True
        # The fast visual clock gates grip even if a native ramp starts early.
        # Do not freeze the reach merely because a native grip knot crossed .5.
        closure_early_blocked = bool(closure > 0 and remaining > cfg.close_lead+cfg.period)
        closure_uncertain_blocked = bool(closure > 0 and target['sigma_time'] > cfg.max_closing_sigma_s)
        if closure_uncertain_blocked or closure_early_blocked:
            closure = 0.
        point = np.clip(p+self.delta, np.asarray(self.spec.center)-self.spec.scale,
                        np.asarray(self.spec.center)+self.spec.scale)
        action = self.spec.encode(point, normal, closure)
        # Hold is triggered only by the issued closure and estimated contact window.
        if closure >= .5 and remaining <= max(.012, cfg.close_lead+cfg.period):
            self.holding = True; self.hold_action = action.copy(); self.hold_action[6] = 1.
        return action, dict(reason='correcting', intercept_t=target['t'], intercept_point=target['point'].tolist(),
            confidence=target['confidence'], sigma=target['sigma'], sigma_xyz=target['sigma_xyz'],
            sigma_time=target['sigma_time'], closure_uncertain_blocked=closure_uncertain_blocked,
            closure_early_blocked=closure_early_blocked, delta_norm=float(np.linalg.norm(self.delta)),
            delta=self.delta.tolist(), time_shift=shift, closure_reflex=closure_reflex, motion_remaining=motion_remaining,
            actuation_lead=cfg.actuation_lead, estimate_age=target['estimate_age'],
            measurement_age=target['measurement_age'], reference_tail_used=bool(self.mode=='hybrid' and
                target['t']>buffer.anchor+self.spec.horizon*self.spec.dt), holding=self.holding)


class PlaneReferenceController(ContinuousReferenceController):
    def __init__(self,spec,cfg=PlaneConfig(),mode='hybrid'):
        super().__init__(spec,cfg,mode)
        self.fast=PlaneResidualController(spec,cfg,'fast')
        self.hybrid=PlaneResidualController(spec,cfg,'hybrid')


class PlaneBodyController(BodyFeedbackController):
    def __init__(self,spec,cfg=PlaneConfig(),mode='hybrid'):
        super().__init__(spec,cfg,mode)
        self.nominal=PlaneReferenceController(spec,cfg,mode)
