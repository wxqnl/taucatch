"""Causal event forecasts and bounded feedback around native VLA action chunks.

This module has no simulator/ball-state access. Its only inputs are timestamped
perception estimates, already delivered VLA chunks, and measured robot state.
"""
from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(frozen=True)
class FastConfig:
    period: float = .004
    min_hits: int = 4
    min_track_age: float = .006
    max_measurement_age: float = .025
    sigma_lo: float = .02
    sigma_hi: float = .12
    max_correction: float = .35
    max_velocity: float = 3.
    max_acceleration: float = 60.
    max_jerk: float = 6000.
    max_time_shift: float = .050
    allow_reflex_closure: bool = True
    actuation_lead: float = .032
    motion_horizon_floor: float = .020
    close_lead: float = .004
    plane_x: float = .70
    region_y: tuple = (-.45, .45)
    region_z: tuple = (.85, 1.65)

    def __post_init__(self):
        if not (0 < self.period and 0 <= self.sigma_lo < self.sigma_hi):
            raise ValueError('Invalid timing/uncertainty configuration')
        if min(self.max_correction, self.max_velocity, self.max_acceleration, self.max_jerk) <= 0:
            raise ValueError('Feedback bounds must be positive')


@dataclass(frozen=True)
class Forecast:
    track_id: int
    observed_t: float
    measured_t: float
    available_t: float
    born_t: float
    hits: int
    offsets: np.ndarray
    positions: np.ndarray
    covariance: np.ndarray

    def validate(self):
        if not self.measured_t <= self.observed_t + 1e-9 <= self.available_t + 2e-9:
            raise ValueError('Noncausal perception timestamps')
        if self.positions.shape != (len(self.offsets), 3) or self.covariance.shape != (6, 6):
            raise ValueError('Invalid perception arrays')
        if not np.all(np.diff(self.offsets) > 0):
            raise ValueError('Forecast time grid must increase')
        if not np.isfinite(self.positions).all() or not np.isfinite(self.covariance).all():
            raise ValueError('Nonfinite perception estimate')
        if np.linalg.eigvalsh((self.covariance+self.covariance.T)/2).min() < -1e-8:
            raise ValueError('Invalid perception covariance')


class DeliveryQueue:
    """Serialized compute cost with delayed, out-of-order-safe estimate delivery."""
    def __init__(self):
        self.heap = []
        self.serial = 0
        self.busy_until = -np.inf
        self.latest = None

    def submit(self, forecast, observed_t, cost_seconds):
        if cost_seconds < 0 or not np.isfinite(cost_seconds):
            raise ValueError('Invalid measured processing cost')
        ready = max(observed_t, self.busy_until) + cost_seconds
        self.busy_until = ready
        if forecast is not None:
            from dataclasses import replace
            forecast = replace(forecast, available_t=ready)
            forecast.validate()
        heapq.heappush(self.heap, (ready, self.serial, forecast))
        self.serial += 1
        return ready

    def deliver(self, t):
        while self.heap and self.heap[0][0] <= t + 1e-9:
            _, _, forecast = heapq.heappop(self.heap)
            # A completed no-detection update clears an older estimate as well.
            self.latest = forecast
        return self.latest


def tracker_forecast(tracker, observed_t):
    """Select one persistent visual track without target IDs or scoring truth."""
    tracks = [tr for tr in tracker.tracks if tr.outs and tr.last_seen <= observed_t + 1e-9]
    if not tracks:
        return None
    tracks.sort(key=lambda tr: (-tr.hits, float(np.trace(tr.ekf.P[:3, :3])), tr.id))
    tr = tracks[0]
    # With multiple strong tracks, do not invent a target binding in a single-ball task.
    if len(tracks) > 1 and tracks[1].hits >= max(4, .8*tr.hits):
        return None
    path = tr.outs[-1]
    positions = np.asarray(path['path'], float).copy()
    from .predictor import HORIZONS
    result = Forecast(tr.id, float(path['t']), float(tr.last_seen), float(path['t']),
                      float(tr.born), int(tr.hits), HORIZONS.copy(), positions,
                      np.asarray(tr.ekf.P[:6, :6], float).copy())
    result.validate()
    return result


def interception(forecast, t, cfg):
    forecast.validate()
    if forecast.available_t > t + 1e-9:
        return None, 'not_delivered'
    if forecast.hits < cfg.min_hits or forecast.observed_t-forecast.born_t < cfg.min_track_age:
        return None, 'warming_track'
    if t-forecast.measured_t > cfg.max_measurement_age:
        return None, 'stale_measurement'
    p = forecast.positions
    crosses = np.flatnonzero((p[:-1, 0] >= cfg.plane_x) & (p[1:, 0] < cfg.plane_x))
    if not len(crosses):
        return None, 'no_crossing'
    k = int(crosses[0]); alpha = (p[k, 0]-cfg.plane_x)/(p[k, 0]-p[k+1, 0])
    h = (1-alpha)*forecast.offsets[k]+alpha*forecast.offsets[k+1]
    at = forecast.observed_t+h
    if not -cfg.period <= at-t <= .35:
        return None, 'crossing_outside_horizon'
    point = (1-alpha)*p[k]+alpha*p[k+1]
    if not cfg.region_y[0] < point[1] < cfg.region_y[1] or not cfg.region_z[0] < point[2] < cfg.region_z[1]:
        return None, 'outside_workspace'
    velocity = (p[k+1]-p[k])/(forecast.offsets[k+1]-forecast.offsets[k])
    cov = forecast.covariance
    endpoint_cov = cov[:3, :3]+h*(cov[:3, 3:]+cov[3:, :3])+h*h*cov[3:, 3:]
    sigma = float(np.sqrt(max(0., np.trace(endpoint_cov)/3)))
    confidence = float(np.clip((cfg.sigma_hi-sigma)/(cfg.sigma_hi-cfg.sigma_lo), 0., 1.))
    if confidence <= 0:
        return None, 'uncertain'
    return dict(t=at, point=point, velocity=velocity, sigma=sigma, confidence=confidence,
                estimate_age=t-forecast.observed_t, measurement_age=t-forecast.measured_t), 'valid'


def limit_norm(value, bound):
    return value*min(1., bound/max(float(np.linalg.norm(value)), 1e-12))


class ResidualController:
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
        point = np.clip(p+self.delta, np.asarray(self.spec.center)-self.spec.scale,
                        np.asarray(self.spec.center)+self.spec.scale)
        action = self.spec.encode(point, normal, closure)
        # Hold is triggered only by the issued closure and estimated contact window.
        if closure >= .5 and remaining <= max(.012, cfg.close_lead+cfg.period):
            self.holding = True; self.hold_action = action.copy(); self.hold_action[6] = 1.
        return action, dict(reason='correcting', intercept_t=target['t'], intercept_point=target['point'].tolist(),
            confidence=target['confidence'], sigma=target['sigma'], delta_norm=float(np.linalg.norm(self.delta)),
            delta=self.delta.tolist(), time_shift=shift, closure_reflex=closure_reflex, motion_remaining=motion_remaining,
            actuation_lead=cfg.actuation_lead, estimate_age=target['estimate_age'],
            measurement_age=target['measurement_age'], reference_tail_used=bool(self.mode=='hybrid' and
                target['t']>buffer.anchor+self.spec.horizon*self.spec.dt), holding=self.holding)


class CommandBuffer:
    """Add delayed feedback commands without changing native chunk provenance."""
    def __init__(self, native, mode):
        self.native, self.mode = native, mode
        self.command = None; self.command_available = np.inf; self.command_expiry = -np.inf

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset(self):
        self.native.reset(); self.command = None
        self.command_available = np.inf; self.command_expiry = -np.inf

    def sample(self, t):
        if self.command is not None and self.command_available <= t+1e-9 <= self.command_expiry+2e-9:
            return self.command
        return self.native.sample(t) if self.mode == 'hybrid' else None

    def set_command(self, action, available, expiry):
        if action is not None and (np.asarray(action).shape != (7,) or not np.isfinite(action).all()):
            raise ValueError('Invalid corrected action')
        if expiry < available:
            raise ValueError('Invalid command lifetime')
        self.command = None if action is None else np.asarray(action, np.float32).copy()
        self.command_available, self.command_expiry = available, expiry
