"""Explicit continuous close-time chunks; never reinterpret V26 grip amplitude.

Positive values advance the delivered fast deadline, negative values delay it.
This CPU interface is not a trained/deployed policy. The caller MUST include the
full measured inference/validation/transport cost in available_t and charge the
timer/reference composition lane before any physical action is delivered.
"""
from dataclasses import asdict, dataclass
import heapq

import numpy as np

from .multiscale_grip_timer_v14 import GripTimer


@dataclass(frozen=True)
class TimingSpec:
    name: str = 'native_close_deadline_advance_seconds_v44'
    horizon: int = 12
    dt: float = .016
    max_shift_s: float = .012
    positive: str = 'advance_current_delivered_fast_deadline'
    expiry: str = 'observation_plus_horizon; zero_fallback_no_reopen'
    late_window: str = 'clamp_delay_to_intercept_plus_period_minus_1ms_servo_tick'

    def contract(self):
        return asdict(self)


TIMING_SPEC = TimingSpec()
CONTEXT_CONTRACT = 'delivered_timer_deadline_intercept_age_sigma_closed_offset7_v44'
CONTEXT_DIM = 7


def require_contract(contract):
    if contract != TIMING_SPEC.contract():
        raise ValueError('Expected explicit V44 timing contract; old grip actions are incompatible')


class TimingChunks:
    """Only already-arrived 12x1 chunks, anchored to causal observation time.

No constant-last extrapolation after 192ms. Newer observations supersede older
ones only on arrival. Repeated reads at the SAME timestamp are supported.
"""
    def __init__(self):
        self.pending = []
        self.seen = set()
        self.serial = 0
        self.anchor = -np.inf
        self.actions = None
        self.last_t = -np.inf
        self.accepted = self.stale = 0

    def enqueue(self, actions, observed_t, available_t, contract):
        require_contract(contract)
        a = np.asarray(actions, np.float32)
        if a.shape != (12, 1) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1:
            raise ValueError('Expected finite normalized 12x1 close-time actions')
        if not np.isfinite([observed_t, available_t]).all() or available_t < observed_t:
            raise ValueError('Noncausal timing reply')
        if available_t < self.last_t:
            raise ValueError('Cannot backdate a newly inserted reply')
        if observed_t in self.seen or observed_t <= self.anchor or available_t >= observed_t + .192:
            self.stale += 1
            return False
        self.seen.add(observed_t)
        heapq.heappush(self.pending, (available_t, self.serial, observed_t, a.copy()))
        self.serial += 1
        return True

    def sample(self, t):
        if not np.isfinite(t) or t < self.last_t:
            raise ValueError('Timing clock must be finite and nondecreasing')
        self.last_t = t
        while self.pending and self.pending[0][0] <= t:
            _, _, observed, a = heapq.heappop(self.pending)
            if observed <= self.anchor or t >= observed + .192:
                self.stale += 1
                continue
            self.anchor = observed
            self.actions = np.r_[np.zeros((1, 1), np.float32), a]
            self.accepted += 1
        if self.actions is None or t >= self.anchor + .192:
            return 0.
        x = np.clip((t - self.anchor) / .016, 0, 12)
        lo = min(int(np.floor(x)), 12)
        hi = min(lo + 1, 12)
        return float(self.actions[lo, 0] + (x-lo)*(self.actions[hi, 0]-self.actions[lo, 0]))


class DeadlineGripTimer(GripTimer):
    """Original delivered forecast/age gates, plus bounded slow timing authority.

Timing cannot invent a target, bypass stale forecasts, or reopen a closed hand.
All state changes occur at advance(t), never retroactively at the shifted time.
Motion remains under the fast controller; this new policy controls timing only.
"""
    def reset(self):
        super().reset()
        self.timing = TimingChunks()
        self.last_advance_t = -np.inf
        self.target_available_t = None
        self.applied_shift_s = 0.
        self.last_detail = None

    def advance(self, t, timing_sample_t=None):
        if not np.isfinite(t) or t < self.last_advance_t:
            raise ValueError('Timer clock must be finite and nondecreasing')
        self.last_advance_t = t
        while self.pending and self.pending[0][0] <= t + 1e-9:
            self.target_available_t, self.target = self.pending.popleft()
        causal_t = t if timing_sample_t is None else timing_sample_t
        if not np.isfinite(causal_t) or causal_t > t:
            raise ValueError('Timing replies must be gated at physical time, not servo lookahead')
        normalized = self.timing.sample(causal_t)
        requested_shift = normalized * TIMING_SPEC.max_shift_s
        r = self.target
        deadline = None
        eligible = False
        self.applied_shift_s = 0.
        if r is not None:
            # The original uncertainty gate is applied in propose(); repeat it
            # here so synthetic/malformed submissions cannot bypass it.
            valid = (r['sigma_time'] <= self.cfg.max_closing_sigma_s
                     and 0 <= t-r['measured_t'] <= self.cfg.max_measurement_age
                     and r['forecast_available_t'] <= t + 1e-9)
            # Leave one servo tick before the original latest-close boundary;
            # clamping exactly to that floating boundary may leave no grid tick.
            deadline = min(r['deadline'] - requested_shift, r['intercept_t'] + self.cfg.period - .001)
            self.applied_shift_s = r['deadline'] - deadline
            eligible = valid and deadline <= t+1e-9 <= r['intercept_t']+self.cfg.period+1e-9
            if not self.closed and eligible:
                self.closed = True
                self.closed_t = t
        self.last_detail = dict(t=t, timing_sample_t=causal_t, target_available_t=self.target_available_t,
            timing_anchor=None if not np.isfinite(self.timing.anchor) else self.timing.anchor,
            normalized=normalized, requested_shift_s=requested_shift,
            effective_shift_s=self.applied_shift_s, effective_deadline=deadline,
            eligible=bool(eligible), closed=self.closed, closed_t=self.closed_t)
        return self.closed

    def context(self, t):
        """Read-only snapshot; no advance/sample and no scoring or future target.

Forecast deadline/intercept are predictions already delivered to this timer.
Normalization scales are fixed ABI choices, not physical rate/force limits.
"""
        if not np.isfinite(t) or t < self.last_advance_t:
            raise ValueError('Invalid context clock')
        if self.closed and (self.closed_t is None or self.closed_t > t):
            raise ValueError('Future closure cannot enter context')
        r = self.target
        if r is None:
            raw = [-1., 0., 0., 1., 1., 1. if self.closed else -1., self.applied_shift_s/.012]
        else:
            if self.target_available_t is None or self.target_available_t > t or not r['measured_t'] <= r['forecast_available_t'] <= t:
                raise ValueError('Future/unknown delivered timer target')
            raw = [1., (r['deadline']-t)/.1, (r['intercept_t']-t)/.1,
                   (t-r['measured_t'])/.1, r['sigma_time']/.02,
                   1. if self.closed else -1., self.applied_shift_s/.012]
        result = np.asarray(raw, np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite timer context')
        return np.clip(result, -1, 1)
