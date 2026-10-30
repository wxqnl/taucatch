"""Causal grip timer independent of the delayed motion-command FIFO.

Simulation models a separate reflex execution resource alongside the existing
1ms servo. This is not a measured hard-real-time/hardware scheduling guarantee.
"""
from collections import deque
import numpy as np
from .multiscale_plane_controller_v13 import interception


class GripTimer:
    def __init__(self,cfg):self.cfg=cfg;self.reset()
    def reset(self):
        self.pending=deque();self.target=None;self.closed=False;self.closed_t=None;self.busy_until=-np.inf
    def propose(self,forecast,t):
        if forecast is None:return None
        target,reason=interception(forecast,t,self.cfg)
        if target is None or target['sigma_time']>self.cfg.max_closing_sigma_s:return None
        return dict(intercept_t=target['t'],deadline=target['t']-self.cfg.close_lead-self.cfg.period,
            measured_t=forecast.measured_t,observed_t=forecast.observed_t,
            forecast_available_t=forecast.available_t,sigma_time=target['sigma_time'])
    def submit(self,proposal,observed_t,cost_seconds):
        if not np.isfinite(cost_seconds) or cost_seconds<0:raise ValueError('Invalid timer processing cost')
        ready=max(observed_t,self.busy_until)+cost_seconds
        if proposal is not None:
            assert proposal['forecast_available_t']<=observed_t+1e-9
        self.pending.append((ready,proposal));self.busy_until=ready;return ready
    def advance(self,t):
        while self.pending and self.pending[0][0]<=t+1e-9:
            _,self.target=self.pending.popleft()
        if not self.closed and self.target is not None:
            r=self.target
            if t-r['measured_t']<=self.cfg.max_measurement_age and r['deadline']<=t+1e-9<=r['intercept_t']+self.cfg.period+1e-9:
                self.closed=True;self.closed_t=t
        return self.closed


class GripTimerBuffer:
    def __init__(self,base,timer):self.base=base;self.timer=timer;self.last_action=None
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self):self.base.reset();self.timer.reset();self.last_action=None
    def sample(self,t):
        action=self.base.sample(t);closed=self.timer.advance(t)
        if action is not None:self.last_action=np.asarray(action,np.float32).copy()
        if self.last_action is None or (action is None and not closed):return None
        result=self.last_action.copy();result[6]=1. if closed else -1.
        return result
