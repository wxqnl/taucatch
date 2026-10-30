"""Causal fast startup followed by a continuous handover to native VLA feedback."""
import numpy as np
from .multiscale_v9 import FastConfig, ResidualController as LeadController


class BootstrapController:
    def __init__(self,spec,cfg=FastConfig(),mode='hybrid'):
        assert mode in ('fast','hybrid')
        self.spec,self.cfg,self.mode=spec,cfg,mode
        self.fast=LeadController(spec,cfg,'fast');self.hybrid=LeadController(spec,cfg,'hybrid')
        self.started=False;self.last_action=None

    def update(self,t,forecast,buffer,measured_action):
        native=buffer.sample(t) if self.mode=='hybrid' else None
        phase='fast_only' if self.mode=='fast' else 'bootstrap'
        # Issued closure is sticky; no contact/target scoring metadata is available.
        if self.mode=='hybrid' and not self.started and not self.fast.holding and native is not None:
            source=measured_action if self.last_action is None else self.last_action
            previous_point=self.spec.decode(source)[0];native_point=self.spec.decode(native)[0]
            delta=previous_point-native_point
            if np.linalg.norm(delta)<=self.cfg.max_correction:
                self.hybrid.delta=delta.copy();self.hybrid.velocity=self.fast.velocity.copy()
                self.hybrid.acceleration=self.fast.acceleration.copy()
                self.started=True
            else:phase='bootstrap_native_outside_residual_range'
        if self.started:
            action,detail=self.hybrid.update(t,forecast,buffer,measured_action);phase='native_feedback'
        else:
            action,detail=self.fast.update(t,forecast,buffer,measured_action)
            if self.fast.holding and self.mode=='hybrid':phase='bootstrap_hold'
        if action is not None:self.last_action=action.copy()
        return action,dict(detail,phase=phase,native_reference_used=self.started)
