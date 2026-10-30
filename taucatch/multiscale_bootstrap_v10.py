"""Rebase the visual residual whenever the asynchronous native reference changes."""
import numpy as np
from .multiscale_bootstrap_v9 import BootstrapController as PreviousController


class ContinuousReferenceController(PreviousController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference_anchor=None

    def update(self,t,forecast,buffer,measured_action):
        previous_anchor=self.reference_anchor
        rebased=False;rebase_norm=None
        native=buffer.sample(t) if self.mode=='hybrid' else None
        if self.started and not self.hybrid.holding and native is not None and buffer.anchor!=previous_anchor:
            source=measured_action if self.last_action is None else self.last_action
            delta=self.spec.decode(source)[0]-self.spec.decode(native)[0]
            rebase_norm=float(np.linalg.norm(delta))
            if rebase_norm>self.cfg.max_correction:
                # Keep issuing the last bounded command rather than admitting a large
                # reference discontinuity. A later admissible chunk can still take over.
                self.hybrid.last_t=t
                return np.asarray(source).copy(),dict(reason='reference_step_outside_residual_range',
                    phase='native_reference_rejected',native_reference_used=True,reference_rebased=False,
                    previous_reference_anchor=previous_anchor,reference_anchor=buffer.anchor,
                    proposed_rebase_norm_m=rebase_norm)
            self.hybrid.delta=delta.copy();rebased=True
        previous_action=None if self.last_action is None else self.last_action.copy()
        dt=self.cfg.period if self.hybrid.last_t is None else t-self.hybrid.last_t
        action,detail=super().update(t,forecast,buffer,measured_action)
        limited=False
        if rebased and action is not None and previous_action is not None:
            before=self.spec.decode(previous_action)[0]
            point,normal,closure=self.spec.decode(action)
            step=point-before;distance=float(np.linalg.norm(step))
            bound=self.cfg.max_velocity*dt
            if distance>bound:
                point=before+step*(bound/distance)
                action=self.spec.encode(point,normal,closure)
                self.hybrid.delta=point-self.spec.decode(native)[0]
                self.last_action=action.copy();limited=True
                if self.hybrid.holding:
                    self.hybrid.hold_action=action.copy();self.hybrid.hold_action[6]=1.
        if self.started and native is not None:self.reference_anchor=buffer.anchor
        return action,dict(detail,reference_rebased=rebased,reference_transition_limited=limited,rebase_norm_m=rebase_norm,
            previous_reference_anchor=previous_anchor,reference_anchor=self.reference_anchor)
