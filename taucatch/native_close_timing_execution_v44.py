"""Measured timing reference lane, before the unchanged 1ms robot servo.

Only this lane advances the timing latch. Context reads must use timer.context,
not base.sample. Pending replies are installed immediately with a FUTURE charged
availability timestamp, so polling cannot backdate delivery. This ABI still
needs full RPC and physical execution audits before new collection.
"""
import time
import numpy as np

from .multiscale_grip_timer_v14 import GripTimerBuffer
from .native_residual_execution_v27 import FinalResidualBuffer


class TimingGripReference(GripTimerBuffer):
    def __init__(self,base,timer,servo_dt):
        super().__init__(base,timer)
        self.servo_dt=servo_dt

    def sample(self,query_t):
        action=self.base.sample(query_t)
        closed=self.timer.advance(query_t,timing_sample_t=query_t-self.servo_dt)
        if action is not None:self.last_action=np.asarray(action,np.float32).copy()
        if self.last_action is None or (action is None and not closed):return None
        result=self.last_action.copy();result[6]=1. if closed else -1.
        return result


class FinalTimingBuffer(FinalResidualBuffer):
    def __init__(self, base, timer, servo_dt=.001, clock=time.perf_counter):
        super().__init__(TimingGripReference(base,timer,servo_dt),servo_dt,clock)

    def enqueue_timing(self, actions, observed_t, available_t, contract):
        return self.base.timer.timing.enqueue(actions,observed_t,available_t,contract)

    def enqueue_residual(self, *args, **kwargs):
        raise ValueError('V44 timing-only lane cannot execute old amplitude/pose residuals')

    def sample(self, query_t):
        result = super().sample(query_t)
        # State snapshot serialization belongs to the owner logging stage. The
        # timed parent already charges timer sampling, reference and composition.
        self.rows[-1]['close_timing'] = dict(self.base.timer.last_detail)
        return result
