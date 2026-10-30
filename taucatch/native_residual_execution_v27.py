"""Measured final residual lane AFTER fast hold/timer, before existing servo.

Every final action, including a neutral one, is delayed by its measured lane
processing cost. Baseline values can be exactly preserved while delivery timing
changes. No simulator truth enters this lane. Existing servo timing assumptions
remain simulation assumptions, not hardware guarantees.
"""
from collections import deque
import time

import numpy as np

from .action_chunk import ActionSpec
from .multiscale_catcher_v1 import MultiscaleCatcher
from .multiscale_grip_timer_v14 import GripTimerBuffer
from .native_residual_v26 import ResidualChunks, ResidualMixer, require_residual_contract


def validate_reply(reply, base_sha256, adapter_sha256):
    require_residual_contract(reply.get('residual_contract'))
    if reply.get('base_checkpoint_sha256') != base_sha256 or reply.get('adapter_sha256') != adapter_sha256:
        raise ValueError('Residual RPC model identity mismatch')


class FinalResidualBuffer:
    def __init__(self, base, servo_dt=.001, clock=time.perf_counter):
        self.base, self.servo_dt, self.clock = base, servo_dt, clock
        self._reset_lane()

    def __getattr__(self, key):
        return getattr(self.base, key)

    def _reset_lane(self):
        self.residual = ResidualChunks()
        self.mixer = ResidualMixer()
        self.pending = deque()
        self.latest = None
        self.busy_until = -np.inf
        self.rows = []
        self.last_t = -np.inf

    def reset(self):
        self.base.reset()
        self._reset_lane()

    def enqueue_residual(self, actions, observed_t, available_t, contract):
        return self.residual.enqueue(actions, observed_t, available_t, contract)

    def sample(self, query_t):
        # ActionCatcher requests t+dt and consumes the result at physical t.
        t = query_t-self.servo_dt
        if t <= self.last_t:
            raise ValueError('Final residual lane must be sampled once per increasing servo tick')
        self.last_t = t
        while self.pending and self.pending[0]['available_t'] <= t+1e-12:
            self.latest = self.pending.popleft()
        started = self.clock()
        baseline = self.base.sample(query_t)
        target = self.residual.sample(t)
        closed = bool(self.base.timer.closed)
        action, detail = self.mixer.mix(t, baseline, target, fast_closed=closed)
        entry = dict(id=len(self.rows), t=t, query_t=query_t,
                     baseline=None if baseline is None else np.asarray(baseline,np.float32).tolist(),
                     generated=None if action is None else action.tolist(),
                     fast_closed=closed, residual_anchor=None if not np.isfinite(self.residual.anchor) else self.residual.anchor,
                     target=None if target is None else target.tolist(), **detail)
        elapsed = self.clock()-started
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid measured final-lane processing cost')
        ready = max(t,self.busy_until)+elapsed
        self.busy_until = ready
        entry.update(available_t=ready,compute_ms=1000*elapsed,expiry_t=ready+.020)
        self.pending.append(entry)
        live = self.latest is not None and self.latest['expiry_t'] >= t-1e-12
        returned = self.latest['generated'] if live else None
        entry['delivered_id'] = self.latest['id'] if live else None
        entry['returned'] = returned
        self.rows.append(entry)
        return None if returned is None else np.array(returned,np.float32)


class ResidualCatcher(MultiscaleCatcher):
    def __init__(self,*args,timer,**kwargs):
        super().__init__(*args,mode='fast',**kwargs)
        self.buffer = FinalResidualBuffer(GripTimerBuffer(self.buffer,timer),self.dt)

    def reset(self,data):
        super().reset(data)
        normal = -data.site_xmat[self.ids.palm_site].reshape(3,3)[:,1]
        self.buffer.base.last_action = self.spec.encode(data.site_xpos[self.ids.palm_site],normal,0.)
