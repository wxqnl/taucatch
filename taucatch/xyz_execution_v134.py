"""Explicit XYZ-only ABI over the preserved, measured final residual lane.

Internal padding reuses V26 translation bounds; no normal or grip residual is
admitted. This is a new executor interface, not a trained VLA policy.
"""
import heapq
import time

import numpy as np

from .native_residual_execution_v27 import FinalResidualBuffer
from .native_residual_v26 import RESIDUAL_SPEC

XYZ_CONTRACT = dict(name='final_fast_relative_xyz_only_v134', horizon=12, dt=.016,
                    action_dim=3, translation_m=.015, translation_rate_mps=.2,
                    normal='fast_unchanged', closure='fast_timer_unchanged',
                    expiry='observation_plus_192ms_then_slew_to_zero')


class XYZFinalBuffer(FinalResidualBuffer):
    def enqueue_residual(self, *args, **kwargs):
        raise ValueError('Use the explicit new XYZ-only ABI')

    def enqueue_xyz(self, actions, observed_t, available_t, contract):
        if contract != XYZ_CONTRACT:
            raise ValueError('Wrong XYZ-only contract')
        a = np.asarray(actions, np.float32)
        if a.shape != (12, 3) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1:
            raise ValueError('Expected normalized finite12x3 XYZ chunk')
        if available_t < self.residual.last_sample_t:
            raise ValueError('Cannot backdate an XYZ packet')
        padded = np.pad(a, ((0, 0), (0, 4)))
        return super().enqueue_residual(padded, observed_t, available_t, RESIDUAL_SPEC.contract())

    def enqueue_billed_xyz(self, actions, observed_t, base_ready, clock=time.perf_counter):
        started = clock()
        serial = self.residual.serial
        accepted = self.enqueue_xyz(actions, observed_t, base_ready, XYZ_CONTRACT)
        elapsed = clock() - started
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid admission processing time')
        ready = base_ready + elapsed
        if accepted:
            i = next(i for i, p in enumerate(self.residual.pending) if p[1] == serial)
            old = self.residual.pending.pop(i)
            if ready >= observed_t + .192:
                self.residual.stale += 1
                accepted = False
            else:
                self.residual.pending.append((ready, *old[1:]))
            heapq.heapify(self.residual.pending)
        return accepted, ready, 1000 * elapsed

    def sample(self, query_t):
        result = super().sample(query_t)
        self.rows[-1]['base_timing_delivered_id'] = self.base.rows[-1]['delivered_id']
        return result
