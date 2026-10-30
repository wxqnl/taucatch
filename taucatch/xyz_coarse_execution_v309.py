"""New explicitly typed6cm/.8m/s residual; unchanged downstream servo limits."""
from dataclasses import replace
import heapq
import time

import numpy as np

from .native_residual_v26 import ResidualMixer, RESIDUAL_SPEC
from .xyz_coarse_contract_v308 import XYZ_CONTRACT
from .xyz_execution_v134 import XYZFinalBuffer, XYZ_CONTRACT as TIMING_CARRIER_CONTRACT


class CoarseXYZMixer(ResidualMixer):
    def __init__(self):
        super().__init__()
        self.spec = replace(RESIDUAL_SPEC, name='coarse_xyz_only_internal_v309',
            translation_m=.06, translation_rate_mps=.8, rotation_rad=0.,
            closure_fraction=0., rotation_rate_radps=0., closure_rate_per_s=0.)

    def mix(self, t, baseline, residual, fast_closed=False):
        if residual is not None:
            r = np.asarray(residual)
            if r.shape != (7,) or np.any(r[3:] != 0):
                raise ValueError('Coarse XYZ cannot change normal or closure')
        return super().mix(t, baseline, residual, fast_closed)


class CoarseXYZFinalBuffer(XYZFinalBuffer):
    def _reset_lane(self):
        super()._reset_lane()
        self.mixer = CoarseXYZMixer()

    def enqueue_xyz(self, actions, observed_t, available_t, contract):
        if contract != XYZ_CONTRACT:
            raise ValueError('Wrong coarse XYZ execution contract')
        # Reuse only the normalized private timing/interpolation carrier.
        # Physical scaling is exclusively performed by CoarseXYZMixer above.
        return super().enqueue_xyz(actions, observed_t, available_t, TIMING_CARRIER_CONTRACT)

    def enqueue_billed_xyz(self, actions, observed_t, base_ready, clock=time.perf_counter):
        start = clock()
        serial = self.residual.serial
        accepted = self.enqueue_xyz(actions, observed_t, base_ready, XYZ_CONTRACT)
        elapsed = clock() - start
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid coarse admission cost')
        ready = base_ready + elapsed
        if accepted:
            index = next(i for i, p in enumerate(self.residual.pending) if p[1] == serial)
            old = self.residual.pending.pop(index)
            if ready >= observed_t + .192:
                self.residual.stale += 1
                accepted = False
            else:
                self.residual.pending.append((ready, *old[1:]))
            heapq.heapify(self.residual.pending)
        return accepted, ready, 1000 * elapsed
