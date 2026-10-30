"""Explicit slow-policy residual contract; independent of frozen absolute actors.

Apply AFTER the fast controller, its hold branch and GripTimerBuffer. This is
an offline-ready interface, not a deployed or trained policy. Ball truth and
contact labels are never inputs. Bounds concern task-space offsets, not physical
acceleration. Expired targets slew toward zero; the logged tail is not a fresh
model action. Final integration must charge the measured composition/delivery.
"""
from dataclasses import asdict, dataclass
import heapq

import numpy as np

from .action_h192_195536 import SPEC


@dataclass(frozen=True)
class ResidualSpec:
    name: str = 'fast_relative_xyz_rotvec_closure_v26'
    horizon: int = 12
    dt: float = .016
    translation_m: float = .015
    rotation_rad: float = float(np.deg2rad(8))
    closure_fraction: float = .15
    translation_rate_mps: float = .2
    rotation_rate_radps: float = float(np.deg2rad(90))
    closure_rate_per_s: float = 2.

    def contract(self):
        return asdict(self)


RESIDUAL_SPEC = ResidualSpec()


def require_residual_contract(contract):
    if contract != RESIDUAL_SPEC.contract():
        raise ValueError('Expected explicit V26 residual contract; absolute-action checkpoints are incompatible')


def _bounded_vector(value, bound):
    value = np.asarray(value, float)
    return value * min(1., bound / max(float(np.linalg.norm(value)), 1e-30))


class ResidualChunks:
    """Future replies stay pending; late/out-of-order replies cannot replace newer ones."""
    def __init__(self):
        self.pending = []
        self.serial = 0
        self.anchor = -np.inf
        self.actions = None
        self.last_sample_t = -np.inf
        self.seen = set()
        self.accepted = self.stale = 0

    def enqueue(self, actions, observed_t, available_t, contract):
        require_residual_contract(contract)
        a = np.asarray(actions, np.float32)
        if a.shape != (12, 7) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1:
            raise ValueError('Expected finite normalized12x7 residual actions')
        if not np.isfinite([observed_t, available_t]).all() or available_t < observed_t:
            raise ValueError('Noncausal residual timestamps')
        if observed_t in self.seen or observed_t <= self.anchor or available_t >= observed_t + .192:
            self.stale += 1
            return False
        self.seen.add(observed_t)
        heapq.heappush(self.pending, (available_t, self.serial, observed_t, a.copy()))
        self.serial += 1
        return True

    def sample(self, t):
        if not np.isfinite(t) or t < self.last_sample_t:
            raise ValueError('Residual clock must be finite and monotonic')
        self.last_sample_t = t
        while self.pending and self.pending[0][0] <= t:
            _, _, observed, a = heapq.heappop(self.pending)
            if observed <= self.anchor or t >= observed + .192:
                self.stale += 1
                continue
            self.actions = np.vstack((np.zeros((1, 7), np.float32), a))
            self.anchor = observed
            self.accepted += 1
        if self.actions is None or t >= self.anchor + .192:
            return None
        x = np.clip((t-self.anchor)/.016, 0, 12)
        lo = min(int(np.floor(x)), 12)
        hi = min(lo+1, 12)
        return self.actions[lo] + (x-lo)*(self.actions[hi]-self.actions[lo])


class ResidualMixer:
    """Offset/rate bounds and bit-exact zero neutrality around final fast actions.

Grip modulation is enabled only by the already-delivered fast close flag. Before
that, it is explicitly masked. After fast closure, negative residuals can relax
closure by at most15%; they are no longer overwritten by the timer. This gives
one-sided authority at saturation and must be logged during future learning.
"""
    def __init__(self, spec=RESIDUAL_SPEC):
        if spec != RESIDUAL_SPEC:
            raise ValueError('Version the contract before changing bounds')
        self.spec = spec
        self.offset = np.zeros(7)
        self.last_t = None

    def mix(self, t, baseline, residual, fast_closed=False):
        if not np.isfinite(t):
            raise ValueError('Invalid mix time')
        dt = .004 if self.last_t is None else t-self.last_t
        if not 0 < dt <= .1:
            raise ValueError('Invalid residual update interval')
        b = None if baseline is None else np.asarray(baseline, np.float32)
        if b is not None and (b.shape != (7,) or not np.isfinite(b).all() or np.max(np.abs(b)) > 1.00001):
            raise ValueError('Invalid final fast action')
        r = np.zeros(7) if residual is None else np.asarray(residual, float)
        if r.shape != (7,) or not np.isfinite(r).all() or np.max(np.abs(r)) > 1.00001:
            raise ValueError('Invalid normalized residual')
        self.last_t = t
        s = self.spec
        target = np.r_[_bounded_vector(r[:3], 1)*s.translation_m,
                       _bounded_vector(r[3:6], 1)*s.rotation_rad,
                       np.clip(r[6], -1, 1)*s.closure_fraction if fast_closed else 0.]
        self.offset[:3] += _bounded_vector(target[:3]-self.offset[:3], s.translation_rate_mps*dt)
        self.offset[3:6] += _bounded_vector(target[3:6]-self.offset[3:6], s.rotation_rate_radps*dt)
        self.offset[6] += np.clip(target[6]-self.offset[6], -s.closure_rate_per_s*dt, s.closure_rate_per_s*dt)
        # Never let a previously enabled grip residual alter an open fast action.
        if not fast_closed:
            self.offset[6] = 0.
        detail = dict(target_available=residual is not None, offset=self.offset.tolist(),
                      grip_masked=not fast_closed, baseline_missing=b is None,
                      decay_tail=residual is None and bool(np.any(self.offset)),
                      physical_limits_scope='Task-space offset and offset-rate only; downstream servo/torque limits still required')
        if b is None:
            return None, detail
        result = b.copy()
        if np.any(self.offset[:3]):
            result[:3] = np.clip(b[:3] + self.offset[:3]/np.asarray(SPEC.scale), -1, 1)
        theta = float(np.linalg.norm(self.offset[3:6]))
        if theta:
            normal = b[3:6].astype(float)
            if np.linalg.norm(normal) < 1e-6:
                raise ValueError('Invalid baseline palm normal')
            normal /= np.linalg.norm(normal)
            axis = self.offset[3:6]/theta
            rotated = normal*np.cos(theta) + np.cross(axis, normal)*np.sin(theta) + axis*(axis@normal)*(1-np.cos(theta))
            result[3:6] = rotated/np.linalg.norm(rotated)
        if self.offset[6]:
            result[6] = np.clip(b[6] + 2*self.offset[6], -1, 1)
        detail['closure_saturated'] = bool(self.offset[6] and result[6] in (-1., 1.))
        return result, detail
