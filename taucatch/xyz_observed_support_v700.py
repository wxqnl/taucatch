"""Offline mask proposal for saved, actually active interpolation samples.

Only add endpoints that affected the original consumer before its scoring cutoff.
This helper does not assign labels: newly covered endpoints need audited complete
teacher or successful-policy chunks. It is not wired into any running trainer.
"""
import numpy as np


def augment_observed_support(point_mask, observed_t, available_t, cutoff_t,
                             active_sample_times, dt=.016):
    mask = np.asarray(point_mask)
    times = np.asarray(active_sample_times, dtype=float)
    if mask.shape != (12,) or mask.dtype != np.bool_:
        raise ValueError('Expected12 boolean point-validity entries')
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('Expected finite saved sample times')
    if not np.isfinite([observed_t, available_t, cutoff_t, dt]).all() or dt <= 0:
        raise ValueError('Invalid timestamps or knot interval')
    if available_t < observed_t or np.any(times < available_t) or np.any(times >= cutoff_t):
        raise ValueError('Samples must be delivered and strictly before cutoff')
    if np.any(times >= observed_t + 12 * dt):
        raise ValueError('Expired chunks have no interpolation support')
    result = mask.copy()
    if len(times):
        phase = (times[:, None] - observed_t) / dt
        # The zero anchor at observed_t is fixed, so only knots1..12 are learned.
        weights = np.maximum(0., 1 - np.abs(phase - np.arange(1, 13)))
        result |= weights.sum(0) > 1e-9
    return result
