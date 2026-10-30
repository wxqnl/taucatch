"""Opt-in causal context preserving distinctions beyond V134's hard bounds.

Only dynamic proprioception/error channels change. Original units/scales stay
3 rad/s, 3 m/s and .05 m; constants are not selected using validation outcomes.
This is a new observation contract, not compatible behavior experience for V239.
No deployed policy or collector imports this experimental module yet.
"""
import numpy as np

from .xyz_context_v134 import CONTEXT_DIM

CONTEXT_CONTRACT = 'delivered_fast7_softsign_qvel7_palmvel3_error3_age_valid_v273'


def softsign(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite dynamic context')
    return value / (1. + np.abs(value))


def context(t, fast_action, fast_available_t, joint_velocity, palm, previous_palm,
            previous_t, measured_t, forecast_available_t):
    if not np.isfinite(t):
        raise ValueError('Invalid observation time')
    qvel = np.asarray(joint_velocity, float)
    palm = np.asarray(palm, float)
    prior = np.asarray(previous_palm, float)
    if qvel.shape != (7,) or palm.shape != (3,) or prior.shape != (3,):
        raise ValueError('Wrong proprioception shape')
    if not np.isfinite(np.r_[qvel, palm, prior, previous_t]).all() or not previous_t < t:
        raise ValueError('Velocity must use a strictly earlier measured pose')
    velocity = (palm - prior) / (t - previous_t)
    if fast_action is None:
        fast, error, valid = np.zeros(7), np.zeros(3), -1.
    else:
        fast = np.asarray(fast_action, float)
        if fast.shape != (7,) or not np.isfinite(fast).all() or np.max(np.abs(fast)) > 1.00001:
            raise ValueError('Invalid fast action')
        if fast_available_t is None or not np.isfinite(fast_available_t) or fast_available_t > t:
            raise ValueError('Fast action has not been delivered')
        from .action_h192_195536 import SPEC
        error = SPEC.decode(fast)[0] - palm
        valid = 1.
    if measured_t is None:
        age = 1.
    else:
        if forecast_available_t is None or not np.isfinite([measured_t, forecast_available_t]).all() or not measured_t <= forecast_available_t <= t:
            raise ValueError('Visual measurement has not been delivered')
        age = 2 * np.clip((t - measured_t) / .1, 0, 1) - 1
    result = np.r_[fast, softsign(qvel / 3.), softsign(velocity / 3.),
                   softsign(error / .05), age, valid].astype(np.float32)
    assert result.shape == (CONTEXT_DIM,) and np.isfinite(result).all()
    return result
