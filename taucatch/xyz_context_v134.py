"""New causal dynamic context, independent of MuJoCo and the scoring sidecar."""
import numpy as np

CONTEXT_CONTRACT = 'delivered_fast7_qvel7_palmvel3_trackingerror3_visualage_valid_v134'
CONTEXT_DIM = 22


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
    result = np.r_[fast, np.clip(qvel / 3., -1, 1), np.clip(velocity / 3., -1, 1),
                   np.clip(error / .05, -1, 1), age, valid].astype(np.float32)
    assert result.shape == (CONTEXT_DIM,) and np.isfinite(result).all()
    return result
