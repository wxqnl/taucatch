"""Native-action RL mathematics, independent of simulator or VLA imports."""
import math
import numpy as np
import torch

DISCOUNT_40MS = .99
NOISE_STD = (.03, .03, .03, .05, .05, .05, .6)
NOISE_RHO = .9


def temporal_cholesky(device='cpu', dtype=torch.float32):
    index = torch.arange(12, device=device)
    covariance = NOISE_RHO ** (index[:, None]-index[None, :]).abs()
    return torch.linalg.cholesky(covariance.to(dtype))


def action_log_prob(action, mean, chol, std):
    # Density of the UNCLIPPED latent action; clipping is an environment transform.
    delta = (action-mean)/std
    white = torch.linalg.solve_triangular(chol, delta, upper=False)
    normalizer = 84*.5*math.log(2*math.pi) + 12*std.log().sum() + 7*chol.diag().log().sum()
    return -.5*white.square().sum((-2, -1))-normalizer


def action_kl(mean, reference, chol, std):
    white = torch.linalg.solve_triangular(chol, (mean-reference)/std, upper=False)
    return .5*white.square().sum((-2, -1))


def sample_action(mean, chol, std, generator):
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return mean + (chol @ noise)*std


def gae(rewards, values, discounts, terminal, lam=.95):
    arrays = [np.asarray(v, dtype=np.float64) for v in (rewards, values, discounts)]
    rewards, values, discounts = arrays
    if not (len(rewards) == len(values) == len(discounts) == len(terminal)):
        raise ValueError('Mismatched rollout arrays')
    advantage = np.zeros(len(rewards)); carry = 0.
    for i in range(len(rewards)-1, -1, -1):
        if terminal[i]:
            next_value = 0.; carry = 0.
        elif i+1 < len(rewards):
            next_value = values[i+1]
        else:
            raise ValueError('Finite pilot rollout must end at an episode terminal')
        delta = rewards[i] + discounts[i]*next_value-values[i]
        carry = delta + discounts[i]*lam*carry
        advantage[i] = carry
    return advantage.astype(np.float32), (advantage+values).astype(np.float32)


def terminal_reward(outcome):
    # Touching, proximity and a wrong-ball hold cannot earn the catch bonus.
    return (10. if outcome['caught'] and not outcome['wrong_held'] else 0.) - (10. if outcome['wrong_held'] else 0.)


def potential(distance, released):
    return float(np.exp(-max(0., distance)/.25)) if released else 0.


def shape_reward(previous, current, dt=.001):
    return 2*((DISCOUNT_40MS ** (dt/.04))*current-previous)


def transitions(requests, reward_ticks, final_t):
    """Drop unconsumed draws; merge their tail into the last executed decision."""
    used = [r for r in requests if r.get('accepted') is True]
    if not used:
        raise RuntimeError('No accepted action: cannot call this an RL training rollout')
    times = np.asarray([t for t, _ in reward_ticks]); rewards = np.asarray([r for _, r in reward_ticks])
    output = []
    for i, row in enumerate(used):
        end = used[i+1]['observed_t'] if i+1 < len(used) else final_t
        start = row['observed_t']; mask = (times >= start-1e-10) & (times < end-1e-10)
        # Include terminal event at final_t exactly once.
        if i+1 == len(used):
            mask |= np.abs(times-final_t) < 1e-10
        weights = DISCOUNT_40MS ** (np.maximum(0, times[mask]-start)/.04)
        output.append(dict(id=row['id'], reward=float((weights*rewards[mask]).sum()),
            discount=float(DISCOUNT_40MS ** ((end-start)/.04)), terminal=i+1 == len(used),
            duration_s=float(end-start)))
    return output
