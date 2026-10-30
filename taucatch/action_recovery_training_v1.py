"""Explicit DVS/native-head scope and fixed mixed TRAIN batch construction."""
from collections import defaultdict

import numpy as np
import torch

from .action_sampling import ActionSampler


TRAIN_PREFIXES = ('event_encoder.', 'vla.action_head.')
SLOTS = ('v3',) * 8 + ('handoff',) * 4 + ('closing',) * 2 + ('flight',) * 2


def training_mode(model):
    # Frozen encoders and their running statistics must remain in evaluation mode.
    model.eval()
    model.event_encoder.train()
    model.vla.action_head.train()


def set_scope(model):
    model.requires_grad_(False)
    model.event_encoder.requires_grad_(True)
    model.vla.action_head.requires_grad_(True)
    training_mode(model)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any(not name.startswith(TRAIN_PREFIXES) for name in names):
        raise ValueError('Unexpected recovery training scope')
    return dict(trainable_names=names,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                event_parameters=sum(p.numel() for p in model.event_encoder.parameters()),
                native_head_parameters=sum(p.numel() for p in model.vla.action_head.parameters()),
                scope='Entire existing DVS encoder and native action head,including state projection/decoder; RGB,text,and condition projectors frozen')


def subset_probabilities(data, indices):
    """Balance actual kind/thrower/scene/rollout within an eligible phase pool."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError('Empty recovery phase pool')
    for local, index in enumerate(indices):
        kind, seed, _, rollout = data.sampling_metadata[index]
        groups[(kind, data.sampling_strata[index])][seed][rollout].append(local)
    strata_per_kind = defaultdict(int)
    for kind, _ in groups:
        strata_per_kind[kind] += 1
    probabilities = np.zeros(len(indices), dtype=np.float64)
    for (kind, _), seeds in groups.items():
        for rollouts in seeds.values():
            for rows in rollouts.values():
                probabilities[rows] = 1 / len(strata_per_kind) / strata_per_kind[kind] / len(seeds) / len(rollouts) / len(rows)
    return indices, probabilities / probabilities.sum()


class MixedSampler:
    def __init__(self, base, recovery):
        self.base = ActionSampler(base, 'balanced')
        valid = recovery.parts['valid']
        flight = [i for i in recovery.flight_indices if valid[i].any()]
        closing = [i for i in flight if np.any((recovery.parts['actions'][i, :, 6] >= 0) & valid[i])]
        self.pools = {name: subset_probabilities(recovery, indices) for name, indices in
                      [('handoff', recovery.first_state_indices), ('closing', closing), ('flight', flight)]}

    def draw(self, rng):
        base = self.base.draw(rng, 8)
        recovery = np.concatenate([rng.choice(self.pools[name][0], size=count, p=self.pools[name][1])
                                   for name, count in [('handoff', 4), ('closing', 2), ('flight', 2)]])
        return base, recovery

    def summary(self, base):
        return dict(batch_slots=list(SLOTS), v3=self.base.summary(base),
                    recovery_pool_sizes={k: len(v[0]) for k, v in self.pools.items()},
                    mixture='8v3+4student_handoff+2precontact_closing+2flight; slot counts,not guaranteed waypoint-loss masses',
                    recovery_balance='Actual kind,thrower/speed,scene,rollout within eligible pool')


def batch(base, recovery, indices):
    bx, by, bm = base.batch(indices[0])
    rx, ry, rm = recovery.batch(indices[1])
    keys = {'over', 'mosaic', 'events', 'state', 'text'}
    if set(bx) != keys or set(rx) != keys:
        raise ValueError('Mixed actor inputs must use the sensor/text whitelist')
    x = {key: torch.cat((bx[key], rx[key])) for key in keys - {'text'}}
    x['text'] = bx['text'] + rx['text']
    return x, torch.cat((by, ry)), torch.cat((bm, rm))


class IndexedView:
    def __init__(self, data, indices):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def batch(self, indices):
        return self.data.batch(self.indices[list(indices)])
