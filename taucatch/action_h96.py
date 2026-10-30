"""Explicit 96 ms contract. Old 4 ms checkpoints are initialization weights only."""
import json
from dataclasses import asdict

from .action_chunk import ActionSpec

SPEC = ActionSpec(dt=.008)
CONTRACT = 'native_spikevla_action_h96_overlay_v1'


def canonical(value):
    return json.loads(json.dumps(value))


def require_spec(value):
    if canonical(value) != canonical(asdict(SPEC)):
        raise ValueError('Expected explicitly trained 12x8ms action contract')


def require_checkpoint(checkpoint):
    require_spec(checkpoint['action_spec'])
    if checkpoint.get('time_contract') != CONTRACT:
        raise ValueError('Missing 96ms training provenance; never retime a 48ms model')
    splits = checkpoint.get('dataset', {})
    if set(splits) != {'train', 'val'}:
        raise ValueError('Missing 96ms dataset identities')
    for split in splits.values():
        require_spec(split['action_spec'])
        for key in ('dataset_sha256', 'source_dataset_sha256', 'overlay_manifest_sha256'):
            value = split.get(key, '')
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Missing dataset provenance: {key}')


def build_h96_policy():
    from .action_policy import build_action_policy
    model, prepare = build_action_policy()
    # Same native head shape and spatial normalization; labels and timing differ.
    # Training from old weights is explicit, with a fresh optimizer and schedule.
    model.spec = SPEC
    return model, prepare
