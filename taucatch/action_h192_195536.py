"""Explicit learned12x16ms contract; never retime an untrained96ms checkpoint."""
from dataclasses import asdict
import json

from .action_chunk import ActionSpec

SPEC = ActionSpec(dt=.016)
CONTRACT = 'native_spikevla_action_h192_teacher_grid_v1'


def require_spec(value):
    if json.loads(json.dumps(value)) != json.loads(json.dumps(asdict(SPEC))):
        raise ValueError('Expected trained12x16ms native action contract')


def require_checkpoint(checkpoint):
    require_spec(checkpoint['action_spec'])
    if checkpoint.get('time_contract') != CONTRACT:
        raise ValueError('Missing192ms training provenance; do not retime96ms weights')
    if set(checkpoint.get('dataset', {})) != {'train', 'val'}:
        raise ValueError('Missing192ms dataset identities')
    for split in checkpoint['dataset'].values():
        require_spec(split['action_spec'])
        for key in ('dataset_sha256', 'source_dataset_sha256', 'overlay_manifest_sha256', 'parent96_manifest_sha256'):
            value = split.get(key, '')
            if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Missing192ms provenance: '+key)


def build_h192_policy():
    from .action_policy import build_action_policy
    model, prepare = build_action_policy()
    model.spec = SPEC
    return model, prepare
