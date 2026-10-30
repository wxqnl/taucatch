"""Versioned native12x7 spatial SFT checkpoint identity; no legacy data forgery."""
from pathlib import Path

import torch

from .action_h192_195536 import build_h192_policy,require_spec

KIND='native_spikevla_h192_servo_space_sft_v90'
TIME_CONTRACT='native_h192_servo_reference12x16ms_v90'
SERVO_CONTRACT=dict(name='bounded_local_position_normal_IK_v72',damping=.01,solver_tol=1e-8,max_iterations=30)


def require_space_checkpoint(ck):
    if ck.get('kind')!=KIND or ck.get('time_contract')!=TIME_CONTRACT:
        raise ValueError('Expected independently versioned servo-space checkpoint')
    require_spec(ck['action_spec'])
    if ck.get('servo_contract')!=SERVO_CONTRACT or not 0<ck['step']<=ck['budget_steps']:
        raise ValueError('Wrong executor contract or training budget')
    if set(ck['dataset'])!={'train','val'} or ck['dataset']['train']['format']!='single_ball_servo_aligned_pilot_v84' or ck['dataset']['val']['format']!='single_ball_servo_aligned_pilot_v84':
        raise ValueError('Wrong spatial training/validation provenance')
    for k in ['dataset_manifest_sha256','initialization_sha256','frozen_upstream_sha256']:
        if len(ck.get(k,''))!=64 or any(x not in '0123456789abcdef' for x in ck[k]):
            raise ValueError('Missing SHA256 identity: '+k)
    if ck['trained_scope_prefixes']!=['event_encoder.','vla.action_head.']:
        raise ValueError('Unexpected trained synapse scope')


def build_space_policy(path):
    ck=torch.load(Path(path),map_location='cpu',weights_only=False);require_space_checkpoint(ck)
    model,prepare=build_h192_policy();model.load_state_dict(ck['state'],strict=True)
    return model,prepare,ck
