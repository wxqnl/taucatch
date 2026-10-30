"""Rich48 SFT checkpoint identity, separate from the rejected eight-scene pilot."""
from pathlib import Path
import torch

from .action_h192_195536 import build_h192_policy,require_spec
from .native_servo_space_v90 import SERVO_CONTRACT

KIND='native_spikevla_h192_servo_rich_sft_v109'
TIME_CONTRACT='native_h192_servo_rich48_reference12x16ms_v109'


def require_rich_checkpoint(ck):
    if ck.get('kind')!=KIND or ck.get('time_contract')!=TIME_CONTRACT:raise ValueError('Expected rich48 native checkpoint provenance')
    require_spec(ck['action_spec'])
    if ck.get('servo_contract')!=SERVO_CONTRACT or ck.get('budget_steps')!=360 or not 0<ck.get('step',0)<=360:raise ValueError('Wrong executor/training budget')
    if set(ck.get('dataset',{}))!={'train','val'}:raise ValueError('Missing rich48 split identities')
    for s,n in [('train',36),('val',12)]:
        if ck['dataset'][s]['format']!='single_ball_servo_rich_v105' or ck['dataset'][s]['all_rollouts']!=n:raise ValueError('Pilot/partial data cannot be disguised as rich48')
        if len(ck['dataset'][s].get('dataset_sha256',''))!=64:raise ValueError('Missing rich dataset identity')
    for name in ['dataset_manifest_sha256','initialization_sha256','frozen_upstream_sha256','training_config_sha256','cache_gate_sha256']:
        v=ck.get(name,'')
        if len(v)!=64 or any(c not in '0123456789abcdef' for c in v):raise ValueError('Missing SHA256 '+name)
    if ck.get('trained_scope_prefixes')!=['event_encoder.','vla.action_head.']:raise ValueError('Wrong rich training scope')


def build_rich_policy(path):
    ck=torch.load(Path(path),map_location='cpu',weights_only=False);require_rich_checkpoint(ck)
    model,prepare=build_h192_policy();model.load_state_dict(ck['state'],strict=True)
    return model,prepare,ck
