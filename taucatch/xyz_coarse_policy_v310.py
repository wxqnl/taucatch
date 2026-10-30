"""New coarse physical action identity; prior weights initialize it only."""
import json
from pathlib import Path
import torch

from .action_horizon_data import sha256
OBSERVATION='prerelease_RGB_text_fresh_DVS_softsign_coarse_XYZ_v310'
from .xyz_learning_v158 import TRAINABLE,STD
from .xyz_coarse_contract_v308 import XYZ_CONTRACT
from .xyz_context_softsign_v273 import CONTEXT_CONTRACT

KIND='rich_native_xyz_coarse_sample_actor_v310'
BEHAVIOR='coarse06_rate08_conditional36D_rho09_std05_v310'
NAMESPACE='new_coarse_sampled_v310_no_old_probability_reuse'


def require_artifact(a,base_sha):
    expected=dict(kind=KIND,behavior_contract=BEHAVIOR,behavior_namespace=NAMESPACE,
        observation_contract=OBSERVATION,base_checkpoint_sha256=base_sha,
        xyz_contract=XYZ_CONTRACT,context_contract=CONTEXT_CONTRACT,
        trainable_keys=list(TRAINABLE),std=list(STD),old_behavior_probability_reuse=False)
    if any(a.get(k)!=v for k,v in expected.items()):
        raise ValueError('Wrong sampled prerelease policy/observation identity')
    for key in ['policy_version','optimization_steps','counter_next']:
        if type(a.get(key))is not int or a[key]<0:
            raise ValueError('Missing nonnegative sampled provenance')
    if (a['policy_version']==0)!=(a['optimization_steps']==0):
        raise ValueError('Initializer/version and new optimization steps disagree')
    if bool(a.get('trained'))!=(a['optimization_steps']>0):
        raise ValueError('Training provenance disagrees')
    for key in ['source_initializer','legacy_runtime_initializer']:
        if sha256(a[key])!=a[key+'_sha256']:
            raise ValueError('Initialization source changed')
    if not isinstance(a.get('rng_resume_state'),torch.Tensor):
        raise ValueError('Private RNG continuation missing')


def build_runtime(c):
    from .xyz_prethrow_runtime_v199 import PrethrowRuntime
    a=torch.load(c['artifact'],map_location='cpu',weights_only=False)
    require_artifact(a,sha256(c['base_checkpoint']))
    gate=json.loads(Path(c['feature_gate']).read_text())
    e=PrethrowRuntime(c['base_checkpoint'],a['legacy_runtime_initializer'],
                     json.loads(Path(gate['runtime_options']).read_text()))
    e.engine.actor.load_state_dict(a['actor'],strict=True)
    e.engine.critic.load_state_dict(a['critic'],strict=True)
    e.engine.generator.set_state(a['rng_resume_state'])
    return e,a
