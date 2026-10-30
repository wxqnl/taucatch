"""A new rate04 learning identity; tensor/distribution math explicitly reuses V139.

No old-rate behavior data or density is accepted as this policy's experience.
The physical ABI is checked against V154 by the initializer in SIM.
"""
from .xyz_policy_v139 import XYZActor, build_critic, value, draw, log_prob, STD, TRAINABLE
from .xyz_context_v134 import CONTEXT_CONTRACT

KIND = 'rich_native_xyz_rate04_context_actor_v158'
XYZ_CONTRACT = dict(name='final_fast_relative_xyz_only_rate04_v154', horizon=12, dt=.016,
                    action_dim=3, translation_m=.015, translation_rate_mps=.4,
                    normal='fast_unchanged', closure='fast_timer_unchanged',
                    expiry='observation_plus_192ms_then_slew_to_zero')
BEHAVIOR_CONTRACT = 'latent_gaussian_36D_rho09_std05_clip_xyz_rate04_v158'


def require_artifact(artifact, base_sha):
    if artifact.get('kind') != KIND or artifact.get('base_checkpoint_sha256') != base_sha:
        raise ValueError('Wrong rate04 policy/base identity')
    if artifact.get('xyz_contract') != XYZ_CONTRACT or artifact.get('context_contract') != CONTEXT_CONTRACT:
        raise ValueError('Wrong rate04 executor/context ABI')
    if artifact.get('behavior_contract') != BEHAVIOR_CONTRACT:
        raise ValueError('Wrong new behavior identity')
    if artifact.get('trainable_keys') != list(TRAINABLE) or artifact.get('std') != list(STD):
        raise ValueError('Wrong trainable scope/distribution')
    if artifact.get('policy_version', -1) < 0 or artifact.get('optimization_steps', -1) < 0:
        raise ValueError('Missing policy version/step provenance')
