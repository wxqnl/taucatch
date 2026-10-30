"""Explicit joint-policy identity; load adapted native weights before tracing."""
import torch

from .action_horizon_data import sha256
from .action_policy import action_runtime
from .native_servo_rich_v109 import build_rich_policy
from .xyz_learning_v158 import build_critic, value
from .xyz_fusion_policy_v596 import FusionXYZActor as XYZActor, FUSION_SPEC
from .native_frozen_condition_v108 import cached_native_prediction
from .xyz_prethrow_runtime_v199 import PrethrowRuntime
from .xyz_runtime_v158 import XYZRuntime

KIND = 'native_xyz_raw_fusion_SFT_v596'
OBSERVATION = 'prerelease_condition_live_DVS_state_softsign_joint_decoder_v404'


class FusionRawEngine(XYZRuntime):
    def __init__(self, path, options):
        a = torch.load(path, map_location='cpu', weights_only=False)
        assert a['kind'] == KIND and a['observation_contract'] == OBSERVATION and a['fusion_spec']==FUSION_SPEC
        assert a['raw_RL_steps'] == 1 and a['selected_joint_gradient_steps']==16
        assert a['supervised_recovery_steps']>0 and a['policy_version']==3
        assert sha256(a['recovery_parent'])==a['recovery_parent_sha256']
        assert sha256(a['parent_raw_joint_artifact'])==a['parent_raw_joint_sha256']
        assert sha256(a['base_initialization']) == a['base_initialization_sha256']
        assert sha256(a['actor_initialization']) == a['actor_initialization_sha256']
        self.model, self.prepare, ck = build_rich_policy(a['base_initialization'])
        parameters = dict(self.model.named_parameters())
        assert set(a['native_trainable_state']) == set(a['native_trainable_names'])
        with torch.no_grad():
            for name, value in a['native_trainable_state'].items():
                assert name.startswith(('event_encoder.', 'vla.action_head.'))
                assert not name.startswith('vla.action_head.decoder.action_projection.')
                parameters[name].copy_(value)
        self.actor = XYZActor(ck)
        self.actor.load_state_dict(a['actor_state'], strict=True)
        initial = torch.load(a['actor_initialization'], map_location='cpu', weights_only=False)
        self.critic = build_critic()
        self.critic.load_state_dict(initial['critic'], strict=True)
        self.model.cuda().eval().requires_grad_(False)
        self.actor.cuda().eval().requires_grad_(False)
        self.critic.cuda().eval().requires_grad_(False)
        self.capture = torch.full((4, 1, 12, 256), float('nan'), device='cuda')
        projection = self.model.vla.action_head.decoder.action_projection
        original = projection.forward_step
        def record(features, state, t):
            self.capture[t].copy_(features)
            return original(features, state, t)
        projection.forward_step = record
        self.runtime = action_runtime(self.model, **options)
        # The artifact binds both native and XYZ weights. Old base identity is
        # explicitly initialization only; old coarse behavior is incompatible.
        self.artifact_sha = sha256(path)
        self.base_sha = self.artifact_sha
        self.base_initialization_sha = a['base_initialization_sha256']
        self.policy_version = a['policy_version']
        self.optimization_steps = a['supervised_recovery_steps']
        self.source_raw_joint_steps=a['selected_joint_gradient_steps']
        self.generator = torch.Generator(device='cuda').manual_seed(404414)


class FusionPrethrowRuntime(PrethrowRuntime):
    def __init__(self, path, options):
        self.engine = FusionRawEngine(path, options)
        self.condition = self.padding = None
        self.original_condition = self.engine.model.vla.encode_condition
        self.graph = None
        self.anchor_observed_t = self.anchor_rgb_t = None
        def capture_condition(*args, **kwargs):
            condition, features = self.original_condition(*args, **kwargs)
            valid = features['text_valid']
            padding = torch.cat((torch.zeros((1, condition.shape[1] - valid.shape[1]),
                                             dtype=torch.bool, device=condition.device), ~valid), 1)
            if self.condition is None:
                self.condition = torch.empty_like(condition)
                self.padding = torch.empty_like(padding)
            assert condition.shape == self.condition.shape
            self.condition.copy_(condition); self.padding.copy_(padding)
            return condition, features
        self.engine.model.vla.encode_condition = capture_condition

    def _body(self):
        e=self.engine;e.capture.fill_(float('nan'))
        cached_native_prediction(e.model,self.condition,self.padding,self.robot,self.events)
        state={};logits=torch.stack([e.actor.readout.forward_step(e.capture[t],state,t)for t in range(e.capture.shape[0])]).mean(0)
        motion=e.capture.mean(0);dynamic=self.dynamic[:,None,:].expand(-1,12,-1)
        delta=e.actor.fusion(torch.cat((motion,dynamic),-1))
        mean=torch.tanh(logits+e.actor.context(self.dynamic)[:,None,:]+delta)
        critic=value(e.critic,e.capture,self.dynamic)
        valid=torch.isfinite(e.capture).all()&torch.isfinite(delta).all()&torch.isfinite(mean).all()&torch.isfinite(critic).all()
        return mean,critic,valid
