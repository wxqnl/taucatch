"""Frozen rich native feature encoder plus the separate XYZ actor/critic."""
import torch

from .action_horizon_data import sha256
from .action_policy import action_runtime
from .native_servo_rich_v109 import build_rich_policy
from .xyz_learning_v158 import XYZActor, build_critic, value, require_artifact


class XYZRuntime:
    def __init__(self, base_path, artifact_path, options):
        self.base_sha = sha256(base_path)
        self.artifact_sha = sha256(artifact_path)
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
        require_artifact(artifact, self.base_sha)
        self.policy_version = artifact['policy_version']
        self.optimization_steps = artifact['optimization_steps']
        self.model, self.prepare, checkpoint = build_rich_policy(base_path)
        self.actor = XYZActor(checkpoint).cuda()
        self.actor.load_state_dict(artifact['actor'], strict=True)
        self.critic = build_critic().cuda()
        self.critic.load_state_dict(artifact['critic'], strict=True)
        self.actor.eval().requires_grad_(False)
        self.critic.eval().requires_grad_(False)
        self.model.cuda().eval().requires_grad_(False)
        self.capture = torch.full((4, 1, 12, 256), float('nan'), device='cuda')
        projection = self.model.vla.action_head.decoder.action_projection
        original = projection.forward_step

        def record(features, state, t):
            self.capture[t].copy_(features)
            return original(features, state, t)

        projection.forward_step = record
        self.runtime = action_runtime(self.model, **options)
        self.generator = torch.Generator(device='cuda')
        self.generator.manual_seed(artifact['behavior_seed'])

    def encode(self, text, sample, cached=True):
        from serve_single_ball_h192_195536 import infer_action
        self.capture.fill_(float('nan'))
        # The absolute7D output is computed and discarded; it never commands motion.
        infer_action(self.model, self.prepare, torch.device('cuda:0'), text, sample,
                     self.runtime if cached else None)
        features = self.capture.detach().clone()
        if not torch.isfinite(features).all():
            raise ValueError('Native features were not captured')
        dynamic = torch.as_tensor(sample['context'], device='cuda', dtype=torch.float32)[None]
        return features, dynamic, self.actor(features, dynamic), value(self.critic, features, dynamic)

    def close(self):
        self.runtime.close()
