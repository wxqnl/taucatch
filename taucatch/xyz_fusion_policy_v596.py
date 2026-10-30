"""Zero-initialized nonlinear fusion of native VLA motion features and live context."""
import torch
from torch import nn
from .xyz_policy_v139 import XYZActor as BaseXYZActor

FUSION_SPEC=dict(name='native_feature_context_nonlinear_residual_v596',native_dim=256,context_dim=22,hidden_dim=128,activation='SiLU',feature_pool='mean_spike_time_per_horizon',initial_output='exact_zero',base_XYZ_readout_and_context_frozen=True)
KIND='native_xyz_raw_fusion_SFT_v596'


class FusionXYZActor(BaseXYZActor):
    def __init__(self,checkpoint):
        super().__init__(checkpoint)
        self.fusion=nn.Sequential(nn.LayerNorm(278),nn.Linear(278,128),nn.SiLU(),nn.Linear(128,3))
        nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)
        self.freeze_base();self.eval()

    def freeze_base(self):
        self.readout.requires_grad_(False);self.context.requires_grad_(False)
        self.fusion.requires_grad_(True)

    def load_base_state(self,state):
        own=self.state_dict();assert set(state)=={k for k in own if not k.startswith('fusion.')}
        own.update(state);self.load_state_dict(own,strict=True)

    def forward(self,features,context):
        if features.ndim!=4 or tuple(features.shape[2:])!=(12,256)or context.shape!=(features.shape[1],22):raise ValueError('Expected native[S,B,12,256] andcontext[B,22]')
        if not torch.isfinite(features).all()or not torch.isfinite(context).all():raise ValueError('Nonfinite fusion inputs')
        if features.shape[1]>1:return torch.cat([self(features[:,i:i+1],context[i:i+1])for i in range(features.shape[1])])
        state={};logits=torch.stack([self.readout.forward_step(features[t],state,t)for t in range(features.shape[0])]).mean(0)
        motion=features.mean(0);dynamic=context[:,None,:].expand(-1,12,-1)
        delta=self.fusion(torch.cat((motion,dynamic),-1))
        return torch.tanh(logits+self.context(context)[:,None,:]+delta)
