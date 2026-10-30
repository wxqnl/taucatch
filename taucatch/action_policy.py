"""SpikeVLA's native action decoder conditioned directly on RGB, language, DVS and proprioception.

No ChoiceTap, choice classifier, target IDs or analytic trajectory output are used here.
The runtime adapter packs DVS tensors alongside robot state only to reuse the existing runtime ABI;
inside the policy the native state encoder receives exactly eight robot-state coordinates.
"""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F

from .action_chunk import ActionSpec
from .chooser import _paths,PRETRAINED,TEXT_PROJECTION

EVENT_SHAPE=(2,6,60,80)


class EventTokens(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(6,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                                nn.AdaptiveAvgPool2d((4,5)))
        self.project=nn.Linear(64,hidden)
        # Average pooling alone loses where a tiny ball lies inside a pooled patch.
        # Preserve raw event-mass centroids for each polarity, without detecting/choosing objects.
        self.moments=nn.Linear(6,hidden,bias=False)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,60),torch.linspace(-1,1,80),indexing='ij')
        self.register_buffer('xy',torch.stack((xx,yy))[None],persistent=False)
        self.position=nn.Parameter(torch.zeros(1,40,hidden))
        self.norm=nn.LayerNorm(hidden)
        nn.init.normal_(self.position,std=.01)

    def forward(self,events):
        b=events.shape[0]
        raw=events.reshape(b*2,6,60,80).float()
        x=self.conv(raw)
        x=x.flatten(2).transpose(1,2).reshape(b,40,64)
        # Python-list advanced indexing materializes a CPU index tensor and copies
        # it to CUDA during capture. Basic slices keep this operation on device.
        mass=torch.stack((raw[:,0],raw[:,3]),dim=1).clamp_min(0)
        pooled=F.adaptive_avg_pool2d(mass,(4,5))
        centroid_x=F.adaptive_avg_pool2d(mass*self.xy[:,0:1],(4,5))/pooled.clamp_min(1e-6)
        centroid_y=F.adaptive_avg_pool2d(mass*self.xy[:,1:2],(4,5))/pooled.clamp_min(1e-6)
        spatial=torch.cat((pooled,centroid_x,centroid_y),1).flatten(2).transpose(1,2).reshape(b,40,6)
        return self.norm(self.project(x)+self.moments(spatial)+self.position)


class DVSActionPolicy(nn.Module):
    def __init__(self,vla,spec=ActionSpec()):
        super().__init__();self.vla=vla;self.spec=spec
        if vla.config.action.horizon!=spec.horizon or vla.config.action.action_dim!=7:
            raise ValueError('Native pretrained action shape differs from the task contract')
        hidden=vla.action_head.decoder.action_queries.weight.shape[-1]
        self.event_encoder=EventTokens(hidden)

    def forward(self,instructions,pixels,robot,events,frozen_text_features=None):
        if robot.shape[-1]!=8 or tuple(events.shape[1:])!=EVENT_SHAPE:
            raise ValueError('Expected state8 and stereo6x60x80 DVS tensors')
        condition,features=self.vla.encode_condition(instructions,{'pixel_values':pixels},
            return_distillation_features=True,frozen_text_features=frozen_text_features)
        tokens=self.event_encoder(events).to(condition.dtype)
        memory=torch.cat((condition,tokens),1)
        text_valid=features['text_valid']
        visual_len=condition.shape[1]-text_valid.shape[1]
        padding=torch.cat((torch.zeros((len(robot),visual_len),dtype=torch.bool,device=robot.device),
                           ~text_valid,torch.zeros((len(robot),40),dtype=torch.bool,device=robot.device)),1)
        return self.vla.action_head(memory,robot,memory_key_padding_mask=padding)


class ActionRuntimeAdapter(nn.Module):
    def __init__(self,policy):super().__init__();self.policy=policy
    @property
    def config(self):return self.policy.vla.config
    @property
    def spike_projectors(self):return self.policy.vla.spike_projectors
    @property
    def text_encoder(self):return self.policy.vla.text_encoder
    @property
    def vision_encoder(self):return self.policy.vla.vision_encoder
    @property
    def vision_projection(self):return self.policy.vla.vision_projection
    @property
    def action_head(self):return self.policy.vla.action_head
    @property
    def action_dim(self):return 7
    def _normalize_samples(self,samples):return self.policy.vla._normalize_samples(samples)
    def forward(self,instructions,pixels,packed_state,frozen_text_features=None):
        robot=packed_state[:,:8]
        events=packed_state[:,8:].reshape(-1,*EVENT_SHAPE)
        return self.policy(instructions,self._normalize_samples(pixels),robot,events,frozen_text_features)


def pack_state(robot,events):return torch.cat((robot,events.flatten(1).to(robot.dtype)),1)


def build_action_policy(run_dir='/tmp/taucatch_action_policy'):
    _paths()
    from experiments.pretrain.common import build_initialized_model,load_recipe,pretraining_args
    from spikevla.data import prepare_images
    vla=build_initialized_model(pretraining_args(load_recipe(),run_dir,TEXT_PROJECTION))
    ck=torch.load(PRETRAINED,map_location='cpu',weights_only=False)
    state=ck.get('ema_model_state_dict') or ck['model_state_dict']
    vla.load_state_dict(state,strict=True)
    return DVSActionPolicy(vla),prepare_images


def action_runtime(policy,backend='cached',spike_mode='none',**overrides):
    _paths()
    from turbovla.evaluation.runtime import InferenceRuntime
    optimized=backend not in ('cached','trace_graph')
    options=dict(batch_views=optimized,cache_neurons=optimized,hoist_action_memory=optimized,gpu_preprocess=optimized)
    allowed=set(options)
    if set(overrides)-allowed:raise ValueError('Unknown action runtime option')
    options.update(overrides)
    trace_graph=backend=='trace_graph'
    runtime=InferenceRuntime(ActionRuntimeAdapter(policy).eval(),backend='cuda_graph' if trace_graph else backend,
                             cache_size=128,spike_mode=spike_mode,**options)
    if trace_graph:
        capture=runtime._capture
        base_function=runtime._get_callable()
        def capture_traced(inputs):
            # Preserve ATen operation boundaries and precision; freeze Python control only.
            # Masks, events, images and robot state remain tensor inputs copied per request.
            with torch.jit.optimized_execution(False):
                runtime._callable=torch.jit.trace(base_function,inputs,check_trace=False,strict=False)
                return capture(inputs)
        runtime._capture=capture_traced
        metadata=runtime.metadata
        runtime.metadata=lambda:dict(metadata(),backend='trace_graph',jit_optimized_execution=False)
    return runtime
