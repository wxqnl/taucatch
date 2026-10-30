"""Exact slow RGB/text condition and a graph with fresh DVS/state/XYZ context.

One complete anchor inference is still charged. This runtime does not make
post-release images available before release and has no sampling/training ABI.
"""
import numpy as np
import torch

from .native_frozen_condition_v108 import cached_native_prediction
from .xyz_learning_v158 import value
from .xyz_runtime_v158 import XYZRuntime

OBSERVATION_CONTRACT = 'causal_RGB_anchor_then_fresh_DVS_state_context_v173'


class TimescaleRuntime:
    def __init__(self, base, artifact, options):
        self.engine = XYZRuntime(base, artifact, options)
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
            if condition.shape != self.condition.shape:
                raise ValueError('Anchor shape changed; construct a new runtime')
            self.condition.copy_(condition)
            self.padding.copy_(padding)
            return condition, features

        self.engine.model.vla.encode_condition = capture_condition

    def anchor(self, text, sample):
        result = self.engine.encode(text, sample)
        self.anchor_observed_t = float(sample['observed_t'])
        self.anchor_rgb_t = float(sample['rgb_t'])
        return result

    def _body(self):
        e = self.engine
        e.capture.fill_(float('nan'))
        cached_native_prediction(e.model, self.condition, self.padding, self.robot, self.events)
        # Same single-item actor operations as XYZActor.forward. Input/output
        # finite checks are fused into one GPU flag, checked after graph replay.
        state = {}
        logits = torch.stack([e.actor.readout.forward_step(e.capture[t], state, t)
                              for t in range(e.capture.shape[0])]).mean(0)
        mean = torch.tanh(logits + e.actor.context(self.dynamic)[:, None, :])
        critic = value(e.critic, e.capture, self.dynamic)
        valid = (torch.isfinite(e.capture).all() & torch.isfinite(mean).all()
                 & torch.isfinite(critic).all())
        return mean, critic, valid

    def build_fast_graph(self, sample):
        if self.graph is not None or self.condition is None:
            raise ValueError('Warm one anchor and build the fast graph exactly once')
        self.robot = torch.as_tensor(sample['state'], dtype=torch.float32, device='cuda')[None].clone()
        self.events = torch.as_tensor(sample['events'], dtype=torch.float32, device='cuda')[None].clone()
        self.dynamic = torch.as_tensor(sample['context'], dtype=torch.float32, device='cuda')[None].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._body()
        return self.fast(sample)

    def fast(self, sample):
        if self.graph is None or self.anchor_observed_t is None:
            raise ValueError('No prepared graph or current anchor')
        for key, shape in [('state', (8,)), ('events', (2, 6, 60, 80)), ('context', (22,))]:
            array = np.asarray(sample[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError('Malformed fast ' + key)
        self.robot.copy_(torch.as_tensor(sample['state'], dtype=torch.float32, device='cuda')[None])
        self.events.copy_(torch.as_tensor(sample['events'], dtype=torch.float32, device='cuda')[None])
        self.dynamic.copy_(torch.as_tensor(sample['context'], dtype=torch.float32, device='cuda')[None])
        self.graph.replay()
        mean, critic, valid = self.output
        if not bool(valid.item()):
            raise ValueError('Nonfinite native features or readout')
        return self.engine.capture.clone(), self.dynamic.clone(), mean.clone(), critic.clone()

    def reference(self, sample):
        """Uncaptured original actor/value on the identical cached RGB condition."""
        e = self.engine
        robot = torch.as_tensor(sample['state'], dtype=torch.float32, device='cuda')[None]
        events = torch.as_tensor(sample['events'], dtype=torch.float32, device='cuda')[None]
        dynamic = torch.as_tensor(sample['context'], dtype=torch.float32, device='cuda')[None]
        e.capture.fill_(float('nan'))
        cached_native_prediction(e.model, self.condition, self.padding, robot, events)
        f = e.capture.clone()
        return f, dynamic, e.actor(f, dynamic), value(e.critic, f, dynamic)

    def reset_episode(self):
        self.anchor_observed_t = self.anchor_rgb_t = None

    def close(self):
        torch.cuda.synchronize()
        self.graph = None
        self.engine.model.vla.encode_condition = self.original_condition
        self.engine.close()
