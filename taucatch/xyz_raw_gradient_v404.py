"""Causal saved inputs and a differentiable DVS/decoder/XYZ path.

This is an offline training adapter, not the deployed runtime. RGB/text remain
frozen; the original native7D output is discarded. No feature detach is used.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .action_horizon_data import sha256
from .native_frozen_condition_v108 import cached_native_prediction
from .xyz_coarse_rpc_v311 import unpack


def load_saved_input(record):
    feature = Path(record['feature_record'])
    assert sha256(feature) == record['feature_record_sha256']
    metadata_path = feature.with_suffix('.json')
    r = json.loads(metadata_path.read_text())
    assert r['op'] == 'fast'
    assert sha256(r['input_record']) == r['input_record_sha256']
    sample, _, op, episode = unpack(Path(r['input_record']).read_bytes())
    assert op == 'fast'
    condition_path = feature.parent / ('%05d.json' % r['condition_record_id'])
    cond = json.loads(condition_path.read_text())
    assert cond['op'] == 'condition' and cond['episode'] == episode
    assert sha256(cond['feature_record']) == cond['feature_record_sha256']
    assert cond['encoded_ns'] <= sample['preparation_complete_ns'] < sample['release_ns']
    frozen = torch.load(cond['feature_record'], map_location='cpu', weights_only=False)
    reference = torch.load(feature, map_location='cpu', weights_only=False)
    return dict(condition=frozen['condition'].detach(), padding=frozen['padding'].detach(),
                robot=torch.from_numpy(np.asarray(sample['state'], np.float32))[None],
                events=torch.from_numpy(np.asarray(sample['events'], np.float32))[None],
                context=torch.from_numpy(np.asarray(sample['context'], np.float32))[None],
                reference_features=reference['features'].detach(),
                input_sha256=r['input_record_sha256'], condition_sha256=cond['feature_record_sha256'])


def set_gradient_scope(model, actor):
    model.eval().requires_grad_(False)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(('event_encoder.', 'vla.action_head.'))
                         and not name.startswith('vla.action_head.decoder.action_projection.'))
    actor.eval().requires_grad_(True)
    return dict(native_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                XYZ_trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
                RGB_text_frozen=True, unused_absolute7D_projection_frozen=True,
                running_statistics_frozen=True)


def differentiable_xyz(model, actor, sample, device):
    captured = []
    projection = model.vla.action_head.decoder.action_projection
    original = projection.forward_step

    def capture(features, state, t):
        captured.append(features.clone())
        return original(features, state, t)

    projection.forward_step = capture
    try:
        cached_native_prediction(model, sample['condition'].to(device), sample['padding'].to(device),
                                 sample['robot'].to(device), sample['events'].to(device))
    finally:
        projection.forward_step = original
    assert len(captured) == 4
    features = torch.stack(captured)
    return actor(features, sample['context'].to(device)), features
