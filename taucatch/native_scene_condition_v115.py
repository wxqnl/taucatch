"""Per-scene frozen condition reuse, bound to source data and upstream weights."""
from collections import OrderedDict
import json
from pathlib import Path

import torch

from .action_horizon_data import sha256
from .native_frozen_condition_v108 import FrozenConditionCache,encode_frozen_condition


class AssembledConditionCache(FrozenConditionCache):
    def __init__(self,condition,padding):
        self.condition=condition;self.padding=padding


def assemble_condition_cache(model,prepare,data,device,initialization_sha256,frozen_digest,index_path=None):
    records={}
    if index_path is not None:
        index=json.loads(Path(index_path).read_text())
        if not index['complete'] or index['initialization_sha256']!=initialization_sha256 or index['frozen_upstream_sha256']!=frozen_digest:raise ValueError('Wrong precomputed condition index')
        records={r['source_dataset_sha256']:r for r in index['records']}
        if len(records)!=len(index['records']):raise ValueError('Duplicate cached scene')
    groups=OrderedDict()
    for i,(_,path,original) in enumerate(data.rows):groups.setdefault(path,[]).append((i,original))
    conditions=[];padding=[];reused=0;encoded=0;order=[]
    for path,rows in groups.items():
        record=records.get(sha256(path))
        if record is not None:
            if sha256(record['path'])!=record['sha256']:raise ValueError('Changed per-scene condition cache')
            payload=torch.load(record['path'],map_location='cpu',weights_only=False)
            if payload['format']!='frozen_rgb_text_scene_v115' or payload['source_dataset_sha256']!=sha256(path) or payload['initialization_sha256']!=initialization_sha256 or payload['frozen_upstream_sha256']!=frozen_digest:raise ValueError('Wrong per-scene cache provenance')
            if payload['condition'].dtype!=torch.float32 or payload['padding'].dtype!=torch.bool or payload['condition'].shape[:2]!=payload['padding'].shape or len(payload['condition'])!=payload['observations']:raise ValueError('Changed cached feature shape/precision')
            ix=[original for _,original in rows];conditions.append(payload['condition'][ix]);padding.append(payload['padding'][ix]);order.extend(i for i,_ in rows);reused+=1
        else:
            for start in range(0,len(rows),8):
                selected=rows[start:start+8];x,_,_=data.batch([i for i,_ in selected]);c,p=encode_frozen_condition(model,prepare,x,device);conditions.append(c.cpu());padding.append(p.cpu());order.extend(i for i,_ in selected)
            encoded+=1
    if order!=list(range(len(data))):raise ValueError('Cache assembly changed dataset row order')
    cache=AssembledConditionCache(torch.cat(conditions),torch.cat(padding))
    return cache,dict(reused_scenes=reused,newly_encoded_scenes=encoded,rows=len(data))
