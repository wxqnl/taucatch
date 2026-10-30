"""Strict completed-pilot loader, same actor whitelist and native192ms ABI.

This has new dataset provenance. Existing legacy checkpoint dataset identities
must not be forged to route this data through the old training launcher.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .action_data import INPUT_KEYS
from .action_h192_195536 import require_spec
from .action_horizon_data import sha256


class ServoNativeData:
    def __init__(self,folder,split):
        if split not in ('train','val'):
            raise ValueError('Only separate train/val extension splits are supported')
        root=Path(folder);path=root/'manifest.json'
        if not path.exists():
            raise ValueError('Finite pilot is not complete; no partial training')
        manifest=json.loads(path.read_text())
        if not manifest.get('complete') or manifest.get('format')!='single_ball_servo_aligned_pilot_v84' or manifest.get('n')!=8:
            raise ValueError('Expected completed independently audited8-scene pilot')
        require_spec(manifest['action_spec'])
        records=manifest['records'];audits={r['seed']:r for r in manifest['audits']}
        if len(records)!=8 or len(audits)!=8 or len({r['seed'] for r in records})!=8:
            raise ValueError('Missing/duplicate physical scene or independent audit')
        for r in records:
            proof=audits[r['seed']]
            if not proof['accepted'] or sha256(proof['path'])!=proof['sha256'] or sha256(r['report'])!=r['report_sha256']:
                raise ValueError('Teacher report/audit identity mismatch')
            if proof['report_sha256']!=r['report_sha256'] or proof['split']!=r['split']:
                raise ValueError('Audit refers to a different physical scene/split')
        if sum(r['split']=='train_extension' for r in records)!=6 or sum(r['split']=='val_extension' for r in records)!=2:
            raise ValueError('Changed TRAIN/VAL split')
        self.seed_set=set();self.rows=[];self.rollouts=0;self.excluded=0;self.all_rollouts=0;self.all_teacher_caught=0
        self.parts={k:[] for k in (*INPUT_KEYS,'actions','valid')}
        identity=hashlib.sha256((sha256(path)+':'+split).encode())
        for r in records:
            if r['split']!=split+'_extension':continue
            self.all_rollouts+=1;good=bool(r['outcome']['caught']);self.all_teacher_caught+=good
            p=Path(r['dataset']['path'])
            if root.resolve() not in p.resolve().parents or sha256(p)!=r['dataset']['sha256']:
                raise ValueError('Dataset path or identity mismatch')
            identity.update(r['dataset']['sha256'].encode())
            with np.load(p,allow_pickle=False) as z:
                require_spec(json.loads(str(z['action_spec'])))
                if int(z['seed'])!=r['seed'] or str(z['split'])!=r['split']:
                    raise ValueError('Wrong stored scene or split')
                n=len(z['state']);valid=z['valid']
                if z['actions'].shape!=(n,12,7) or valid.shape!=(n,12) or valid.dtype!=bool:
                    raise ValueError('Invalid native target dimensions')
                if z['over'].dtype!=np.uint8 or z['mosaic'].dtype!=np.uint8 or z['state'].dtype!=np.float32:
                    raise ValueError('Changed image or state encoding')
                if np.any(z['rgb_t']+.003>z['observed_t']+1e-9):
                    raise ValueError('RGB readout looks ahead')
                if any(not np.isfinite(z[k]).all() for k in ['state','events','actions']) or np.max(np.abs(z['actions']))>1.00001:
                    raise ValueError('Invalid actor values or targets')
                if np.any(z['actions'][~valid]!=0) or (not good and valid.any()):
                    raise ValueError('Failed or missing imitation labels were not masked')
                keep=np.flatnonzero(valid.any(axis=1))
                if not len(keep):self.excluded+=1;continue
                self.seed_set.add(r['seed']);self.rollouts+=1
                for k in self.parts:self.parts[k].append(z[k][keep].copy())
                self.rows.extend((str(z['text']),str(p),int(i)) for i in keep)
        if not self.rows:raise ValueError('No usable successful teacher labels in '+split)
        self.parts={k:np.concatenate(v) for k,v in self.parts.items()};self.fingerprint=identity.hexdigest()

    def __len__(self):return len(self.rows)

    def batch(self,indices):
        ix=np.asarray(list(indices),dtype=np.int64)
        x={k:torch.from_numpy(self.parts[k][ix]) for k in INPUT_KEYS}
        x['text']=[self.rows[i][0] for i in ix]
        return x,torch.from_numpy(self.parts['actions'][ix]),torch.from_numpy(self.parts['valid'][ix])

    def summary(self):
        return dict(format='single_ball_servo_aligned_pilot_v84',samples=len(self),seeds=len(self.seed_set),
                    retained_rollouts=self.rollouts,excluded_failed_or_empty_rollouts=self.excluded,
                    all_teacher_caught=self.all_teacher_caught,all_rollouts=self.all_rollouts,dataset_sha256=self.fingerprint)


def load_servo_native_data(folder):
    data={s:ServoNativeData(folder,s) for s in ['train','val']}
    if data['train'].seed_set & data['val'].seed_set:raise ValueError('Training/validation leakage')
    return data
