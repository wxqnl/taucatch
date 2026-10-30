"""Strictly separated demonstration inputs, future action labels, and scoring metadata."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .action_chunk import ActionSpec

INPUT_KEYS=('over','mosaic','state','events')


class ActionData:
    def __init__(self,folder,split,successful_only=True):
        if split not in ('train','val'):
            raise ValueError('Initial action experiments use train/val only')
        root=Path(folder);manifest=json.loads((root/'manifest.json').read_text())
        if not manifest.get('complete'):
            raise ValueError('Demonstration collection is not complete')
        paths=sorted(root.glob('*.npz'))
        if 'results' in manifest:
            listed={Path(row['file']).name for row in manifest['results']}
            if listed!={p.name for p in paths}:raise ValueError('Manifest demonstration files differ')
        identity=hashlib.sha256((root/'manifest.json').read_bytes())
        for path in paths:
            identity.update(path.name.encode())
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(2**20),b''):identity.update(block)
        self.fingerprint=identity.hexdigest()
        self.rows=[];self.sampling_metadata=[];self.sampling_strata=[];self.parts={k:[] for k in (*INPUT_KEYS,'actions','valid')}
        self.seed_set=set();self.rollouts=0;self.excluded=0;self.all_teacher_caught=0;self.all_rollouts=0
        for path in paths:
            with np.load(path,allow_pickle=False) as z:
                if str(z['split'])!=split:continue
                self.all_rollouts+=1;good=bool(json.loads(str(z['outcome']))['caught'])
                self.all_teacher_caught+=good
                if successful_only and not good:self.excluded+=1;continue
                spec=json.loads(str(z['action_spec']))
                if spec!=json.loads(json.dumps(asdict(ActionSpec()))):
                    raise ValueError(f'Action contract mismatch in {path}')
                n=len(z['state']);target=z['actions'];valid=z['valid'].astype(bool)
                if target.shape!=(n,12,7) or valid.shape!=(n,12):
                    raise ValueError(f'Bad future-action shape in {path}')
                for k in ('state','events','actions'):
                    if not np.isfinite(z[k]).all():raise ValueError(f'Nonfinite {k}: {path}')
                if np.any(z['rgb_t']+.003>z['observed_t']+1e-9):
                    raise ValueError(f'RGB readout looks ahead: {path}')
                if np.max(np.abs(target))>1.00001:raise ValueError(f'Action range: {path}')
                self.seed_set.add(int(z['seed']));self.rollouts+=1
                for k in self.parts:self.parts[k].append(z[k].copy())
                self.rows.extend((str(z['text']),str(path),i) for i in range(n))
                outcome=json.loads(str(z['outcome']));contact=outcome.get('contact_t')
                release=(float(z['release'][int(z['scoring_target'])]) if 'release' in z.files
                         else max(0.,float(outcome.get('t_choose',.02))-.02))
                kind=str(z['kind']) if 'kind' in z.files else 'unknown'
                scene=json.loads(str(z['scene_metadata'])) if 'scene_metadata' in z.files else {}
                profile=scene.get('profile',{})
                stratum=(profile.get('thrower','launcher'),profile.get('speed_bin','legacy'))
                for i,t in enumerate(z['observed_t']):
                    phase=('hold' if contact is not None and t>=contact else 'pre_release' if t<release
                           else 'early' if t<release+.08 else 'flight')
                    self.sampling_metadata.append((kind,int(z['seed']),phase,str(path)))
                    self.sampling_strata.append(stratum)
        if not self.rows:raise ValueError(f'No usable {split} action demonstrations')
        self.parts={k:np.concatenate(v) for k,v in self.parts.items()}

    def __len__(self):return len(self.rows)

    def batch(self,indices):
        ix=np.asarray(list(indices),dtype=np.int64)
        # This explicit whitelist is the only route from stored metadata into the policy.
        x={k:torch.from_numpy(self.parts[k][ix]) for k in INPUT_KEYS}
        x['text']=[self.rows[i][0] for i in ix]
        return x,torch.from_numpy(self.parts['actions'][ix]),torch.from_numpy(self.parts['valid'][ix])

    def summary(self):
        return dict(samples=len(self),seeds=len(self.seed_set),retained_rollouts=self.rollouts,
                    excluded_failed_rollouts=self.excluded,all_teacher_caught=self.all_teacher_caught,
                    all_rollouts=self.all_rollouts,dataset_sha256=self.fingerprint)


def load_action_data(folder,successful_only=True):
    data={s:ActionData(folder,s,successful_only) for s in ('train','val')}
    if data['train'].seed_set & data['val'].seed_set:
        raise ValueError('Volley seed leakage between training and validation')
    return data
