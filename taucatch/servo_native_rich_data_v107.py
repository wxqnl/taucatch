"""New48-scene provenance; never disguise richer data as the old8pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .action_data import INPUT_KEYS
from .action_h192_195536 import require_spec
from .action_horizon_data import sha256
from .servo_native_data_v88 import ServoNativeData
from .native_servo_space_v90 import SERVO_CONTRACT


class RichServoNativeData(ServoNativeData):
    def __init__(self,folder,split):
        if split not in ('train','val'):raise ValueError('Separate train/val required')
        root=Path(folder);path=root/'manifest.json'
        if not path.exists():raise ValueError('Finite rich48 collection is not complete; no partial training')
        m=json.loads(path.read_text())
        if not m.get('complete') or m.get('format')!='single_ball_servo_rich_v105' or m.get('n')!=48 or m.get('servo_contract')!=SERVO_CONTRACT:raise ValueError('Wrong rich48 provenance/executor')
        require_spec(m['action_spec']);records=m['records'];audits={r['seed']:r for r in m['audits']}
        if len(records)!=48 or len(audits)!=48 or len({r['seed'] for r in records})!=48:raise ValueError('Missing/duplicate scene or audit')
        if sum(r['split']=='train_extension' for r in records)!=36 or sum(r['split']=='val_extension' for r in records)!=12:raise ValueError('Changed36TRAIN/12VAL split')
        self.seed_set=set();self.rows=[];self.rollouts=0;self.excluded=0;self.all_rollouts=0;self.all_teacher_caught=0
        self.parts={k:[] for k in (*INPUT_KEYS,'actions','valid')};identity=hashlib.sha256((sha256(path)+':'+split).encode())
        for r in records:
            proof=audits[r['seed']]
            if not proof['accepted'] or sha256(proof['path'])!=proof['sha256'] or sha256(r['report'])!=r['report_sha256'] or proof['report_sha256']!=r['report_sha256'] or proof['split']!=r['split']:raise ValueError('Changed independent physical-scene audit')
            if r['split']!=split+'_extension':continue
            self.all_rollouts+=1;good=bool(r['outcome']['caught']);self.all_teacher_caught+=good;p=Path(r['dataset']['path'])
            if root.resolve() not in p.resolve().parents or sha256(p)!=r['dataset']['sha256']:raise ValueError('Dataset path or identity mismatch')
            identity.update(r['dataset']['sha256'].encode())
            with np.load(p,allow_pickle=False) as z:
                require_spec(json.loads(str(z['action_spec'])));n=len(z['state']);valid=z['valid']
                if int(z['seed'])!=r['seed'] or str(z['split'])!=r['split'] or bool(z['teacher_caught'])!=good:raise ValueError('Wrong scene/split/teacher label')
                if z['actions'].shape!=(n,12,7) or valid.shape!=(n,12) or valid.dtype!=bool or z['actions'].dtype!=np.float32:raise ValueError('Changed native label contract')
                if z['over'].shape!=(n,224,224,3) or z['mosaic'].shape!=(n,224,224,3) or z['over'].dtype!=np.uint8 or z['mosaic'].dtype!=np.uint8 or z['state'].shape!=(n,8) or z['state'].dtype!=np.float32 or z['events'].shape!=(n,2,6,60,80):raise ValueError('Changed sensor contract')
                if np.any(np.diff(z['observed_t'])<=0) or np.any(z['rgb_t']+.003>z['observed_t']+1e-9):raise ValueError('Noncausal sensor delivery')
                if any(not np.isfinite(z[k]).all() for k in ['state','events','actions']) or np.max(np.abs(z['actions']))>1.00001 or np.any(z['actions'][~valid]!=0) or (not good and valid.any()):raise ValueError('Invalid actor/labels/failure mask')
                keep=np.flatnonzero(valid.any(axis=1))
                if not len(keep):self.excluded+=1;continue
                self.seed_set.add(r['seed']);self.rollouts+=1
                for k in self.parts:self.parts[k].append(z[k][keep].copy())
                self.rows.extend((str(z['text']),str(p),int(i)) for i in keep)
        if not self.rows:raise ValueError('No usable successful teacher labels in '+split)
        self.parts={k:np.concatenate(v) for k,v in self.parts.items()};self.fingerprint=identity.hexdigest()

    def summary(self):
        return dict(super().summary(),format='single_ball_servo_rich_v105')


def load_rich_servo_native_data(folder):
    data={s:RichServoNativeData(folder,s) for s in ['train','val']}
    if data['train'].seed_set & data['val'].seed_set:raise ValueError('Training/validation leakage')
    return data
