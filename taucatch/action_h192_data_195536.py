"""Checked192ms future labels; the original causal sensor arrays remain shared."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .action_data import ActionData, INPUT_KEYS
from .action_h192_195536 import SPEC, require_spec
from .action_horizon_data import sha256


def exact_teacher_grid(source, overlay):
    """Reconstruct shared physical references from either TRAIN or VAL labels."""
    with np.load(source,allow_pickle=False) as x,np.load(overlay,allow_pickle=False) as y:
        assert str(x['split']) in ('train','val')
        assert np.array_equal(x['observed_t'],y['observed_t'])
        spec=json.loads(str(y['action_spec']))
        assert spec['dt']==.008 and spec['horizon']==12
        refs={}
        for t,actions,valid in zip(y['observed_t'],y['actions'],y['valid']):
            for index in np.flatnonzero(valid):
                tick=round((t+.008*(index+1))/.001)
                if tick in refs:
                    assert np.array_equal(refs[tick],actions[index]),'Conflicting physical reference'
                else:refs[tick]=actions[index].copy()
    assert all(t in refs for t in range(min(refs),max(refs)+1,8))
    return refs


class Data192:
    def __init__(self, source, overlay, split):
        source, overlay = Path(source), Path(overlay)
        m = json.loads((overlay/'manifest.json').read_text())
        assert m['complete'] and m['format']=='native_h192_exact_teacher_grid_v1'
        require_spec(m['action_spec'])
        assert source.resolve()==Path(m['source_root']).resolve()
        assert sha256(source/'manifest.json')==m['source_manifest_sha256']
        base = ActionData(source, split)
        assert base.fingerprint==m['source_dataset_sha256']
        self.base, self.manifest, self.split = base, m, split
        self.overlay_sha256 = sha256(overlay/'manifest.json')
        for name in ('rows', 'sampling_metadata', 'sampling_strata', 'seed_set', 'rollouts', 'excluded', 'all_teacher_caught', 'all_rollouts'):
            setattr(self, name, getattr(base, name))
        self.parts = {k:base.parts[k] for k in INPUT_KEYS}
        self.parts['actions'] = np.zeros_like(base.parts['actions'])
        self.parts['valid'] = np.zeros_like(base.parts['valid'], dtype=bool)
        groups = {}
        for i, (_, path, local) in enumerate(base.rows):groups.setdefault(path, []).append((i, local))
        expected = {n for n,r in m['label_files'].items() if r['split']==split and r['teacher_caught']}
        assert {Path(p).name for p in groups}==expected
        for path, rows in groups.items():
            filename = Path(path).name
            record = m['label_files'][filename]
            label = overlay/filename
            assert sha256(label)==record['sha256']
            with np.load(label, allow_pickle=False) as z, np.load(path, allow_pickle=False) as raw:
                require_spec(json.loads(str(z['action_spec'])))
                assert np.array_equal(z['observed_t'], raw['observed_t'])
                assert str(raw['split'])==split and int(raw['seed'])==record['seed']
                actions, valid = z['actions'], z['valid']
                n=len(rows)
                assert actions.shape==(n,12,7) and actions.dtype==np.float32 and valid.shape==(n,12) and valid.dtype==bool
                assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1.00001
                assert np.all(actions[~valid]==0) and int(valid.sum())==record['valid_waypoints']
                indices, local=np.asarray(rows).T
                assert np.array_equal(local,np.arange(n))
                self.parts['actions'][indices]=actions
                self.parts['valid'][indices]=valid
        assert len(self)==m['counts'][split]['observations']
        assert int(self.parts['valid'].sum())==m['counts'][split]['valid_waypoints']
        identity=dict(source=base.fingerprint,overlay=self.overlay_sha256,action_spec=asdict(SPEC))
        self.fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def __len__(self):return len(self.rows)

    def batch(self, indices):
        ix=np.asarray(list(indices),dtype=np.int64)
        x={k:torch.from_numpy(self.parts[k][ix]) for k in INPUT_KEYS}
        x['text']=[self.rows[i][0] for i in ix]
        return x,torch.from_numpy(self.parts['actions'][ix]),torch.from_numpy(self.parts['valid'][ix])

    def summary(self):
        return dict(samples=len(self),seeds=len(self.seed_set),retained_rollouts=self.rollouts,
            excluded_failed_rollouts=self.excluded,all_rollouts=self.all_rollouts,
            action_spec=json.loads(json.dumps(asdict(SPEC))),dataset_sha256=self.fingerprint,
            source_dataset_sha256=self.base.fingerprint,overlay_manifest_sha256=self.overlay_sha256,
            parent96_manifest_sha256=self.manifest['parent96_manifest_sha256'],
            valid_waypoints=int(self.parts['valid'].sum()),input_arrays_shared=True)
