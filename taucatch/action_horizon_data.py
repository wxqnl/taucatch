"""Verified label overlay; sensor arrays remain shared with the frozen ActionData."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .action_data import ActionData, INPUT_KEYS
from .action_h96 import SPEC, canonical, require_spec


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


class HorizonOverlay:
    def __init__(self, source, overlay):
        self.source = Path(source).resolve()
        self.root = Path(overlay).resolve()
        self.manifest_sha = sha256(self.root / 'manifest.json')
        self.manifest = m = json.loads((self.root / 'manifest.json').read_text())
        if m.get('format') != 'action_horizon_label_overlay_v1' or not m.get('complete'):
            raise ValueError('Incomplete or unknown horizon overlay')
        require_spec(m['action_spec'])
        if Path(m['source_root']).resolve() != self.source:
            raise ValueError('Overlay source namespace differs')
        if sha256(self.source / 'manifest.json') != m['source_manifest_sha256']:
            raise ValueError('Overlay source manifest differs')
        self.labels = {}
        for relative, row in m['label_files'].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or sha256(path) != row['sha256']:
                raise ValueError(f'Label identity differs: {relative}')
            with np.load(path, allow_pickle=False) as z:
                require_spec(json.loads(str(z['action_spec'])))
                actions, valid, observed = z['actions'], z['valid'], z['observed_t']
                n = row['observations']
                if (actions.shape != (n, 12, 7) or actions.dtype != np.float32 or
                        valid.shape != (n, 12) or valid.dtype != np.bool_ or observed.shape != (n,)):
                    raise ValueError(f'Label shape/dtype differs: {relative}')
                if (not np.isfinite(actions).all() or not np.isfinite(observed).all() or
                        np.any(np.diff(observed) <= 0) or np.max(np.abs(actions)) > 1.00001 or
                        np.any(actions[~valid] != 0)):
                    raise ValueError(f'Invalid overlay values: {relative}')
                if (int(valid.sum()) != row['valid_waypoints'] or
                        int(valid.all(axis=1).sum()) != row['full_horizon_observations']):
                    raise ValueError(f'Label counts differ: {relative}')
                self.labels[relative] = actions, valid, observed

    def attach(self, base, split):
        return HorizonActionData(base, split, self)


class HorizonActionData:
    def __init__(self, base, split, overlay):
        m = overlay.manifest
        if split not in ('train', 'val') or base.fingerprint != m['source_dataset_sha256']:
            raise ValueError('Original demonstration fingerprint differs')
        self.base, self.overlay = base, overlay
        for key in ('rows', 'sampling_metadata', 'sampling_strata', 'seed_set', 'rollouts',
                    'excluded', 'all_teacher_caught', 'all_rollouts'):
            setattr(self, key, getattr(base, key))
        # Preserve every input array by identity; future labels never enter x.
        self.parts = {k: base.parts[k] for k in INPUT_KEYS}
        self.parts['actions'] = np.zeros_like(base.parts['actions'])
        self.parts['valid'] = np.zeros_like(base.parts['valid'], dtype=bool)
        paths = {}
        for i, (_, path, local) in enumerate(self.rows):
            paths.setdefault(path, []).append((i, local))
        expected = {name for name, entry in m['entries'].items() if entry['split'] == split}
        if {Path(p).name for p in paths} != expected:
            raise ValueError('Overlay language rollout coverage differs')
        for path, rows in paths.items():
            if Path(path).resolve().parent != overlay.source:
                raise ValueError('Original rollout source namespace differs')
            entry = m['entries'][Path(path).name]
            file_row = m['label_files'][entry['label_file']]
            if (entry['split'] != split or file_row['split'] != split or
                    entry['trajectory'] != file_row['trajectory']):
                raise ValueError('Overlay physical trajectory mapping differs')
            actions, valid, observed = overlay.labels[entry['label_file']]
            indices, local = np.asarray(rows).T
            if not np.array_equal(local, np.arange(len(observed))):
                raise ValueError('Overlay observation row mapping differs')
            with np.load(path, allow_pickle=False) as z:
                trajectory = (str(z['trajectory_id']) if 'trajectory_id' in z.files else
                              f'{int(z["seed"])}_{int(z["scoring_target"])}')
                if (not np.array_equal(z['observed_t'], observed) or int(z['seed']) != entry['seed'] or
                        str(z['split']) != split or trajectory != entry['trajectory']):
                    raise ValueError('Overlay observation timeline differs')
            self.parts['actions'][indices] = actions[local]
            self.parts['valid'][indices] = valid[local]
        counts = m['counts'][split]
        if (len(self) != counts['observations'] or self.rollouts != counts['language_rollouts'] or
                int(self.parts['valid'].sum()) != counts['valid_waypoints'] or
                int(self.parts['valid'].all(axis=1).sum()) != counts['full_horizon_observations']):
            raise ValueError('Overlay split counts differ')
        if self.excluded != sum(r['split'] == split for r in m['excluded_failed']):
            raise ValueError('Failed demonstration retention differs')
        identity = dict(source=base.fingerprint, overlay=overlay.manifest_sha,
                        action_spec=canonical(asdict(SPEC)))
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def batch(self, indices):
        import torch
        ix = np.asarray(list(indices), dtype=np.int64)
        x = {k: torch.from_numpy(self.parts[k][ix]) for k in INPUT_KEYS}
        x['text'] = [self.rows[i][0] for i in ix]
        return x, torch.from_numpy(self.parts['actions'][ix]), torch.from_numpy(self.parts['valid'][ix])

    def summary(self):
        return dict(self.base.summary(), dataset_sha256=self.fingerprint,
                    source_dataset_sha256=self.base.fingerprint,
                    overlay_manifest_sha256=self.overlay.manifest_sha,
                    action_spec=canonical(asdict(SPEC)),
                    valid_waypoints=int(self.parts['valid'].sum()),
                    full_horizon_observations=int(self.parts['valid'].all(axis=1).sum()),
                    input_arrays_shared=True)


def load_horizon_data(source, overlay):
    catalog = HorizonOverlay(source, overlay)
    data = {s: catalog.attach(ActionData(source, s), s) for s in ('train', 'val')}
    if data['train'].seed_set & data['val'].seed_set:
        raise ValueError('Volley seed leakage between training and validation')
    return data
