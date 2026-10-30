"""Finite single-ball expert/replay check or causal dual-horizon demonstration collection."""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from taucatch.action_chunk import ActionSpec, current_action, reference_action
from taucatch.action_catcher import ActionCatcher
from taucatch.action_h96 import SPEC
from taucatch.action_horizon_data import sha256
from taucatch.single_ball_v1 import INSTRUCTION, PROTOCOL, TRAIN_SEEDS, VAL_SEEDS, setup


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def worker(job):
    seed, split, root, mode = job
    torch.set_num_threads(1)
    sim, volley = setup(seed)
    refs = {}
    samples = []
    observation = None
    if mode == 'data':
        from taucatch.action_observation import ActionObservation
        observation = ActionObservation(sim, seed=seed)
        observation.reset()
    def record(s, t):
        k = round(t / .001)
        refs[k] = reference_action(s, SPEC)
        if observation is not None:
            observation.step(t)
            stride = 16 if s.catcher.contact_t is not None else 8
            if k % stride == 0:
                sample = observation.sample(t)
                if sample is not None:
                    samples.append(sample)
    try:
        teacher = sim.run(volley, .02, hold=.5, frame_cb=record)
    finally:
        if observation is not None:
            observation.close()
    result = dict(seed=seed, split=split, teacher=teacher, speed_mps=volley.throws[0].speed)
    if mode == 'audit':
        end = max(refs)
        # Fixed request times, future privileged labels, 45ms simulated delay.
        # This tests the executor ceiling, never the VLA or sensor pipeline.
        sim.catcher = ActionCatcher(sim.model, sim.ids, sim.ball, sim.home_q, arm='agile', spec=SPEC)
        pending = None
        def replay(s, t):
            nonlocal pending
            k = round(t / .001)
            if pending is not None and k >= pending[0] + 45:
                observed, actions, origin = pending
                s.catcher.buffer.submit(actions, observed*.001, (observed+45)*.001, origin)
                pending = None
            if pending is None and k >= 8 and k % 48 == 8:
                ix = [min(k + 8*(i+1), end) for i in range(12)]
                pending = (k, np.stack([refs[j] for j in ix]), current_action(s, SPEC))
            return None
        result['replay45ms'] = sim.run(volley, 0., hold=.5, state_fn=replay)
        result['replay_chunks'] = sim.catcher.buffer.accepted
        result['replay_stale'] = sim.catcher.buffer.stale
        result['privileged_replay'] = True
        return result
    root = Path(root)
    source = root/'source'/f'{seed}_0.npz'
    overlay = root/'overlay'/f'{seed}_0.npz'
    if source.exists() or overlay.exists():
        raise FileExistsError(source)
    if not samples:
        raise ValueError('No causal samples')
    end = max(refs)
    labels = {}
    for name, spec in [('source', ActionSpec()), ('overlay', SPEC)]:
        actions, valid = [], []
        for sample in samples:
            ix = [round((sample['observed_t'] + (i+1)*spec.dt)/.001) for i in range(12)]
            mask = np.array([j <= end for j in ix], bool)
            chunk = np.stack([refs[min(j, end)] for j in ix])
            if name == 'overlay':
                chunk[~mask] = 0
            actions.append(chunk)
            valid.append(mask)
        labels[name] = dict(actions=np.stack(actions), valid=np.stack(valid),
                            observed_t=np.array([s['observed_t'] for s in samples]),
                            action_spec=json.dumps(asdict(spec)))
    # Identical spatial normalization. Both future horizons sampled from the same 1ms trace.
    np.savez_compressed(source, **{k:np.stack([s[k] for s in samples]) for k in ('over','mosaic','events','state','rgb_t')},
        **labels['source'], text=INSTRUCTION, kind='single_ball', split=split, seed=seed, branch=0,
        scoring_target=0, types=np.array(['tennis']), colors=np.array(['yellow']),
        release=volley.release, outcome=json.dumps(teacher), trajectory_id=f'{seed}_0',
        scene_metadata=json.dumps(dict(profile=dict(thrower='launcher', speed_bin='12-18'), protocol=PROTOCOL)))
    np.savez_compressed(overlay, **labels['overlay'])
    result.update(file=str(source), label_file=overlay.name, samples=len(samples),
                  valid_waypoints=int(labels['overlay']['valid'].sum()),
                  full_horizon_observations=int(labels['overlay']['valid'].all(1).sum()),
                  source_sha256=sha256(source), label_sha256=sha256(overlay))
    return result


def finish_data(root, rows):
    from hashlib import sha256 as digest
    source, overlay = root/'source', root/'overlay'
    save(source/'manifest.json', dict(complete=True, action_spec=asdict(ActionSpec()),
         protocol=PROTOCOL, results=rows, teacher='Privileged state, release+20ms; failures retained',
         observations='Causal RGB/stereo DVS/state8; fixed text; no true ball states'))
    identity = digest((source/'manifest.json').read_bytes())
    for path in sorted(source.glob('*.npz')):
        identity.update(path.name.encode())
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(2**20), b''):
                identity.update(block)
    entries, labels, excluded, counts = {}, {}, [], {}
    for split in ('train','val'):
        good = [r for r in rows if r['split']==split and r['teacher']['caught']]
        counts[split] = dict(observations=sum(r['samples'] for r in good), language_rollouts=len(good),
            valid_waypoints=sum(r['valid_waypoints'] for r in good),
            full_horizon_observations=sum(r['full_horizon_observations'] for r in good))
    for row in rows:
        if not row['teacher']['caught']:
            excluded.append(row)
            continue
        name = Path(row['file']).name
        trajectory = f"{row['seed']}_0"
        entries[name] = dict(split=row['split'], seed=row['seed'], trajectory=trajectory, label_file=row['label_file'])
        labels[row['label_file']] = dict(split=row['split'], trajectory=trajectory, sha256=row['label_sha256'],
            observations=row['samples'], valid_waypoints=row['valid_waypoints'],
            full_horizon_observations=row['full_horizon_observations'])
    save(overlay/'manifest.json', dict(format='action_horizon_label_overlay_v1', complete=True,
        action_spec=asdict(SPEC), source_root=str(source.resolve()), source_manifest_sha256=sha256(source/'manifest.json'),
        source_dataset_sha256=identity.hexdigest(), entries=entries, label_files=labels, counts=counts,
        excluded_failed=excluded, all_attempts=rows, protocol=PROTOCOL))
    from taucatch.action_horizon_data import load_horizon_data
    data = load_horizon_data(source, overlay)
    summary = {s:d.summary() for s,d in data.items()}
    for split in data:
        x, y, valid = data[split].batch([0])
        assert set(x)=={'over','mosaic','events','state','text'} and x['text']==[INSTRUCTION]
        assert y.shape==(1,12,7) and valid.shape==(1,12)
    save(root/'audit.json', dict(complete=True, protocol=PROTOCOL, dataset=summary,
        train_val_disjoint=True, confirmation_not_collected=True, raw_failures_retained=True,
        teacher_caught=sum(r['teacher']['caught'] for r in rows), attempts=len(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('audit','data'), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seeds', type=int, nargs='+')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    assert not (a.root/'progress.json').exists(), 'No overwrite or automatic retry'
    if a.mode=='data':
        for name in ('source','overlay'):
            (a.root/name).mkdir()
        jobs = [(s, split, str(a.root), a.mode) for split, seeds in [('train',TRAIN_SEEDS),('val',VAL_SEEDS)] for s in seeds]
    else:
        jobs = [(s, 'val', str(a.root), a.mode) for s in (a.seeds or VAL_SEEDS)]
    rows = []
    start = time.monotonic()
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(worker, jobs):
            rows.append(row)
            save(a.root/'progress.json', dict(complete=False, expected=len(jobs), results=rows))
            print(json.dumps(dict(done=len(rows), expected=len(jobs), seconds=time.monotonic()-start,
                                  seed=row['seed'], caught=row['teacher']['caught'])), flush=True)
    rows.sort(key=lambda r:r['seed'])
    if a.mode=='data':
        finish_data(a.root, rows)
    save(a.root/'progress.json', dict(complete=True, expected=len(jobs), protocol=PROTOCOL, results=rows,
         teacher_caught=sum(r['teacher']['caught'] for r in rows),
         replay45ms_caught=sum(r.get('replay45ms',{}).get('caught',False) for r in rows)))


if __name__=='__main__':
    main()
