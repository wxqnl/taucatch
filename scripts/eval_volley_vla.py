"""Selective catching with the chooser in the loop (plan §5.10, stage 3).

Each volley gets one instruction (val or test templates); the named ball is the instruction's
answer. The event tracker follows every ball. The RGB camera (240 fps, 2 ms exposure, 3 ms
readout) gives a frame; the tracks in flight at that frame are the candidates; the chooser's
answer takes effect after its latency (simulation time) and binds the catcher to that track. The
chooser is asked at every new frame until two answers in a row name the same track (or until
150 ms after the first release).

Choosers: "vla" (scripts/chooser_server.py must be running), "oracle" (the named ball's track as
soon as it exists), "color" (a color filter on the crops; it can only read color instructions).
Usage: python scripts/eval_volley_vla.py tennis --balls 3 --n 64 --workers 24 --chooser vla --latency 0.006
       (--extra-delay 0.05 or 0.1 acts as a slower VLA; --split test for unseen wording and colors)
"""
import argparse
import dataclasses
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from taucatch.catcher import CatcherConfig
from taucatch.choice import MAX_CANDIDATES, NONE, REF_RADIUS, SHIRTS, instructions, state_vector, views
from taucatch.predictor import HORIZONS
from taucatch.rgb import RGBCam
from taucatch.scene import WINDOW_X
from taucatch.throws import window_crossing
from taucatch.volley import COLORS, VolleySim, sample_volley
from taucatch.volley_tracker import VolleyTracker

STOP_ASKING = 0.15      # s after the first release
_W = {}


def _init(ball, n_balls, thrower, arm, source, finder, chooser, sigma, types=None, hold=0.5, prepare_instruction=False,
          latency_source='server', reset_text_cache=False):
    torch.set_num_threads(1)
    cfg = CatcherConfig() if sigma is None else dataclasses.replace(CatcherConfig(), sigma_lo=sigma[0], sigma_hi=sigma[1])
    sim = VolleySim(ball, n_balls, cfg=cfg, arm=arm, thrower=thrower, types=types)
    client = None
    if chooser == "vla":
        from taucatch.vla_link import ChooserClient
        client = ChooserClient()
    _W.update(sim=sim, tracker=VolleyTracker(sim, finder, source, oracle=False), rgb=RGBCam(sim), client=client,
              chooser=chooser, hold=hold, prepare_instruction=prepare_instruction,
              latency_source=latency_source, reset_text_cache=reset_text_cache)


def volley_facts(sim, volley, shirts, thrower):
    balls = []
    for i, th in enumerate(volley.throws):
        cross, t_cross = window_crossing(th.p0, th.v0, th.w0, sim.ball_types[i].params)
        balls.append(dict(color=volley.colors[i], type=sim.ball_types[i].name, shirt=shirts[i], side_y=float(th.p0[1]),
                          released=True, speed=float(np.linalg.norm(th.v0)), arrival=float(volley.release[i] + t_cross),
                          aim=(float(cross[1]), float(cross[2]))))
    return {"thrower": thrower, "balls": balls}


def track_candidates(tracker, geom, t_frame, rng):
    """The tracks in flight at the frame time as chooser candidates (random mark order, at most 4)."""
    # RGB readout is delayed. Use only event evidence available at exposure time.
    entry = next((l for l in reversed(tracker.log) if l["t"] <= t_frame + 1e-9), None)
    if entry is None:
        return []
    cands = []
    for tid, snap in entry["candidates"].items():
        if snap["hits"] < 3:
            continue
        u, v, z = geom.project(entry["tracks"][tid])
        if not (z > 0.2 and 0 <= u < geom.width and 0 <= v < geom.height):
            continue
        path = snap["path"]
        ahead = np.nonzero(path[:, 0] <= WINDOW_X)[0]
        stop = int(ahead[0]) + 1 if len(ahead) else len(path)
        pu, pv, pz = geom.project(path[:stop])
        cands.append(dict(uv=np.array([u, v]), r_px=geom.f * REF_RADIUS / z, released=True, track_id=tid,
                          speed=snap["speed"],
                          time_to_window=float(HORIZONS[stop - 1]) if len(ahead) else float(HORIZONS[-1]),
                          path_uv=[(a, b) for a, b, c in zip(pu, pv, pz) if c > 0.2]))
    order = rng.permutation(len(cands))[:MAX_CANDIDATES]
    return [cands[i] for i in order]


def color_filter(text, frame, cands):
    """Pick the candidate whose crop center is nearest in color to the color word (None if no color word)."""
    word = next((c for c in COLORS if c in text.split()), None)
    if word is None or not cands:
        return None
    target = np.array(COLORS[word][:3]) * 255
    dist = []
    for c in cands:
        u, v = int(round(c["uv"][0])), int(round(c["uv"][1]))
        patch = frame[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3].reshape(-1, 3).astype(float)
        dist.append(np.min(np.linalg.norm(patch - target, axis=1)) if len(patch) else np.inf)
    return int(np.argmin(dist))


def oracle_choice(sim, tracker, ids):
    target = sim.ball_pos(sim.volley.target)
    distances = {tr.id: np.linalg.norm(tr.ekf.x[:3] - target) for tr in tracker.tracks}
    # An exposure may contain a track that was lost during readout. Keep mark indices intact.
    d = [distances.get(tid, np.inf) for tid in ids]
    return int(np.argmin(d)) if d and min(d) < .15 else NONE


def _run(args, frame_cb=None):
    seed, split, latency, extra = args
    sim, tr, rgb, client, chooser = _W["sim"], _W["tracker"], _W["rgb"], _W["client"], _W["chooser"]
    rng = np.random.default_rng(seed)
    volley = sample_volley(rng, sim.ball.name, sim.n, types=[b.name for b in sim.ball_types])
    shirts = list(rng.choice(list(SHIRTS), sim.n, replace=False))
    sim.dress([SHIRTS[c] for c in shirts])
    ins = instructions(volley_facts(sim, volley, shirts, sim.thrower), rng, split, k=1)
    text, named, kind = ins[0]
    reset = client.clear_text_cache() if chooser == 'vla' and _W.get('reset_text_cache', False) else None
    preparation, preparation_ms = None, None
    if chooser == "vla" and _W.get("prepare_instruction", False):
        if hasattr(client, 'prepare_detailed'):
            preparation = client.prepare_detailed(text)
            preparation_ms = preparation['prepared_ms']
            if reset is not None and preparation.get('text_cache') != dict(hits=0, misses=1):
                raise RuntimeError('Isolated preflight preparation did not encode exactly one cold instruction')
        else:
            preparation_ms = client.prepare(text)
    volley.target = named
    st = dict(last=None, pending=[], done=False, bound=[], bind_t=None, correct=None, queries=0, vla_ms=[],
              query_details=[])

    def apply(s, t, k, ids):
        if k == NONE or k < 0 or k >= len(ids):
            st["bound"].append(None)
            return
        if not any(track.id == ids[k] for track in tr.tracks):
            st["bound"].append(None)
            return
        tr.bind(ids[k])
        st["bound"].append(ids[k])
        if st["bind_t"] is None:
            st["bind_t"] = t
        st["correct"] = tr.bound_on_target(s)
        if len(st["bound"]) >= 2 and st["bound"][-1] == st["bound"][-2]:
            st["done"] = True

    def update(s, t):
        tr.step(s, t)
        if st["done"] and not any(track.id == tr.bound for track in tr.tracks):
            st["done"] = False
            rgb.reset()
        for item in [p for p in st["pending"] if t >= p[0] - 1e-9]:
            st["pending"].remove(item)
            apply(s, t, *item[1:])
        if st["done"] or t > STOP_ASKING:
            return
        rgb.step(s, t)
        # One chooser serves one frame at a time within this volley. Frames may arrive while
        # it computes, but they cannot all finish after the same fixed delay in parallel.
        if st["pending"]:
            return
        f = rgb.latest(t)
        if f is None or f[0] == st["last"] or not tr.log:
            return
        st["last"] = f[0]
        cands = track_candidates(tr, rgb.geom, f[0], rng)
        if not cands:
            return
        ids = [c["track_id"] for c in cands]
        over, mosaic = views(f[1], cands)
        st["overview"], st["mark_ids"] = over, ids
        detail = None
        if chooser == "vla":
            if hasattr(client, 'choose_detailed'):
                detail = client.choose_detailed(text, over, mosaic, state_vector(cands))
                scores, ms = detail['scores'], detail['ms']
                if reset is not None:
                    cold = st['queries'] == 0 and preparation_ms is None
                    expected = dict(hits=int(not cold), misses=int(cold))
                    if detail.get('text_cache') != expected:
                        raise RuntimeError(f'Isolated in-flight text-cache provenance failed: {detail}')
            else:
                scores, ms = client.choose(text, over, mosaic, state_vector(cands))
            st["vla_ms"].append(ms)
            k = int(np.argmax(scores))
        elif chooser == "oracle":
            k = oracle_choice(s, tr, ids)
        else:
            k = color_filter(text, f[1], cands)
            k = int(rng.integers(len(cands))) if k is None else k
        st["queries"] += 1
        compute = latency
        if chooser == "vla" and latency is None:
            source = _W.get('latency_source', 'server')
            measured_ms = detail['roundtrip_ms'] if source == 'roundtrip' else ms
            if not np.isfinite(measured_ms) or measured_ms < 0:
                raise RuntimeError('Invalid measured chooser latency')
            compute = measured_ms / 1000
        if detail is not None:
            st['query_details'].append(dict(detail, issued_t=float(t), frame_t=float(f[0]),
                                            mark_ids=ids, selected_mark=k, charged_delay_ms=1000*(compute+extra)))
        st["pending"].append((t + compute + extra, k, ids))

    def cb(s, t):
        update(s, t)
        if frame_cb is not None:
            frame_cb(s, t, tr, rgb, st, text)

    tr.rig.set_config(tr.rig.cfg)
    tr.reset()
    rgb.reset()
    res = sim.run(volley, 0.0, hold=_W.get("hold", 0.5), frame_cb=cb, state_fn=tr.path_state)
    res.update(seed=seed, kind=kind, text=text, correct=st["correct"], queries=st["queries"],
               choice_t=None if st["bind_t"] is None else st["bind_t"] - volley.release[named],
               vla_ms=float(np.median(st["vla_ms"])) if st["vla_ms"] else None)
    res["vla_times_ms"] = st["vla_ms"]
    res["instruction_prepare_ms"] = preparation_ms
    res["instruction_prepared_before_release"] = preparation_ms is not None
    res['instruction_preparation'] = preparation
    res['text_cache_reset_before_trial'] = reset
    res['query_details'] = st['query_details']
    res['latency_source'] = 'fixed hypothetical delay' if latency is not None else _W.get('latency_source', 'server')
    res.update(types=[b.name for b in sim.ball_types], target=named, target_type=sim.ball_types[named].name)
    res["hold_s"] = _W.get("hold", 0.5)
    return res


def summarize(res):
    by = defaultdict(list)
    for r in res:
        by["all"].append(r)
        by[r["kind"]].append(r)
    lines = ["kind        n   catch  right choice  no choice  choice after release  wrong touch"]
    for k, rs in sorted(by.items(), key=lambda x: (x[0] != "all", x[0])):
        ct = [r["choice_t"] for r in rs if r["choice_t"] is not None]
        lines.append(f"{k:10s} {len(rs):3d}  {np.mean([r['caught'] for r in rs]):5.0%}  "
                     f"{np.mean([bool(r['correct']) for r in rs]):12.0%}  {np.mean([r['choice_t'] is None for r in rs]):9.0%}"
                     f"  {1000 * np.median(ct) if ct else np.nan:17.0f} ms  {np.mean([r['wrong_touch'] for r in rs]):11.0%}")
    ms = [r["vla_ms"] for r in res if r["vla_ms"] is not None]
    if ms:
        lines.append(f"chooser compute (server, median): {np.median(ms):.1f} ms")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ball", nargs="?", default="tennis")
    ap.add_argument("--balls", type=int, default=3)
    ap.add_argument("--types", nargs="+", choices=["tennis", "pingpong"], help="ball type per throw; overrides --balls")
    ap.add_argument("--thrower", default="launcher")
    ap.add_argument("--arm", default="agile", choices=["fast", "panda", "agile"])
    ap.add_argument("--source", default="fused:outputs/models/core_spread_v1.pt")
    ap.add_argument("--finder", default="outputs/models/finder_v2.pt")
    ap.add_argument("--chooser", default="vla", choices=["vla", "oracle", "color"])
    ap.add_argument("--latency", type=float, default=None,
                    help="s, explicit hypothetical compute delay; default: server measurement for VLA, 0.006 for baselines")
    ap.add_argument('--latency-source', choices=['server', 'roundtrip'], default='server',
                    help='measured VLA delay: server compute or full local client request, excluding connection/startup')
    ap.add_argument('--reset-text-cache', action='store_true',
                    help='isolated diagnostics only: clear shared cache before each trial; requires exclusive server and one worker')
    ap.add_argument("--extra-delay", type=float, default=0.0, help="s, added to act as a slower chooser")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--sigma", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--hold", type=float, default=0.5, help="s retained after contact; 0.3 reproduces historical tables")
    ap.add_argument("--prepare-instruction", action="store_true",
                    help="instruction is known before release; encode it separately and report that time")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=700000)
    ap.add_argument("--out", type=Path, help="save configuration and per-volley outcomes as JSON")
    a = ap.parse_args()
    if a.types:
        a.balls = len(a.types)
    if a.reset_text_cache and (a.workers != 1 or a.chooser != 'vla'):
        ap.error('text-cache reset requires one VLA worker and an exclusive server')
    if a.latency_source == 'roundtrip' and (a.latency is not None or a.chooser != 'vla'):
        ap.error('roundtrip charging requires measured VLA latency, without --latency')
    if a.out and a.out.exists():
        ap.error('output evidence exists; choose a new --out path')
    if not (0 < a.hold < np.inf):
        ap.error("hold must be positive and finite")
    if a.latency is None and a.chooser != "vla":
        a.latency = 0.006
    if (a.latency is not None and (not np.isfinite(a.latency) or a.latency < 0)) or not (0 <= a.extra_delay < np.inf):
        ap.error("latency and extra delay must be finite and nonnegative")
    jobs = [(a.seed0 + i, a.split, a.latency, a.extra_delay) for i in range(a.n)]
    res = []
    def save_results(complete=False, error=None):
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                           timing=f'{a.latency_source} measured' if a.latency is None else 'fixed hypothetical delay',
                           complete=complete, results=res)
            if error is not None:
                payload.update(error=repr(error), next_uncollected_seed=a.seed0 + len(res))
            temp = a.out.with_suffix(a.out.suffix + '.tmp')
            temp.write_text(json.dumps(payload, indent=2))
            temp.replace(a.out)
    save_results()
    try:
        with mp.Pool(a.workers, initializer=_init,
                     initargs=(a.ball, a.balls, a.thrower, a.arm, a.source, a.finder, a.chooser, a.sigma, a.types,
                               a.hold, a.prepare_instruction, a.latency_source, a.reset_text_cache)) as pool:
            for result in pool.imap(_run, jobs, chunksize=1):
                res.append(result)
                save_results()
                print(f'completed {len(res)}/{a.n} seed={result["seed"]} caught={result["caught"]} '
                      f'right={result["correct"]}', flush=True)
    except BaseException as error:
        save_results(error=error)
        raise
    save_results(complete=True)
    print(f"{a.ball} balls={a.balls} thrower={a.thrower} chooser={a.chooser} latency={a.latency} "
          f"extra={a.extra_delay} split={a.split} sigma={a.sigma} n={a.n} hold={a.hold}")
    print(summarize(res))
