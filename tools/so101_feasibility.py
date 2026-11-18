"""Can the SO-101 catch? The true-state expert (the planner every arm uses) on the SO-101 with a cup, low-speed tosses
(2-4 m/s, 0.75-1.25 m in front of the hand, flight 0.45-0.75 s), under variants of the servo model, the cup, the
catch behaviour and the toss range. Every variant runs in fresh processes (so101's module settings do not leak), on
the same seeds. No rendering: about 0.1 s per toss.

    python tools/so101_feasibility.py --tosses 200 --out outputs/checks/so101/feasibility.json
    python tools/so101_feasibility.py --only default "absorb off" --tosses 100
    python tools/so101_feasibility.py --scan-base --tosses 60        # pedestal placement around the ready point

`throws` replaces the drawn flight time by flat throws at a release speed from a distance ahead of the ready point
(sim.set_throws, as collect.py --throw-speed --throw-from). `start` delays what the expert sees: it holds the ready
pose until the ball has flown this long (s), as a tracker
needs time to lock on (SpikeVLA's track holds 45-85 ms after release; 20 ms more for the delay to the arm).
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FULL = ((-0.06, 0.10), (-0.18, 0.18), (-0.14, 0.14))
SIDE12 = ((-0.06, 0.10), (-0.12, 0.12), (-0.14, 0.14))

VARIANTS = {               # name: (so101 module settings, SERVO settings, sim attributes, toss box, start s)
    "default": ({}, {}, {}, FULL, 0.0),
    "axial cup": ({"CUP_MOUNT": "axial"}, {}, {}, FULL, 0.0),
    "axial cup, tracker delay": ({"CUP_MOUNT": "axial"}, {}, {}, FULL, 0.105),
    "soft cup contact (MuJoCo default)": ({"CUP_SOLREF": (0.02, 1.0)}, {}, {}, FULL, 0.0),
    "cup into the ball (blend .75)": ({}, {}, {"EXPERT_BLEND": 0.75}, FULL, 0.0),
    "cup into the ball (blend 1)": ({}, {}, {"EXPERT_BLEND": 1.0}, FULL, 0.0),
    "throws 3 m/s from 1.5 m": ({}, {}, {"throws": ((2.9, 3.1), (1.5,))}, FULL, 0.0),
    "throws 3 m/s from 2 m": ({}, {}, {"throws": ((2.9, 3.1), (2.0,))}, FULL, 0.0),
    "throws 4-5 m/s from 2.5 m": ({}, {}, {"throws": ((4.0, 5.0), (2.5,))}, FULL, 0.0),
    "throws 4-5 m/s from 2.5 m, tracker delay": ({}, {}, {"throws": ((4.0, 5.0), (2.5,))}, FULL, 0.105),
    "throws 5-6 m/s from 3 m": ({}, {}, {"throws": ((5.0, 6.0), (3.0,))}, FULL, 0.0),
    "throws 6-8 m/s from 3 m": ({}, {}, {"throws": ((6.0, 8.0), (3.0,))}, FULL, 0.0),
    "tracker delay (start 105 ms)": ({}, {}, {}, FULL, 0.105),
    "tracker delay, +-12 cm sideways": ({}, {}, {}, SIDE12, 0.105),
    "+-12 cm sideways": ({}, {}, {}, SIDE12, 0.0),
    "absorb off": ({}, {"absorb": 0.0}, {}, FULL, 0.0),
    "absorb 0.3 s": ({}, {"absorb": 0.3}, {}, FULL, 0.0),
    "catch signal 50 ms early": ({}, {}, {"EXPERT_LEAD": 0.05}, FULL, 0.0),
    "catch signal at arrival": ({}, {}, {"EXPERT_LEAD": 0.0}, FULL, 0.0),
    "cup faces the ready way (blend .15)": ({}, {}, {"EXPERT_BLEND": 0.15}, FULL, 0.0),
    "7.4 V supply": ({"SUPPLY": 7.4}, {}, {}, FULL, 0.0),
    "stiffer P loop (sat 1 deg)": ({}, {"sat_deg": 1.0}, {}, FULL, 0.0),
    "softer P loop (sat 8 deg)": ({}, {"sat_deg": 8.0}, {}, FULL, 0.0),
    "less damping (zeta .3)": ({}, {"zeta": 0.3}, {}, FULL, 0.0),
    "heavier rotor (armature .05)": ({}, {"armature": 0.05}, {}, FULL, 0.0),
    "servo goal ramp 40 rad/s2": ({}, {"setpoint_accel": 40.0}, {}, FULL, 0.0),
    "50 Hz bus": ({}, {"send_every": 0.02}, {}, FULL, 0.0),
    "no sag compensation": ({}, {"sag_comp": False}, {}, FULL, 0.0),
    "no wrist camera": ({"WRIST_CAMERA": False}, {}, {}, FULL, 0.0),
    "heavier cup (120 g)": ({"CUP_MASS": 0.12}, {}, {}, FULL, 0.0),
}


def job(args):
    name, module, servo, attrs, box, start, seeds, base = args
    from taucatch.vla_catch import so101
    if base is not None:
        so101.BASE = base
    for k, v in module.items():
        setattr(so101, k, v)
    so101.SERVO.update(servo)
    from taucatch.vla_catch.collect import make_sim
    from taucatch.vla_catch.expert import Expert
    from taucatch.vla_catch.rollout import run_expert
    try:
        sim = make_sim("so101")
    except RuntimeError as e:
        return name, base, None, str(e)
    for k, v in attrs.items():
        if k == "throws":                      # (speed range, distance ahead of the ready point): flat throws
            sim.set_throws(*v)
        else:
            setattr(sim, k, v)
    expert = Expert(sim)
    if start:
        act = expert.act

        def late(released, t=0.0):
            toss = sim.toss_now
            return act(released and t - toss.hold >= start, t)
        expert.act = late
    caught = []
    for seed in seeds:
        toss = sim.sample_toss(np.random.default_rng(seed), box=box)
        sim.toss_now = toss
        caught.append(bool(run_expert(sim, expert, toss)))
    return name, base, caught, None


def run(jobs, workers):
    out = {}
    with mp.Pool(workers, maxtasksperchild=1) as pool:
        for name, base, caught, err in pool.imap_unordered(job, jobs):
            key = (name, base)
            out.setdefault(key, {"caught": [], "error": err})
            if caught is not None:
                out[key]["caught"].extend(caught)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tosses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000000)
    ap.add_argument("--only", nargs="*", default=None, help="variant names")
    ap.add_argument("--scan-base", action="store_true", help="pedestal offsets behind and below the ready point")
    ap.add_argument("--variant", default="default", help="the variant --scan-base places")
    ap.add_argument("--behind", type=float, nargs="*", default=[0.26, 0.30, 0.34, 0.38, 0.42, 0.46])
    ap.add_argument("--below", type=float, nargs="*", default=[0.20, 0.26, 0.32, 0.38, 0.44])
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = list(range(args.seed, args.seed + args.tosses))
    split = max(1, min(10, args.tosses // 20))
    rows = []
    if args.scan_base:
        from taucatch.vla_catch.so101 import READY_POINT
        bases = [(round(READY_POINT[0] - dx, 3), round(READY_POINT[1], 3), round(READY_POINT[2] - dz, 3))
                 for dx in args.behind for dz in args.below]
        v = args.variant
        jobs = [(v, *VARIANTS[v], seeds[k::split], b) for b in bases for k in range(split)]
        res = run(jobs, args.workers)
        for b in bases:
            r = res[(v, b)]
            dx, dz = READY_POINT[0] - b[0], READY_POINT[2] - b[2]
            rate = np.mean(r["caught"]) if r["caught"] else None
            rows.append({"variant": v, "base": b, "behind": round(dx, 2), "below": round(dz, 2), "caught": rate,
                         "error": r["error"]})
            said = f"caught {rate * 100:5.1f}%" if rate is not None else r["error"]
            print(f"behind {dx:.2f} below {dz:.2f}: {said}", flush=True)
    else:
        names = args.only or list(VARIANTS)
        jobs = [(n, *VARIANTS[n], seeds[k::split], None) for n in names for k in range(split)]
        res = run(jobs, args.workers)
        for n in names:
            c = res[(n, None)]["caught"]
            rows.append({"variant": n, "caught": float(np.mean(c)), "tosses": len(c)})
            print(f"{n:34s} expert caught {np.mean(c) * 100:5.1f}%  ({len(c)} tosses)", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"tosses": args.tosses, "seed": args.seed, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
