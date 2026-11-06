"""Expert demonstrations for the SpikeVLA catcher.

CPU workers roll out the true-state expert and log the joint state every 4 ms; the main process
renders both event cameras from those states on one GPU (EGL) and writes shards.

Per toss, every 20 ms policy step stores state8 and a 12 x 7 label chunk at 20 ms spacing:
grasp-target offset from the current grasp centre (3), rotation target relative to the current hand
rotation as a rotation vector (3), and a finger ramp that crosses 0.5 at the expert's close time.
Events are stored sparse per 4 ms slice: pixel index, positive and negative counts.
With --distractors (selective catching) the shards also hold the instruction, the ball colours and the
RGB frames of both RGB cameras at 30 Hz (random phase per toss).

    MUJOCO_EGL_DEVICE_ID=7 python -m taucatch.vla_catch.collect --out data_v2/bc --seed 0 --tosses 2000
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .events import CAMERAS, RES, EventStream

TUNE = None                # parsed arguments of the run, read by the workers
EV_DT = 0.004              # event slice
STEP = 0.020               # policy step and chunk spacing
HORIZON = 12
PRE = 0.04                 # policy steps start this long before release
POST = 0.12                # and end this long after the planned arrival
RGB_DT = 1 / 30            # RGB frame interval


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = np.arccos(c)
    if a < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - a < 1e-4:          # near pi: axis from the symmetric part
        k = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0)) * np.sign(w + 1e-12)
        return a * k / np.linalg.norm(k)
    return a * w / (2 * np.sin(a))


ARM = "openarm"
DISTRACTORS = 0


def make_sim(arm, distractors=0):
    if arm == "b601":
        from .b601 import B601Sim
        return B601Sim(distractors=distractors)
    if arm == "openarm_v1":
        from .openarm_v1 import OpenArmV1Sim
        return OpenArmV1Sim(distractors=distractors)
    from .sim import CatchSim
    return CatchSim(distractors=distractors)


def chunk_labels(times, commands, t, g, R, t_close):
    """12 x 7 label chunk at time t from the expert's command stream (times, (t, target, rot, close))."""
    labels = np.zeros((HORIZON, 7), np.float32)
    mask = np.zeros(HORIZON, bool)
    for k in range(HORIZON):
        tk = t + k * STEP
        if tk > times[-1] + 1e-9:
            break
        _, target, rot, _ = commands[int(np.searchsorted(times, tk + 1e-9)) - 1]
        labels[k, :3] = target - g
        labels[k, 3:6] = rotvec(rot @ R.T)
        labels[k, 6] = 0.0 if t_close is None else np.clip(0.5 + (tk - t_close) / STEP, 0.0, 1.0)
        mask[k] = True
    return labels, mask


def tune(sim):
    """Fast throws, catcher and motors of this run, when asked for."""
    a = TUNE
    if a is None:
        return
    if a.throw_speed:
        sim.set_throws(a.throw_speed, a.throw_from)
    if a.bounce:
        sim.set_bounce(a.bounce)
    if a.volts:
        sim.limit_motors(a.volts)
    if a.blend is not None:
        sim.EXPERT_BLEND = a.blend
    if a.lead is not None:
        sim.EXPERT_LEAD = a.lead


def simulate(seed):
    """Expert rollout of one toss; returns the 4 ms joint states and the policy-step records."""
    from .expert import Expert
    from .rollout import CONTROL_EVERY, PHYSICS_DT, run_expert
    global _SIM, _EXPERT
    if "_SIM" not in globals():
        _SIM = make_sim(ARM, DISTRACTORS)
        tune(_SIM)
        _EXPERT = Expert(_SIM)
    sim, expert = _SIM, _EXPERT
    rng = np.random.default_rng(seed)
    toss = sim.sample_scene(rng) if DISTRACTORS else sim.sample_toss(rng)
    release = toss.hold
    t_first = release - PRE
    t_last = release + toss.flight + POST
    ev_every = int(round(EV_DT / PHYSICS_DT))
    commands, qpos, steps = [], [], []
    close_time = [None]

    def on_tick(t, released, target, rot, close):
        commands.append((t, target.copy(), rot.copy(), close))
        if close and close_time[0] is None:
            close_time[0] = t
        k = int(round(t / PHYSICS_DT))
        if k % ev_every == 0:
            qpos.append(sim.d.qpos.copy())
        if k % int(round(STEP / PHYSICS_DT)) == 0 and t_first - 1e-9 <= t <= t_last + 1e-9:
            steps.append((t, sim.state8(), sim.grasp_point().copy(), sim.site_rot().copy()))

    assert ev_every % CONTROL_EVERY == 0
    caught = run_expert(sim, expert, toss, on_tick)
    times = np.array([c[0] for c in commands])
    t_close = close_time[0]
    chunks = [chunk_labels(times, commands, t, g, R, t_close) for t, _, g, R in steps]
    labels = np.stack([c[0] for c in chunks])
    mask = np.stack([c[1] for c in chunks])
    return dict(
        seed=seed, caught=caught, release=release, flight=toss.flight,
        launch=toss.launch, velocity=toss.velocity, target=toss.target,
        others=[[np.asarray(x).tolist() for x in o] for o in toss.others], colors=list(toss.colors),
        instruction=toss.instruction, close_time=-1.0 if t_close is None else t_close,
        step_times=np.array([s[0] for s in steps], np.float32),
        state8=np.stack([s[1] for s in steps]).astype(np.float32),
        labels=labels, mask=mask, qpos=np.stack(qpos),
    )


def as_toss(r):
    from .sim import Toss
    return Toss(np.asarray(r["launch"]), np.asarray(r["velocity"]), np.asarray(r["target"]), r["flight"],
                r["release"], tuple(tuple(np.asarray(x) for x in o[:2]) + (o[2],) for o in r.get("others", ())),
                tuple(r.get("colors", ())), r.get("instruction", "catch the ball"))


def set_state(sim, qpos, toss, j):
    """Scene at slice j: the logged joint state, distractors on their paths."""
    sim.d.qpos[:] = qpos[j]
    if toss.others:
        sim.move_distractors(toss, j * EV_DT)


def render_events(stream, qpos, seed, first, last, toss=None):
    """Sparse events of each camera for slices first..last (slice j covers ((j-1)*4 ms, j*4 ms])."""
    sim = stream.sim
    toss = toss or as_toss({"launch": 0, "velocity": 0, "target": 0, "flight": 0, "release": 0})
    sim.paint(toss)
    set_state(sim, qpos, toss, first - 1)
    stream.reset(seed)
    parts = {cam: [] for cam in CAMERAS}
    for j in range(first, last + 1):
        set_state(sim, qpos, toss, j)
        for cam, (pix, pos, neg) in zip(CAMERAS, stream.slice()):
            parts[cam].append((np.full(pix.size, j, np.uint16), pix, pos, neg))
    return {cam: tuple(np.concatenate(x) for x in zip(*p)) for cam, p in parts.items()}


def render_rgb(stream, qpos, toss, seed, t_from, t_to):
    """RGB frames of both RGB cameras every RGB_DT (phase drawn per toss), each from the scene at its slice.
    -> times [n], frames [n, 2, RES, RES, 3] uint8."""
    import mujoco
    from .sim import RGB_CAMERAS
    sim, r = stream.sim, stream.r
    phase = np.random.default_rng((seed, 30)).uniform(0, RGB_DT)
    times = np.arange(t_from - RGB_DT + phase, t_to, RGB_DT)
    times = times[times >= 0]
    sim.paint(toss)
    frames = np.zeros((len(times), len(RGB_CAMERAS), RES, RES, 3), np.uint8)
    for i, t in enumerate(times):
        set_state(sim, qpos, toss, min(int(t / EV_DT), len(qpos) - 1))
        mujoco.mj_kinematics(sim.m, sim.d)
        mujoco.mj_camlight(sim.m, sim.d)
        for c, cam in enumerate(RGB_CAMERAS):
            r.update_scene(sim.d, camera=cam)
            frames[i, c] = r.render()
    return times.astype(np.float32), frames


def write_shard(path, records):
    arrays, meta = {}, []
    offsets = {"steps": [0], **{c: [0] for c in CAMERAS}}
    if "rgb" in records[0]:
        offsets["rgb"] = [0]
    for r in records:
        if "rgb" in r:
            arrays.setdefault("rgb_times", []).append(r["rgb"][0])
            arrays.setdefault("rgb", []).append(r["rgb"][1])
            offsets["rgb"].append(offsets["rgb"][-1] + len(r["rgb"][0]))
        for key in ("step_times", "state8", "labels", "mask", "labels_true", "mask_true"):
            if key in r:
                arrays.setdefault(key, []).append(r[key])
        offsets["steps"].append(offsets["steps"][-1] + len(r["step_times"]))
        for cam in CAMERAS:
            sl, px, po, ne = r["events"][cam]
            for key, val in (("slice", sl), ("pix", px), ("pos", po), ("neg", ne)):
                arrays.setdefault(f"{cam}_{key}", []).append(val)
            offsets[cam].append(offsets[cam][-1] + len(sl))
        meta.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
                     if k in ("seed", "caught", "release", "flight", "launch", "velocity", "target", "close_time",
                              "dagger", "others", "colors", "instruction")})
    (np.savez_compressed if "rgb" in arrays else np.savez)(path, **{k: np.concatenate(v) for k, v in arrays.items()},
             **{f"offset_{k}": np.array(v, np.int64) for k, v in offsets.items()},
             meta=np.array(json.dumps(meta)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tosses", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arm", default="openarm", choices=["openarm", "openarm_v1", "b601"])
    ap.add_argument("--distractors", type=int, default=0, help="selective catching: balls besides the named one")
    ap.add_argument("--throw-speed", type=float, nargs="+", default=None, help="flat throws: release speed, m/s (one value or low, high)")
    ap.add_argument("--throw-from", type=float, nargs="+", default=[3.0], help="with --throw-speed: metres ahead of the ready point")
    ap.add_argument("--bounce", type=float, default=None, help="damping ratio of ball on cup")
    ap.add_argument("--cup-depth", type=float, default=None, help="m; B601 cup")
    ap.add_argument("--cup-radius", type=float, default=None, help="m; B601 cup, a straight tube")
    ap.add_argument("--catcher", default=None, choices=["cup", "hand"], help="what the B601 catches with")
    ap.add_argument("--hand-rate", type=float, default=285.0, help="deg/s per finger joint")
    ap.add_argument("--hand-open", type=float, default=0.26, help="rad the fingers wait bent back")
    ap.add_argument("--lead", type=float, default=None, help="s before arrival the hand starts to close")
    ap.add_argument("--volts", type=int, default=None, choices=[24, 48], help="hold the arm to its motors' torque-speed line")
    ap.add_argument("--blend", type=float, default=None, help="how far the expert turns the catcher to face the ball")
    args = ap.parse_args()
    global ARM, DISTRACTORS, TUNE
    ARM, DISTRACTORS, TUNE = args.arm, args.distractors, args
    if args.cup_depth:
        from . import b601
        b601.CUP_DEPTH = args.cup_depth
    if args.cup_radius:
        from . import b601
        b601.CUP_RADIUS = b601.CUP_MOUTH = args.cup_radius
    if args.catcher:
        from . import b601
        b601.CATCHER, b601.HAND_RATE, b601.HAND_OPEN = args.catcher, np.radians(args.hand_rate), args.hand_open
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.tosses))
    records, shard, caught, t0 = [], 0, 0, time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:     # fork before any GL context exists
        stream = EventStream(make_sim(args.arm, args.distractors))
        for i, r in enumerate(pool.imap(simulate, seeds, chunksize=2)):
            first = int(round(r["step_times"][0] / EV_DT)) - int(round(STEP / EV_DT)) - 1
            last = int(round(r["step_times"][-1] / EV_DT))
            toss = as_toss(r)
            r["events"] = render_events(stream, r["qpos"], r["seed"], first, last, toss)
            if args.distractors:
                r["rgb"] = render_rgb(stream, r["qpos"], toss, r["seed"], r["step_times"][0], r["step_times"][-1])
            del r["qpos"]
            caught += r["caught"]
            records.append(r)
            if len(records) == args.shard or i == len(seeds) - 1:
                write_shard(out / f"shard_{args.seed:07d}_{shard:03d}.npz", records)
                records, shard = [], shard + 1
                print(f"{i + 1}/{len(seeds)} caught {caught} ({caught / (i + 1):.1%}) "
                      f"{(time.time() - t0) / (i + 1):.2f} s/toss", flush=True)
    os._exit(0)          # skip EGL teardown errors at interpreter exit


if __name__ == "__main__":
    main()
