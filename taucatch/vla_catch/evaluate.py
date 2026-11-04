"""Closed-loop catching with a trained SpikeVLA policy in the OpenArm scene.

The policy sees what it was trained on: two event frames (20 ms window) and state8, every 20 ms.
Its 12 x 7 chunk runs at 500 Hz through the expert's task-space tracker: entries are interpolated
in time, and the finger closes when the ramp crosses 0.5 after release and then stays closed.

    MUJOCO_EGL_DEVICE_ID=7 CUDA_VISIBLE_DEVICES=0 python -m taucatch.vla_catch.evaluate \
        --ckpt runs_v2/bc_v1_f7/ema_8000.pt --tosses 200

Render and infer on different GPUs: on one GPU, every render after a CUDA call stalls (0.1-19 s).
A policy call is bound by kernel launches (about 15k per call), so --batch tosses run in lockstep and
share each call; --fast replays each call as a CUDA Graph (fast.py, same outputs as eager).
A policy with the in-model ball locator (locator.py) keeps one track memory per toss; it acquires the ball with
its own token scorer.

With --record DIR the visited states are written as training shards (DAgger): every policy step is
relabelled with the expert's next 12 commands, simulated on a copy of the scene from that state.
With --distractors (selective catching) every toss has further balls of other colours, the instruction names the one to
catch, and the policy also gets the two RGB views (30 Hz, the newest frame RGB_DELAY old).
"""
import argparse
import dataclasses
import json
import os
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import ball as ball_meter
from . import objects
from .collect import EV_DT, HORIZON, POST, PRE, RGB_DT, STEP, chunk_labels, make_sim, write_shard
from .data import RGB_DELAY, WINDOW
from .events import CAMERAS, RES, EventStream, event_frame, trace_frame
from .expert import EstimateSim, Expert
from .fovea import sensor_images
from .ground import Grounder, ball_paths, order_sentence, peak, relation_sentence
from .locator import LOST, SliceTrack, sentence
from .policy import INSTRUCTION, build_model, pixel_values, set_numerics
from .rollout import AFTER, CONTROL_EVERY, PHYSICS_DT, run_expert
from .sim import GRAVITY, RGB_CAMERAS, SENSOR, ballistic, use_kinds

EVAL_SEED = 1_000_000


def rotation(v):
    a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class Policy:
    def __init__(self, ckpt, fast=False, **fast_options):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model, _ = build_model(ck["model_config"], condition=ck.get("condition", False))
        self.locator = ck.get("locator", False)
        if self.locator:
            from .locator import CatchPolicy
            self.model = CatchPolicy(self.model, ck["stats"], ck.get("selective", False))
            self.model.no_image = ck.get("no_image", False)
            self.model.anchor = ck.get("anchor", False)
        self.model.load_state_dict(ck["ema_state_dict"])
        self.model.cuda().eval()
        self.stats = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
        self.step = ck["step"]
        self.frames = ck.get("frames", "window")
        self.ball = ck.get("ball", False)
        self.no_velocity = ck.get("no_velocity", False)
        self.selective = ck.get("selective", False)         # RGB views and the toss's instruction
        self.graphs = {} if fast else None       # one CUDA Graph per batch size
        self.fast_options = fast_options         # fast.GraphPolicy: compile, batch_views, half, autotune
        self.tracks = {}                         # locator memory per toss

    def reset(self):
        self.tracks = {}

    @torch.no_grad()
    def __call__(self, frames, state8, ids=None, instructions=None, gate=None):
        """frames [B, V, H, W, 3] uint8, state8 [B, S] -> chunks [B, 12, 7] in physical units.
        ids name the tosses whose locator memories this call reads and updates; gate [B, 2, 2] (px) replaces the
        locator's scorer for acquisition (eager)."""
        if not (self.selective and instructions):            # single-ball policies keep their trained instruction
            instructions = [INSTRUCTION] * len(frames)
        instructions = list(instructions)
        s = self.stats
        n = state8.shape[-1]
        if self.no_velocity:
            state8 = state8.copy()
            state8[:, 3:6] = 0.0
        inputs = [torch.from_numpy(frames).cuda(),
                  torch.from_numpy((state8 - s["state_mean"][:n]) / s["state_std"][:n]).float().cuda()]
        if self.locator:
            m = self.model.locator.memory
            empty = (torch.zeros(m, 3, device="cuda"), torch.zeros(m, dtype=torch.bool, device="cuda"))
            inputs += [torch.stack(x) for x in zip(*(self.tracks.get(i, empty) for i in ids))]
        if self.graphs is None or gate is not None:
            if gate is not None:
                inputs.append(torch.from_numpy(gate).cuda())
            out = self.model(instructions, pixel_values(inputs[0]), *inputs[1:])
            out = (out[0], out[2], out[3]) if self.locator else (out,)
        else:
            from .fast import GraphPolicy
            if len(frames) not in self.graphs:
                self.graphs[len(frames)] = GraphPolicy(self.model, len(frames), frames.shape[1:], **self.fast_options)
            with torch.inference_mode():
                out = self.graphs[len(frames)](*inputs, instructions=instructions)
            out = out if self.locator else (out,)
        if self.locator:
            for i, memory, valid in zip(ids, out[1], out[2]):
                self.tracks[i] = (memory, valid)
            locator = self.model.locator
            self.track = [x.float().cpu().numpy() for x in locator.fit(out[1].float(), out[2], locator.ages[:-1])]
        a = out[0].float().cpu().numpy()
        out = a * s["action_std"] + s["action_mean"]
        out[..., 6] = (a[..., 6] + 1) / 2
        return out


def expert_relabel(sim, toss, t, closed):
    """The expert's label chunk from the current state, simulated on a copy of the scene."""
    m, live, q_ref = sim.m, sim.d, sim.q_ref.copy()
    g, R = sim.grasp_point().copy(), sim.site_rot().copy()
    fork = mujoco.MjData(m)
    mujoco.mj_copyData(fork, m, live)
    sim.d = fork
    expert = Expert(sim)
    expert.reset()
    expert.closed = closed
    release = int(round(toss.hold / PHYSICS_DT))
    k0 = int(round(t / PHYSICS_DT))
    commands, t_close = [], (t - 1.0 if closed else None)
    for k in range(k0, k0 + HORIZON * int(round(STEP / PHYSICS_DT))):
        if k < release:
            sim.place_ball(toss.launch)
        elif k == release:
            sim.place_ball(toss.launch, toss.velocity)
        if k % CONTROL_EVERY == 0:
            target, rot, close = expert.act(k >= release, k * PHYSICS_DT)
            commands.append((k * PHYSICS_DT, target.copy(), rot.copy(), close))
            if close and t_close is None:
                t_close = k * PHYSICS_DT
            sim.track(target, rot, CONTROL_EVERY * PHYSICS_DT)
            sim.set_finger(close)
        mujoco.mj_step(m, fork)
    sim.d, sim.q_ref = live, q_ref
    return chunk_labels(np.array([c[0] for c in commands]), commands, t, g, R, t_close)


def rollout(sim, stream, toss, seed, latency=0.0, record=False, oracle=False, mode="window", ball=False, rgb_views=False,
            ensemble=0.0, teacher_gate=False, close_track=False, event_gate=None, selected=False, erase_others=False,
            clock=None, raw_views=False, plan=False, erase_tracks=False, reader=None, mix=None, slice_track=None,
            query_every=None):
    """One toss as a generator: yields (frames, state8) at every policy step and is sent the chunk.
    oracle: execute the expert's label chunk from the current state instead (checks the action interface).
    ensemble: grasp target and finger value averaged over the chunks that cover the moment, weight exp(-age / ensemble)
    by query age (0: the newest chunk alone); orientation from the newest chunk.
    teacher_gate: also yield the training-time acquisition gate (named ball's true projection + noise) per step.
    close_track: the gripper closes the expert's lead before the policy's own ball track (sent with each chunk as
    (position, velocity, measurements)) meets the hand's planned path, not on the chunk's finger channel.
    event_gate: (band, disk, look) px: drop events outside a disk around the named ball (its true place until the track
    holds, then the track) and a band along the arm, so the policy sees a one-ball scene.
    selected: a selector acquires the ball: it sends its peak with each chunk, and until the track holds the gate is a
    disk of radius look around that peak (look 0: open).
    erase_others: no events from the distractors at all (what a perfect gate would leave).
    clock: a one-item list that holds the time of the current policy step.
    raw_views: the two trace frames of the ungated events follow the RGB views, for a selector that reads them.
    plan: the expert's planner runs on the policy's ball track and drives the arm; the action head's output is unused.
    erase_tracks: a multi-ball event tracker follows everything that flies, and once the policy holds its ball the events
    around the other tracks are dropped (not where one crosses the policy's ball in the image).
    reader: the backbone reads the sensor crops around the tracks; the track it picks acquires the policy's ball.
    mix: "rot" / "pos": until the closure the planner on the policy's ball track gives the hand's orientation / grasp
    target and the action head the rest (which half of the head's output loses).
    slice_track: a locator.SliceTrack that measures the ball at every event slice once the policy's track holds.
    query_every: s between policy calls once the slice track feeds the locator memory (its fit rewrites the memory at
    every call, so the calls need not keep the STEP grid of the model's own track); STEP before that."""
    sim.reset_arm()
    sim.paint(toss)
    rgb_times = deque()                    # RGB exposures still to come, and the frames taken so far
    if rgb_views:
        phase = np.random.default_rng((seed, 30)).uniform(0, RGB_DT)
        rgb_times.extend(np.arange(toss.hold - PRE - RGB_DT + phase, toss.hold + toss.flight + POST, RGB_DT))
    rgb = []
    end = toss.hold + toss.flight + AFTER
    n = int(round(end / PHYSICS_DT))
    release = int(round(toss.hold / PHYSICS_DT))
    step_every = int(round(STEP / PHYSICS_DT))
    query_k = int(round((query_every or STEP) / PHYSICS_DT))
    ev_every = int(round(EV_DT / PHYSICS_DT))
    t_first, t_last = toss.hold - PRE, toss.hold + toss.flight + POST
    k_first = int(np.ceil((t_first - 1e-9) / STEP)) * step_every
    j_first = k_first // ev_every - WINDOW - 1           # first event slice, as in collect.py
    slices = deque(maxlen=WINDOW)
    chunks, closed, close_time, queries, gap_other, approach_err = [], False, None, 0, None, None
    kept, steps, drift, gap_min, gap_close = [], [], [], np.inf, None
    track, gate_rng = [], np.random.default_rng((seed, 99))
    arrival, locks, named_gap = None, [], []     # time the track meets the hand's plan; per step ball followed, its gap
    track_start = None                           # first step with a usable track, s after release
    held, band = None, [np.ones(RES * RES, bool) for _ in CAMERAS]   # event gate: (t, position, velocity) of the track
    looking = None                               # the selector's newest peak [2, 2] px
    view, planner, planned = EstimateSim(sim), None, None
    tracker, recent = None, deque(maxlen=2)
    if erase_tracks or reader is not None:
        from .baseline import ALL_SEEN, ARM_BAND, ARM_STILL, Tracker
        tracker = Tracker()
    shots, chosen, first_seen = deque(), None, None        # sensor frames not read yet, the track picked
    shot_times = deque()
    if reader is not None:
        earliest = min([toss.hold] + [o[2] for o in toss.others])
        shot_times.extend(np.arange(earliest, toss.hold + toss.flight, RGB_DT))
        shoot = np.random.default_rng((seed, 31))
    if plan or mix:
        planner = Expert(view)
        planner.reset()
    for k in range(n):
        t = k * PHYSICS_DT
        released = k >= release
        if k < release:
            sim.place_ball(toss.launch)
        elif k == release:
            sim.place_ball(toss.launch, toss.velocity)
        if toss.others:
            sim.move_distractors(toss, t)
        while rgb_times and rgb_times[0] <= t + 1e-9:          # exposure: the scene now, both RGB cameras
            mujoco.mj_kinematics(sim.m, sim.d)
            mujoco.mj_camlight(sim.m, sim.d)
            pair = []
            for cam in RGB_CAMERAS:
                stream.r.update_scene(sim.d, camera=cam)
                pair.append(stream.r.render().copy())
            rgb.append((rgb_times.popleft(), np.stack(pair)))
        while shot_times and shot_times[0] <= t + 1e-9:
            shots.append((shot_times.popleft(), [objects.expose(x, sim.gain, shoot) for x in sensor_images(sim, sim.sensor)]))
        if k % ev_every == 0:
            j = k // ev_every
            if j == j_first - 1:
                stream.reset(seed)
                last = [np.full(RES * RES, -np.inf) for _ in CAMERAS]
                raw = [np.full(RES * RES, -np.inf) for _ in CAMERAS]
            elif j >= j_first:
                events = stream.slice()
                if raw_views:
                    for ci in range(len(CAMERAS)):
                        raw[ci][events[ci][0]] = j
                if tracker is not None:
                    recent.append(events)
                    if np.linalg.norm(sim.grasp_velocity()) > ARM_STILL:
                        pts = arm_points(sim)
                        masks = [arm_band(cam, pts, ARM_BAND) for cam in CAMERAS]
                    else:
                        masks = [np.zeros(RES * RES, bool) for _ in CAMERAS]
                    fresh = [(np.concatenate([s[ci][0] for s in recent]).astype(np.int64),
                              np.concatenate([s[ci][1].astype(np.float64) + s[ci][2] for s in recent]))
                             for ci in range(len(CAMERAS))]
                    tracker.update(t, fresh, masks)
                    if reader is not None and chosen is None:
                        live = tracker.confirmed()
                        first_seen = t if first_seen is None and live else first_seen
                        while shots and shots[0][0] <= t - RGB_DELAY + 1e-6:
                            taken, images = shots.popleft()
                            if live:
                                reader(live, taken, images, toss.instruction)
                        if live and (len(live) > len(toss.others) or t - first_seen > ALL_SEEN):
                            chosen = reader.pick(live, t - first_seen > ALL_SEEN + 0.1)
                    if held is not None:
                        own = ballistic(held[1], held[2], t - held[0])
                        rest = [x.predict(t) for x in tracker.confirmed() if np.linalg.norm(x.predict(t) - own) > 0.08]
                        kept_events = []
                        for e, cam in zip(events, CAMERAS):
                            uv = ball_meter.project(cam, own)
                            away = [q for q in (ball_meter.project(cam, p) for p in rest) if np.linalg.norm(q - uv) > 12.0]
                            kept_events.append(erase_events(e, away, 12.0) if away else e)
                        events = kept_events
                if erase_others:
                    others = [sim.d.mocap_pos[i] for i in sim.mocap[:len(toss.others)]]
                    events = [erase_events(e, [ball_meter.project(cam, p) for p in others])
                              for e, cam in zip(events, CAMERAS)]
                if event_gate and held is None and selected:
                    if event_gate[2]:
                        centres = [None] * len(CAMERAS) if looking is None else looking
                        events = [keep_events(e, c, b, event_gate[2]) for e, c, b in zip(events, centres, band)]
                elif event_gate:
                    if held is None:
                        tb = np.clip(t - toss.hold, 0.0, toss.flight)
                        gate_ball = np.asarray(toss.launch) + np.asarray(toss.velocity) * tb + 0.5 * GRAVITY * tb ** 2
                    else:
                        dt = t - held[0]
                        gate_ball = held[1] + held[2] * dt + 0.5 * GRAVITY * dt ** 2
                    events = [keep_events(e, ball_meter.project(cam, gate_ball), b, event_gate[1])
                              for e, cam, b in zip(events, CAMERAS, band)]
                slices.append((j, events))
                for ci in range(len(CAMERAS)):
                    last[ci][slices[-1][1][ci][0]] = j
                if slice_track is not None:          # pixels whose latest event is in slice j-1 or j, as the locator's
                    slice_track.add(t - EV_DT, np.stack([(j - a) < 2 for a in last]).reshape(len(CAMERAS), RES, RES))
                if record:
                    kept.append(slices[-1])
        on_grid = k % step_every == 0 or (k % query_k == 0 and slice_track is not None and slice_track.ready())
        if on_grid and k >= k_first and t <= t_last + 1e-9:
            frames = []
            for ci in range(2):
                if mode == "trace":
                    frames.append(trace_frame(((slices[-1][0] - last[ci]) * EV_DT).astype(np.float32)))
                    continue
                pix = np.concatenate([s[ci][0] for _, s in slices])
                pos = np.concatenate([s[ci][1] for _, s in slices])
                neg = np.concatenate([s[ci][2] for _, s in slices])
                newest = np.concatenate([np.full(s[ci][0].size, jj == slices[-1][0]) for jj, s in slices])
                frames.append(event_frame(pix, pos, neg, newest))
            if rgb_views:
                ready = [f for tr, f in rgb if tr <= t - RGB_DELAY + 1e-6]
                frames += list(ready[-1] if ready else np.zeros((len(RGB_CAMERAS), RES, RES, 3), np.uint8))
            if raw_views:
                frames += [trace_frame(((slices[-1][0] - r) * EV_DT).astype(np.float32)) for r in raw]
            mujoco.mj_forward(sim.m, sim.d)          # state8 from fresh kinematics, as collect.py logs it
            state8 = sim.state8()
            if clock is not None:
                clock[0] = t
            state = state8
            if ball:                                  # ball.py measurement from the two newest slices
                t_obs = (slices[-1][0] - 1) * EV_DT
                tb = np.clip(t_obs - toss.hold, 0.0, toss.flight)
                truth = np.asarray(toss.launch) + np.asarray(toss.velocity) * tb + 0.5 * GRAVITY * tb ** 2
                events = [(np.concatenate([s[ci][0] for _, s in list(slices)[-2:]]),
                           np.concatenate([s[ci][1].astype(np.float64) + s[ci][2] for _, s in list(slices)[-2:]]))
                          for ci in range(len(CAMERAS))]
                p = ball_meter.measure(events, ball_meter.gate_centres(truth, gate_rng))
                if p is not None:
                    track.append((t_obs, p))
                state = np.concatenate([state8, ball_meter.features(track, t, state8[:3])])
            if reader is not None:
                gate = np.zeros((len(CAMERAS), 2), np.float32)
                if chosen is not None:
                    gate = np.array([ball_meter.project(cam, chosen.predict(t)) for cam in CAMERAS], np.float32)
                    looking = gate
                actions = yield np.stack(frames), state, gate
            elif teacher_gate:
                t_obs = (slices[-1][0] - 1) * EV_DT
                tb = np.clip(t_obs - toss.hold, 0.0, toss.flight)
                truth = np.asarray(toss.launch) + np.asarray(toss.velocity) * tb + 0.5 * GRAVITY * tb ** 2
                gate = []
                for cam in CAMERAS:
                    a, r = gate_rng.uniform(0, 2 * np.pi), gate_rng.uniform(0, ball_meter.ACQUIRE_NOISE)
                    gate.append(ball_meter.project(cam, truth) + r * np.array([np.cos(a), np.sin(a)]))
                actions = yield np.stack(frames), state, np.array(gate, np.float32)
            else:
                actions = yield np.stack(frames), state
            if event_gate:
                pts = arm_points(sim)
                band = [arm_band(cam, pts, event_gate[0]) for cam in CAMERAS]
            if isinstance(actions, tuple):
                actions, estimate, *peaks = actions
                if peaks and peaks[0].any():             # all zero: this toss kept its track and did not look
                    looking = peaks[0]
                if event_gate:
                    held = (t, estimate[0], estimate[1]) if estimate[2] >= 3 else None
                # "hand": when the track meets the hand where it is now, not the chunk's planned targets
                arrival = track_arrival(t, sim.grasp_point(), 0 * actions if close_track == "hand" else actions, *estimate)
                if slice_track is not None and not slice_track.active and estimate[2] >= 3:
                    slice_track.start(t, np.asarray(estimate[0], float), np.asarray(estimate[1], float))
                if track_start is None and estimate[2] >= 4:
                    track_start = t - toss.hold
                if estimate[2] >= 3:
                    planned = (t, estimate[0], estimate[1])
                if toss.others and estimate[2] >= 4:     # which ball the track follows
                    mujoco.mj_kinematics(sim.m, sim.d)
                    balls = [sim.d.xpos[sim.ball]] + [sim.d.mocap_pos[i] for i in sim.mocap[:len(toss.others)]]
                    gaps = [np.linalg.norm(estimate[0] - b) for b in balls]
                    arm = min(np.linalg.norm(estimate[0] - p) for p in arm_points(sim))
                    locks.append(int(np.argmin(gaps)) if min(gaps) < 0.1 else -2 if arm < 0.15 else -1)
                    named_gap.append(float(gaps[0]))
            if record or oracle:
                label, mask = expert_relabel(sim, toss, t, closed)
                drift.append(float(np.linalg.norm(actions[0, :3] - label[0, :3])))
                if record:
                    steps.append((t, state8, label, mask))
                if oracle:
                    actions = label
            chunks.append((t + latency, t, sim.grasp_point().copy(), sim.site_rot().copy(), actions))
            queries += 1
        if k % CONTROL_EVERY == 0:
            active = [c for c in chunks if c[0] <= t + 1e-9]
            if not active:
                target, rot, close = sim.ready_grasp, sim.ready_rot, False
            else:
                _, t0, g, R, A = active[-1]
                a = chunk_at(A, (t - t0) / STEP)
                target, rot, finger = g + a[:3], rotation(a[3:6]) @ R, a[6]
                if ensemble > 0:
                    cover = [c for c in active[-HORIZON:] if t - c[1] <= (HORIZON - 1) * STEP + 1e-9] or active[-1:]
                    w = np.array([np.exp(-(t0 - c[1]) / ensemble) for c in cover])
                    part = np.stack([np.r_[c[2] + chunk_at(c[4], (t - c[1]) / STEP)[:3],
                                           chunk_at(c[4], (t - c[1]) / STEP)[6]] for c in cover])
                    mean = w @ part / w.sum()
                    target, finger = mean[:3], mean[3]
                close = closed or (released and finger > 0.5)
                if close_track:
                    close = closed or (released and arrival is not None and t >= arrival - sim.EXPERT_LEAD)
            if planner is not None:
                if plan:
                    target, rot, close = sim.ready_grasp, sim.ready_rot, False
                if planner.closed:
                    view.estimate = sim.grasp_point().copy(), np.zeros(3)
                elif planned is not None:
                    view.estimate = ballistic(planned[1], planned[2], t - planned[0]), planned[2] + GRAVITY * (t - planned[0])
                if released and view.estimate is not None and plan:
                    target, rot, close = planner.act(released, t)
                elif released and view.estimate is not None and not closed:
                    aim, turn, _ = planner.act(released, t)
                    planner.closed = False                 # the policy's rule closes the hand
                    target, rot = (target, turn) if mix == "rot" else (aim, rot)
            if close and not closed:
                close_time = t
                gap_close = float(np.linalg.norm(sim.d.xpos[sim.ball] - sim.grasp_point()))
                incoming = -sim.ball_state()[1]
                approach_err = float(np.degrees(np.arccos(np.clip(
                    -sim.site_rot()[:, 2] @ incoming / np.linalg.norm(incoming), -1, 1))))
                if toss.others:
                    gap_other = min(float(np.linalg.norm(sim.d.mocap_pos[i] - sim.grasp_point()))
                                    for i in sim.mocap[:len(toss.others)])
            closed = close
            sim.track(target, rot, CONTROL_EVERY * PHYSICS_DT)
            sim.set_finger(close)
        if released:
            gap_min = min(gap_min, float(np.linalg.norm(sim.d.xpos[sim.ball] - sim.grasp_point())))
        sim.step_physics()
    end_ball, end_vel = sim.ball_state()
    result = {"caught": sim.caught(), "close_after_release": None if close_time is None else close_time - toss.hold,
              "approach_err": approach_err, "end_gap": float(np.linalg.norm(end_ball - sim.grasp_point())),
              "end_rel_speed": float(np.linalg.norm(end_vel - sim.grasp_velocity())), "end_height": float(end_ball[2]),
              "flight": toss.flight, "queries": queries, "gap_min": gap_min, "gap_at_close": gap_close,
              "wrong_ball": gap_other is not None and gap_close is not None and gap_other < gap_close,
              "locks": [locks.count(i) / max(len(locks), 1) for i in (0, 1, 2, -1, -2)] if locks else None,
              "named_gap": float(np.median(named_gap)) if named_gap else None, "track_start": track_start,
              "instruction": toss.instruction}
    if record or oracle:
        result["label_err"] = drift          # first-entry position error against the expert, along the toss
    if record:
        events = {}
        for ci, cam in enumerate(CAMERAS):
            parts = [(np.full(s[ci][0].size, j, np.uint16), *s[ci]) for j, s in kept]
            events[cam] = tuple(np.concatenate(x) for x in zip(*parts))
        result["record"] = dict(
            seed=seed, caught=result["caught"], dagger=True, release=toss.hold, flight=toss.flight,
            launch=toss.launch, velocity=toss.velocity, target=toss.target, close_time=-1.0,
            others=[[np.asarray(x).tolist() for x in o] for o in toss.others], colors=list(toss.colors),
            instruction=toss.instruction,
            step_times=np.array([s[0] for s in steps], np.float32), state8=np.stack([s[1] for s in steps]),
            labels=np.stack([s[2] for s in steps]), mask=np.stack([s[3] for s in steps]), events=events)
        if rgb_views and not selected:           # a selector's RGB stays with it: the policy trains on events alone
            result["record"]["rgb"] = (np.array([tr for tr, _ in rgb], np.float32), np.stack([f for _, f in rgb]))
    return result


_COLS, _ROWS = (x.ravel().astype(np.float32) for x in np.meshgrid(np.arange(RES), np.arange(RES)))


def arm_points(sim):
    """World points along the arm, base to grasp centre."""
    chain, b = [], sim.ee_body
    while b > 0:
        chain.append(b)
        b = sim.m.body_parentid[b]
    return [sim.d.xpos[i].copy() for i in chain[::-1]] + [sim.grasp_point().copy()]


def arm_band(cam, points, width):
    """Pixels [RES ** 2] within width of the arm's projected segments."""
    uv = [ball_meter.project(cam, p) for p in points]
    keep = np.zeros(RES * RES, bool)
    for a, b in zip(uv[:-1], uv[1:]):
        d = b - a
        u = np.clip(((_COLS - a[0]) * d[0] + (_ROWS - a[1]) * d[1]) / max(float(d @ d), 1e-6), 0, 1)
        keep |= (_COLS - a[0] - u * d[0]) ** 2 + (_ROWS - a[1] - u * d[1]) ** 2 < width ** 2
    return keep


def erase_events(event, balls, radius=14.0):
    """One view's (pix, pos, neg) without the events within radius px of these balls' projections."""
    pix = event[0]
    keep = np.ones(pix.size, bool)
    for uv in balls:
        keep &= (_COLS[pix] - uv[0]) ** 2 + (_ROWS[pix] - uv[1]) ** 2 >= radius ** 2
    return tuple(x[keep] for x in event)


def keep_events(event, uv, band, disk):
    """One view's (pix, pos, neg) restricted to the ball's disk (uv None: no disk yet) and the arm band."""
    pix = event[0]
    keep = band[pix].copy()
    if uv is not None:
        keep |= (_COLS[pix] - uv[0]) ** 2 + (_ROWS[pix] - uv[1]) ** 2 < disk ** 2
    return tuple(x[keep] for x in event)


def track_arrival(t, g, A, p, v, n):
    """When the tracked ball comes closest to the hand's planned path; None without a track or if it passes 10 cm off."""
    if n < 4:
        return None
    taus = np.arange(0.0, 0.6, 0.002)
    balls = p + np.outer(taus, v) + 0.5 * np.outer(taus ** 2, GRAVITY)
    plan = g + np.array([chunk_at(A, u)[:3] for u in taus / STEP])
    gaps = np.linalg.norm(balls - plan, axis=1)
    i = int(np.argmin(gaps))
    return t + taus[i] if gaps[i] < 0.1 else None


def chunk_at(A, u):
    """Chunk entry at fractional step u (clamped to the horizon), linear between entries."""
    u = min(max(u, 0.0), HORIZON - 1)
    i0 = int(np.floor(u))
    i1 = min(i0 + 1, HORIZON - 1)
    return (1 - (u - i0)) * A[i0] + (u - i0) * A[i1]


class Selector:
    """Where to look for the named ball over the RGB cells: the policy's own instruction head (ground.py checkpoint),
    or the match of a separate selective policy."""

    def __init__(self, ckpt, policy):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.head = None
        if "head" in ck:
            self.model = policy.model
            self.head = Grounder(self.model.vla.text_encoder.bert.config.hidden_size, relations="place.weight" in ck["head"],
                                 events="trace.0.weight" in ck["head"]).cuda().eval()
            self.head.load_state_dict(ck["head"])
        else:
            self.model = Policy(ckpt).model

    @torch.no_grad()
    def __call__(self, frames, instructions):
        """frames [B, V, H, W, 3] uint8: gated events, RGB, and with an event-reading head the raw events
        -> [B, 2, 2] px, match peak per view."""
        vla = self.model.vla
        x = pixel_values(torch.from_numpy(frames).cuda())
        def stage2(y):
            return tuple(torch.cat(p) for p in zip(*(vla.vision_encoder(y[i:i + 1], fine=True) for i in range(len(y)))))
        text = sentence(vla, list(instructions), x.device)
        if self.head is not None:                # the policy's own backbone, RGB views and raw event views
            events = stage2(x[:, 4:6])[1] if self.head.events else None
            return peak(self.head(stage2(x[:, 2:4])[1], text, events)).float().cpu().numpy()
        tokens, fine = stage2(x)
        return self.model.locator.scores(tokens, text, fine)[1].float().cpu().numpy()


def word_scene(sim, seed, held_out, words):
    """A selective scene whose named ball is told by words(paths, rng, held_out) rather than by its colour."""
    for k in range(200):
        rng = np.random.default_rng((seed, k))
        toss = sim.sample_scene(rng, held_out)
        text = words(ball_paths(dict(launch=toss.launch, velocity=toss.velocity, release=toss.hold,
                                     flight=toss.flight, others=toss.others)), rng, held_out)
        if text is not None:
            return dataclasses.replace(toss, instruction=text)
    raise RuntimeError(f"no scene for seed {seed}")


def run_batch(sims, streams, policy, tosses, seeds, latency=0.0, record=False, oracle=False, ensemble=0.0,
              teacher_gate=False, close_track=False, event_gate=None, selector=None, erase_others=False, switch=None,
              plan=False, erase_tracks=False, reader=None, mix=None, slice_track=False, query_every=None):
    """Tosses in lockstep; each policy call serves every toss waiting for a chunk.
    switch: s after release at which the instruction changes from another ball to the named one; until then the
    selector follows the other ball, and at the change the track is dropped and looked for again."""
    mode = getattr(policy, "frames", "window")
    rgb_views = selector is not None or getattr(policy, "selective", bool(tosses[0].others and oracle))
    clocks = [[0.0] for _ in tosses]
    located = getattr(policy, "locator", False)
    tracks = [SliceTrack(policy.model.locator.shift.cpu()) if slice_track and located else None for _ in tosses]
    switched = [False] * len(tosses)
    rollouts = [rollout(*a, latency, record, oracle, mode, getattr(policy, "ball", False), rgb_views, ensemble,
                        teacher_gate, close_track, event_gate, selector is not None or reader is not None, erase_others, clock,
                        getattr(getattr(selector, "head", None), "events", False), plan, erase_tracks, reader, mix, st, query_every)
                for *a, clock, st in zip(sims, streams, tosses, seeds, clocks, tracks)]
    if located:
        policy.reset()
    waiting = {i: next(r) for i, r in enumerate(rollouts)}
    results, infer = [None] * len(rollouts), []
    while waiting:
        idx = list(waiting)
        tic = time.perf_counter()
        kw = {"ids": idx} if located else {}
        if tosses[0].others:
            kw["instructions"] = [tosses[i].instruction for i in idx]
        if switch is not None:
            for n, i in enumerate(idx):
                if clocks[i][0] < tosses[i].hold + switch:
                    kw["instructions"][n] = tosses[i].instruction.replace(tosses[i].colors[0], tosses[i].colors[1])
                elif not switched[i]:
                    switched[i] = True
                    policy.tracks.pop(i, None)
        if teacher_gate or reader is not None:
            kw["gate"] = np.stack([waiting[i][2] for i in idx])
        frames = np.stack([waiting[i][0] for i in idx])
        for i in idx:                             # the slice track's fit as the policy's locator memory
            held = tracks[i] and tracks[i].memory(clocks[i][0], policy.model.locator.ages[1:].cpu().numpy())
            if held:
                policy.tracks[i] = tuple(torch.from_numpy(x).cuda() for x in held)
        if selector is not None:                 # only tosses without a track need to look
            valid = [policy.tracks[i][1] if i in policy.tracks else None for i in idx]
            look = [n for n, v in enumerate(valid) if v is None or v.sum() < 2 or not v[:LOST].any()]
            kw["gate"] = np.zeros((len(idx), 2, 2), np.float32)
            if look:
                kw["gate"][look] = selector(frames[look], [kw["instructions"][n] for n in look])
            frames = frames[:, :2]
        chunks = policy(frames, np.stack([waiting[i][1] for i in idx]), **kw)
        if close_track or (located and tosses[0].others):
            chunks = list(zip(chunks, zip(*policy.track)))
        if selector is not None:
            chunks = [(*c, g) for c, g in zip(chunks, kw["gate"])]
        infer.append(time.perf_counter() - tic)
        waiting = {}
        for i, a in zip(idx, chunks):
            try:
                waiting[i] = rollouts[i].send(a)
            except StopIteration as stop:
                results[i] = stop.value
    return results, infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tosses", type=int, default=200)
    ap.add_argument("--latency", type=float, default=0.0, help="seconds from observation to chunk use")
    ap.add_argument("--ensemble", type=float, default=0.0, help="chunk averaging time constant (s); 0: newest chunk")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default="openarm", choices=["openarm", "openarm_v1", "b601"])
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--record", default=None, help="write visited states with expert labels here (DAgger)")
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--batch", type=int, default=20, help="tosses run in lockstep")
    ap.add_argument("--oracle", action="store_true", help="execute expert label chunks (--ckpt only names the output)")
    ap.add_argument("--shadow", action="store_true", help="with --oracle: query the policy and log its error")
    ap.add_argument("--fast", action="store_true", help="CUDA Graph replay of the policy call")
    ap.add_argument("--distractors", type=int, default=0, help="selective catching: balls besides the named one")
    ap.add_argument("--held-out-phrase", action="store_true", help="with --distractors: phrasings not trained on")
    ap.add_argument("--painted", action="store_true", help="one ball in a named colour, as a selective scene has it")
    ap.add_argument("--relation", action="store_true", help="with --distractors: the instruction names the ball by where "
                    "it is among the others (left, right, highest, lowest)")
    ap.add_argument("--order", action="store_true", help="with --distractors: the instruction names the ball by when it "
                    "was thrown (first, last); needs an instruction head trained with --events")
    ap.add_argument("--drag", type=float, default=0.0, help="ball air drag, 1/m (deceleration = drag * speed ** 2)")
    ap.add_argument("--event-noise", type=float, default=0.0, help="background events per pixel per second")
    ap.add_argument("--hot-pixels", type=float, default=0.0, help="share of pixels stuck firing")
    ap.add_argument("--throw-speed", type=float, nargs="+", default=None, help="flat throws at this release speed, m/s (one value or low, high)")
    ap.add_argument("--throw-from", type=float, nargs="+", default=[3.0], help="with --throw-speed: metres ahead of the ready point")
    ap.add_argument("--blend", type=float, default=None, help="how far the expert turns the catcher to face the ball")
    ap.add_argument("--bounce", type=float, default=None, help="damping ratio of ball on cup")
    ap.add_argument("--cup-depth", type=float, default=None, help="m; B601 cup")
    ap.add_argument("--cup-radius", type=float, default=None, help="m; B601 cup, a straight tube")
    ap.add_argument("--catcher", default=None, choices=["cup", "hand"], help="what the B601 catches with")
    ap.add_argument("--hand-rate", type=float, default=285.0, help="deg/s per finger joint")
    ap.add_argument("--hand-open", type=float, default=0.26, help="rad the fingers wait bent back")
    ap.add_argument("--lead", type=float, default=None, help="s before arrival the hand starts to close")
    ap.add_argument("--volts", type=int, default=None, choices=[24, 48], help="hold the arm to its motors' torque-speed line")
    ap.add_argument("--rig", type=int, default=0, help="which random direction the camera rigs moved in")
    ap.add_argument("--hand-look", action="store_true", help="one look at the hand gives the rigs' error")
    ap.add_argument("--ray-gap", type=float, default=None, help="largest gap between the two views' rays, m")
    ap.add_argument("--hand-read", default=None, help="handlook.py checkpoint: the backbone finds the hand in the RGB views")
    ap.add_argument("--camera-cm", type=float, default=0.0, help="camera rigs off their calibration, cm")
    ap.add_argument("--camera-deg", type=float, default=0.0, help="camera rigs off their calibration, degrees")
    ap.add_argument("--switch", type=float, default=None, help="with --selector: s after release at which the instruction "
                    "changes from another ball to the one to catch")
    ap.add_argument("--compile", action="store_true", help="with --fast: torch.compile the graph")
    ap.add_argument("--batch-views", action="store_true", help="with --fast: both views in one backbone pass")
    ap.add_argument("--half", action="store_true", help="with --fast: FP16 spike-input convolutions")
    ap.add_argument("--autotune", action="store_true", help="with --compile: kernel autotuning")
    ap.add_argument("--tf32", action="store_true", help="with --fast: TF32 for the remaining FP32 GEMMs")
    ap.add_argument("--fold", action="store_true", help="with --fast: spiking timesteps folded into the batch")
    ap.add_argument("--close-track", action="store_true", help="gripper from the expert's rule on the policy's ball track")
    ap.add_argument("--snap", action="store_true", help="locator policies: move the head's arm targets onto the "
                    "located ball's path (locator.CatchPolicy.snapped)")
    ap.add_argument("--close-hand", action="store_true", help="with --close-track: time the closure on the hand's present "
                    "position instead of the chunk's planned targets")
    ap.add_argument("--slice-track", action="store_true", help="locator policies: measure the ball at every 4 ms event "
                    "slice once the policy's track holds (locator.SliceTrack), not only at the policy step")
    ap.add_argument("--query-every", type=float, default=None, help="with --slice-track: s between policy calls once the "
                    "slice track holds (default STEP)")
    ap.add_argument("--mix", choices=("rot", "pos"), default=None, help="with --close-track: the planner on the policy's "
                    "ball track gives the orientation (rot) or the grasp target (pos), the head the rest")
    ap.add_argument("--reader", default=None, help="fovea.py checkpoint: the backbone reads the crops around event tracks")
    ap.add_argument("--kinds", action="store_true", help="objects named by kind, a look-alike in every scene")
    ap.add_argument("--patterns", action="store_true", help="objects named by colour and figure")
    ap.add_argument("--unseen-objects", action="store_true", help="with --patterns: combinations never trained on")
    ap.add_argument("--erase-tracks", action="store_true", help="with --event-gate: drop the events around the other balls' "
                    "tracks once the policy holds its ball")
    ap.add_argument("--blind", action="store_true", help="the action head sees empty frames: state tokens alone (not with --fast)")
    ap.add_argument("--plan-track", action="store_true", help="with --close-track: the expert's planner on the policy's "
                    "ball track drives the arm (where the loss is: track or action head)")
    ap.add_argument("--event-gate", type=float, default=0.0, help="with --close-track: keep only the named ball's events "
                    "and the arm's, in a band of this many px")
    ap.add_argument("--gate-disk", type=float, default=30.0, help="with --event-gate: radius around the ball, px")
    ap.add_argument("--look-disk", type=float, default=30.0, help="with --event-gate and --selector: radius around the "
                    "selector's peak until the track holds, px (0: gate open until then)")
    ap.add_argument("--teacher-gate", action="store_true", help="locator acquisition from the named ball's true "
                    "projection (+ training noise), eager")
    ap.add_argument("--erase-others", action="store_true", help="remove the distractors' events (upper bound of a gate)")
    ap.add_argument("--selector", default=None, help="selective checkpoint whose instruction match acquires the ball "
                    "for this policy's locator (eager)")
    args = ap.parse_args()
    if args.cup_depth:
        from . import b601
        b601.CUP_DEPTH = args.cup_depth
    if args.cup_radius:
        from . import b601
        b601.CUP_RADIUS = b601.CUP_MOUTH = args.cup_radius
    if args.catcher:
        from . import b601
        b601.CATCHER, b601.HAND_RATE, b601.HAND_OPEN = args.catcher, np.radians(args.hand_rate), args.hand_open
    set_numerics()
    if args.oracle and not args.shadow:
        policy = lambda frames, state8, **kw: np.zeros((len(frames), HORIZON, 7), np.float32)
        policy.step = None
    else:
        speed = {k: getattr(args, k) for k in ("compile", "batch_views", "half", "autotune", "tf32", "fold")}
        policy = Policy(args.ckpt, args.fast, **speed)
    selector = Selector(args.selector, policy) if args.selector else None
    if args.blind:
        policy.model.blind = True
    if args.snap:
        policy.model.snap = True
    if args.hand_read or args.kinds or args.patterns:
        use_kinds()                              # the RGB cameras and their sensor size, also for one ball
    sims = [make_sim(args.arm, args.distractors) for _ in range(min(args.batch, args.tosses))]
    for sim in sims:
        sim.drag = args.drag
        if args.throw_speed:
            sim.set_throws(args.throw_speed, args.throw_from)
        if args.bounce:
            sim.set_bounce(args.bounce)
        if args.blend is not None:
            sim.EXPERT_BLEND = args.blend
        if args.lead is not None:
            sim.EXPERT_LEAD = args.lead
        if args.volts:
            sim.limit_motors(args.volts)
        sim.miscalibrate(args.camera_cm, args.camera_deg, args.rig)
    shift = np.zeros((2, 2))
    if args.hand_look or args.hand_read:
        sims[0].reset_arm()
        shift = ball_meter.view_shift(sims[0])
    if args.hand_read:
        from .handlook import HandLook
        truth = shift
        look = HandLook(args.hand_read, policy.model.vla)
        shift = look(sims[0], mujoco.Renderer(sims[0].m, SENSOR, SENSOR), np.random.default_rng(args.rig))
        print("views off", np.round(truth, 2).tolist(), "read", np.round(shift, 2).tolist(), "px", flush=True)
    ball_meter.set_shift(shift)
    if getattr(policy, "locator", False):
        policy.model.locator.shift.copy_(torch.as_tensor(shift, dtype=torch.float32))
        if args.ray_gap:
            policy.model.locator.first_gap = args.ray_gap
    reader = None
    if args.reader:
        from .baseline import TrackReader
        reader = TrackReader(args.reader, policy.model.vla)
        for sim in sims:
            sim.sensor = mujoco.Renderer(sim.m, SENSOR, SENSOR)
    streams = [EventStream(sim, noise=args.event_noise, hot=args.hot_pixels) for sim in sims]
    expert = Expert(sims[0])
    results, records, infer, t0 = [], [], [], time.time()
    for first in range(0, args.tosses, args.batch):
        seeds = [args.seed + i for i in range(first, min(first + args.batch, args.tosses))]
        words = relation_sentence if args.relation else order_sentence if args.order else None
        tosses = [word_scene(sims[0], seed, args.held_out_phrase, words) if words
                  else sims[0].sample_patterns(np.random.default_rng(seed), args.held_out_phrase, args.unseen_objects) if args.patterns
                  else sims[0].sample_kinds(np.random.default_rng(seed), args.held_out_phrase) if args.kinds
                  else sims[0].sample_scene(np.random.default_rng(seed), args.held_out_phrase) if args.distractors or args.painted
                  else sims[0].sample_toss(np.random.default_rng(seed)) for seed in seeds]
        batch, times = run_batch(sims, streams, policy, tosses, seeds, args.latency, args.record is not None,
                                 args.oracle, args.ensemble, args.teacher_gate, "hand" if args.close_hand else args.close_track,
                                 (args.event_gate, args.gate_disk, args.look_disk) if args.event_gate else None, selector,
                                 args.erase_others, args.switch, args.plan_track, args.erase_tracks, reader, args.mix,
                                 args.slice_track, args.query_every)
        infer += times
        for seed, toss, r in zip(seeds, tosses, batch):
            if args.record is not None:
                records.append(r.pop("record"))
            r["expert"] = run_expert(sims[0], expert, toss)
            r["seed"] = seed
            results.append(r)
        while args.record is not None and records and (len(records) >= args.shard or len(results) == args.tosses):
            Path(args.record).mkdir(parents=True, exist_ok=True)
            n = (len(results) - len(records)) // args.shard
            write_shard(Path(args.record) / f"shard_{args.seed:07d}_{n:03d}.npz", records[:args.shard])
            records = records[args.shard:]
        print(f"{len(results)}/{args.tosses} policy {np.mean([x['caught'] for x in results]):.1%} "
              f"expert {np.mean([x['expert'] for x in results]):.1%} {(time.time() - t0) / len(results):.2f} s/toss",
              flush=True)
    summary = {"ckpt": args.ckpt, "arm": args.arm, "step": policy.step, "latency": args.latency, "ensemble": args.ensemble, "tosses": args.tosses,
               "policy": float(np.mean([x["caught"] for x in results])),
               "expert": float(np.mean([x["expert"] for x in results])),
               "batch": args.batch, "batch_infer_ms_median": 1000 * float(np.median(infer)), "oracle": args.oracle,
               "gap_min_cm_median": 100 * float(np.median([x["gap_min"] for x in results])),
               "speed": [k for k in ("compile", "batch_views", "half", "autotune", "tf32", "fold") if getattr(args, k)],
               "distractors": args.distractors, "held_out_phrase": args.held_out_phrase, "relation": args.relation,
               "switch": args.switch, "order": args.order,
               "teacher_gate": args.teacher_gate, "close_track": args.close_track, "close_hand": args.close_hand, "snap": args.snap, "mix": args.mix, "slice_track": args.slice_track, "query_every": args.query_every, "event_gate": args.event_gate,
               "selector": args.selector,
               "wrong_ball": float(np.mean([x["wrong_ball"] for x in results]))}
    locks = np.array([x["locks"] for x in results if x["locks"]])
    if len(locks):              # tosses whose track mostly follows the named ball / a distractor / nothing; no track
        top = locks.argmax(1)
        summary["track_named"], summary["track_distractor"] = float((top == 0).sum() / len(results)), float(np.isin(top, (1, 2)).sum() / len(results))
        summary["track_arm"] = float((top == 4).sum() / len(results))
        summary["track_none"] = 1 - (summary["track_named"] + summary["track_distractor"] + summary["track_arm"])
    if "label_err" in results[0] and not (args.oracle and not args.shadow):
        summary["label_err_cm_mean"] = 100 * float(np.mean([e for x in results for e in x["label_err"]]))
    print(json.dumps(summary), flush=True)
    out = Path(args.out or Path(args.ckpt).with_suffix(f".eval_{args.arm}_lat{int(round(args.latency * 1000))}.json"))
    out.write_text(json.dumps({"summary": summary, "tosses": results}, indent=1))
    os._exit(0)


if __name__ == "__main__":
    main()
