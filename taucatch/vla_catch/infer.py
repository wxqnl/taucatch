"""Real-time catcher: the policy loop of the judge setting (evaluate --fast --close-track --close-hand --slice-track
--query-every 0.004) with the simulator's rendering and physics replaced by three hardware interfaces.

    event slices (4 ms) -> trace frames + SliceTrack -> SpikeVLA (every 4 ms once the track holds, else 20 ms)
    -> 12 x 20 ms chunk -> 500 Hz damped-least-squares IK -> joint reference
    ball track -> the hand closes 50 ms (EXPERT_LEAD) before the track reaches the hand

Modes: head (the action head's chunk drives the arm) and plan (the expert's planner on the policy's ball track drives
arm and hand; the head's chunk is unused). The arm's MuJoCo model is the kinematic model: measured joint angles in,
the joint reference of sim.track out.

Check against the simulator (same seeds, same numbers as `evaluate --batch 1`):

    VLA_AIM_X=1.3 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m taucatch.vla_catch.infer \
        --policy outputs/hf/spikevla-catch-b601 --check 10 --out outputs/checks/vla_catch/infer_check.json
"""
import argparse
import json
import threading
import time
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np
import torch

from .collect import EV_DT, PRE, POST, STEP, make_sim
from .data import WINDOW
from .events import CAMERAS, RES, EventStream, trace_frame
from .evaluate import Policy, chunk_at, rotation, track_arrival
from .expert import EstimateSim, Expert
from .locator import SliceTrack
from .policy import set_numerics
from .rollout import AFTER, CONTROL_EVERY, PHYSICS_DT
from .sim import GRAVITY, ballistic

GRID_SLICES = int(round(STEP / EV_DT))          # slices per 20 ms policy step


class EventSource(Protocol):
    def slice(self) -> tuple[float, list]:
        """Block until the next 4 ms slice -> (its end time, s on time.monotonic's clock; per camera (pixel, positive
        count, negative count) arrays, pixel = row * 224 + column on the undistorted, cropped 224 x 224 grid)."""


class ArmIO(Protocol):
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint angles [6] (rad) and velocities [6] (rad/s)."""

    def send(self, q_ref: np.ndarray) -> None:
        """Joint reference [6]; the driver runs the joint PD with gravity compensation (MIT mode)."""


class HandIO(Protocol):
    def read(self) -> float:
        """Finger angle (rad, as the simulator's finger joint)."""

    def send(self, close: bool) -> None:
        """Open or close the hand."""


class Catcher:
    """The policy side of one throw. add_slice for every event slice, query when due(), control at 500 Hz.

    policy: evaluate.Policy (fast) of a located model; sim: the arm's model (make_sim), kept at the measured state;
    plan: the planner drives arm and hand; latency: s added to a chunk's start (the simulator's model of the delay;
    0 on the real robot, where the delay passes by itself)."""

    def __init__(self, policy, sim, plan=False, latency=0.0, watchdog=None, box=None):
        self.policy, self.sim, self.plan, self.latency = policy, sim, plan, latency
        self.watchdog, self.box = watchdog, box     # s without a fresh chunk before the ready pose; m around it
        self.ages = policy.model.locator.ages[1:].cpu().numpy()
        self.reset()

    def reset(self):
        self.last = [np.full(RES * RES, -np.inf) for _ in CAMERAS]      # newest slice with an event, per pixel
        self.j = None
        self.track = SliceTrack(self.policy.model.locator.shift.cpu())
        self.policy.reset()
        self.chunks, self.arrival, self.planned, self.closed = [], None, None, False
        self.view = EstimateSim(self.sim)
        self.planner = Expert(self.view)
        self.planner.reset()

    def add_slice(self, j, t, events):
        """Slice j (counted on the EV_DT grid) ending at t: per camera (pixel, positive, negative)."""
        for ci, e in enumerate(events):
            self.last[ci][e[0]] = j
        self.j = j
        self.track.add(t - EV_DT, np.stack([(j - a) < 2 for a in self.last]).reshape(len(CAMERAS), RES, RES))

    def due(self):
        return self.j is not None and (self.j % GRID_SLICES == 0 or self.track.ready())

    def query(self, t, state8, grasp, rot):
        """Policy call at t (the newest slice's end) with the arm state then -> chunk [12, 7] (grasp offsets,
        rotation vector, hand), in physical units."""
        frames = np.stack([trace_frame(((self.j - a) * EV_DT).astype(np.float32)) for a in self.last])
        held = self.track.memory(t, self.ages)        # the slice track's fit as the locator memory
        if held:
            self.policy.tracks[0] = tuple(torch.from_numpy(x).cuda() for x in held)
        chunk = self.policy(frames[None], np.asarray(state8, np.float32)[None], ids=[0])[0]
        p, v, n = (x[0] for x in self.policy.track)
        self.arrival = track_arrival(t, grasp, 0 * chunk, p, v, n)     # the track meets the hand where it is
        if not self.track.active and n >= 3:
            self.track.start(t, np.asarray(p, float), np.asarray(v, float))
        if n >= 3:
            self.planned = (t, p, v)
        self.chunks.append((t + self.latency, t, grasp, rot, chunk))
        return chunk

    def control(self, t, dt):
        """Arm and hand at t -> joint reference [6], close. One IK step of dt on the model's reference."""
        sim = self.sim
        active = [c for c in self.chunks if c[0] <= t + 1e-9]
        stale = self.watchdog is not None and active and t - active[-1][1] > self.watchdog and not self.closed
        if not active or stale:
            target, rot, close = sim.ready_grasp, sim.ready_rot, self.closed
        else:
            _, t0, g, R, A = active[-1]
            a = chunk_at(A, (t - t0) / STEP)
            target, rot = g + a[:3], rotation(a[3:6]) @ R
            close = self.closed or (self.arrival is not None and t >= self.arrival - sim.EXPERT_LEAD)
        if self.plan:
            target, rot, close = sim.ready_grasp, sim.ready_rot, False
            if self.planner.closed:
                self.view.estimate = sim.grasp_point().copy(), np.zeros(3)
            elif self.planned is not None:
                t0, p, v = self.planned
                self.view.estimate = ballistic(p, v, t - t0), v + GRAVITY * (t - t0)
            if self.view.estimate is not None:
                target, rot, close = self.planner.act(True, t)
        if self.box is not None:
            target = np.clip(target, sim.ready_grasp - self.box, sim.ready_grasp + self.box)
        self.closed = close
        sim.track(target, rot, dt)
        sim.set_finger(close)
        return sim.q_ref.copy(), close


def observe(sim, q, qd, finger):
    """Measured arm and hand state into the kinematic model -> (state8, grasp point, grasp rotation)."""
    d = sim.d
    d.qpos[sim.qadr], d.qvel[sim.vadr] = q, qd
    if sim.finger_q is not None:
        d.qpos[sim.finger_q] = finger
    mujoco.mj_forward(sim.m, d)
    return sim.state8(), sim.grasp_point().copy(), sim.site_rot().copy()


def run(catcher, events: EventSource, arm: ArmIO, hand: HandIO, seconds, rate=500):
    """One armed window of `seconds` on the robot: a perception thread (slices, SliceTrack, policy calls) and the
    500 Hz control loop here. The arm should be at the ready pose; afterwards it returns there."""
    sim, dt = catcher.sim, 1.0 / rate
    catcher.reset()
    q, qd = arm.read()
    sim.q_ref = np.asarray(q, float).copy()
    state = {"now": observe(sim, q, qd, hand.read())}
    stop = threading.Event()

    def perceive():
        while not stop.is_set():
            t, ev = events.slice()
            catcher.add_slice(int(round(t / EV_DT)), t, ev)
            if catcher.due():
                catcher.query(t, *state["now"])
    worker = threading.Thread(target=perceive, daemon=True)
    worker.start()
    end = time.monotonic() + seconds
    tick = time.monotonic()
    try:
        while tick < end:
            q, qd = arm.read()
            state["now"] = observe(sim, q, qd, hand.read())
            q_ref, close = catcher.control(time.monotonic(), dt)
            arm.send(q_ref)
            hand.send(close)
            tick += dt
            time.sleep(max(0.0, tick - time.monotonic()))
    finally:
        stop.set()
        hand.send(False)


def simulate(catcher, sim, stream, toss, seed):
    """One toss in MuJoCo with every decision made by the Catcher, in evaluate.rollout's order: the same caught /
    missed as evaluate --batch 1 with the same options."""
    sim.reset_arm()
    sim.paint(toss)
    catcher.reset()
    n = int(round((toss.hold + toss.flight + AFTER) / PHYSICS_DT))
    release = int(round(toss.hold / PHYSICS_DT))
    step_every, ev_every = int(round(STEP / PHYSICS_DT)), int(round(EV_DT / PHYSICS_DT))
    t_first, t_last = toss.hold - PRE, toss.hold + toss.flight + POST
    k_first = int(np.ceil((t_first - 1e-9) / STEP)) * step_every
    j_first = k_first // ev_every - WINDOW - 1
    started, queries = False, 0
    for k in range(n):
        t = k * PHYSICS_DT
        if k < release:
            sim.place_ball(toss.launch)
        elif k == release:
            sim.place_ball(toss.launch, toss.velocity)
        if k % ev_every == 0:
            j = k // ev_every
            if j == j_first - 1:
                stream.reset(seed)
            elif j >= j_first:
                catcher.add_slice(j, t, stream.slice())
                started = True
        if started and catcher.due() and k >= k_first and t <= t_last + 1e-9 and k % ev_every == 0:
            mujoco.mj_forward(sim.m, sim.d)
            catcher.query(t, sim.state8(), sim.grasp_point().copy(), sim.site_rot().copy())
            queries += 1
        if k % CONTROL_EVERY == 0:
            catcher.control(t, CONTROL_EVERY * PHYSICS_DT)
        sim.step_physics()
    return {"seed": seed, "caught": sim.caught(), "queries": queries}


def judge_sim():
    """The arm model of the judge setting: B601 with the five-finger hand, 24 V motor limits, hand closing 50 ms
    ahead, planner blend 0.5; 8 m/s throws from 4 m for the simulator check."""
    from . import b601
    b601.CATCHER, b601.HAND_RATE, b601.HAND_OPEN = "hand", np.radians(285), 0.26
    sim = make_sim("b601")
    sim.set_throws([8.0], [4.0])
    sim.set_bounce(0.5)
    sim.EXPERT_BLEND, sim.EXPERT_LEAD = 0.5, 0.05
    sim.limit_motors(24)
    sim.miscalibrate(0.0, 0.0, 0)
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="released model directory or Hugging Face repo id")
    ap.add_argument("--plan", action="store_true", help="the planner on the policy's ball track drives arm and hand")
    ap.add_argument("--check", type=int, default=10, help="tosses in the simulator check")
    ap.add_argument("--seed", type=int, default=1000000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_numerics()
    policy = Policy(args.policy, fast=True)
    sim = judge_sim()
    stream = EventStream(sim)
    catcher = Catcher(policy, sim, plan=args.plan, latency=0.02)
    results = []
    for seed in range(args.seed, args.seed + args.check):
        toss = sim.sample_toss(np.random.default_rng(seed))
        results.append(simulate(catcher, sim, stream, toss, seed))
        print(results[-1], flush=True)
    summary = {"policy": args.policy, "plan": args.plan, "tosses": len(results),
               "caught": float(np.mean([r["caught"] for r in results]))}
    Path(args.out).write_text(json.dumps({"summary": summary, "tosses": results}, indent=1) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
