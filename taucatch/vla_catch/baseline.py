"""Motion-control baseline: no network. The same event stream goes through a multi-ball tracker (event blobs -> stereo ->
gravity fit, with the locator's windows and physics checks, at the 250 Hz of the event slices), and the expert's
intercept planner and timed closure run on the estimate instead of the true state.

Which ball it goes for:
  none      no language: the first confirmed track with an intercept inside the reach
  rule      hand-written rules on the instruction: colour word -> colour of the track in the RGB views; left, right,
            high, low -> image order of the tracks; first, last -> order the tracks appeared in
  named     the true named ball acquires the track (what the arm and tracker can do with selection solved)

    python -m taucatch.vla_catch.baseline --select rule --distractors 2 --tosses 100 --out runs_v2/baseline/rule.json
"""
import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage

from . import ball as bm
from .collect import EV_DT, PRE, RGB_DT, make_sim
from .data import RGB_DELAY
from .evaluate import EVAL_SEED, arm_band, arm_points, word_scene
from .events import CAMERAS, RES, EventStream
from .expert import EstimateSim, Expert
from .ground import RELATIONS, ORDERS, order_sentence, relation_sentence
from .rollout import AFTER, CONTROL_EVERY, PHYSICS_DT, run_expert
from . import objects
from .sim import COLORS, GRAVITY, RGB_CAMERAS, SENSOR, ballistic, use_kinds

WINDOWS = (8, 6, 6)         # px: tracked measuring windows, as the locator's
RAY_GAP = 0.03              # m: two views' blobs belong together when their rays pass this close
TRACK_GATE = 0.04           # m: a measurement this far from the prediction is refused once 3 points exist
FIRST_GAP = 0.08            # m: step limit between the first points of a new track (15 m/s over 2 slices + margin)
LOST_AFTER = 0.08           # s without an accepted point before a track is dropped
FIT_LIMIT = 0.02            # m: rms of the gravity fit of a confirmed track
CONFIRM = 5                 # points of a confirmed track
MIN_SPEED = 1.0             # m/s: slower tracks are the arm or noise (a toss near its top is still above 1.3)
PLACE_GAP = 0.012           # s: a track must have a measurement this close to an RGB exposure to read it
COLOUR_VOTES = 2            # RGB looks a track needs before its colour counts
ALL_SEEN = 0.25             # s after the first track when a comparison stops waiting for the last ball
ARM_BAND = 8                # px: blobs this close to the moving arm are the arm
ARM_STILL = 0.05            # m/s
RELOCK = 0.08               # m: a new track this close to the lost track's path is the same ball
MIN_EVENTS = 3
BOX = ((-0.6, 5.5), (-2.0, 2.0), (0.2, 3.0))     # m: where a ball can be
INCOMING_FROM = 1.0         # m: an incoming track starts this far out along the throw (the arm never reaches it)
INCOMING_SPEED = 2.0        # m/s toward the robot


class Track:
    def __init__(self, t, p, order):
        self.points, self.last, self.order, self.votes = [(t, p)], t, order, {}

    def fit(self, t):
        """Gravity fit -> position and velocity at t, rms of the fit."""
        pts = [(tk, p) for tk, p in self.points if tk >= t - bm.SPAN]
        dt = np.array([tk - t for tk, _ in pts])
        y = np.stack([p for _, p in pts]) - 0.5 * GRAVITY * dt[:, None] ** 2
        A = np.stack([np.ones_like(dt), dt], 1)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        return x[0], x[1], float(np.sqrt(np.mean((A @ x - y) ** 2)))

    def predict(self, t):
        if len(self.points) == 1:
            return self.points[0][1]
        p, v, _ = self.fit(self.last)
        return ballistic(p, v, t - self.last)

    @property
    def confirmed(self):
        return len(self.points) >= CONFIRM and np.linalg.norm(self.fit(self.last)[1]) > MIN_SPEED

    def state(self, t):
        p, v, rms = self.fit(t)
        return p, v, rms


def in_box(p):
    return all(lo < x < hi for x, (lo, hi) in zip(p, BOX))


def blobs(pix, weight, mask):
    """Connected groups of this view's fresh events away from the arm -> centres (col, row)."""
    keep = ~mask[pix]
    image = np.zeros((RES, RES), bool)
    image.flat[pix[keep]] = True
    labels, n = ndimage.label(ndimage.binary_dilation(image, iterations=1))
    if n == 0:
        return []
    w = np.bincount(pix[keep], weights=weight[keep], minlength=RES * RES).reshape(RES, RES)
    index = np.arange(1, n + 1)
    size = ndimage.sum(image, labels, index)
    centres = ndimage.center_of_mass(w, labels, index)
    return [np.array([c[1] + 0.5, c[0] + 0.5]) for c, s in zip(centres, size) if s >= MIN_EVENTS]


def window(pix, weight, guess):
    """Mean shift in the tracked windows; None with fewer than MIN_EVENTS inside."""
    col, row = pix % RES, pix // RES
    uv = np.asarray(guess, float)
    for radius in WINDOWS:
        near = np.hypot(col - uv[0], row - uv[1]) < radius
        if near.sum() < MIN_EVENTS:
            return None
        uv = np.array([np.average(col[near], weights=weight[near]), np.average(row[near], weights=weight[near])])
    return uv + 0.5


class Tracker:
    ray_gap = RAY_GAP

    """Every moving ball the two event cameras see, as 3D tracks."""

    def __init__(self):
        self.tracks, self.count = [], 0

    def update(self, t, fresh, masks):
        """t: time of the newest events; fresh: per view (pix, weight) of the two newest slices."""
        used = [np.zeros(p.size, bool) for p, _ in fresh]
        for track in sorted(self.tracks, key=lambda x: -len(x.points)):       # longest tracks measure first
            guess = track.predict(t)
            uv = [window(pix, w, bm.project(cam, guess)) for (pix, w), cam in zip(fresh, CAMERAS)]
            if any(u is None for u in uv):
                continue
            p = bm.triangulate(uv)
            gate = TRACK_GATE if len(track.points) >= 3 else FIRST_GAP
            if np.linalg.norm(p - guess) > gate:
                continue
            track.points.append((t, p))
            track.last = t
            for i, (pix, _) in enumerate(fresh):
                used[i] |= np.hypot(pix % RES - uv[i][0], pix // RES - uv[i][1]) < WINDOWS[-1]
        left = [blobs(pix[~u], w[~u], m) for (pix, w), u, m in zip(fresh, used, masks)]
        for a in left[0]:
            for b in left[1]:
                (c1, d1), (c2, d2) = (bm._ray(cam, uv) for cam, uv in zip(CAMERAS, (a, b)))
                s, r = np.linalg.lstsq(np.stack([d1, -d2], 1), c2 - c1, rcond=None)[0]
                gap = np.linalg.norm(c1 + s * d1 - c2 - r * d2)
                p = (c1 + s * d1 + c2 + r * d2) / 2
                if gap < self.ray_gap and in_box(p) and all(np.linalg.norm(p - x.predict(t)) > 0.06 for x in self.tracks):
                    self.tracks.append(Track(t, p, self.count))
                    self.count += 1
        kept = []
        for track in self.tracks:
            if t - track.last > LOST_AFTER:
                continue
            if track.confirmed and track.fit(track.last)[2] > FIT_LIMIT:
                continue
            kept.append(track)
        self.tracks = kept

    def confirmed(self):
        return [x for x in self.tracks if x.confirmed]


class Rule:
    """Hand-written reading of the instruction."""

    def __init__(self, sentence):
        words = sentence.lower().replace("leftmost", "left").replace("rightmost", "right").split()
        self.colour = next((w for w in words if w in COLORS), None)
        self.colour = next((w for w in objects.KINDS if w in sentence), self.colour)   # a kind: the mean colour it has
        self.relation = None
        for word, (axis, sign, phrases, unseen) in RELATIONS.items():
            if sentence in phrases or sentence == unseen:
                self.relation = axis, sign
        self.order = None
        for word, (place, phrases, unseen) in ORDERS.items():
            if sentence in phrases or sentence == unseen:
                self.order = place

    def pick(self, tracks, t, late=False):
        """The track the sentence names, from the confirmed ones (None: no word it knows, or nothing to compare yet)."""
        if not tracks:
            return None
        if self.colour is not None:
            seen = [x for x in tracks if x.votes.get(self.colour, 0.0) >= COLOUR_VOTES
                    and x.votes[self.colour] > sum(x.votes.values()) / 2]
            if not seen and late:                                      # no track holds a clear majority: the most votes
                seen = [x for x in tracks if x.votes.get(self.colour, 0.0) > 0]
            return max(seen, key=lambda x: x.votes[self.colour]) if seen else tracks[0] if late else None
        if self.relation is not None:
            axis, sign = self.relation
            score = []
            for x in tracks:
                p = x.state(t)[0]
                score.append(sum(sign * bm.project_rgb(cam, p)[axis] for cam in CAMERAS))
            return tracks[int(np.argmax(score))]
        if self.order is not None:
            return sorted(tracks, key=lambda x: x.points[0][0])[self.order]
        return None


COLOUR_AXES = {k: np.asarray(v) / np.linalg.norm(v) for k, v in COLORS.items()}
KIND_AXES = {k: v / np.linalg.norm(v) for k, v in objects.mean_colours().items()}


def vote_colours(track, t, image_pair, axes=COLOUR_AXES):
    """Add one RGB exposure at time t to a track's colour votes: nearest named colour of the pixels around the ball."""
    guess = track.predict(t)
    for cam, image in zip(CAMERAS, image_pair):
        c, r = bm.project_rgb(cam, guess)
        c, r = int(round(c - 0.5)), int(round(r - 0.5))
        if not (2 <= c < RES - 2 and 2 <= r < RES - 2):
            continue
        rgb = image[r - 1:r + 2, c - 1:c + 2].reshape(-1, 3).astype(float).mean(0) / 255.0
        if np.linalg.norm(rgb) < 0.05:
            continue
        unit = rgb / np.linalg.norm(rgb)
        k = max(axes, key=lambda name: axes[name] @ unit)
        track.votes[k] = track.votes.get(k, 0.0) + 1.0


class TrackReader:
    """SpikeVLA's backbone and instruction head on the sensor crops around the tracks (fovea.py)."""
    FEWEST, MOST = 3, 6          # exposures a track needs; after which the best one is taken
    MARGIN = 4.0                 # lead in mean match that settles it earlier

    def __init__(self, ckpt, vla=None):
        import torch
        from .fovea import load_policy, make_reader
        from .locator import sentence
        self.torch, self.sentence = torch, sentence
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.vla = vla if vla is not None else load_policy(ck["ckpt"]).model.vla
        self.head = make_reader(self.vla.text_encoder.bert.config.hidden_size).cuda().eval()
        self.head.load_state_dict(ck["head"])
        self.renderer = None

    @staticmethod
    def place(track, t):
        """The track's own measurement of the exposure time: a fit on a few points flies off the object."""
        near = min(track.points, key=lambda point: abs(point[0] - t))
        return near[1] if abs(near[0] - t) < PLACE_GAP else None

    def __call__(self, tracks, t, images, text):
        """One exposure: adds what each track measured at that moment to its score, the head's match to the sentence."""
        from .fovea import blocks, crop
        torch = self.torch
        places = [(x, self.place(x, t)) for x in tracks]
        places = [(x, p) for x, p in places if p is not None]
        if not places:
            return
        pairs = np.stack([np.stack([crop(image, cam, p) for cam, image in zip(CAMERAS, images)]) for _, p in places])
        with torch.no_grad():
            fine = blocks(self.vla, torch.from_numpy(pairs)).cuda()
            words = self.sentence(self.vla, [text], "cuda").float().repeat(2 * len(places), 1)
            scores = self.head(fine.flatten(0, 1).float(), words).view(len(places), 2).sum(1).cpu().numpy()
        for (x, _), score in zip(places, scores):
            x.reads = getattr(x, "reads", 0) + 1
            x.score = getattr(x, "score", 0.0) + float(score)

    def pick(self, tracks, late):
        read = [x for x in tracks if getattr(x, "reads", 0) >= self.FEWEST]
        if not read:
            return tracks[0] if late and tracks else None
        ranked = sorted(read, key=lambda x: x.score / x.reads, reverse=True)
        mean = lambda x: x.score / x.reads
        settled = len(ranked) > 1 and mean(ranked[0]) - mean(ranked[1]) > self.MARGIN
        return ranked[0] if settled or late or ranked[0].reads >= self.MOST else None


def rollout(sim, stream, expert, view, toss, seed, select, switch=None, reader=None):
    sim.reset_arm()
    sim.paint(toss)
    expert.reset()
    view.estimate = None
    end = toss.hold + toss.flight + AFTER
    n = int(round(end / PHYSICS_DT))
    release = int(round(toss.hold / PHYSICS_DT))
    ev_every = int(round(EV_DT / PHYSICS_DT))
    j_first = int(round((toss.hold - PRE) / EV_DT))
    earliest = min([toss.hold] + [o[2] for o in toss.others])
    j_first = min(j_first, int(round((earliest - PRE) / EV_DT)))
    rgb_times = deque(np.arange(earliest, toss.hold + toss.flight, RGB_DT) if select in ("rule", "reader") and toss.others else [])
    frames = []                                    # (time, images): RGB exposures not used yet
    tracker, rule = Tracker(), Rule(toss.instruction) if select == "rule" else None
    if switch is not None:                         # the instruction names another ball until release + switch
        rule = Rule(f"catch the {toss.colors[1]} ball")
    slices, masks = deque(maxlen=2), [np.zeros(RES * RES, bool) for _ in CAMERAS]
    chosen, chosen_time, lost_chosen = None, None, 0
    target, rot, close = sim.ready_grasp, sim.ready_rot, False
    gap_min, first_lock, picks = np.inf, None, []
    first_seen, coast = None, None
    for k in range(n):
        t = k * PHYSICS_DT
        released = k >= release
        if k < release:
            sim.place_ball(toss.launch)
        elif k == release:
            sim.place_ball(toss.launch, toss.velocity)
        if toss.others:
            sim.move_distractors(toss, t)
        while rgb_times and rgb_times[0] <= t + 1e-9:
            mujoco.mj_kinematics(sim.m, sim.d)
            mujoco.mj_camlight(sim.m, sim.d)
            pair = []
            for cam in RGB_CAMERAS:
                r = stream.r if reader is None else reader.renderer
                r.update_scene(sim.d, camera=cam)
                pair.append(objects.expose(r.render(), sim.gain, np.random.default_rng((seed, k))) if toss.colors and
                            toss.colors[0] not in COLORS else r.render().copy())
            frames.append((rgb_times.popleft(), pair))
        if k % ev_every == 0:
            j = k // ev_every
            if j == j_first:
                stream.reset(seed)
            elif j > j_first:
                events = stream.slice()
                slices.append(events)
                if np.linalg.norm(sim.grasp_velocity()) > ARM_STILL:      # a still arm makes no events
                    pts = arm_points(sim)
                    masks = [arm_band(cam, pts, ARM_BAND) for cam in CAMERAS]
                else:
                    masks = [np.zeros(RES * RES, bool) for _ in CAMERAS]
                fresh = []
                for ci in range(len(CAMERAS)):
                    pix = np.concatenate([s[ci][0] for s in slices]).astype(np.int64)
                    w = np.concatenate([s[ci][1].astype(np.float64) + s[ci][2] for s in slices])
                    fresh.append((pix, w))
                tracker.update(t - EV_DT, fresh, masks)   # two slices of edges centre one slice back, as in ball.py
                while frames and frames[0][0] <= t - RGB_DELAY + 1e-6:
                    tr, pair = frames.pop(0)
                    if reader is not None:
                        if chosen is None and tracker.confirmed():
                            reader(tracker.confirmed(), tr, pair, toss.instruction)
                        continue
                    for x in tracker.confirmed():
                        vote_colours(x, tr, pair, KIND_AXES if toss.colors and toss.colors[0] in objects.KINDS else COLOUR_AXES)
                live = tracker.confirmed()
                first_seen = t if first_seen is None and live else first_seen
                if switch is not None and t >= toss.hold + switch:
                    rule, chosen, coast, switch = Rule(toss.instruction), None, None, None
                if chosen is not None and chosen not in tracker.tracks:
                    coast = (t, *chosen.state(t)[:2])       # keep flying on the last fit until the ball is found again
                    chosen, lost_chosen = None, lost_chosen + 1
                if chosen is None and coast is not None:
                    dt = t - coast[0]
                    near = [x for x in live if np.linalg.norm(x.predict(t) - ballistic(coast[1], coast[2], dt)) < RELOCK]
                    if near:
                        chosen, coast = near[0], None
                    elif dt > LOST_AFTER * 3:
                        coast = None
                if select == "named":
                    tb = max(t - toss.hold, 0.0)
                    truth = ballistic(np.asarray(toss.launch), np.asarray(toss.velocity), tb)
                    live = [x for x in live if np.linalg.norm(x.predict(t) - truth) < 0.1]
                    chosen = live[0] if live else chosen
                elif select == "rule":
                    complete = rule.colour is not None or len(live) > len(toss.others) or (
                        first_seen is not None and t - first_seen > ALL_SEEN)
                    if chosen is None and complete:
                        chosen = rule.pick(live, t, first_seen is not None and t - first_seen > ALL_SEEN + 0.1)
                elif select == "reader":
                    if chosen is None and live and (len(live) > len(toss.others) or t - first_seen > ALL_SEEN):
                        chosen = reader.pick(live, t - first_seen > ALL_SEEN + 0.1)
                elif select == "incoming":
                    if chosen is None:
                        for x in sorted(live, key=lambda x: -x.points[0][1][0]):
                            p, v, _ = x.state(t)
                            if x.points[0][1][0] > INCOMING_FROM and v[0] < -INCOMING_SPEED:
                                view.estimate = p, v
                                if expert.intercept(p, v) is not None:
                                    chosen = x
                                    break
                elif chosen is None:                 # none: first track the planner can reach
                    for x in sorted(live, key=lambda x: x.points[0][0]):
                        p, v, _ = x.state(t)
                        view.estimate = p, v
                        if expert.intercept(p, v) is not None:
                            chosen = x
                            break
                if chosen is None and coast is not None:
                    view.estimate = ballistic(coast[1], coast[2], t - coast[0]), coast[2] + GRAVITY * (t - coast[0])
                elif chosen is not None:
                    chosen_time = chosen_time or t
                    if first_lock is None:
                        first_lock = t - toss.hold
                    p, v, _ = chosen.state(t)
                    view.estimate = p, v
                    balls = [sim.d.xpos[sim.ball]] + [sim.d.mocap_pos[i] for i in sim.mocap[:len(toss.others)]]
                    gaps = [np.linalg.norm(p - b) for b in balls]
                    picks.append(int(np.argmin(gaps)) if min(gaps) < 0.1 else -1)
                else:
                    view.estimate = None
        if k % CONTROL_EVERY == 0:
            if expert.closed:                         # the ball is in the hand or gone: hold still
                view.estimate = (sim.grasp_point().copy(), np.zeros(3))
            if view.estimate is None:
                target, rot, close = sim.ready_grasp, sim.ready_rot, False
            else:
                target, rot, close = expert.act(released, t)
            sim.track(target, rot, CONTROL_EVERY * PHYSICS_DT)
            sim.set_finger(close)
        if released:
            gap_min = min(gap_min, float(np.linalg.norm(sim.d.xpos[sim.ball] - sim.grasp_point())))
        sim.step_physics()
    return {"caught": sim.caught(), "gap_min": gap_min, "first_lock": first_lock, "lost_chosen": lost_chosen,
            "picks": [picks.count(i) / max(len(picks), 1) for i in (0, 1, 2, -1)] if picks else None,
            "tracks": tracker.count, "instruction": toss.instruction, "flight": toss.flight, "closed": expert.closed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default="named", choices=["none", "incoming", "rule", "named", "reader"],
                    help="named: the true path picks the track (sim only); incoming: a track from the thrower's side "
                         "flying at the robot that the planner can reach (no ground truth)")
    ap.add_argument("--kinds", action="store_true", help="objects named by kind, with a look-alike in every scene")
    ap.add_argument("--reader", default=None, help="fovea.py checkpoint, with --select reader")
    ap.add_argument("--patterns", action="store_true", help="objects named by colour and figure")
    ap.add_argument("--unseen-objects", action="store_true", help="with --patterns: combinations never trained on")
    ap.add_argument("--arm", default="openarm", choices=["openarm", "openarm_v1", "b601"])
    ap.add_argument("--tosses", type=int, default=100)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--distractors", type=int, default=0)
    ap.add_argument("--held-out-phrase", action="store_true")
    ap.add_argument("--relation", action="store_true")
    ap.add_argument("--order", action="store_true")
    ap.add_argument("--drag", type=float, default=0.0, help="ball air drag, 1/m")
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
    ap.add_argument("--ray-gap", type=float, default=RAY_GAP, help="largest gap between the two views' rays, m")
    ap.add_argument("--hand-read", default=None, help="handlook.py checkpoint: the backbone finds the hand in the RGB views")
    ap.add_argument("--camera-cm", type=float, default=0.0, help="camera rigs off their calibration, cm")
    ap.add_argument("--camera-deg", type=float, default=0.0, help="camera rigs off their calibration, degrees")
    ap.add_argument("--switch", type=float, default=None, help="s after release at which the instruction changes")
    ap.add_argument("--out", required=True)
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
    if args.kinds or args.patterns or args.hand_read:
        use_kinds()
    sim = make_sim(args.arm, args.distractors)
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
    Tracker.ray_gap = args.ray_gap
    stream, view = EventStream(sim, noise=args.event_noise, hot=args.hot_pixels), EstimateSim(sim)
    reader = TrackReader(args.reader) if args.select == "reader" else None    # one model per process: a second one reads wrong
    shift = np.zeros((2, 2))
    if args.hand_look or args.hand_read:
        sim.reset_arm()
        shift = bm.view_shift(sim)
    if args.hand_read:
        from .handlook import HandLook
        look = HandLook(args.hand_read, reader.vla if reader else None)
        real, shift = shift, look(sim, mujoco.Renderer(sim.m, SENSOR, SENSOR), np.random.default_rng(args.rig))
        print("views off", np.round(real, 2).tolist(), "read", np.round(shift, 2).tolist(), "px", flush=True)
    bm.set_shift(shift)
    expert, truth = Expert(view), Expert(sim)
    if reader is not None:
        reader.renderer = mujoco.Renderer(sim.m, SENSOR, SENSOR)
    results, t0 = [], time.time()
    words = relation_sentence if args.relation else order_sentence if args.order else None
    for seed in range(args.seed, args.seed + args.tosses):
        toss = (word_scene(sim, seed, args.held_out_phrase, words) if words
                else sim.sample_patterns(np.random.default_rng(seed), args.held_out_phrase, args.unseen_objects) if args.patterns
                else sim.sample_kinds(np.random.default_rng(seed), args.held_out_phrase) if args.kinds
                else sim.sample_scene(np.random.default_rng(seed), args.held_out_phrase) if args.distractors
                else sim.sample_toss(np.random.default_rng(seed)))
        r = rollout(sim, stream, expert, view, toss, seed, args.select, args.switch, reader)
        r["seed"] = seed
        r["expert"] = run_expert(sim, truth, toss)
        results.append(r)
        if len(results) % 20 == 0:
            print(f"{len(results)}/{args.tosses} baseline {np.mean([x['caught'] for x in results]):.1%} "
                  f"expert {np.mean([x['expert'] for x in results]):.1%} {(time.time() - t0) / len(results):.2f} s/toss",
                  flush=True)
    summary = {"select": args.select, "kinds": args.kinds, "patterns": args.patterns, "unseen_objects": args.unseen_objects, "arm": args.arm, "tosses": args.tosses, "distractors": args.distractors,
               "relation": args.relation, "order": args.order, "held_out_phrase": args.held_out_phrase, "switch": args.switch,
               "policy": float(np.mean([x["caught"] for x in results])),
               "expert": float(np.mean([x["expert"] for x in results])),
               "gap_min_cm_median": 100 * float(np.median([x["gap_min"] for x in results]))}
    locks = np.array([x["picks"] for x in results if x["picks"]])
    if len(locks):
        top = locks.argmax(1)
        summary["track_named"] = float((top == 0).sum() / len(results))
        summary["track_distractor"] = float(np.isin(top, (1, 2)).sum() / len(results))
    print(json.dumps(summary), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "tosses": results}, indent=1, default=float))
    os._exit(0)


if __name__ == "__main__":
    main()
