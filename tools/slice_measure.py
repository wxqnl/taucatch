"""Would a higher-resolution event sensor sharpen the per-slice ball measurement? Events rendered at R x R (same 62 deg
view, contrast thresholds as now), the locator's tracking mean shift (radii 8/6/6 px at 224, scaled with R) on the pixels
whose latest event is in the newest 2 slices, gate at the ball's true projection, triangulation with the focal length
of R. The expert on the true state moves the arm. Measurement error against the truth, and the
velocity of a gravity fit over every 4 ms measurement so far.

    python tools/slice_measure.py measure R SEED N OUT.pkl
    python tools/slice_measure.py report OUT1.pkl OUT2.pkl ..."""
import pickle, sys
import numpy as np

if sys.argv[1] == "measure":
    R, seed0, n, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    from taucatch.vla_catch import events as ev
    ev.RES = R                                   # before ball.py computes its focal length from it
    from taucatch.vla_catch import b601
    b601.CATCHER, b601.HAND_RATE, b601.HAND_OPEN = "hand", np.radians(285.0), 0.26
    from taucatch.vla_catch import ball as bm
    assert bm.RES == R
    from taucatch.vla_catch.collect import make_sim, EV_DT
    from taucatch.vla_catch.events import EventStream, CAMERAS
    from taucatch.vla_catch.expert import Expert
    from taucatch.vla_catch.rollout import CONTROL_EVERY, AFTER, PHYSICS_DT
    from taucatch.vla_catch.sim import ballistic
    class Stream(EventStream):          # pixel indices past 65535 at R > 255: keep them as int64, not uint16
        def slice(self):
            out = []
            for level, ref, c in zip(self._levels(), self.ref, self.c):
                n = np.trunc((level - ref) / c)
                ref += n * c
                out.append((np.flatnonzero(n),))
            return out
    sim = make_sim("b601", 0)
    sim.set_throws([8.0], [4.0]); sim.set_bounce(.5); sim.EXPERT_BLEND = .5; sim.EXPERT_LEAD = .05; sim.limit_motors(24)
    stream, expert = Stream(sim), Expert(sim)
    cols, rows_ = np.meshgrid(np.arange(R), np.arange(R))
    cols, rows_ = cols.ravel().astype(float), rows_.ravel().astype(float)
    scale = R / 224
    ev_every = int(round(EV_DT / PHYSICS_DT))

    def shift(fresh, uv):
        for radius in (8 * scale, 6 * scale, 6 * scale):
            near = fresh & (np.hypot(cols - uv[0], rows_ - uv[1]) < radius)
            if near.sum() < 3:
                return None
            uv = np.array([cols[near].mean(), rows_[near].mean()])
        return uv + 0.5

    rows = []
    for seed in range(seed0, seed0 + n):
        toss = sim.sample_toss(np.random.default_rng(seed))
        sim.reset_arm(); sim.paint(toss); expert.reset(); stream.reset(seed)
        last = [np.full(R * R, -np.inf) for _ in CAMERAS]
        L0, V0 = np.asarray(toss.launch), np.asarray(toss.velocity)
        release = int(round(toss.hold / PHYSICS_DT))
        for k in range(int(round((toss.hold + toss.flight + AFTER) / PHYSICS_DT))):
            t = k * PHYSICS_DT
            if k < release:
                sim.place_ball(toss.launch)
            elif k == release:
                sim.place_ball(toss.launch, toss.velocity)
            if k % CONTROL_EVERY == 0:
                target, rot, close = expert.act(k >= release, t)
                sim.track(target, rot, CONTROL_EVERY * PHYSICS_DT)
                sim.set_finger(close)
            if k % ev_every == 0:
                j = k // ev_every
                events = stream.slice()
                for ci in range(len(CAMERAS)):
                    last[ci][events[ci][0]] = j
                tb = t - toss.hold
                if 0.0 < tb < toss.flight - 0.08 and not expert.closed:
                    p = ballistic(L0, V0, tb)
                    uv = [shift((j - last[ci]) < 2, bm.project(cam, p)) for ci, cam in enumerate(CAMERAS)]
                    if all(u is not None for u in uv):
                        rows.append((seed, tb, bm.triangulate(uv), L0, V0, toss.flight, np.linalg.norm(p - sim.ready_grasp)))
            sim.step_physics()
        print(seed, len(rows), flush=True)
    pickle.dump(rows, open(out, "wb"))
    import os; os._exit(0)

from taucatch.vla_catch.sim import GRAVITY, ballistic
groups = {}
for f in sys.argv[2:]:
    groups.setdefault(f.split("_r")[-1].split("_")[0], []).extend(pickle.load(open(f, "rb")))
for R, rows in sorted(groups.items(), key=lambda x: int(x[0])):
    best = None
    for lag in np.arange(-0.008, 0.0081, 0.0005):
        err = np.array([r[2] - ballistic(r[3], r[4], r[1] + lag) for r in rows])
        s = np.median(np.linalg.norm(err, axis=1))
        if best is None or s < best[0]:
            best = (s, lag, err)
    s, lag, err = best
    dist = np.array([r[6] for r in rows])
    print(f"R {R}: {len({r[0] for r in rows})} tosses, {len(rows)} pts, offset {1000 * lag:+.1f} ms")
    for lo, hi in ((2.0, 4.0), (1.0, 2.0), (0.0, 1.0)):
        m = (dist >= lo) & (dist < hi); e = err[m]
        print(f"  {lo:.0f}-{hi:.0f} m: {m.sum()} pts, |err| median {100 * np.median(np.linalg.norm(e, axis=1)):.2f} cm, "
              f"bias {np.round(100 * e.mean(0), 2)} cm, std {np.round(100 * e.std(0), 2)} cm")
    out = {a: [] for a in (0.06, 0.08, 0.1, 0.14, 0.2)}
    for seed in sorted({r[0] for r in rows}):
        S = sorted([r for r in rows if r[0] == seed], key=lambda r: r[1])
        ts = np.array([r[1] for r in S]) + lag; P = np.stack([r[2] for r in S]); V0 = S[0][4]
        for a in out:
            m = (ts <= a) & (ts > a - 0.4)
            if m.sum() >= 3:
                A = np.stack([np.ones(m.sum()), ts[m]], 1)
                c = np.linalg.lstsq(A, P[m] - 0.5 * GRAVITY * ts[m, None] ** 2, rcond=None)[0]
                out[a].append(np.linalg.norm(c[1] - V0))
    print("  velocity error median cm/s (fit over every 4 ms slice): " +
          " ".join(f"{int(1000 * a)} ms {100 * np.median(v):.1f}" for a, v in out.items() if v))
