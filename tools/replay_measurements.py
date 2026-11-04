"""The expert planner on a ball track built from the measurements tools/slice_measure.py recorded (every 4 ms event slice, the
locator's mean shift, triangulated) for the same tosses: real per-slice noise, correlation and bias, replayed. Gravity
least squares over the newest 0.4 s of measurements, used from `start` s after release, each measurement usable `lat` s
after its slice. Variants scale the error per axis (a side-view camera shrinks the error along the throw), take off
the mean error per distance band, or start earlier. B601 hand at 24 V, 8 m/s from 4 m.

    python tools/replay_measurements.py NAME 'res_r224_*.pkl' start lat sx sy sz debias [step]"""
import glob, pickle, sys
import numpy as np
from taucatch.vla_catch import b601
b601.CATCHER, b601.HAND_RATE, b601.HAND_OPEN = "hand", np.radians(285.0), 0.26
from taucatch.vla_catch.collect import make_sim
from taucatch.vla_catch.expert import EstimateSim, Expert
from taucatch.vla_catch.rollout import run_expert
from taucatch.vla_catch.sim import GRAVITY, ballistic

name, pattern = sys.argv[1], sys.argv[2]
start, lat = float(sys.argv[3]), float(sys.argv[4])
scale = np.array([float(x) for x in sys.argv[5:8]])
debias = sys.argv[8] == "1"
step = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0       # s: the estimate refreshed only at policy steps
OFFSET = 0.0035                  # the 2-slice smear sits this far behind the slice time (slice_measure.py fit)
rows = [r for f in sorted(glob.glob(pattern)) for r in pickle.load(open(f, "rb"))]
by_seed = {}
for seed, tb, p, L0, V0, flight, dist in rows:
    truth = ballistic(np.asarray(L0), np.asarray(V0), tb - OFFSET)
    by_seed.setdefault(seed, []).append((tb, np.asarray(p) - truth, truth, dist))
bands = (0.0, 1.0, 2.0, 9.0)
mean_err = [np.mean([e for S in by_seed.values() for _, e, _, d in S if lo <= d < hi], 0) for lo, hi in zip(bands, bands[1:])]

sim = make_sim("b601", 0)
sim.set_throws([8.0], [4.0]); sim.set_bounce(.5); sim.EXPERT_BLEND = .5; sim.EXPERT_LEAD = .05; sim.limit_motors(24)


class Track(EstimateSim):
    def ball_state(self):
        now = self.sim.d.time - self.toss.hold
        t = np.floor(now / step + 1e-9) * step if step else now       # the newest policy step
        if t < start:
            return None
        S = [x for x in self.meas if x[0] <= t - lat]
        S = [x for x in S if S and x[0] > S[-1][0] - 0.4]          # the newest 0.4 s of measurements
        if len(S) < 3:
            return None
        ts = np.array([x[0] for x in S]) - OFFSET
        P = np.stack([x[1] for x in S])
        A = np.stack([np.ones_like(ts), ts], 1)
        c = np.linalg.lstsq(A, P - 0.5 * GRAVITY * ts[:, None] ** 2, rcond=None)[0]
        self.err.append((t, *(c[1] - self.toss.velocity)))
        return c[0] + c[1] * now + 0.5 * GRAVITY * now ** 2, c[1] + GRAVITY * now


class Waiting(Expert):
    def act(self, released, t=0.0):
        if released and self.sim.ball_state() is None:
            released = False
        return super().act(released, t)


view = Track(sim)
view.err = []
expert = Waiting(view)
caught = []
for seed in sorted(by_seed):
    toss = sim.sample_toss(np.random.default_rng(seed))
    meas = []
    for tb, e, truth, d in by_seed[seed]:
        if debias:
            e = e - mean_err[np.searchsorted(bands, d, side="right") - 1]
        meas.append((tb, truth + scale * e))
    view.toss, view.meas = toss, meas
    caught.append(run_expert(view, expert, toss))
e = np.array(view.err)
v = {ms: 100 * np.median(np.abs(e[(e[:, 0] >= ms / 1000) & (e[:, 0] < ms / 1000 + 0.02), 1:]), 0) for ms in (100, 140, 200)}
print(f"{name}: {len(caught)} tosses caught {np.mean(caught):.2f} | |vel err| x,y,z cm/s at " +
      " ".join(f"{ms} ms {np.round(x, 1)}" for ms, x in v.items()), flush=True)
