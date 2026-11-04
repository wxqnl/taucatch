"""Ball measurement for the policy state: per view, the centroid of the two newest event slices inside a gate,
triangulated, then a gravity-aware fit over the track since release -> ball position (relative to the grasp
centre) and velocity at the policy step.

On the recorded tosses the triangulated centroid is 0.64 cm from the ball (median 0.50 cm), and it stays at
that level with the gate centre 12 px off. The gate centre here is the true projection plus a random offset of
up to GATE_NOISE px, standing in for a coarse ball locator.
"""
import numpy as np

from .collect import EV_DT, STEP
from .events import CAMERAS, RES
from .sim import EVENT_CAMERAS, EVENT_FOVY, GRAVITY, look_at, rgb_pose

BALL_DIM = 7                  # position relative to the grasp centre, velocity, seen
GATE, GATE_NOISE = 20, 12     # px
SPAN = 0.4                    # s of track in the fit
FOCAL = (RES / 2) / np.tan(np.radians(EVENT_FOVY) / 2)
_CAMS, _RGB = {}, {}           # event cameras, and the RGB camera beside each (same orientation, 3 cm along x)
for _name, (_pos, _target) in EVENT_CAMERAS.items():
    for _table, (_p, _t) in ((_CAMS, (_pos, _target)), (_RGB, rgb_pose(_name))):
        _xy = look_at(_p, _t)
        _table[_name] = (np.asarray(_p, float), np.stack([_xy[:3], _xy[3:], np.cross(_xy[:3], _xy[3:])], 1))


SHIFT = {}                     # px per view: where moved rigs show what the calibration puts elsewhere (set_shift)


def set_shift(shift):
    """Offsets [2, 2] px of the views: added to projections, taken off pixels before they become rays."""
    SHIFT.update({cam: np.asarray(x, float) for cam, x in zip(CAMERAS, shift)})


def _project(c, R, p):
    q = R.T @ (np.asarray(p) - c)
    return np.array([RES / 2 + FOCAL * q[0] / -q[2], RES / 2 - FOCAL * q[1] / -q[2]])      # (col, row)


def project(cam, p):
    return _project(*_CAMS[cam], p) + SHIFT.get(cam, 0.0)


def project_rgb(cam, p):
    """Pixel (col, row) of p in the RGB camera beside event camera cam."""
    return _project(*_RGB[cam], p) + SHIFT.get(cam, 0.0)


def _ray(cam, uv):
    c, R = _CAMS[cam]
    uv = np.asarray(uv) - SHIFT.get(cam, 0.0)
    d = R @ np.array([(uv[0] - RES / 2) / FOCAL, -(uv[1] - RES / 2) / FOCAL, -1.0])
    return c, d / np.linalg.norm(d)


def triangulate(uvs):
    (c1, d1), (c2, d2) = (_ray(cam, uv) for cam, uv in zip(CAMERAS, uvs))
    s, t = np.linalg.lstsq(np.stack([d1, -d2], 1), c2 - c1, rcond=None)[0]
    return (c1 + s * d1 + c2 + t * d2) / 2


def view_shift(sim):
    """Where the hand shows in each view, less where the calibration puts it [2, 2] px: what one look at the hand tells
    about rigs or an arm base that have moved. A pixel offset undoes a rig's turn at every depth, its move at the hand's."""
    import mujoco
    mujoco.mj_kinematics(sim.m, sim.d)
    mujoco.mj_camlight(sim.m, sim.d)
    g = sim.grasp_point()
    out = []
    for cam in CAMERAS:
        i = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        out.append(_project(sim.d.cam_xpos[i].copy(), sim.d.cam_xmat[i].reshape(3, 3), g) - _project(*_CAMS[cam], g))
    return np.stack(out)


def centroid(pix, weight, guess):
    """Event-weighted mean shift from the gate centre; None with fewer than 3 events in the gate."""
    col, row = pix % RES, pix // RES
    uv = np.asarray(guess, float)
    for radius in (GATE, 10, 10):
        near = np.hypot(col - uv[0], row - uv[1]) < radius
        if near.sum() < 3:
            return None
        uv = np.array([np.average(col[near], weights=weight[near]), np.average(row[near], weights=weight[near])])
    return uv + 0.5


def gate_centres(ball, rng):
    """Gate centre per view: true projection off by up to GATE_NOISE px."""
    out = []
    for cam in CAMERAS:
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, GATE_NOISE)
        out.append(project(cam, ball) + r * np.array([np.cos(a), np.sin(a)]))
    return out


def measure(events, guesses):
    """events: per view (pix, weight) of the two newest slices -> triangulated ball, or None."""
    uvs = []
    for (pix, weight), guess in zip(events, guesses):
        uv = centroid(pix.astype(np.int64), weight.astype(np.float64), guess)
        if uv is None:
            return None
        uvs.append(uv)
    return triangulate(uvs)


def features(track, t, grasp):
    """track: [(time, xyz)] measured so far -> [BALL_DIM] at time t (zeros until two measurements)."""
    pts = [(tk, p) for tk, p in track if tk >= t - SPAN]
    if len(pts) < 2:
        return np.zeros(BALL_DIM, np.float32)
    dt = np.array([tk - t for tk, _ in pts])
    y = np.stack([p for _, p in pts]) - 0.5 * GRAVITY * dt[:, None] ** 2
    A = np.stack([np.ones_like(dt), dt], 1)
    (pos, vel), *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.concatenate([pos - grasp, vel, [1.0]]).astype(np.float32)


# The in-model locator (locator.py): one memory slot per policy call, gates from the track's own prediction,
# binary fresh pixels (the trace frame keeps no event counts). track_memory rebuilds that memory from recorded
# events for training, with the acquisition gate taken from the true projection (the scorer's job at run time).
MEMORY = 20                   # slots: 20 calls = 0.4 s
ACQUIRE_NOISE = 20            # px


def camera(cam):
    return _CAMS[cam]


def fresh_pixels(events, j):
    """Pixels whose latest event is in slice j-1 or j, per view: (pix, ones)."""
    out = []
    for cam in CAMERAS:
        sl, pix = events[cam][0], events[cam][1]
        p = np.unique(pix[np.searchsorted(sl, j - 1):np.searchsorted(sl, j + 1)].astype(np.int64))
        out.append((p, np.ones(p.size)))
    return out


def predict(track, t):
    """Gravity fit of the track (ages up to SPAN + one step before t) -> position at t, or None."""
    pts = [(tk, p) for tk, p in track if tk >= t - SPAN - 0.0045]
    if len(pts) < 2:
        return None
    f = features(pts, t, np.zeros(3))
    return f[:3]


def track_memory(toss, s, rng):
    """Memory the locator holds at step s: slot i = measurement of call s-1-i (world xyz), valid flags."""
    memory, valid = np.zeros((MEMORY, 3), np.float32), np.zeros(MEMORY, bool)
    track = []
    for k in range(s):
        j = int(round(toss.times[k] / EV_DT))
        t_obs = (j - 1) * EV_DT
        recent = [(tk, p) for tk, p in track if tk > toss.times[k] - STEP * (MEMORY + 1)]
        p = predict(recent, t_obs)
        if p is not None:
            guesses = [project(cam, p) for cam in CAMERAS]
        else:
            truth = toss.ball_at(t_obs)
            guesses = []
            for cam in CAMERAS:
                a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, ACQUIRE_NOISE)
                guesses.append(project(cam, truth) + r * np.array([np.cos(a), np.sin(a)]))
        m = measure(fresh_pixels(toss.events, j), guesses)
        if m is not None:
            track.append((t_obs, m))
            slot = s - 1 - k
            if slot < MEMORY:
                memory[slot], valid[slot] = m, True
    return memory, valid


def acquisition(toss, s, rng):
    """Teacher gate (true projection + noise) and the scorer's target patch (196 = no moving ball), per view."""
    j = int(round(toss.times[s] / EV_DT))
    t_obs = (j - 1) * EV_DT
    truth = toss.ball_at(t_obs)
    moving = toss.meta["release"] + 2 * EV_DT <= t_obs <= toss.meta["release"] + toss.meta["flight"]
    gates, targets = [], []
    for cam in CAMERAS:
        uv = project(cam, truth)
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, ACQUIRE_NOISE)
        gates.append(uv + r * np.array([np.cos(a), np.sin(a)]))
        col, row = int(uv[0] // (RES // 14)), int(uv[1] // (RES // 14))
        targets.append(row * 14 + col if moving and 0 <= col < 14 and 0 <= row < 14 else 196)
    return np.array(gates, np.float32), np.array(targets, np.int64)
