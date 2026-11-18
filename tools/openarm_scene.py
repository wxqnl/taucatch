"""The OpenArm v2 real-site scene (docs/openarm_scene.md): screenshots and layout, the expert check, the cameras.

    python tools/openarm_scene.py render docs/figures/openarm_scene     # camera views, overviews, layout drawing
    python tools/openarm_scene.py check --tosses 200 --out outputs/checks/openarm_scene/check.json
    python tools/openarm_scene.py cameras --out outputs/checks/openarm_scene/cameras.json [--search]
    python tools/openarm_scene.py tracker --tosses 600 --out outputs/checks/openarm_scene/tracker.json

check: the true-state expert (drag-aware intercept) on launcher tosses, under variants of the catcher, the palm's
rebound, the toss spread, the closing time, a tracker that only reports the ball late or with a fixed error; plus
the palm drop test (rebound of a ball dropped 30 cm onto the palm). No rendering: about 1 s per toss.

cameras: how well a camera set pins the catch. Linearised (Fisher) analysis of a tracker that fits the flight (outlet
position, 1 cm prior; launch velocity, 0.5 m/s prior around the launcher setting) to the ball's image positions:
1 px noise in a 224 px, 62 deg view for every camera, only while the ball is in view; the DVS every 4 ms, exact time,
20 ms latency; RGB at 30 Hz, random phase, 2 ms frame-time jitter, 80 ms latency (USB webcam); the wrist camera held
at its ready pose. Output: the miss in the palm plane (cm) and the arrival-time error (ms), 1 sigma, median over
tosses, of the estimate made 150 ms and 250 ms after launch and 150 ms before arrival. --search: greedy RGB picks,
the best partner for rgb2 on a stand, and the DVS pose with the RGB fixed.

tracker: end to end. The expert acts on that tracker's estimate instead of the true ball: per toss the measurement
noise is drawn once, every control tick uses the frames that have arrived, and the arm holds the ready pose until
the predicted miss (1 sigma) is below the gate (--gate 0: it moves at once).
"""
import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from taucatch.vla_catch import pingpong as P  # noqa: E402
from taucatch.vla_catch.sim import look_at  # noqa: E402

WIDE = ((-0.06, 0.10), (-0.18, 0.18), (-0.14, 0.14))      # the sim's default toss box (sim.sample_toss)
Y0 = float(P.READY_POINT[1])
LABELS = {"dvs": "DVS", "rgb1": "RGB-1", "rgb2": "RGB-2", "wrist": "腕部 RGB"}
COLORS = {"dvs": "#2a78d6", "rgb1": "#1baf7a", "rgb2": "#d6457a"}


def nominal():
    v = P.aim(P.LAUNCHER, P.READY_POINT, P.FLIGHT)
    traj, vel = P.fly(P.LAUNCHER, v, P.FLIGHT)
    return v, traj, vel


# ----- render ------------------------------------------------------------------------------------------------------
def render(out):
    import matplotlib
    matplotlib.use("Agg")
    import mujoco
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    for f in ("Noto Sans CJK SC", "Noto Sans CJK JP"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = [f, "DejaVu Sans"]
            break
    out = Path(out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    _, traj, _ = nominal()
    apex = traj[np.argmax(traj[:, 2])]
    sim = P.PingPongSim("hand", trail=traj[50::50], rigs=True)
    m, d = sim.m, sim.d
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1200
    sim.reset_arm()
    sim.place_ball(P.LAUNCHER)
    mujoco.mj_forward(m, d)

    def shot(w, h, camera):
        r = mujoco.Renderer(m, h, w)
        r.update_scene(d, camera=camera)
        img = r.render()
        r.close()
        return img

    def pixel(camera, p, w, h):
        i = m.camera(camera).id
        R = d.cam_xmat[i].reshape(3, 3)
        c = R.T @ (np.asarray(p) - d.cam_xpos[i])
        f = h / 2 / np.tan(np.radians(m.cam_fovy[i]) / 2)
        return (w / 2 + f * c[0] / -c[2], h / 2 - f * c[1] / -c[2]) if c[2] < 0 else None

    views = {name: (LABELS[name], *spec) for name, spec in P.CAMERAS.items()}
    views["wrist"] = (LABELS["wrist"], None, None, P.WRIST_FOVY, *P.WRIST_SIZE)
    for name, (label, pos, tgt, fovy, w, h) in views.items():
        camera = "camera_wrist_right" if name == "wrist" else name
        img = shot(w, h, camera)
        Image.fromarray(img).save(out / "raw" / f"{name}.png")
        fig = plt.figure(figsize=(w / 160, h / 160 + 0.45), dpi=160)
        ax = fig.add_axes([0, 0, 1, h / (h + 72)])
        ax.imshow(img)
        ax.axhline(h / 2, color="white", lw=0.5, alpha=0.5)
        ax.axvline(w / 2, color="white", lw=0.5, alpha=0.5)
        marks = {"出球口": P.LAUNCHER} if name == "wrist" else \
            {"出球口": P.LAUNCHER, "手的待命点": P.READY_POINT, "最高点": apex, "瞄准点": tgt}
        for text, p in marks.items():
            uv = pixel(camera, p, w, h)
            if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
                continue
            ax.plot(*uv, marker="o", ms=9, mfc="none", mec="#ffd54a", mew=1.6)
            ax.annotate(text, uv, (uv[0] + 0.03 * w, uv[1] - 0.04 * h), color="#ffd54a", fontsize=9,
                        arrowprops=dict(arrowstyle="-", color="#ffd54a", lw=0.8))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        fig.text(0.01, 1 - 0.3 / (h / 160 + 0.45), f"{label}（{w}×{h}，竖直视场 {fovy:.0f}°）", fontsize=9,
                 va="center", color="#111a24")
        fig.savefig(out / f"view_{name}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)

    cam = mujoco.MjvCamera()
    for name, (az, el, dist, at, w, h) in {
            "overview": (-135, -22, 4.6, (1.1, -0.1, 0.8), 1920, 1200),
            "overview_back": (20, -25, 2.4, (1.2, -0.4, 0.9), 1920, 1200),
            "hand": (-150, -12, 0.75, tuple(P.READY_POINT), 1200, 800)}.items():
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = az, el, dist, at
        r = mujoco.Renderer(m, h, w)
        r.update_scene(d, camera=cam)
        Image.fromarray(r.render()).save(out / f"{name}.png")
        r.close()
    layout(out / "layout.png", traj, apex, plt)
    print("wrote", sorted(str(p.relative_to(out)) for p in out.rglob("*.png")))


LABEL_AT = {   # top view: camera label offset (m) and alignment
    "dvs": (0.12, -0.16, "left"), "rgb1": (0.12, 0.14, "left"), "rgb2": (0.12, 0.14, "left")}


def layout(path, traj, apex, plt):
    from matplotlib.patches import Rectangle, Wedge
    ORANGE, INK, MUTED, AQUA = "#eb6834", "#111a24", "#7a8595", "#0f8a63"
    (bx, sx, ry, ly), bh = P.BACKDROP
    L, Rp, y0 = P.LAUNCHER, P.READY_POINT, Y0
    fig, (top, side) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax in (top, side):
        ax.set_aspect("equal")
        ax.grid(True, color="#e6eaef", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # top view
    top.plot([bx, bx], [ry, ly], color="black", lw=4)
    top.plot([bx, sx], [ry, ry], color="black", lw=4)
    top.plot([bx, sx], [ly, ly], color="black", lw=4)
    top.text(bx + 0.08, ly - 0.12, "黑色幕布（三面，高 2.2 m）", fontsize=9, color=INK, va="top")
    top.add_patch(Rectangle((-0.1, -0.13), 0.2, 0.26, fc=INK, ec="none", zorder=3))
    top.text(0.08, 0.2, "OpenArm 底座（原点）", fontsize=9, color=INK, ha="left", va="bottom")
    top.plot([0, Rp[0]], [-0.092, Rp[1]], color=MUTED, lw=3, solid_capstyle="round", zorder=2)
    top.plot(*Rp[:2], marker="o", ms=9, color=AQUA, zorder=4)
    top.text(Rp[0] - 0.05, Rp[1] - 0.25, f"手的待命点\n({Rp[0]:.2f}, {Rp[1]:+.2f}, {Rp[2]:.2f})", fontsize=9,
             color=INK, ha="center", va="top")
    top.add_patch(Rectangle((L[0] - 0.11, y0 - 0.2), 0.4, 0.4, fc="#8a6a4a", alpha=0.35, ec="none"))
    top.plot(*L[:2], marker="o", ms=7, color=ORANGE, zorder=4)
    top.text(L[0] + 0.05, y0 - 0.3, f"发球机出球口\n({L[0]:.2f}, {L[1]:+.2f}, {L[2]:.2f})", fontsize=9, color=INK,
             ha="left", va="top")
    top.plot(traj[:, 0], traj[:, 1], color=ORANGE, lw=2, ls=(0, (4, 3)))
    top.annotate("", (Rp[0], y0 - 0.95), (L[0], y0 - 0.95), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    top.text((Rp[0] + L[0]) / 2 - 0.3, y0 - 1.0, "2.00 m（球的飞行线）", fontsize=9, color=INK, ha="center", va="top")
    for name, (pos, tgt, fovy, w, h) in P.CAMERAS.items():
        color = COLORS[name]
        hfov = np.degrees(2 * np.arctan(w / h * np.tan(np.radians(fovy / 2))))
        ang = np.degrees(np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0]))
        top.add_patch(Wedge(pos[:2], 3.6, ang - hfov / 2, ang + hfov / 2, fc=color, alpha=0.05, ec="none"))
        top.plot([pos[0], tgt[0]], [pos[1], tgt[1]], color=color, lw=0.9, ls=(0, (2, 2)))
        top.plot(*tgt[:2], marker="+", ms=10, mew=1.8, color=color, zorder=4)
        top.plot(*pos[:2], marker="s", ms=10, color=color, zorder=4)
        dx, dy, ha = LABEL_AT[name]
        top.text(pos[0] + dx, pos[1] + dy, f"{LABELS[name]}\n({pos[0]:.2f}, {pos[1]:+.2f}, {pos[2]:.2f})", fontsize=9,
                 color=INK, ha=ha, va="center")
    top.text(1.5, y0 - 0.12, "+ 各相机的瞄准点", fontsize=8, color=MUTED, ha="center", va="top")
    top.annotate("", (bx, -2.35), (0, -2.35), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    top.text(bx / 2, -2.3, f"{-bx:.1f} m", fontsize=9, color=INK, ha="center", va="bottom")
    top.set_xlim(bx - 0.3, max(sx, max(c[0][0] for c in P.CAMERAS.values())) + 0.9)
    top.set_ylim(ry - 0.2, ly + 0.2)
    top.set_xlabel("x（m，朝发球机）")
    top.set_ylabel("y（m，机器人的左边为正）")
    top.set_title("俯视", loc="left", fontsize=13, fontweight="bold", color=INK)
    # side view
    side.plot([bx, sx + 1.0], [0, 0], color=MUTED, lw=1.5)
    side.plot([bx, bx], [0, bh], color="black", lw=4)
    side.add_patch(Rectangle((-0.1, 0), 0.2, 0.697, fc=INK, ec="none"))
    side.text(0.13, 0.35, "臂根高 0.697 m", fontsize=9, color=INK, va="center")
    side.plot([0, Rp[0]], [0.697, Rp[2]], color=MUTED, lw=3, solid_capstyle="round")
    side.plot(Rp[0], Rp[2], marker="o", ms=9, color=AQUA, zorder=4)
    side.text(Rp[0] + 0.1, Rp[2] - 0.1, f"待命点 z {Rp[2]:.2f}", fontsize=9, color=INK, ha="left", va="top")
    for t in P.TOSS_FLIGHT:
        a, _ = P.fly(L, P.aim(L, Rp, t), t)
        side.plot(a[:, 0], a[:, 2], color=ORANGE, lw=1, alpha=0.45)
    side.plot(traj[:, 0], traj[:, 2], color=ORANGE, lw=2, ls=(0, (4, 3)))
    side.annotate(f"最高点 {apex[2]:.2f} m（飞行 {P.FLIGHT:.2f} s）\n浅色：0.55 s / 0.75 s", apex[[0, 2]], (0.3, 1.85),
                  fontsize=9, color=INK, ha="center", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    side.add_patch(Rectangle((L[0] - 0.11, 0), 0.4, 0.54, fc="#8a6a4a", alpha=0.35, ec="none"))
    side.plot(L[0], L[2], marker="o", ms=7, color=ORANGE, zorder=4)
    side.text(L[0] - 0.05, L[2] - 0.12, f"出球口 z {L[2]:.2f}", fontsize=9, color=INK, ha="right", va="top")
    heights = {}
    for name, (pos, *_rest) in P.CAMERAS.items():
        heights.setdefault((pos[0], pos[2]), []).append(name)
    for (x, z), names in heights.items():
        side.plot(x, z, marker="s", ms=10, color=COLORS[names[0]], zorder=4)
        side.text(x + 0.12, z, " / ".join(LABELS[n] for n in names) + f"\nz {z:.2f}", fontsize=9, color=INK,
                  va="center")
    side.set_xlim(bx - 0.3, sx + 1.3)
    side.set_ylim(-0.1, 2.4)
    side.set_xlabel("x（m）")
    side.set_ylabel("z（m，离地）")
    side.set_title("侧视（从机器人右边看）", loc="left", fontsize=13, fontweight="bold", color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


# ----- check -------------------------------------------------------------------------------------------------------
def variant(kw=None, attrs=None, box=P.TOSS_BOX, flight=P.TOSS_FLIGHT, start=0.0, drag=True, off_cm=0.0, off_ms=0.0):
    """kw: PingPongSim arguments; attrs: sim attributes; start: tracker reports from this time after launch;
    drag: drag-aware expert; off_cm, off_ms: the tracker's fixed error per toss (1 sigma), in the palm plane and in
    time, until the hand closes."""
    return dict(kw=kw or {}, attrs=attrs or {}, box=box, flight=flight, start=start, drag=drag, off_cm=off_cm,
                off_ms=off_ms)


VARIANTS = {
    "hand, lined palm (zeta .5), +-10 cm": variant(),
    "hand, toss box +-18 / +-14 cm": variant(box=WIDE),
    "hand, palm without rebound (zeta 1)": variant(kw={"bounce": 1.0}),
    "hand, firmer palm (zeta .3)": variant(kw={"bounce": 0.3}),
    "hand, hard palm (zeta .2)": variant(kw={"bounce": 0.2}),
    "hand, close 30 ms early": variant(attrs={"EXPERT_LEAD": 0.03}),
    "hand, close 80 ms early": variant(attrs={"EXPERT_LEAD": 0.08}),
    "hand, flatter tosses (flight .45-.6 s)": variant(flight=(0.45, 0.60)),
    "hand, tracker reports after 100 ms": variant(start=0.10),
    "hand, tracker reports after 150 ms": variant(start=0.15),
    "hand, tracker reports after 250 ms": variant(start=0.25),
    "hand, tracker off by 1 cm (fixed)": variant(off_cm=1.0),
    "hand, tracker off by 2 cm (fixed)": variant(off_cm=2.0),
    "hand, tracker off by 3 cm (fixed)": variant(off_cm=3.0),
    "hand, tracker off by 10 ms (fixed)": variant(off_ms=10.0),
    "hand, tracker off by 20 ms (fixed)": variant(off_ms=20.0),
    "hand, expert ignores drag": variant(drag=False),
    "v2 gripper (ball does not rebound)": variant(kw={"catcher": "gripper", "bounce": 1.0}),
}


def palm_axes(toss):
    """Two unit vectors across the ball's arrival direction."""
    _, V = P.fly(toss.launch, toss.velocity, toss.flight)
    u = V[-1] / np.linalg.norm(V[-1])
    a = np.cross(u, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(u, a)


def check_job(args):
    name, seeds = args
    var = VARIANTS[name]
    from taucatch.vla_catch.expert import Expert
    from taucatch.vla_catch.rollout import run_expert
    sim = P.PingPongSim(**var["kw"])
    for k, v in var["attrs"].items():
        setattr(sim, k, v)
    expert = (P.DragExpert if var["drag"] else Expert)(sim)
    act, true_state = expert.act, sim.ball_state
    caught = []
    for seed in seeds:
        toss = sim.sample_toss(np.random.default_rng(seed), box=var["box"], flight=var["flight"])
        rng = np.random.default_rng(seed + 7777)
        a, b = palm_axes(toss)
        off = var["off_cm"] / 100 * (rng.normal() * a + rng.normal() * b)
        dt = var["off_ms"] / 1000 * rng.normal()

        def wrapped(released, t=0.0, toss=toss, off=off, dt=dt):
            released = released and t - toss.hold >= var["start"]
            if not released or expert.closed or not (var["off_cm"] or var["off_ms"]):
                return act(released, t)
            def seen():
                p, v = true_state()
                return p + off + v * dt, v
            sim.ball_state = seen
            try:
                return act(True, t)
            finally:
                sim.ball_state = true_state
        expert.act = wrapped
        caught.append(bool(run_expert(sim, expert, toss)))
    return name, caught


def drop_test(bounce, height=0.30):
    """Rebound height (cm) of a ball dropped `height` onto the palm, along the palm normal (gravity along it too)."""
    sim = P.PingPongSim(bounce=bounce)
    sim.drag = 0.0
    n = -sim.site_rot()[:, 2]
    sim.m.opt.gravity[:] = -9.81 * n
    speed = np.sqrt(2 * 9.81 * height)
    sim.place_ball(sim.ready_grasp + 0.01 * n, -speed * n)
    out = 0.0
    for _ in range(300):
        sim.step_physics()
        _, v = sim.ball_state()
        out = max(out, float(v @ n))
    return 100 * out ** 2 / (2 * 9.81)


def check(args):
    seeds = list(range(args.seed, args.seed + args.tosses))
    names = args.only or list(VARIANTS)
    jobs = [(n, seeds[k::args.split]) for n in names for k in range(args.split)]
    res = {n: [] for n in names}
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for name, caught in pool.imap_unordered(check_job, jobs):
            res[name].extend(caught)
    rows = []
    for n in names:
        rate = float(np.mean(res[n]))
        rows.append({"variant": n, "caught": rate, "tosses": len(res[n])})
        print(f"{n:44s} {100 * rate:5.1f}%  ({len(res[n])} tosses)", flush=True)
    drops = {z: drop_test(z) for z in (1.0, 0.5, 0.3, 0.2)}
    for z, cm in drops.items():
        print(f"palm drop test, zeta {z}: a ball dropped 30 cm rebounds {cm:.1f} cm")
    v, traj, vel = nominal()
    toss = {"speed": float(np.linalg.norm(v)), "elevation_deg": float(np.degrees(np.arctan2(v[2], -v[0]))),
            "apex_z": float(traj[:, 2].max()), "arrival_speed": float(np.linalg.norm(vel[-1])),
            "arrival_angle_deg": float(np.degrees(np.arctan2(vel[-1][2], -vel[-1][0])))}
    print("nominal toss:", {k: round(x, 2) for k, x in toss.items()})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows, "drop_rebound_cm": drops, "nominal": toss,
                                              "seed": args.seed}, indent=1))


# ----- cameras: a linearised tracker ---------------------------------------------------------------------------------
SIGMA = np.tan(np.radians(31)) / 112        # 1 px in a 224 px, 62 deg view, in normalised image units
JITTER = 0.002                              # s, RGB frame-time jitter (1 sigma); DVS events carry exact times
LATENCY = {"dvs": 0.02, "rgb": 0.08}        # s from exposure to the tracker
PRIOR = np.diag([1e4] * 3 + [4.0] * 3)      # information: outlet 1 cm, launch velocity 0.5 m/s
NORMAL = -P.READY_ROT[:, 2]                 # palm normal at the ready pose
DECIDE = (0.15, 0.25, -0.15)                # s after launch the estimate is made (negative: before arrival)
SETUPS = {   # name: (cameras, RGB frame rate, RGB latency); None: the true ball
    "true state": None,
    "dvs": (("dvs",), 30, 0.08),
    "dvs + wrist": (("dvs", "wrist"), 30, 0.08),
    "dvs + wrist + rgb1": (("dvs", "wrist", "rgb1"), 30, 0.08),
    "dvs + wrist + rgb1 + rgb2 (chosen)": (("dvs", "wrist", "rgb1", "rgb2"), 30, 0.08),
    "wrist + rgb1 + rgb2 (no dvs)": (("wrist", "rgb1", "rgb2"), 30, 0.08),
    "chosen, RGB at 60 Hz / 50 ms": (("dvs", "wrist", "rgb1", "rgb2"), 60, 0.05),
}
PLACES = {   # DVS poses: (position, look-at)
    "chosen: right, by the launcher": P.CAMERAS["dvs"][:2],
    "behind the arm, right (sim ev_left)": ((-0.25, -1.25, 1.55), (1.3, Y0, 0.95)),
    "overhead behind the arm": ((-0.5, Y0, 1.90), (1.4, Y0, 0.9)),
    "side, square to the flight": ((1.4, -1.9, 1.10), (1.4, Y0, 1.10)),
    "right, halfway, high": ((1.4, -1.72, 1.90), (1.2, Y0, 1.05)),
}


class Cam:
    """A fixed pinhole camera that reports the ball's image position."""

    def __init__(self, pos, target=None, rate=30.0, fovy=43.0, aspect=16 / 9, jitter=JITTER,
                 latency=LATENCY["rgb"], R=None, up=(0, 0, 1)):
        self.pos = np.asarray(pos, float)
        if R is None:
            x, y = look_at(pos, target, up).reshape(2, 3)
            R = np.stack([x, y, np.cross(x, y)])
        self.R, self.rate, self.jitter, self.latency = R, rate, jitter, latency     # R rows: image x, image y, back
        self.tv = np.tan(np.radians(fovy / 2))
        self.tu = aspect * self.tv

    def frames(self, lin, rng):
        """Frames of one flight with the ball in view: (arrival times at the tracker, H n x 2 x 6, cov n x 2 x 2)."""
        t0 = 0.0 if self.rate > 100 else rng.uniform(0, 1 / self.rate)
        ts = np.arange(t0, lin["t"], 1 / self.rate)
        k = np.minimum(np.round(ts / 0.001).astype(int), len(lin["P"]) - 1)
        c = (lin["P"][k] - self.pos) @ self.R.T
        depth = -c[:, 2]
        ok = depth > 0.2
        depth = np.where(ok, depth, 1.0)
        u, v = c[:, 0] / depth, c[:, 1] / depth
        ok &= (np.abs(u) < 0.97 * self.tu) & (np.abs(v) < 0.97 * self.tv)
        k, depth, u, v = k[ok], depth[ok], u[ok], v[ok]
        x, y, z = self.R
        J = np.stack([(x + u[:, None] * z) / depth[:, None], (y + v[:, None] * z) / depth[:, None]], 1)
        udot = J @ lin["V"][k][:, :, None]
        cov = SIGMA ** 2 * np.eye(2) + self.jitter ** 2 * udot @ udot.transpose(0, 2, 1)
        return ts[ok] + self.latency, J @ lin["S"][k], cov

    def sees(self, pts):
        c = (pts - self.pos) @ self.R.T
        depth = -c[:, 2]
        return bool(np.all(depth > 0.2) and np.all(np.abs(c[:, 0] / depth) < self.tu)
                    and np.all(np.abs(c[:, 1] / depth) < self.tv))


@functools.lru_cache(None)
def wrist_pose():
    import mujoco
    sim = P.PingPongSim()
    sim.reset_arm()
    mujoco.mj_forward(sim.m, sim.d)
    i = sim.m.camera("camera_wrist_right").id
    return sim.d.cam_xpos[i].copy(), sim.d.cam_xmat[i].reshape(3, 3).T.copy()


def make_cams(names, rate=30.0, latency=LATENCY["rgb"]):
    out = []
    for n in names:
        if n == "dvs":
            pos, tgt, fovy, w, h = P.CAMERAS["dvs"]
            out.append(Cam(pos, tgt, 250.0, fovy, w / h, 0.0, LATENCY["dvs"]))
        elif n == "wrist":
            pos, R = wrist_pose()
            out.append(Cam(pos, rate=rate, fovy=P.WRIST_FOVY, aspect=P.WRIST_SIZE[0] / P.WRIST_SIZE[1],
                           latency=latency, R=R))
        else:
            pos, tgt = P.CAMERAS[n][:2]
            out.append(Cam(pos, tgt, rate, latency=latency))
    return out


def linearise(launch, velocity, flight, extra=0.0):
    """Flight every 1 ms to flight + extra, its sensitivities to x = (outlet, launch velocity), and the maps from an
    error in x to the miss in the palm plane (M, 3 x 6) and to the arrival time (g, 6)."""
    x0 = np.concatenate([launch, velocity])
    T = flight + extra
    Pn, Vn = P.fly(launch, velocity, T)
    S, SV = np.zeros((len(Pn), 3, 6)), np.zeros((len(Pn), 3, 6))
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = 1e-5
        a, b = P.fly(*np.split(x0 + dx, 2), T), P.fly(*np.split(x0 - dx, 2), T)
        S[:, :, i], SV[:, :, i] = (a[0] - b[0]) / 2e-5, (a[1] - b[1]) / 2e-5
    k = round(flight / 0.001)
    nv = NORMAL @ Vn[k]
    M = (np.eye(3) - np.outer(Vn[k], NORMAL) / nv) @ S[k]
    return dict(P=Pn, V=Vn, S=S, SV=SV, t=flight, M=M, g=-(NORMAL @ S[k]) / nv)


def fisher_tosses(n=8, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        target = P.READY_POINT + np.array([rng.uniform(*b) for b in P.TOSS_BOX])
        t = rng.uniform(*P.TOSS_FLIGHT)
        out.append(linearise(P.LAUNCHER, P.aim(P.LAUNCHER, target, t), t))
    return out


def infos(cam, tosses, draws, rng):
    """Information (6 x 6) from one camera at each decision time, per toss x frame-phase draw."""
    out = np.zeros((len(DECIDE), len(tosses) * draws, 6, 6))
    for i, lin in enumerate(tosses):
        for j in range(draws):
            T, H, C = cam.frames(lin, rng)
            if not len(T):
                continue
            F = np.einsum("nai,nab,nbj->nij", H, np.linalg.inv(C), H)
            for s, t in enumerate(DECIDE):
                out[s, i * draws + j] = F[T <= (t if t > 0 else lin["t"] + t)].sum(0)
    return out


def errors(info, tosses, draws):
    """Median miss in the palm plane (cm) and arrival-time error (ms), 1 sigma, per decision time."""
    res = []
    for s in range(info.shape[0]):
        m = []
        for n in range(info.shape[1]):
            lin = tosses[n // draws]
            C = np.linalg.inv(PRIOR + info[s, n])
            m.append((100 * np.sqrt(np.trace(lin["M"] @ C @ lin["M"].T)), 1000 * np.sqrt(lin["g"] @ C @ lin["g"])))
        res.append(np.median(m, 0))
    return np.array(res)


def candidates():
    """RGB poses to search: beside the flight, behind the robot, past the launcher, overhead, by the hand."""
    tg = {"early": (2.0, Y0, 1.15), "apex": (1.3, Y0, 1.25), "mid": (1.5, Y0, 1.05), "catch": (0.6, Y0, 1.0)}
    pos = [(x, Y0 + dy, z) for x in (-0.6, 0.0, 0.6, 1.2, 1.8, 2.4) for dy in (-2.3, -1.8, -1.2, 1.2, 1.8, 2.3)
           for z in (0.5, 1.0, 1.5, 2.0)]
    pos += [(-0.8, Y0 + dy, z) for dy in (-1.2, -0.6, 0.6, 1.2) for z in (1.4, 2.0)]
    pos += [(2.9, Y0 + dy, z) for dy in (-1.2, -0.6, 0.6, 1.2) for z in (1.0, 1.5, 2.0)]
    pos += [(x, Y0 + dy, 2.15) for x in (0.9, 1.5, 2.1) for dy in (-0.6, 0.0, 0.6)]
    out = [((x, Y0 + dy, z), t) for x in (0.1, 0.4) for dy in (-0.55, -0.4, 0.4) for z in (0.7, 1.0, 1.3)
           for t in ((2.2, Y0, 0.95), (2.0, Y0, 1.15), (1.3, Y0, 1.25))]
    out += [(p, t) for p in pos for t in tg.values() if np.linalg.norm(np.subtract(p, t)) >= 0.8]
    return [(p, t, Cam(p, t, up=(0, 0, 1) if abs(p[0] - t[0]) + abs(p[1] - t[1]) > 0.3 else (1, 0, 0)))
            for p, t in out]


def fmt(e):
    return "  ".join(f"{a:5.1f} cm {b:5.1f} ms" for a, b in e)


def cameras(args):
    tosses, draws = fisher_tosses(), 4

    def seeded(*key):
        """Frame phases fixed per camera, so adding a row does not change the others."""
        return np.random.default_rng(zlib.crc32(repr(key).encode()))

    def info(names, rate=30, lat=LATENCY["rgb"]):
        return sum(infos(c, tosses, draws, seeded(n, rate)) for n, c in zip(names, make_cams(names, rate, lat)))

    print("estimate made 150 ms after launch / 250 ms after launch / 150 ms before arrival: "
          "palm-plane miss, arrival time (1 sigma, median)")
    rows = {"setups": [], "dvs_places": []}
    for name, setup in SETUPS.items():
        if setup is None:
            continue
        e = errors(info(*setup), tosses, draws)
        rows["setups"].append({"setup": name, "errors": e.tolist()})
        print(f"{name:44s} {fmt(e)}", flush=True)
    fixed = info(("wrist",) + RGB)
    pts = np.vstack([lin["P"][::10] for lin in tosses] + [P.READY_POINT[None]])
    print("\nDVS pose: alone, and with the wrist + rgb1 + rgb2 (estimate 150 ms after launch)")
    for name, (pos, tgt) in PLACES.items():
        cam = Cam(pos, tgt, 250.0, 62.0, 4 / 3, 0.0, LATENCY["dvs"])
        own = infos(cam, tosses, draws, seeded(name))
        alone, both = errors(own, tosses, draws), errors(own + fixed, tosses, draws)
        rows["dvs_places"].append({"place": name, "sees_everything": cam.sees(pts), "alone": alone.tolist(),
                                   "with_rgb": both.tolist()})
        print(f"{name:40s} sees all {str(cam.sees(pts)):5s}  {fmt(alone[:1])}   {fmt(both[:1])}", flush=True)
    if args.search:
        rows["search"] = search(tosses, draws, seeded("search"), info, fixed, pts)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))


RGB = ("rgb1", "rgb2")


def search(tosses, draws, rng, info, fixed, pts, k=3, apart=0.6):
    """Greedy RGB picks from the DVS + wrist (one camera per spot, apart from each other and from the DVS); the best
    partner for rgb2 on a stand (<= 2 m up, seeing the outlet, the apex and the hand); the DVS pose with the RGB
    fixed."""
    def score(i):
        e = errors(i, tosses, draws)
        return np.log(e[0, 0]) + np.log(e[1, 0]) + np.log(e[2, 0]) + np.log(e[2, 1] / 10), e

    cands = candidates()
    cand_info = [infos(c, tosses, draws, rng) for _, _, c in cands]
    out = {}
    for start in (("dvs", "wrist"),):
        cur, taken, picks = info(start), [np.array(P.CAMERAS["dvs"][0])], []
        print(f"\ngreedy RGB from {' + '.join(start)}: {fmt(errors(cur, tosses, draws))}")
        for _ in range(k):
            best = None
            for i, (p, t, _) in enumerate(cands):
                if all(np.linalg.norm(np.subtract(p, q)) >= apart for q in taken):
                    s, e = score(cur + cand_info[i])
                    if best is None or s < best[0]:
                        best = (s, i, e)
            _, i, e = best
            cur, p, t = cur + cand_info[i], cands[i][0], cands[i][1]
            taken.append(np.array(p))
            picks.append({"pos": p, "look_at": t, "errors": e.tolist()})
            print(f"  + {p} -> {t}  {fmt(e)}", flush=True)
        out[" + ".join(start)] = picks
    _, traj, _ = nominal()
    key = np.vstack([P.LAUNCHER, traj[np.argmax(traj[:, 2])], P.READY_POINT])
    base = info(("dvs", "wrist", "rgb2"))
    partners = []
    for i, (p, t, c) in enumerate(cands):
        if p[2] <= 2.0 and c.sees(key) and all(np.linalg.norm(np.subtract(p, P.CAMERAS[n][0])) >= apart
                                               for n in ("dvs", "rgb2")):
            partners.append({"pos": p, "look_at": t, "errors": errors(base + cand_info[i], tosses, draws).tolist()})
    partners.sort(key=lambda r: r["errors"][0][0])
    print(f"\npartner for rgb2 (dvs + wrist + rgb2 + one more), {len(partners)} poses on a stand; best:")
    for r in partners[:5]:
        print(f"  {r['pos']} -> {r['look_at']}  {fmt(np.array(r['errors']))}")
    print(f"  rgb1 {P.CAMERAS['rgb1'][0]}  {fmt(errors(base + info(('rgb1',)), tosses, draws))}")
    out["partner"] = partners[:20]
    found = []
    for x in np.arange(0.0, 3.01, 0.3):
        for dy in (-2.1, -1.5, -1.2, 1.2, 1.5, 2.1):
            for z in (1.0, 1.6, 2.1):
                for ax in (0.9, 1.5):
                    pos, tgt = (round(float(x), 2), Y0 + dy, z), (ax, Y0, 1.05)
                    cam = Cam(pos, tgt, 250.0, 62.0, 4 / 3, 0.0, LATENCY["dvs"])
                    near = any(np.linalg.norm(np.subtract(pos, P.CAMERAS[n][0])) < apart for n in RGB)
                    if near or not cam.sees(pts):
                        continue
                    e = errors(infos(cam, tosses, draws, rng) + fixed, tosses, draws)
                    found.append({"pos": pos, "look_at": tgt, "errors": e.tolist()})
    found.sort(key=lambda r: np.log(r["errors"][0][0]) + np.log(r["errors"][1][0]))
    print(f"\nDVS poses that see every flight and the hand, with the wrist + rgb1 + rgb2: {len(found)}; best:")
    for r in found[:8]:
        print(f"  {r['pos']} -> {r['look_at']}  {fmt(np.array(r['errors']))}")
    out["dvs"] = found[:20]
    return out


# ----- tracker: the expert on the estimate ---------------------------------------------------------------------------
def estimates(cams, toss, rng):
    """The linearised tracker over one flight: when each estimate becomes available, its state error, and the
    predicted miss (1 sigma); plus the flight's sensitivities."""
    lin = linearise(toss.launch, toss.velocity, toss.flight, extra=0.15)
    T, H, C, N = [], [], [], []
    for cam in cams:
        t, h, c = cam.frames(lin, rng)
        T.append(t)
        H.append(h)
        C.append(c)
        N.append(np.array([rng.multivariate_normal(np.zeros(2), ci) for ci in c]).reshape(-1, 2))
    T, H, C, N = (np.concatenate(a) for a in (T, H, C, N))
    o = np.argsort(T, kind="stable")
    Ci = np.linalg.inv(C[o])
    F = np.einsum("nai,nab,nbj->nij", H[o], Ci, H[o])
    b = np.einsum("nai,nab,nb->ni", H[o], Ci, N[o])
    n0 = np.concatenate([rng.normal(0, 0.01, 3), rng.normal(0, 0.5, 3)])
    info = PRIOR + np.concatenate([np.zeros((1, 6, 6)), np.cumsum(F, 0)])
    vec = PRIOR @ n0 + np.concatenate([np.zeros((1, 6)), np.cumsum(b, 0)])
    err = np.linalg.solve(info, vec[:, :, None])[:, :, 0]
    cov = np.linalg.inv(info)
    miss = np.sqrt(np.einsum("ij,njk,ik->n", lin["M"], cov, lin["M"]))
    return T[o], err, miss, lin


def tracker_job(args):
    name, gate, seeds = args
    from taucatch.vla_catch.rollout import run_expert
    sim = P.PingPongSim()
    expert = P.DragExpert(sim)
    setup = SETUPS[name]
    cams = make_cams(*setup) if setup else []
    act, true_state = expert.act, sim.ball_state
    caught = []
    for seed in seeds:
        toss = sim.sample_toss(np.random.default_rng(seed))
        if setup is None:
            expert.act = act
            caught.append(bool(run_expert(sim, expert, toss)))
            continue
        T, err, miss, lin = estimates(cams, toss, np.random.default_rng(seed + 99))
        ok = np.nonzero(miss < gate / 100)[0] if gate > 0 else np.array([0])      # gate 0: move at once
        start = max(T[0] + 0.01 if len(T) else 9.0, (T[ok[0] - 1] if ok[0] else 0.0) if len(ok) else 9.0)

        def wrapped(released, t=0.0, toss=toss, T=T, err=err, lin=lin, start=start):
            tr = t - toss.hold
            if not released or tr < start:
                return act(False, t)
            if expert.closed:                     # the ball is in the hand: the tracker sees it there
                return act(True, t)
            e = err[np.searchsorted(T, tr, side="right")]
            k = min(round(tr / 0.001), len(lin["S"]) - 1)
            def seen():
                p, v = true_state()
                return p + lin["S"][k] @ e, v + lin["SV"][k] @ e
            sim.ball_state = seen
            try:
                return act(True, t)
            finally:
                sim.ball_state = true_state
        expert.act = wrapped
        caught.append(bool(run_expert(sim, expert, toss)))
    return name, gate, caught


def tracker(args):
    seeds = list(range(args.seed, args.seed + args.tosses))
    names = args.only or list(SETUPS)
    jobs = [(n, g, seeds[k::args.split]) for g in args.gate for n in names for k in range(args.split)
            if SETUPS[n] is not None or g == args.gate[0]]
    res = {}
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for name, gate, caught in pool.imap_unordered(tracker_job, jobs):
            res.setdefault((name, gate), []).extend(caught)
    rows = []
    print(f"{'':44s} " + "  ".join(f"gate {g:g} cm" for g in args.gate))
    for n in names:
        cells = [res.get((n, g), res.get((n, args.gate[0]))) for g in args.gate]
        rows.append({"setup": n, "caught": {str(g): float(np.mean(c)) for g, c in zip(args.gate, cells)},
                     "tosses": len(cells[0])})
        print(f"{n:44s} " + "  ".join(f"{100 * np.mean(c):8.1f}%" for c in cells) + f"   ({len(cells[0])} tosses)",
              flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows, "seed": args.seed, "gates_cm": args.gate}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("out", nargs="?", default=str(ROOT / "docs/figures/openarm_scene"))
    for name, n in (("check", 200), ("tracker", 600)):
        c = sub.add_parser(name)
        c.add_argument("--tosses", type=int, default=n)
        c.add_argument("--seed", type=int, default=3000000)
        c.add_argument("--only", nargs="*", default=None)
        c.add_argument("--split", type=int, default=10)
        c.add_argument("--workers", type=int, default=100)
        c.add_argument("--out", default=None)
        if name == "tracker":
            c.add_argument("--gate", type=float, nargs="+", default=[3.0, 5.0],
                           help="cm: the arm holds the ready pose until the predicted miss is below this (0: no gate)")
    p = sub.add_parser("cameras")
    p.add_argument("--search", action="store_true", help="greedy RGB picks and the DVS pose search")
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    {"render": lambda: render(args.out), "check": lambda: check(args), "cameras": lambda: cameras(args),
     "tracker": lambda: tracker(args)}[args.cmd]()


if __name__ == "__main__":
    main()
