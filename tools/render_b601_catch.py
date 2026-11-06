"""GIF of SpikeVLA catching an 8 m/s ball with the B601 five-finger hand: overview + the two event views.

Two processes, since OSMesa gives one renderer per process:
    record: run the judge setting for one seed (evaluate.main), keep qpos and both event-camera trace frames
            at every 4 ms slice -> NPZ
    render: replay the qpos into the scene, render an overview, compose with the trace frames -> GIF

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa VLA_AIM_X=1.3 CUDA_VISIBLE_DEVICES=5 \\
        python tools/render_b601_catch.py record 1000086 weights/spikevla-catch-b601 OUT.npz
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa VLA_AIM_X=1.3 python tools/render_b601_catch.py render OUT.npz OUT.gif
"""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
JUDGE = ("--fast --close-track --close-hand --slice-track --query-every 0.004 --catcher hand --hand-rate 285 "
         "--hand-open .26 --lead .05 --bounce .5 --volts 24 --blend .5 --arm b601 --latency .02 "
         "--throw-speed 8 --throw-from 4 --tosses 1 --batch 1").split()


def record(seed, ckpt, out):
    from taucatch.vla_catch import evaluate as E
    from taucatch.vla_catch import events as EV
    from taucatch.vla_catch.sim import CatchSim
    frames, states, times = [], [], []
    last = {}
    original = EV.EventStream.slice

    def slice_(self):
        events = original(self)
        if not last:
            last.update({c: np.full(EV.RES * EV.RES, -np.inf) for c in range(len(events))})
            last["j"] = 0
        last["j"] += 1
        for c, ev in enumerate(events):
            last[c][ev[0].astype(np.int64)] = last["j"]
        age = [((last["j"] - last[c]) * EV.SLICE_S).astype(np.float32) for c in range(len(events))]
        frames.append(np.stack([EV.trace_frame(a) for a in age]))
        states.append(self.sim.d.qpos.copy())
        times.append(self.sim.d.time)
        return events
    EV.EventStream.slice = slice_
    E.EventStream = EV.EventStream

    def exit_(code):
        result = json.loads(Path(out + ".json").read_text())
        toss = result["tosses"][0]
        np.savez_compressed(out, frames=np.stack(frames), qpos=np.stack(states), time=np.array(times),
                            seed=seed, caught=toss["caught"], flight=toss["flight"],
                            release=_release(seed))
        print("saved", out, len(frames), "slices, caught", toss["caught"], flush=True)
        os._exit(code)
    E.os = types.SimpleNamespace(_exit=exit_)
    sys.argv = ["evaluate", "--ckpt", ckpt, *JUDGE, "--seed", str(seed), "--out", out + ".json"]
    E.main()


def _release(seed):
    from taucatch.vla_catch.collect import make_sim
    sim = make_sim("b601", 0)
    sim.set_throws([8.0], [4.0])
    return sim.sample_toss(np.random.default_rng(seed)).hold


def render(path, out):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from taucatch.vla_catch import b601
    b601.CATCHER, b601.HAND_RATE, b601.HAND_OPEN = "hand", np.radians(285.0), 0.26
    from taucatch.vla_catch.collect import make_sim
    z = np.load(path)
    sim = make_sim("b601", 0)
    W, H = 640, 400
    r = mujoco.Renderer(sim.m, H, W)
    wide = mujoco.MjvCamera()
    wide.lookat[:] = (2.0, -0.3, 1.0)
    wide.distance, wide.azimuth, wide.elevation = 3.1, -103.0, -9.0
    close = mujoco.MjvCamera()
    close.lookat[:] = (0.42, -0.22, 0.97)
    close.distance, close.azimuth, close.elevation = 0.8, 150.0, -12.0
    font, small = _font(18), _font(14)
    t_rel = z["time"] - float(z["release"])
    keep = np.flatnonzero((t_rel > -0.06) & (t_rel < float(z["flight"]) + 0.12))
    band, side, gap = 54, 200, 8
    trail, out_frames = [], []
    for k in keep:
        sim.d.qpos[:] = z["qpos"][k]
        mujoco.mj_forward(sim.m, sim.d)
        trail.append(sim.d.xpos[sim.ball].copy())
        r.update_scene(sim.d, camera=wide)
        for p in trail[-60:-1:3]:                     # the ball's last 0.24 s as small dots
            _dot(r.scene, p)
        img = Image.fromarray(r.render())
        r.update_scene(sim.d, camera=close)
        zoom = Image.fromarray(r.render()).resize((240, 150), Image.LANCZOS)
        img.paste(zoom, (gap, H - 150 - gap))
        views = [Image.fromarray(v).resize((side, side), Image.NEAREST) for v in z["frames"][k]]
        canvas = Image.new("RGB", (W + side + 3 * gap, H + band), (252, 252, 251))
        canvas.paste(img, (gap, band))
        for i, v in enumerate(views):
            canvas.paste(v, (W + 2 * gap, band + i * (side + 0)))
        d = ImageDraw.Draw(canvas)
        d.text((gap, 6), f"出手后 {1000 * t_rel[k]:+4.0f} ms", fill=(17, 26, 36), font=font)
        d.text((gap, 31), "SpikeVLA 控制 B601 五指手，球 8 m/s，从 4 m 外抛出（20 倍慢放）", fill=(82, 81, 78), font=small)
        d.rectangle((gap, band + H - 150 - gap, gap + 240, band + H - gap), outline=(252, 252, 251), width=2)
        d.text((2 * gap + 2, band + H - 150 - gap + 4), "手部特写", fill=(240, 239, 236), font=small)
        d.text((W + 2 * gap + 6, band + 4), "事件相机（左）", fill=(240, 239, 236), font=small)
        d.text((W + 2 * gap + 6, band + side + 4), "事件相机（右）", fill=(240, 239, 236), font=small)
        out_frames.append(canvas)
    hold = [out_frames[-1]] * 12
    (out_frames + hold)[0].save(out, save_all=True, append_images=(out_frames + hold)[1:], duration=80, loop=0,
                                optimize=True)
    print("wrote", out, len(out_frames), "frames", flush=True)


def _dot(scene, p):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]), np.asarray(p, float),
                        np.eye(3).ravel(), np.array([0.92, 0.41, 0.2, 0.55], np.float32))
    scene.ngeom += 1


def _font(size):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        render(sys.argv[2], sys.argv[3])
