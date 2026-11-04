"""One look at the hand: SpikeVLA's backbone finds the gripper in the RGB views, nothing else is calibrated.

The forward kinematics say where the hand is, the nominal camera poses say where it should show, and the difference in
each view is what moved: camera rigs or the arm's base. Taken off the event pixels as an image offset it undoes a rig's
turn at every depth and its move at the hand's depth, where the catch happens: at 4 cm 2 deg a track starts 3.0 cm off
and ends 0.7 cm off, and the two views' rays meet again. The look happens at the ready pose.

collect: crops around the nominal projection of the hand under random rigs (software rendering) -> data_v2/handlook/*.npz
features, train: a heat map over the 28 x 28 stage-2 cells, soft-argmax.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from . import ball as bm
from . import objects
from .events import CAMERAS, RES
from .fovea import load_policy, sensor_images
from .sim import SENSOR, use_kinds

SIDE = 256                       # sensor px of the crop around the nominal projection
TEMPERATURE = 0.05


def rgb_nominal(cam, p):
    return np.asarray(bm.project_rgb(cam, p)) * SENSOR / RES


def rgb_true(sim, cam, p):
    import mujoco
    from .sim import RGB_CAMERAS
    name = next(k for k, v in RGB_CAMERAS.items() if v == cam)
    i = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    return bm._project(sim.d.cam_xpos[i].copy(), sim.d.cam_xmat[i].reshape(3, 3), p) * SENSOR / RES


def crop_at(image, centre):
    """Crop [SIDE, SIDE, 3] around sensor pixel `centre`, black outside the image."""
    x0, y0 = int(round(centre[0] - SIDE / 2)), int(round(centre[1] - SIDE / 2))
    box = np.zeros((SIDE, SIDE, 3), np.uint8)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + SIDE, SENSOR), min(y0 + SIDE, SENSOR)
    if xb > xa and yb > ya:
        box[ya - y0:yb - y0, xa - x0:xb - x0] = image[ya:yb, xa:xb]
    return box, np.array([x0 + SIDE / 2, y0 + SIDE / 2])


def collect_part(job):
    first, n, out, arm, most = job
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2
    import mujoco
    from .collect import make_sim
    use_kinds()
    sim = make_sim(arm, 2)
    renderer = mujoco.Renderer(sim.m, SENSOR, SENSOR)
    crops, offsets, moved = [], [], []
    for seed in range(first, first + n):
        rng = np.random.default_rng((seed, 43))
        cm = rng.uniform(0, most)
        sim.miscalibrate(cm, cm / 2, seed)
        sim.reset_arm()
        gain = objects.look(rng)[0]
        mujoco.mj_kinematics(sim.m, sim.d)
        mujoco.mj_camlight(sim.m, sim.d)
        g = sim.grasp_point()
        images = [objects.expose(x, gain, rng) for x in sensor_images(sim, renderer)]
        row, off = [], []
        for cam, image in zip(CAMERAS, images):
            box, centre = crop_at(image, rgb_nominal(cam, g))
            row.append(cv2.resize(box, (RES, RES), interpolation=cv2.INTER_AREA))
            off.append((rgb_true(sim, cam, g) - centre) / (SIDE / 2))
        crops.append(np.stack(row)), offsets.append(np.stack(off)), moved.append(cm)
    np.savez(out, crops=np.stack(crops), offsets=np.array(offsets, np.float32), moved=np.array(moved, np.float32),
             seeds=np.arange(first, first + n))
    return len(crops)


def collect(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    size = args.rigs // args.workers
    jobs = [(args.seed + i * size, size, f"{args.out}/part{i:03d}.npz", args.arm, args.most) for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        print(sum(pool.map(collect_part, jobs, 1)), "looks", time.time() - t0, "s", flush=True)


def make_finder(cell=128):
    import torch
    import torch.nn as nn

    class Finder(nn.Module):
        """Stage-2 blocks [B, 196, 512] -> place of the hand in the crop [B, 2], in (-1, 1)."""

        def __init__(self):
            super().__init__()
            self.map = nn.Sequential(nn.LayerNorm(cell), nn.Linear(cell, 96), nn.GELU(), nn.Linear(96, 96), nn.GELU(), nn.Linear(96, 1))
            n = 28
            ramp = (torch.arange(n) + 0.5) / n * 2 - 1
            self.register_buffer("grid", torch.stack(torch.meshgrid(ramp, ramp, indexing="xy"), -1).flatten(0, 1), persistent=False)

        def forward(self, fine):
            from .ground import Grounder
            cells = Grounder.unfold(fine[:, None])                     # [B, 28, 28, 128]
            w = torch.softmax(self.map(cells).flatten(1) / TEMPERATURE, 1)
            return w @ self.grid

    return Finder()


def blocks(vla, crops, batch=64):
    """Crop pairs [N, 2, 224, 224, 3] uint8 -> stage-2 blocks [N, 2, 196, 512] half."""
    import torch
    from .policy import pixel_values
    out = []
    for i in range(0, len(crops), batch):
        with torch.no_grad():
            out.append(vla.vision_encoder(pixel_values(crops[i:i + batch].cuda(non_blocking=True)), fine=True)[1].half().cpu())
    return torch.cat(out)


def features(args):
    import torch
    parts = sorted(Path(args.data).glob("part*.npz"))
    z = [np.load(p) for p in parts]
    crops = np.concatenate([x["crops"] for x in z])
    vla = load_policy(args.ckpt).model.vla
    t0 = time.time()
    fine = blocks(vla, torch.from_numpy(crops))
    print("blocks", fine.shape, time.time() - t0, "s", flush=True)
    torch.save({"fine": fine, "offsets": torch.from_numpy(np.concatenate([x["offsets"] for x in z])),
                "moved": torch.from_numpy(np.concatenate([x["moved"] for x in z])),
                "seeds": torch.from_numpy(np.concatenate([x["seeds"] for x in z])), "ckpt": args.ckpt}, args.out)


def train(args):
    import torch
    z = torch.load(args.data, weights_only=False)
    fine, offsets, moved, seeds = z["fine"], z["offsets"], z["moved"], z["seeds"]
    val = seeds % 20 == 0
    head = make_finder().cuda()
    print(sum(p.numel() for p in head.parameters()), "parameters;", int((~val).sum()), "train,", int(val.sum()), "val rigs", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    pool = torch.nonzero(~val)[:, 0]
    t0, window = time.time(), []
    for step in range(1, args.steps + 1):
        index = pool[torch.randint(len(pool), (args.batch,))]
        view = torch.randint(2, (args.batch,))
        x = fine[index, view].cuda(non_blocking=True).float()
        y = offsets[index, view].cuda(non_blocking=True)
        loss = (head(x) - y).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        window.append(loss.item())
        if step % 500 == 0:
            print(f"step {step} loss {np.mean(window):.4f} {(time.time() - t0) / step:.3f} s/step", flush=True)
            window = []
        if step % 2000 == 0 or step == args.steps:
            head.eval()
            with torch.no_grad():
                idx = torch.nonzero(val)[:, 0]
                preds = torch.cat([head(fine[idx[i:i + 64]].flatten(0, 1).cuda().float()).cpu() for i in range(0, len(idx), 64)])
            head.train()
            errors = (preds.view(-1, 2, 2) - offsets[idx]).norm(dim=-1) * SIDE / 2 * RES / SENSOR      # px of the 224 view
            out = {"median_view_px": round(float(errors.median()), 3), "p90_view_px": round(float(errors.quantile(0.9)), 3)}
            for low, high in ((0, 2), (2, 4), (4, 8), (8, 16)):
                pick = (moved[idx] >= low) & (moved[idx] < high)
                if pick.any():
                    out[f"cm_{low}_{high}"] = round(float(errors[pick].median()), 3)
            print(step, json.dumps(out), flush=True)
            torch.save({"ckpt": z["ckpt"], "head": head.state_dict(), "step": step}, args.out)


class HandLook:
    """How far the views are off, from the RGB views at the ready pose."""

    def __init__(self, ckpt, vla=None):
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.vla = vla if vla is not None else load_policy(ck["ckpt"]).model.vla
        self.head = make_finder().cuda().eval()
        self.head.load_state_dict(ck["head"])

    def __call__(self, sim, renderer, rng, gain=np.ones(3)):
        """-> offsets of the two views [2, 2] px."""
        import cv2
        import torch
        import mujoco
        bm.set_shift(np.zeros((2, 2)))
        sim.reset_arm()
        mujoco.mj_kinematics(sim.m, sim.d)
        mujoco.mj_camlight(sim.m, sim.d)
        g = sim.grasp_point()
        images = [objects.expose(x, gain, rng) for x in sensor_images(sim, renderer)]
        boxes, centres = zip(*(crop_at(image, rgb_nominal(cam, g)) for cam, image in zip(CAMERAS, images)))
        pair = np.stack([cv2.resize(b, (RES, RES), interpolation=cv2.INTER_AREA) for b in boxes])[None]
        with torch.no_grad():
            fine = blocks(self.vla, torch.from_numpy(pair))
            place = self.head(fine[0].cuda().float()).cpu().numpy()
        seen = [(c + p * SIDE / 2) * RES / SENSOR for c, p in zip(centres, place)]
        return np.stack([x - rgb_nominal(cam, g) * RES / SENSOR for x, cam in zip(seen, CAMERAS)])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("collect")
    a.add_argument("--out", default="data_v2/handlook")
    a.add_argument("--rigs", type=int, default=6000)
    a.add_argument("--workers", type=int, default=40)
    a.add_argument("--seed", type=int, default=7500000)
    a.add_argument("--most", type=float, default=8.0, help="largest rig move, cm (degrees: half of it)")
    a.add_argument("--arm", default="openarm")
    a = sub.add_parser("features")
    a.add_argument("ckpt")
    a.add_argument("--data", default="data_v2/handlook")
    a.add_argument("--out", default="data_v2/handlook_blocks.pt")
    a = sub.add_parser("train")
    a.add_argument("--data", default="data_v2/handlook_blocks.pt")
    a.add_argument("--out", required=True)
    a.add_argument("--steps", type=int, default=8000)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    {"collect": collect, "features": features, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
