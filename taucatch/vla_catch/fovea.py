"""Reading what a track sits on: SpikeVLA's backbone on the full-resolution RGB around the track.

The 224 px view shows an object 7 px wide, which holds its colour and little else. The sensor holds four times that, and
the event track says where to look: a crop of 4.8 object radii around the track, scaled by its distance, enters the
backbone at the usual 224 px. The instruction head scores each crop against the sentence.

collect: crops of kinds scenes (software rendering, no arm motion needed)   -> data_v2/fovea/*.npz
features: frozen stage-2 blocks of the crops                                -> data_v2/fovea_blocks.pt
train: the head; picks the named object among the three crops of a moment.
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
from .sim import BALL_RADIUS, EVENT_FOVY, RGB_CAMERAS, SENSOR, ballistic, use_kinds

CROP = 112                       # stored px per side; the backbone gets it at 224
SPAN = 2.4                       # half side of the crop, object radii
JITTER = 8.0                     # sensor px: where the track puts the crop, off the object's centre
SENSOR_FOCAL = (SENSOR / 2) / np.tan(np.radians(EVENT_FOVY) / 2)


def crop(image, cam, p, shift=(0.0, 0.0)):
    """Crop [CROP, CROP, 3] of a sensor image around world point p as RGB camera beside `cam` sees it."""
    import cv2
    c, r = np.asarray(bm.project_rgb(cam, p)) * SENSOR / RES + np.asarray(shift)
    half = SPAN * BALL_RADIUS * SENSOR_FOCAL / np.linalg.norm(np.asarray(p) - bm.camera(cam)[0])
    side = max(int(round(2 * half)), 8)
    x0, y0 = int(round(c - side / 2)), int(round(r - side / 2))
    box = np.zeros((side, side, 3), np.uint8)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, SENSOR), min(y0 + side, SENSOR)
    if xb > xa and yb > ya:
        box[ya - y0:yb - y0, xa - x0:xb - x0] = image[ya:yb, xa:xb]
    return cv2.resize(box, (CROP, CROP), interpolation=cv2.INTER_AREA if side > CROP else cv2.INTER_LINEAR)


def sensor_images(sim, renderer):
    import mujoco
    mujoco.mj_kinematics(sim.m, sim.d)
    mujoco.mj_camlight(sim.m, sim.d)
    out = []
    for name in RGB_CAMERAS:
        renderer.update_scene(sim.d, camera=name)
        out.append(renderer.render().copy())
    return out


def collect_part(job):
    first, n, out, arm, patterns = job
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["OMP_NUM_THREADS"] = "1"
    import mujoco
    from .collect import make_sim
    use_kinds()
    sim = make_sim(arm, 2)
    renderer = mujoco.Renderer(sim.m, SENSOR, SENSOR)
    words = list(objects.all_textures())
    crops, names, seeds = [], [], []
    for seed in range(first, first + n):
        rng = np.random.default_rng((seed, 41))
        toss = sim.sample_patterns(rng, unseen=seed % 5 == 0) if patterns else sim.sample_kinds(rng)
        sim.reset_arm()
        sim.paint(toss)
        for t in rng.uniform(toss.hold, toss.hold + 0.6 * toss.flight, 4):
            p = ballistic(np.asarray(toss.launch), np.asarray(toss.velocity), t - toss.hold)
            sim.place_ball(p)
            sim.move_distractors(toss, t)
            places = [p] + [sim.d.mocap_pos[i].copy() for i in sim.mocap]
            images = [objects.expose(x, sim.gain, rng) for x in sensor_images(sim, renderer)]
            for cam, image in zip(CAMERAS, images):
                crops.append(np.stack([crop(image, cam, q, rng.uniform(-JITTER, JITTER, 2)) for q in places]))
                names.append([words.index(w) for w in toss.colors])
                seeds.append(seed)
    np.savez(out, crops=np.stack(crops), names=np.array(names, np.int8), seeds=np.array(seeds))
    return len(crops)


def collect(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    size = args.scenes // args.workers
    jobs = [(args.seed + i * size, size, f"{args.out}/part{i:03d}.npz", args.arm, args.patterns) for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        print(sum(pool.map(collect_part, jobs, 1)), "crop sets", time.time() - t0, "s", flush=True)


def load_policy(ckpt):
    from .evaluate import Policy
    from .policy import set_numerics
    set_numerics()
    return Policy(ckpt)


def blocks(vla, crops, batch=64):
    """Left and right crops of one object [N, 2, H, W, 3] uint8 -> stage-2 blocks [N, 2, 196, 512] half."""
    import torch
    from .policy import pixel_values
    out = []
    for i in range(0, len(crops), batch):
        x = crops[i:i + batch].cuda(non_blocking=True)
        b = x.shape[0]
        x = x.flatten(0, 1).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).clamp(0, 255).round().to(torch.uint8).view(b, 2, RES, RES, 3)
        with torch.no_grad():
            out.append(vla.vision_encoder(pixel_values(x), fine=True)[1].half().cpu())
    return torch.cat(out)


def features(args):
    import torch
    parts = sorted(Path(args.data).glob("part*.npz"))
    crops, names, seeds = [], [], []
    for p in parts:
        z = np.load(p)
        crops.append(z["crops"]), names.append(z["names"]), seeds.append(z["seeds"])
    crops, names, seeds = np.concatenate(crops), np.concatenate(names), np.concatenate(seeds)
    print(crops.shape, flush=True)
    if args.shrink:                     # what the 224 px view holds of the same crop
        import cv2
        flat = crops.reshape(-1, CROP, CROP, 3)
        crops = np.stack([cv2.resize(cv2.resize(x, (args.shrink, args.shrink), interpolation=cv2.INTER_AREA), (CROP, CROP),
                                     interpolation=cv2.INTER_LINEAR) for x in flat]).reshape(crops.shape)
    vla = load_policy(args.ckpt).model.vla
    t0 = time.time()
    pairs = torch.from_numpy(crops).view(-1, 2, *crops.shape[1:]).transpose(1, 2).flatten(0, 1)   # [3 M, 2, H, W, 3]
    fine = blocks(vla, pairs)
    print("blocks", fine.shape, time.time() - t0, "s", flush=True)
    fine = fine.view(-1, 3, *fine.shape[1:]).transpose(1, 2).flatten(0, 1)                        # [2 M, 3, 196, 512]
    sentences = {}
    from .locator import sentence
    for word in objects.all_textures():
        for phrase in objects.PHRASES + objects.HELD_OUT_PHRASES:
            text = phrase.format(word)
            with torch.no_grad():
                sentences[text] = sentence(vla, [text], "cuda")[0].float().cpu()
    torch.save({"fine": fine, "names": torch.from_numpy(names), "seeds": torch.from_numpy(seeds),
                "sentences": sentences, "ckpt": args.ckpt}, args.out)


def make_reader(text_dim, cell=128, dim=128):
    import torch.nn as nn

    class Reader(nn.Module):
        """Stage-2 blocks [B, 196, 512], sentence [B, D] -> score [B]."""

        def __init__(self):
            super().__init__()
            self.cell = nn.Sequential(nn.LayerNorm(cell), nn.Linear(cell, dim), nn.GELU())
            self.pool = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.word = nn.Linear(text_dim, dim)

        def forward(self, fine, text):
            import torch
            from .ground import Grounder
            cells = self.cell(Grounder.unfold(fine[:, None])).flatten(1, 2)
            whole = self.pool(torch.cat([cells.mean(1), cells.amax(1)], -1))
            return (whole * self.word(text.float())).sum(-1) / dim ** 0.5

    return Reader()


def train(args):
    import torch
    import torch.nn as nn
    z = torch.load(args.data, weights_only=False)
    fine, names, seeds, sentences = z["fine"], z["names"].long(), z["seeds"], z["sentences"]
    val = seeds % 20 == 1                 # seeds % 5 == 0 are the scenes that name untrained combinations
    words = list(objects.all_textures())
    phrases, held = objects.PHRASES, objects.HELD_OUT_PHRASES
    text = torch.stack([torch.stack([sentences[p.format(w)] for w in words]) for p in phrases + held]).cuda()  # [P, W, D]
    head = make_reader(text.shape[-1]).cuda()
    print(sum(p.numel() for p in head.parameters()), "parameters;", int((~val).sum()), "train,", int(val.sum()), "val crop sets", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    hidden = torch.tensor([words.index(w) for w in objects.UNSEEN_PATTERNS])
    unseen = torch.isin(names[:, 0], hidden)                # scenes that name a combination never trained on
    pool = torch.nonzero(~val & ~unseen)[:, 0]
    fine = fine.pin_memory() if args.pin else fine

    def scores(index, which, phrase):
        b = len(index)
        x = fine[index].cuda(non_blocking=True).float()                  # [B, 3, 196, 512]
        word = names[index].cuda()[torch.arange(b), which]
        t = text[phrase, word]
        return head(x.flatten(0, 1), t.repeat_interleave(3, 0)).view(b, 3)

    t0, window = time.time(), []
    for step in range(1, args.steps + 1):
        index = pool[torch.randint(len(pool), (args.batch,))]
        which = torch.randint(3, (args.batch,)).cuda()
        phrase = torch.randint(len(phrases), (args.batch,)).cuda()
        loss = nn.functional.cross_entropy(scores(index, which, phrase), which)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        window.append(loss.item())
        if step % 200 == 0:
            print(f"step {step} loss {np.mean(window):.3f} {(time.time() - t0) / step:.3f} s/step", flush=True)
            window = []
        if step % 1000 == 0 or step == args.steps:
            out = {}
            head.eval()
            with torch.no_grad():
                idx = torch.nonzero(val & ~unseen)[:, 0]
                tests = (("seen", 0, idx), ("unseen", len(phrases), idx))
                if unseen.any():
                    tests += (("new_object", 0, torch.nonzero(unseen)[:, 0]),)
                for tag, ph, idx in tests:
                    hit, per = [], {w: [] for w in words}
                    for i in range(0, len(idx), 256):
                        index = idx[i:i + 256]
                        which = torch.zeros(len(index), dtype=torch.long).cuda()
                        pick = scores(index, which, torch.full((len(index),), ph).cuda()).argmax(1) == 0
                        hit.append(pick.cpu())
                        for w, ok in zip(names[index, 0].tolist(), pick.cpu().tolist()):
                            per[words[w]].append(ok)
                    out[tag] = round(float(torch.cat(hit).float().mean()), 4)
                    out[tag + "_kind"] = {w: round(float(np.mean(v)), 3) for w, v in per.items()}
            head.train()
            print(step, json.dumps(out), flush=True)
            torch.save({"ckpt": z["ckpt"], "head": head.state_dict(), "step": step}, args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("collect")
    a.add_argument("--out", default="data_v2/fovea")
    a.add_argument("--scenes", type=int, default=4000)
    a.add_argument("--workers", type=int, default=40)
    a.add_argument("--seed", type=int, default=7000000)
    a.add_argument("--arm", default="openarm")
    a.add_argument("--patterns", action="store_true", help="objects named by colour and figure")
    a = sub.add_parser("features")
    a.add_argument("ckpt")
    a.add_argument("--data", default="data_v2/fovea")
    a.add_argument("--out", default="data_v2/fovea_blocks.pt")
    a.add_argument("--shrink", type=int, default=0)
    a = sub.add_parser("train")
    a.add_argument("--data", default="data_v2/fovea_blocks.pt")
    a.add_argument("--out", required=True)
    a.add_argument("--steps", type=int, default=6000)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--pin", action="store_true")
    args = ap.parse_args()
    {"collect": collect, "features": features, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
