"""Instruction head on a trained catch policy: the policy's own backbone sees the two RGB views, a word-by-cell match on
its stage-2 map points at the ball the sentence names. Backbone and text encoder stay frozen, so the catch path is the
trained policy, unchanged. Every ball of a recorded scene can be named, by its colour or by where it is among the others
(on the left, the highest) or by the order they were thrown in, so a toss gives many grounding problems. A word that
compares balls reads the cell's place; with --events the head also reads the stage-2 map of the raw event time surfaces,
where a release shows at once and the older trail tells which ball left first, gaps a 30 Hz RGB camera cannot resolve.

    CUDA_VISIBLE_DEVICES=0 python -m taucatch.vla_catch.ground runs_v2/r12_loc_f7_novel/ema_4000.pt \
        --data data_v2/sel_bc data_v2/dagger_s3_sel --out runs_v2/r12_loc_f7_novel/ground.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import ball as ball_meter
from .data import CELLS, RGB_DELAY, load_tosses
from .events import CAMERAS, RES
from .locator import sentence
from .policy import pixel_values, set_numerics
from .sim import BALL_RADIUS, GRAVITY, HELD_OUT_PHRASES, PHRASES

LOOK = (-0.10, 0.25)      # s around release: when a toss still has to pick its ball
MARGIN = 12.0             # px the named ball must stand out by in both views for a relation to name it
COLOUR_PHRASES = (*PHRASES, "pick the {} ball", "go for the {} one", "the {} ball", "i want the {} ball",
                  "reach for the {} ball")
RELATIONS = {             # word -> image axis, sign, phrases, held-out phrase
    "left": (0, -1, ("catch the ball on the left", "get the leftmost ball", "grab the left one", "the leftmost one"),
             "take the ball on the left"),
    "right": (0, 1, ("catch the ball on the right", "get the rightmost ball", "grab the right one", "the rightmost one"),
              "take the ball on the right"),
    "high": (1, -1, ("catch the highest ball", "get the top ball", "grab the upper one", "the highest one"),
             "take the ball on top"),
    "low": (1, 1, ("catch the lowest ball", "get the bottom ball", "grab the lower one", "the lowest one"),
            "take the ball at the bottom"),
}
ORDERS = {                # word -> index in release order, phrases, held-out phrase
    "first": (0, ("catch the first ball", "get the ball thrown first", "grab the first one", "the first ball"),
              "take the ball that came first"),
    "last": (-1, ("catch the last ball", "get the ball thrown last", "grab the last one", "the last ball"),
             "take the ball that came last"),
}
ORDER_GAP = 0.03          # s: releases closer than this have no order to name
ORDER_SEEN = 0.02         # s after a release before it can be seen


def ball_paths(m):
    """(launch, velocity, release, flight or None) of a scene's balls, the named one first."""
    return [(m["launch"], m["velocity"], m["release"], m["flight"])] + [(*o[:3], None) for o in m["others"]]


def ball_place(path, t):
    launch, velocity, hold, flight = path
    tb = np.clip(t - hold, 0.0, flight) if flight is not None else max(t - hold, 0.0)
    p = np.asarray(launch) + np.asarray(velocity) * tb + 0.5 * GRAVITY * tb ** 2
    p[2] = max(p[2], BALL_RADIUS)
    return p


def in_order(paths, word):
    """The ball a word of ORDERS names, and when the word can first be told; None if releases are too close."""
    holds = np.array([p[2] for p in paths])
    rank = np.argsort(holds)
    gaps = np.diff(holds[rank])
    place = ORDERS[word][0]
    if gaps[0 if place == 0 else -1] < ORDER_GAP:
        return None
    return int(rank[place]), float(holds[rank[place]]) + ORDER_SEEN


def extreme(places, axis, sign):
    """The ball that lies furthest along an image axis in both RGB views by MARGIN, or None."""
    picks = set()
    for cam in CAMERAS:
        x = sign * np.array([ball_meter.project_rgb(cam, p)[axis] for p in places])
        top = np.sort(x)
        if top[-1] - top[-2] < MARGIN or not all(0 <= ball_meter.project_rgb(cam, p).min() and
                                                 ball_meter.project_rgb(cam, p).max() < RES for p in places):
            return None
        picks.add(int(x.argmax()))
    return picks.pop() if len(picks) == 1 else None


def relation_sentence(paths, rng, held_out=False):
    """A sentence naming ball 0 by a relation that holds all through the look window, or None."""
    words = [w for w in RELATIONS if all(extreme([ball_place(b, paths[0][2] + a) for b in paths], *RELATIONS[w][:2]) == 0
                                         for a in np.linspace(*LOOK, 8))]
    if not words:
        return None
    axis, sign, phrases, unseen = RELATIONS[str(rng.choice(words))]
    return unseen if held_out else str(rng.choice(phrases))


def order_sentence(paths, rng, held_out=False):
    """A sentence naming ball 0 by when it was thrown, or None."""
    words = [w for w in ORDERS if in_order(paths, w) and in_order(paths, w)[0] == 0]
    if not words:
        return None
    _, phrases, unseen = ORDERS[str(rng.choice(words))]
    return unseen if held_out else str(rng.choice(phrases))


class Grounder(nn.Module):
    """Stage-2 blocks [B, V, N, 512] of the RGB views (and of the raw event views) and a sentence [B, D] -> match logits
    [B, V, 28 ** 2]."""

    def __init__(self, text_dim, cell=128, dim=64, relations=False, events=False):
        super().__init__()
        self.events = events
        if events:
            self.trace = nn.Sequential(nn.LayerNorm(cell), nn.Linear(cell, dim))
        self.cell = nn.Sequential(nn.LayerNorm(cell), nn.Linear(cell, dim))
        self.word = nn.Linear(text_dim, dim)
        if relations:                       # a cell's place in the image, for words that compare the balls
            self.place = nn.Linear(2, dim)
            ramp = (torch.arange(CELLS) + 0.5) / CELLS * 2 - 1
            self.register_buffer("grid", torch.stack(torch.meshgrid(ramp, ramp, indexing="xy"), -1), persistent=False)

    @staticmethod
    def unfold(fine):
        b, v = fine.shape[:2]
        side = int(fine.shape[2] ** 0.5)
        blocks = fine.float().reshape(b * v, side, side, -1).permute(0, 3, 1, 2)
        return nn.functional.pixel_shuffle(blocks, 2).permute(0, 2, 3, 1)

    def forward(self, fine, text, fine_events=None):
        b, v = fine.shape[:2]
        word = self.word(text.float()).repeat_interleave(v, 0)
        cells = self.cell(self.unfold(fine))
        if hasattr(self, "place"):
            cells = cells + self.place(self.grid)
        if self.events:
            cells = cells + self.trace(self.unfold(fine_events))
        return ((cells * word[:, None, None]).sum(-1) / self.word.out_features ** 0.5).view(b, v, -1)


def peak(match):
    """Match logits [B, V, 28 ** 2] -> pixel (col, row) of the best cell [B, V, 2]."""
    cell = match.argmax(-1)
    return torch.stack([cell % CELLS, cell // CELLS], -1).float() * RES / CELLS + RES / CELLS / 2


class Scenes(torch.utils.data.Dataset):
    """Steps of the selective shards. A sample names one ball of the scene at random (val: the toss's own), by its colour
    or, when a word of the given kinds holds at that moment, by the word. kinds: colour, relation, order."""

    def __init__(self, tosses, val, held_out=False, kinds=("colour", "relation"), events=False):
        self.tosses, self.val, self.held_out, self.kinds, self.events = tosses, val, held_out, kinds, events
        self.items = [(i, s) for i, t in enumerate(tosses) if (t.meta["seed"] % 20 == 0) == val and t.meta.get("colors")
                      for s in range(0, len(t.times), 2 if val else 1)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        i, s = self.items[n]
        toss, m = self.tosses[i], self.tosses[i].meta
        rng = np.random.default_rng((n, 7)) if self.val else np.random.default_rng()
        paths = ball_paths(m)
        k = int(np.searchsorted(toss.rgb_times, toss.times[s] - RGB_DELAY + 1e-6)) - 1
        t = float(toss.rgb_times[max(k, 0)])                    # the scene the RGB frame shows
        places = [ball_place(path, t) for path in paths]
        project = ball_meter.project_rgb
        which, text = 0 if self.val else int(rng.integers(len(paths))), None
        kind = str(rng.choice(self.kinds))
        if kind == "relation":
            axis, sign, phrases, unseen = RELATIONS[str(rng.choice(list(RELATIONS)))]
            top = extreme(places, axis, sign)
            if top is not None:
                which, text = top, unseen if self.held_out else str(rng.choice(phrases))
        elif kind == "order":
            word = str(rng.choice(list(ORDERS)))
            found = in_order(paths, word)
            if found and toss.times[s] >= found[1]:             # a word of time points at the events, which are newer
                which, text = found[0], ORDERS[word][2] if self.held_out else str(rng.choice(ORDERS[word][1]))
                places, project = [ball_place(path, float(toss.times[s])) for path in paths], ball_meter.project
        if text is None:
            text = str(rng.choice(HELD_OUT_PHRASES if self.held_out else COLOUR_PHRASES)).format(m["colors"][which])
        p = places[which]
        out, uv = np.full(len(CAMERAS), CELLS ** 2, np.int64), np.zeros((len(CAMERAS), 2), np.float32)
        for v, cam in enumerate(CAMERAS):
            uv[v] = project(cam, p)
            col, row = (uv[v] // (RES // CELLS)).astype(int)
            if k >= 0 and 0 <= col < CELLS and 0 <= row < CELLS:
                out[v] = row * CELLS + col
        rgb = torch.from_numpy(toss.rgb_frames(s).copy())
        trace = toss.frames(s, mode="trace")[:len(CAMERAS)] if self.events else np.zeros(0, np.uint8)
        return rgb, torch.from_numpy(trace), torch.from_numpy(out), torch.from_numpy(uv), text, float(toss.times[s]) - m["release"]


def encode(vla, head, frames, trace, text):
    """Match logits of one batch of RGB pairs and raw trace pairs."""
    with torch.no_grad():
        fine = vla.vision_encoder(pixel_values(frames.cuda(non_blocking=True)), fine=True)[1]
        fine_events = vla.vision_encoder(pixel_values(trace.cuda(non_blocking=True)), fine=True)[1] if head.events else None
        words = sentence(vla, list(text), fine.device)
    return head(fine, words, fine_events)


@torch.no_grad()
def validate(vla, head, data, batch=64):
    head.eval()
    loader = torch.utils.data.DataLoader(data, batch, num_workers=4)
    hit, near, err, when = [], [], [], []
    for frames, trace, ground, uv, text, age in loader:
        match = encode(vla, head, frames, trace, text)
        seen = ground.cuda() < CELLS ** 2
        hit.append((match.argmax(-1) == ground.cuda())[seen].cpu())
        d = (peak(match) - uv.cuda()).norm(dim=-1)[seen].cpu()
        err.append(d)
        near.append(d < 16)
        when.append(age[:, None].repeat(1, 2)[seen.cpu()])
    head.train()
    hit, near, err, when = (torch.cat(x).float().numpy() for x in (hit, near, err, when))
    look = (when > LOOK[0]) & (when < LOOK[1])
    return {"hit": float(hit.mean()), "near": float(near.mean()), "err_px": float(np.median(err)),
            "look_hit": float(hit[look].mean()), "look_near": float(near[look].mean()),
            "look_err_px": float(np.median(err[look]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", nargs="+", default=["data_v2/sel_bc", "data_v2/dagger_s3_sel"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--events", action="store_true", help="also read the raw event views; adds the order words")
    args = ap.parse_args()
    set_numerics()
    from .evaluate import Policy
    policy = Policy(args.ckpt)
    vla = policy.model.vla
    tosses = [t for d in args.data for t in load_tosses(d)]
    kinds = ("colour", "relation", "order") if args.events else ("colour", "relation")
    train = Scenes(tosses, False, kinds=kinds, events=args.events)
    tests = {k: Scenes(tosses, True, kinds=(k,), events=args.events) for k in kinds}
    tests.update({f"unseen {k}": Scenes(tosses, True, True, (k,), args.events) for k in kinds})
    print(f"{len(train)} train steps, {len(tests['colour'])} val steps", flush=True)
    head = Grounder(vla.text_encoder.bert.config.hidden_size, relations=True, events=args.events).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    loader = torch.utils.data.DataLoader(train, args.batch, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    t0, it, window = time.time(), iter(loader), []
    for step in range(1, args.steps + 1):
        try:
            frames, trace, ground, *_, text, _ = next(it)
        except StopIteration:
            it = iter(loader)
            frames, trace, ground, *_, text, _ = next(it)
        match = encode(vla, head, frames, trace, text)
        loss = nn.functional.cross_entropy(match.flatten(0, 1), ground.cuda().flatten(), ignore_index=CELLS ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        window.append(loss.item())
        if step % 200 == 0:
            print(f"step {step} loss {np.mean(window):.3f} {(time.time() - t0) / step:.2f} s/step", flush=True)
            window = []
        if step % 2000 == 0 or step == args.steps:
            for k in kinds:
                print(step, k, json.dumps(validate(vla, head, tests[k])), flush=True)
            torch.save({"ckpt": args.ckpt, "head": head.state_dict(), "step": step}, args.out)
    for name, data in tests.items():
        print("final", name, json.dumps(validate(vla, head, data)), flush=True)


if __name__ == "__main__":
    main()
