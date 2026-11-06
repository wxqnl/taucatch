"""Behaviour cloning of the SpikeVLA catcher on collect.py shards (one GPU, full fp32 as the checkpoints).

    CUDA_VISIBLE_DEVICES=0 python -m taucatch.vla_catch.train --arm f7 --data data_v2/bc_v0 --out runs_v2/bc_v0_f7
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .ball import BALL_DIM
from .data import CatchDataset, denormalize_actions, label_stats, load_tosses
from .policy import INSTRUCTION, VENDOR, build_model, model_config, pixel_values, set_numerics

SCORER_WEIGHT = 0.1           # cross-entropy of the locator's token scorer next to the action L1
GROUND_WEIGHT = 0.1           # selective: cross-entropy of the instruction match over the RGB cells


def masked_l1(pred, target, mask):
    err = (pred - target).abs().mean(-1)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def batches(loader):
    while True:
        yield from loader


def step_inputs(model, batch):
    """One loader batch -> (actions, scorer logits or None, labels, mask, scorer target or None, grounding target or
    None), on the GPU."""
    instructions = [INSTRUCTION] * len(batch[0])
    if isinstance(batch[-1], (list, tuple)):          # selective shards: one instruction per sample
        *batch, instructions = batch
        instructions = list(instructions)
    frames, state, labels, mask, *extra = (x.cuda(non_blocking=True) for x in batch)
    if not extra:
        return model(instructions, pixel_values(frames), state), None, labels, mask.float(), None, None
    memory, valid, gate, target, *ground = extra
    actions, logits, *_ = model(instructions, pixel_values(frames), state, memory, valid, gate)
    return actions, logits, labels, mask.float(), target, ground[0] if ground else None


def take_match(model):
    """The selector's last match map [B, 2, cells], taken off the module (a graph tensor there blocks deepcopy)."""
    match, model.locator.match = model.locator.match, None
    return match.float()


def ground_loss(match, ground):
    return torch.nn.functional.cross_entropy(match.flatten(0, 1), ground.flatten(), ignore_index=match.shape[-1])


def scorer_loss(logits, target):
    return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), target.flatten())


def refresh_norms(model, loader, n):
    """EMA weights need running norm statistics from training data (ICML training_stat_refresh)."""
    from turbovla.training.wta_norms import training_stat_refresh
    it = iter(loader)
    with training_stat_refresh(getattr(model, "vla", model)):
        for _ in range(n):
            step_inputs(model, next(it))


@torch.no_grad()
def evaluate(model, loader, stats, n):
    """Mean error in physical units over n validation batches: first chunk entry, and position 220 ms ahead
    (the last entry, which carries the intercept)."""
    model.eval()
    pos, far, ball, rot, finger, l1, hit, grounded = [], [], [], [], [], [], [], []
    for i, batch in enumerate(loader):
        if i == n:
            break
        pred, logits, labels, mask, target, ground = step_inputs(model, batch)
        if ground is not None:                           # the match points at the named ball's cell
            match = take_match(model)
            seen = ground < match.shape[-1]
            grounded.append((match.argmax(-1) == ground)[seen].float().mean().item())
        pred, labels, mask = pred.float().cpu(), labels.cpu(), mask.cpu()
        l1.append(masked_l1(pred, labels, mask).item())
        if logits is not None:                           # scorer picks the ball's patch (or "no ball")
            hit.append((logits.argmax(-1) == target).float().mean().item())
        p_all, t_all = denormalize_actions(pred.numpy(), stats), denormalize_actions(labels.numpy(), stats)
        last = mask[:, -1].numpy().astype(bool)
        far.append(np.linalg.norm(p_all[last, -1, :3] - t_all[last, -1, :3], axis=-1))
        if p_all.shape[-1] > 7:
            ball.append(np.linalg.norm(p_all[last, -1, 7:10] - t_all[last, -1, 7:10], axis=-1))
        p, t = p_all[:, 0], t_all[:, 0]
        pos.append(np.linalg.norm(p[:, :3] - t[:, :3], axis=-1))
        rot.append(np.linalg.norm(p[:, 3:6] - t[:, 3:6], axis=-1))
        finger.append(np.abs(p[:, 6] - t[:, 6]))
    model.train()
    return {"l1": float(np.mean(l1)), "pos_cm": 100 * float(np.concatenate(pos).mean()),
            "pos_220ms_cm": 100 * float(np.concatenate(far).mean()),
            "ball_220ms_cm": 100 * float(np.concatenate(ball).mean()) if ball else None,
            "rot_deg": float(np.degrees(np.concatenate(rot).mean())), "finger": float(np.concatenate(finger).mean()),
            "scorer_hit": float(np.mean(hit)) if hit else None,
            "ground_hit": float(np.mean(grounded)) if grounded else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["v1", "v1_sgsa", "f7"])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--vision-lr", type=float, default=1e-5)
    ap.add_argument("--new-lr", type=float, default=1e-4, help="parameters absent from the start checkpoint")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--all-tosses", action="store_true", help="also train on tosses the expert dropped")
    ap.add_argument("--frames", default="window", choices=["window", "trace"], help="event frame encoding")
    ap.add_argument("--forecast", action="store_true", help="chunk also predicts the ball position (3 more channels)")
    ap.add_argument("--ball", action="store_true", help=f"state also carries the ball measurement ({BALL_DIM} tokens)")
    ap.add_argument("--locator", action="store_true", help="the ball tokens come from the in-model event locator")
    ap.add_argument("--condition", action="store_true", help="the state also enters the action queries")
    ap.add_argument("--no-velocity", action="store_true", help="hide the grasp velocity from the policy (copycat)")
    ap.add_argument("--no-image", action="store_true", help="with --locator: the action head gets no image tokens, "
                    "only the hand and the located ball")
    ap.add_argument("--anchor", action="store_true", help="with --locator: arm targets on the located ball's path "
                    "(the head picks the time along it and an offset)")
    ap.add_argument("--intercept", action="store_true", help="with --locator: the head also reads where its own "
                    "track meets the hand (the expert's intercept on the located ball: time, point - grasp, found)")
    ap.add_argument("--own-labels", action="store_true", help="arm-position labels from the expert planner on the "
                    "policy's own (slice) track, not the true ball (ownlabels.py cache)")
    ap.add_argument("--slice-memory", action="store_true", help="with --locator: once ready, the track memory is "
                    "the one closed loop with --slice-track gives (slicemem.py cache), not one raw point per call")
    ap.add_argument("--selective", action="store_true", help="RGB views + instruction-conditioned locator "
                    "(collect.py --distractors shards; needs --locator)")
    ap.add_argument("--init", default=None, help="start from a trained catch checkpoint (its arm and frames)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_numerics()
    torch.manual_seed(0)

    keep = (lambda m: True) if args.all_tosses else (lambda m: m["caught"] or m.get("dagger", False))
    tosses = [t for root in args.data for t in load_tosses(root, keep)]
    val = [t for t in tosses if t.meta["seed"] % 20 == 0]
    train = [t for t in tosses if t.meta["seed"] % 20 != 0]
    ball = args.ball or args.locator
    # Scales from the expert's own tosses: DAgger labels far off the expert's path would stretch them.
    stats = label_stats([t for t in train if not t.meta.get("dagger", False)], args.forecast, ball)
    kw = dict(mode=args.frames, forecast=args.forecast, ball=args.ball, locator=args.locator,
              no_velocity=args.no_velocity, slice_memory=args.slice_memory)
    train_set = CatchDataset(train, stats, augment=True, own_labels=args.own_labels, **kw)
    val_set = CatchDataset(val, stats, **kw)          # validation keeps the true-state labels
    print(f"train {len(train)} tosses {len(train_set)} steps, val {len(val)} tosses {len(val_set)} steps", flush=True)
    loader = torch.utils.data.DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, 32, shuffle=True, num_workers=4,
                                             generator=torch.Generator().manual_seed(0))
    refresh_loader = torch.utils.data.DataLoader(train_set, 128, shuffle=True, num_workers=8, drop_last=True)

    config, weights, located = model_config(args.arm) + ({},)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        config, weights = copy.deepcopy(ck["model_config"]), ck["ema_state_dict"]
        args.condition = args.condition or ck.get("condition", False)
        args.no_image = args.no_image or ck.get("no_image", False)
        args.anchor = args.anchor or ck.get("anchor", False)
        if ck.get("locator"):
            located = {k[8:]: v for k, v in weights.items() if k.startswith("locator.")}
            weights = {k[4:]: v for k, v in weights.items() if k.startswith("vla.")}
    if args.forecast:
        config["action"]["action_dim"] = 10
    if args.selective:            # the two RGB views follow the event views; their view parameters start fresh
        config["vision"]["num_views"] = 4
    icept = None
    if args.intercept:                # B601 geometry as the expert uses it; the ready grasp where the hand waits
        from .b601 import B601Sim
        from .locator import INTERCEPT_DIM
        ready = np.median([t.state8[0, :3] for t in train if not t.meta.get("dagger", False)], 0)
        icept = [*B601Sim.SHOULDER, B601Sim.REACH, *ready]
        print("intercept geometry", np.round(icept, 3).tolist(), flush=True)
    state_dim = 8 + BALL_DIM + (INTERCEPT_DIM if args.intercept else 0)
    widen = ball and config["action"]["state_dim"] != state_dim
    if ball:
        config["action"]["state_dim"] = config["action"]["num_state_tokens"] = state_dim
    model, result = build_model(config, weights, args.condition)
    fresh = set(result.missing_keys)
    if widen:                 # the state encoder gains the ball tokens: train it at the new-parameter rate
        fresh |= {n for n, _ in model.named_parameters() if n.startswith("action_head.state_projection")}
    if args.locator:
        from .locator import CatchPolicy
        model = CatchPolicy(model, stats, args.selective)
        model.no_image, model.anchor = args.no_image, args.anchor
        if icept is not None:
            model.set_intercept(icept)
        new = {n for n, _ in model.locator.named_parameters()}
        if located:
            missing = model.locator.load_state_dict(located, strict=False).missing_keys
            new = {n for n in new if n in missing}
        fresh = {"vla." + n for n in fresh} | {"locator." + n for n in new}
    print(f"{args.arm}: {len(fresh)} fresh tensors {sorted(fresh)[:8]}, {len(result.unexpected_keys)} unused "
          f"{result.unexpected_keys[:8]}", flush=True)
    model.cuda().train()
    groups = {"vision": [], "head": [], "new": []}
    for name, p in model.named_parameters():
        bare = name.removeprefix("vla.")
        if bare.startswith("text_encoder.bert"):
            p.requires_grad = False
            continue
        if not p.requires_grad:
            continue
        groups["new" if name in fresh else "vision" if bare.startswith("vision_encoder.backbone") else "head"].append(p)
    lrs = {"vision": args.vision_lr, "head": args.lr, "new": args.new_lr}
    optimizer = torch.optim.AdamW([{"params": v, "lr": lrs[k], "name": k} for k, v in groups.items() if v],
                                  weight_decay=0.0)
    print({k: sum(p.numel() for p in v) for k, v in groups.items()}, flush=True)

    def schedule(step):
        if step < args.warmup:
            return 0.1 + 0.9 * (step + 1) / args.warmup
        progress = min((step - args.warmup) / max(1, args.steps - args.warmup), 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = None             # copied after the first step: some norms register their buffers on first use

    log = open(out / "log.jsonl", "a")
    meta = {"arm": args.arm, "args": vars(args), "stats": {k: v.tolist() for k, v in stats.items()},
            "train_tosses": len(train), "val_tosses": len(val), "vendor": str(VENDOR)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    it = batches(loader)
    t0, window = time.time(), []
    for step in range(1, args.steps + 1):
        pred, logits, labels, mask, target, ground = step_inputs(model, next(it))
        loss = masked_l1(pred.float(), labels, mask)
        if logits is not None:
            loss = loss + SCORER_WEIGHT * scorer_loss(logits, target)
        if ground is not None:
            loss = loss + GROUND_WEIGHT * ground_loss(take_match(model), ground)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], 1.0)
        optimizer.step()
        scheduler.step()
        if ema is None:
            ema = copy.deepcopy(model).eval()
            ema.requires_grad_(False)
            ema_params = [(e, p) for e, p in zip(ema.parameters(), model.parameters()) if p.requires_grad]
        with torch.no_grad():
            for e, p in ema_params:
                e.lerp_(p, 1 - args.ema)
        window.append(loss.item())
        if step % 50 == 0:
            rec = {"step": step, "loss": float(np.mean(window)), "grad": float(grad),
                   "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
                   "s_per_step": (time.time() - t0) / step}
            window = []
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.val_every == 0 or step in (100, args.steps):       # 100: exercise the path early
            buffers = dict(model.named_buffers())
            for name, buf in ema.named_buffers():
                buf.copy_(buffers[name])
            refresh_norms(ema, refresh_loader, 16)
            rec = {"step": step, "val_ema": evaluate(ema, val_loader, stats, 16),
                   "val_raw": evaluate(model, val_loader, stats, 16)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % args.save_every == 0 or step in (100, args.steps):
                torch.save({"arm": args.arm, "step": step, "model_config": config, "frames": args.frames,
                            "forecast": args.forecast, "ball": args.ball, "locator": args.locator,
                            "condition": args.condition, "no_velocity": args.no_velocity, "no_image": args.no_image,
                            "anchor": args.anchor, "intercept": icept,
                            "selective": args.selective,
                            "ema_state_dict": ema.state_dict(), "stats": meta["stats"]}, out / f"ema_{step}.pt")


if __name__ == "__main__":
    main()
