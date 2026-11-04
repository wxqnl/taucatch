"""Ball locator inside the catch policy: SpikeVLA finds the ball, its own events measure it, a track memory carries it.

Per call (every 20 ms), for each event camera:
  1. gate: while the memory holds fewer than two measurements, a token scorer on SpikeVLA's backbone tokens picks
     the ball's patch (or "no ball"); afterwards the gate is the track's own prediction;
  2. measure: event-pixel mean shift inside the gate (radius 20, 10, 10 px) on the fresh pixels of the trace frame,
     i.e. pixels whose latest event is in the two newest 4 ms slices (red channel >= 143);
then the two views are triangulated, the measurement enters an N-slot memory (one slot per call) unless it lies off the
track's prediction (another ball crossing the gate in the image is far away in space), and a fit with known gravity over
the memory gives the ball's position (relative to the grasp centre) and velocity: seven state tokens of the action head,
as ball.py's measurement in round 4. ball.py's track_memory builds the same memory from recorded
events for training.

On held-out tosses the scorer's patch is 6.5 px from the ball (median, 100% within 20 px); with the gate inside 20 px
the track measures the ball to 0.48 cm and 2.5 cm/s.

Selective catching (SelectiveLocator): views 2 and 3 are RGB cameras 3 cm beside the event cameras, so an RGB pixel
shows what the event pixel shows, in colour (newest 30 Hz frame, 10 ms old: the ball is 5 px behind, median). Colour
survives in SDT-V3's stage-2 map (28 x 28, 8 px cells: a linear probe names the ball's colour 97%) but not in the
final 14 x 14 tokens (68%). Each RGB cell is matched against the instruction (mean SpikeBERT hidden state); the match
map is trained to point at the named ball's cell (grounding cross-entropy over the 784 cells, train.py). Its peak lands
on the named ball in 90% of views at every stage of the toss. Without a track each view looks for fresh events around
its peak, and the first measurements count only if the two views' rays meet (a one-view mistake does not triangulate), so
the track starts on the named ball when it moves and follows it from there.
"""
import numpy as np
import torch
import torch.nn as nn

from . import ball as bm
from .collect import EV_DT, STEP
from .events import CAMERAS, RES
from .policy import IMAGE_MEAN, IMAGE_STD
from .sim import GRAVITY

GRID = 14
PATCH = RES // GRID
FRESH = 143                     # red channel of a pixel whose latest event is 0 or 1 slice old (255, 171; 2 slices: 115)
LOST = 4                        # calls without an accepted measurement after which a track is dropped and looked for again
TAU_MAX = 0.6                   # s: anchored arm targets (CatchPolicy.anchor) lie on the located path up to this far ahead
OFF_MAX = 0.10                  # m: their largest offset across that path


class BallLocator(nn.Module):
    first_gap = 0.03                  # largest gap (m) between the two views' rays while the track holds < 3 points
    track_gate = 0.04                 # largest distance (m) from the track's prediction, once it holds 3 points
    fit_limit = 0.02                  # largest rms distance (m) of a track's points from its ballistic fit

    def __init__(self, channels, memory=bm.MEMORY):
        super().__init__()
        self.memory = memory
        self.scorer = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 128), nn.GELU(), nn.Linear(128, 1))
        self.null = nn.Parameter(torch.zeros(1))       # "no ball" logit, one per view and call
        centres, axes = zip(*(bm.camera(cam) for cam in CAMERAS))
        self.register_buffer("centre", torch.tensor(np.stack(centres), dtype=torch.float32), persistent=False)
        self.register_buffer("axes", torch.tensor(np.stack(axes), dtype=torch.float32), persistent=False)
        self.register_buffer("gravity", torch.tensor(GRAVITY, dtype=torch.float32), persistent=False)
        self.register_buffer("shift", torch.zeros(2, 2), persistent=False)   # px per view, from a look at the hand (ball.set_shift)
        # ages (s, before the policy step) of the current measurement and of the memory slots
        ages = EV_DT + STEP * np.arange(memory + 1)
        self.register_buffer("ages", torch.tensor(ages, dtype=torch.float32), persistent=False)
        ii, jj = torch.meshgrid(torch.arange(RES), torch.arange(RES), indexing="ij")
        self.register_buffer("rows", ii.float(), persistent=False)
        self.register_buffer("cols", jj.float(), persistent=False)
        self.register_buffer("mean_r", torch.tensor(IMAGE_MEAN[0]), persistent=False)
        self.register_buffer("std_r", torch.tensor(IMAGE_STD[0]), persistent=False)

    def scores(self, tokens, text=None, fine=None):
        """Backbone tokens [B, V, N, C] -> logits [B, 2, N + 1] of the event views, the last one "no ball"; and where to
        look without a track [B, 2, 2] (None: the scorer's top patch)."""
        logits = self.scorer(tokens[:, :len(CAMERAS)].float()).squeeze(-1)
        return torch.cat([logits, self.null.expand(*logits.shape[:2], 1)], -1), None

    def fit(self, points, valid, ages):
        """Least squares with known gravity: points [B, K, 3] at ages [K] (s before now) -> position, velocity, n."""
        w = valid.float()
        dt = -ages
        y = points - 0.5 * self.gravity * (dt ** 2)[None, :, None]
        s0, s1, s2 = w.sum(1), (w * dt).sum(1), (w * dt ** 2).sum(1)
        sy, syt = (w[..., None] * y).sum(1), (w[..., None] * y * dt[None, :, None]).sum(1)
        det = (s0 * s2 - s1 ** 2).clamp_min(1e-9)[:, None]
        pos = (s2[:, None] * sy - s1[:, None] * syt) / det
        vel = (s0[:, None] * syt - s1[:, None] * sy) / det
        return pos, vel, s0

    def project(self, p):
        """World points [B, 3] -> pixel (col, row) per view [B, V, 2]."""
        q = torch.einsum("vji,bvj->bvi", self.axes, p[:, None] - self.centre)
        f = (RES / 2) / np.tan(np.radians(bm.EVENT_FOVY) / 2)
        return torch.stack([RES / 2 + f * q[..., 0] / -q[..., 2], RES / 2 - f * q[..., 1] / -q[..., 2]], -1) + self.shift

    def triangulate(self, uv):
        """uv [B, V=2, 2] -> midpoint of the two rays [B, 3], gap between them [B]."""
        f = (RES / 2) / np.tan(np.radians(bm.EVENT_FOVY) / 2)
        uv = uv - self.shift
        cam = torch.stack([(uv[..., 0] - RES / 2) / f, -(uv[..., 1] - RES / 2) / f, -torch.ones_like(uv[..., 0])], -1)
        d = torch.einsum("vij,bvj->bvi", self.axes, cam)
        d = d / d.norm(dim=-1, keepdim=True)
        b = self.centre[1] - self.centre[0]
        d1, d2 = d[:, 0], d[:, 1]
        a11, a12, a22 = (d1 * d1).sum(-1), -(d1 * d2).sum(-1), (d2 * d2).sum(-1)
        r1, r2 = (d1 * b).sum(-1), -(d2 * b).sum(-1)
        det = (a11 * a22 - a12 ** 2).clamp_min(1e-9)
        s, t = (a22 * r1 - a12 * r2) / det, (a11 * r2 - a12 * r1) / det
        p1, p2 = self.centre[0] + s[:, None] * d1, self.centre[1] + t[:, None] * d2
        return (p1 + p2) / 2, (p1 - p2).norm(dim=-1)

    def measure(self, fresh, gate, tracking):
        """fresh [B, V, H, W] bool, gate [B, V, 2] -> uv [B, V, 2], found [B, V] (as ball.centroid). With a track
        (tracking [B]) the gate is its prediction and the window stays about the ball's own size: a wider one lets a
        neighbouring ball pull the mean by centimetres."""
        uv, found = gate, torch.ones(gate.shape[:2], dtype=torch.bool, device=gate.device)
        for wide, close in ((bm.GATE, 8.0), (10.0, 6.0), (10.0, 6.0)):
            radius = torch.where(tracking, close, float(wide))[:, None, None, None]
            near = fresh & (torch.hypot(self.cols - uv[..., 0, None, None], self.rows - uv[..., 1, None, None]) < radius)
            n = near.sum((-1, -2)).float()
            found = found & (n >= 3)
            c = torch.stack([(near * self.cols).sum((-1, -2)), (near * self.rows).sum((-1, -2))], -1) / n.clamp_min(1)[..., None]
            uv = torch.where(found[..., None], c, uv)
        return uv + 0.5, found

    def forward(self, tokens, pixel_values, grasp, memory, valid, gate=None, text=None, fine=None):
        """tokens [B, V, N, C]; pixel_values [B, V, 3, H, W] (normalised trace frames first); grasp [B, 3] (m);
        memory [B, M, 3], valid [B, M]; gate [B, 2, 2] replaces the scorer for acquisition (training);
        text [B, D] instruction features and fine [B, V, N, 4 * 128] stage-2 blocks (SelectiveLocator).
        -> ball features [B, 7], logits [B, 2, N + 1], memory, valid (current measurement in slot 0)."""
        logits, peak = self.scores(tokens, text, fine)
        valid = valid & valid[:, :LOST].any(1, keepdim=True)
        pos, vel, n = self.fit(memory, valid, self.ages[1:])
        a = self.ages[0]
        expected = pos - vel * a + 0.5 * self.gravity * a ** 2
        predicted = self.project(expected)
        tracking = (n >= 2)[:, None]
        if peak is not None and gate is None:
            patch, acquire = peak, torch.ones(peak.shape[:2], dtype=torch.bool, device=peak.device)
        elif gate is None:
            k = logits.argmax(-1)
            patch = torch.stack([(k % GRID + 0.5) * PATCH, (k // GRID + 0.5) * PATCH], -1).float()
            acquire = (k < GRID * GRID)
        else:
            patch, acquire = gate, torch.ones(gate.shape[:2], dtype=torch.bool, device=gate.device)
        centre = torch.where(tracking[..., None], predicted, patch)
        red = (pixel_values[:, :len(CAMERAS), 0].float() * self.std_r + self.mean_r) * 255
        uv, found = self.measure(red >= FRESH, centre, tracking[:, 0])
        point, gap = self.triangulate(uv)
        starting = (tracking[:, 0] | acquire.all(1)) & (gap < self.first_gap)
        seen = found.all(1) & torch.where(n >= 3, (point - expected).norm(dim=-1) < self.track_gate, starting)
        memory = torch.cat([point[:, None], memory[:, :-1]], 1)
        valid = torch.cat([seen[:, None], valid[:, :-1]], 1)
        pos, vel, n = self.fit(memory, valid, self.ages[:-1])
        dt = -self.ages[:-1][None, :, None]              # a ball obeys gravity; the arm, or two balls paired up, do not
        off = (memory - (pos[:, None] + vel[:, None] * dt + 0.5 * self.gravity * dt ** 2)).norm(dim=-1)
        rms = ((off * valid) ** 2).sum(1).div(n.clamp_min(1)).sqrt()
        valid = valid & ~((n >= 5) & (rms > self.fit_limit))[:, None]
        pos, vel, n = self.fit(memory, valid, self.ages[:-1])
        known = (n >= 2)[:, None]
        features = torch.cat([torch.where(known, pos - grasp, 0), torch.where(known, vel, 0), known.float()], -1)
        return features, logits, memory, valid


class SliceTrack:
    """The locator's measurement at every 4 ms event slice instead of every 20 ms policy step, as a loop beside the
    policy runs it on the event stream (mean shift and triangulation only, no network). Once the policy's own track
    holds, the buffered slices of the last 0.4 s are measured around its path; from then each new slice around this
    track's own gravity fit. Each policy call then gets a memory whose points lie on this fit, so the model reads its
    position and velocity: four times the measurements, the early velocity five times finer at the same noise."""
    window = 0.4                      # s of slices and points kept
    span = 0.04                       # s of points before the fit takes over from the policy's own track

    def __init__(self, shift=None):
        self.meter = BallLocator(8)
        if shift is not None:
            self.meter.shift.copy_(torch.as_tensor(shift, dtype=torch.float32))
        self.slices = []              # (time of the measurement, fresh pixels [V, H, W] bool)
        self.points = []              # (time, point) accepted
        self.coef = None              # (position, velocity) at time 0 of the gravity fit
        self.active = False

    def predict(self, t):
        p, v = self.coef
        return p + v * t + 0.5 * GRAVITY * t ** 2

    def add(self, t, fresh):
        """One slice: fresh [V, H, W] bool, its measurement at time t."""
        self.slices = [x for x in self.slices if x[0] > t - self.window] + [(t, fresh)]
        if self.active:
            self.measure(t, fresh, self.predict(t))
            if self.points[-1][0] < t - LOST * STEP:          # the ball is gone: wait for the policy's track again
                self.active, self.points, self.coef = False, [], None

    def start(self, t, p, v):
        """The policy's track (position p, velocity v at time t) holds: measure the buffered slices around it."""
        self.active, self.points, self.coef = True, [], None
        for ts, fresh in self.slices:            # around the policy's path: a fit over the first few slices drifts
            self.measure(ts, fresh, p + v * (ts - t) + 0.5 * GRAVITY * (ts - t) ** 2)
        if len(self.points) < 3:
            self.active, self.points, self.coef = False, [], None

    def measure(self, t, fresh, guess):
        m = self.meter
        g = torch.as_tensor(np.asarray(guess, np.float32))[None]
        uv, found = m.measure(torch.from_numpy(fresh)[None], m.project(g), torch.tensor([True]))
        if not found.all():
            return
        point = m.triangulate(uv)[0][0].double().numpy()
        if np.linalg.norm(point - guess) < m.track_gate:
            self.points = [x for x in self.points if x[0] > t - self.window] + [(t, point)]
            self.fit()

    def fit(self):
        """Gravity least squares; while the rms passes the locator's fit limit, the worst point goes."""
        while True:
            ts = np.array([x[0] for x in self.points]); P = np.stack([x[1] for x in self.points])
            if len(ts) < 3:
                return
            A = np.stack([np.ones_like(ts), ts], 1)
            c = np.linalg.lstsq(A, P - 0.5 * GRAVITY * ts[:, None] ** 2, rcond=None)[0]
            off = np.linalg.norm(P - (c[0] + np.outer(ts, c[1]) + 0.5 * GRAVITY * ts[:, None] ** 2), axis=1)
            if len(ts) < 5 or np.sqrt(np.mean(off ** 2)) <= BallLocator.fit_limit:
                self.coef = c[0], c[1]
                return
            del self.points[int(np.argmax(off))]

    def ready(self):
        return self.active and self.coef is not None and len(self.points) >= 5 and self.points[-1][0] - self.points[0][0] >= self.span

    def memory(self, t, ages):
        """Points on the fit at times t - ages [M], all valid, so that the model's own newest measurement weighs 1 / (M + 1)
        in its fit; None until the fit spans `span` s of points (a fit over a few slices has a poor velocity)."""
        if not self.ready():
            return None
        times = t - np.asarray(ages)
        return np.stack([self.predict(x) for x in times]).astype(np.float32), np.ones(len(times), bool)


class SelectiveLocator(BallLocator):
    def __init__(self, channels, text_dim, memory=bm.MEMORY, cell=128, dim=64):
        super().__init__(channels, memory)
        self.cell = nn.Sequential(nn.LayerNorm(cell), nn.Linear(cell, dim))
        self.word = nn.Linear(text_dim, dim)
        nn.init.zeros_(self.word.weight)                  # starts as the plain scorer: any moving ball
        nn.init.zeros_(self.word.bias)

    def scores(self, tokens, text, fine):
        n = len(CAMERAS)
        logits = self.scorer(tokens[:, :n].float()).squeeze(-1)
        b = fine.shape[0]
        blocks = fine[:, n:2 * n].float().reshape(b * n, GRID, GRID, -1).permute(0, 3, 1, 2)
        cells = nn.functional.pixel_shuffle(blocks, 2).permute(0, 2, 3, 1)            # [B * 2, 28, 28, 128]
        word = self.word(text.float()).repeat_interleave(n, 0)                         # [B * 2, dim]
        match = (self.cell(cells) * word[:, None, None]).sum(-1)[:, None] / self.word.out_features ** 0.5
        if not torch.compiler.is_compiling():
            self.match = match.view(b, n, -1)                                         # grounding logits
        top = match.amax((-1, -2), keepdim=True).detach()
        pooled = top + torch.log(nn.functional.avg_pool2d((match - top).exp(), 4, 2, 1) + 1e-12)
        cell = match.view(b, n, -1).argmax(-1)
        side = 2 * GRID
        peak = torch.stack([cell % side, cell // side], -1).float() * RES / side + RES / side / 2
        return torch.cat([logits + pooled.view(b, n, GRID * GRID), self.null.expand(b, n, 1)], -1), peak


def sentence(vla, instructions, device):
    """Mean SpikeBERT hidden state over the instruction's tokens [B, D]."""
    hidden, mask, _ = vla.text_encoder.encode_bert_hidden(instructions, device)
    w = mask.float()[..., None]
    return (hidden.float() * w).sum(1) / w.sum(1).clamp_min(1)


class CatchPolicy(nn.Module):
    """SpikeVLA with the ball locator: state8 + the locator's seven ball tokens drive the action head."""
    blind = False                     # the action head sees empty frames: its state tokens are all it has (evaluation)
    no_image = False                  # the action head gets zero image tokens: it reads the located ball and the hand only
    anchor = False                    # arm targets on the located ball's path: the head picks the time along it and an offset
    snap = False                      # the head's arm targets moved to the nearest point of the located ball's path

    def __init__(self, vla, stats, selective=False):
        super().__init__()
        self.vla = vla
        self.selective = selective
        if selective:
            self.locator = SelectiveLocator(vla.vision_encoder.hidden_size, vla.text_encoder.bert.config.hidden_size)
        else:
            self.locator = BallLocator(vla.vision_encoder.hidden_size)
        for k in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(k, torch.as_tensor(np.asarray(stats[k], np.float32)), persistent=False)
        self.register_buffer("up", torch.tensor([0.0, 0.0, 1.0]), persistent=False)
        self.register_buffer("taus", torch.arange(0.0, TAU_MAX + 1e-6, 0.002), persistent=False)
        _cache_vision()

    def anchored(self, actions, ball):
        """Arm channels of the chunk [B, H, D] -> targets on the located ball's path (ball features [B, 7]): channel 0
        picks the time tau along it (0 to TAU_MAX), channels 1-2 an offset across it (OFF_MAX at +-1). A target that
        slides along the path still meets the ball; only the offset and the track's error put it off. Until the track
        holds, and from the step the head closes the hand (the ball then rides in it, off the path), the channels stay
        the plain target offset."""
        raw = actions[..., :3].float()
        known = (ball[:, None, 6:7] > 0.5) & (actions[..., 6:7] < 0)
        p = torch.where(known, ball[:, None, :3], 0.0)
        v = torch.where(known, ball[:, None, 3:6], -self.up.roll(-2))   # other rows: any finite direction
        tau = TAU_MAX * (raw[..., :1] + 1) / 2
        g = self.locator.gravity
        point = p + v * tau + 0.5 * g * tau ** 2
        d = v + g * tau
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        across = torch.cross(d, self.up.expand_as(d), dim=-1)
        across = across / across.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        target = point + OFF_MAX * (raw[..., 1:2] * across + raw[..., 2:3] * torch.cross(across, d, dim=-1))
        pos = torch.where(known, (target - self.action_mean[:3]) / self.action_std[:3], raw)
        return torch.cat([pos.to(actions.dtype), actions[..., 3:]], -1)

    def snapped(self, actions, ball):
        """Arm targets of the chunk [B, H, D] moved to the nearest point (2 ms grid up to TAU_MAX ahead) of the located
        ball's path (ball features [B, 7]): the head keeps where along the path it meets the ball, the track sets where
        the path is. Plain targets until the track holds and from the step the head closes the hand."""
        raw = actions[..., :3].float()
        known = (ball[:, None, 6:7] > 0.5) & (actions[..., 6:7] < 0)
        target = raw * self.action_std[:3] + self.action_mean[:3]
        tau = self.taus[None, :, None]
        path = ball[:, None, :3] + ball[:, None, 3:6] * tau + 0.5 * self.locator.gravity * tau ** 2      # [B, T, 3]
        i = (target[:, :, None] - path[:, None]).norm(dim=-1).argmin(-1)                                # [B, H]
        on = path.gather(1, i[..., None].expand(-1, -1, 3))
        pos = torch.where(known, (on - self.action_mean[:3]) / self.action_std[:3], raw)
        return torch.cat([pos.to(actions.dtype), actions[..., 3:]], -1)

    def forward(self, instructions, pixel_values, state8, memory, valid, gate=None):
        """state8 normalised [B, 8] -> actions [B, 12, D], logits, memory, valid, ball features (physical)."""
        vla = self.vla
        calibrating = vla.text_encoder.text_projection.input_norm.calibrating
        if self.selective:                              # the stage-2 map as well, for the RGB cells
            encode = lambda x: vla.vision_encoder(x, fine=True)
        else:
            encode = vla._encode_vision_tokens
        if vla.training or calibrating:
            tokens = encode(pixel_values)
            rows = [tokens]
        else:                                           # eval encodes one observation at a time, as the VLA does
            rows = [encode(pixel_values[i:i + 1]) for i in range(len(pixel_values))]
            tokens = tuple(torch.cat(parts) if parts[0] is not None else None for parts in zip(*rows))
        grasp = state8[:, :3] * self.state_std[:3] + self.state_mean[:3]
        text = sentence(vla, instructions, pixel_values.device) if self.selective else None
        ball, logits, memory, valid = self.locator(tokens[0], pixel_values, grasp, memory, valid, gate, text, tokens[1])
        state = torch.cat([state8, (ball - self.state_mean[8:]) / self.state_std[8:]], -1)
        if self.selective:                              # the VLA itself reads no stage-2 blocks
            rows = [(row[0], None) for row in rows]
        if self.blind:
            empty = (torch.zeros(3, device=pixel_values.device) - torch.tensor(IMAGE_MEAN, device=pixel_values.device)) \
                / torch.tensor(IMAGE_STD, device=pixel_values.device)
            empty = empty[None, None, :, None, None].expand_as(pixel_values[:1])
            rows = [encode(empty)] * len(pixel_values)
        if self.no_image:
            rows = [tuple(None if x is None else torch.zeros_like(x) for x in row) for row in rows]
        vla._vision_rows = list(rows)                     # the VLA reuses these encoder outputs
        try:
            actions = vla(instructions, pixel_values, state)
        finally:
            vla._vision_rows = None
        if self.anchor:
            actions = self.anchored(actions, ball)
        if self.snap:
            actions = self.snapped(actions, ball)
        return actions, logits, memory, valid, ball


def _cache_vision():
    """Let TurboVLA take encoder outputs computed by CatchPolicy (each backbone pass runs once per observation)."""
    from turbovla.models.turbovla import TurboVLA
    if getattr(TurboVLA, "_catch_cached", False):
        return
    original = TurboVLA._encode_vision_tokens

    def encode(self, pixel_values):
        rows = getattr(self, "_vision_rows", None)
        if not rows:
            return original(self, pixel_values)
        tokens, blocks = rows.pop(0)
        if tokens.shape[0] != pixel_values.shape[0]:
            raise RuntimeError("cached encoder rows do not match the VLA's call")
        return tokens, blocks
    TurboVLA._encode_vision_tokens = encode
    TurboVLA._catch_cached = True


def main():
    """Fit the locator's token scorer on a trained ball-state policy (backbone frozen) -> a CatchPolicy checkpoint.

        CUDA_VISIBLE_DEVICES=7 python -m taucatch.vla_catch.locator runs_v2/ball_v1_f7/ema_2000.pt \
            --out runs_v2/ball_v1_f7/located_2000.pt
    """
    import argparse
    from .data import load_tosses
    from .policy import build_model, pixel_values, set_numerics
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data_v2/bc_v1")
    ap.add_argument("--samples", type=int, default=24000)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    set_numerics()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not ck.get("ball"):
        raise ValueError("the policy must read the ball state (train.py --ball)")
    vla, _ = build_model(ck["model_config"], condition=ck.get("condition", False))
    vla.load_state_dict(ck["ema_state_dict"])
    model = CatchPolicy(vla, ck["stats"]).cuda().eval()
    rng = np.random.default_rng(0)
    tosses = load_tosses(args.data, lambda m: m["caught"])
    sets = {"fit": [t for t in tosses if t.meta["seed"] % 20 != 0], "test": [t for t in tosses if t.meta["seed"] % 20 == 0]}
    data = {}
    with torch.no_grad():
        for name, pool in sets.items():
            picks = [(t, s) for t in pool for s in range(len(t.times))]
            n = args.samples if name == "fit" else args.samples // 8
            picks = [picks[j] for j in rng.choice(len(picks), n, replace=False)]
            X, Y = [], []
            for b in range(0, n, 64):
                part = picks[b:b + 64]
                frames = torch.from_numpy(np.stack([t.frames(s, mode=ck.get("frames", "window")) for t, s in part]))
                X.append(vla._encode_vision_tokens(pixel_values(frames.cuda()))[0].half().cpu())
                Y.append(np.stack([bm.acquisition(t, s, rng)[1] for t, s in part]))
            data[name] = torch.cat(X), torch.from_numpy(np.concatenate(Y))
            print(f"{name}: {n} samples, no ball in {(data[name][1] == GRID * GRID).float().mean():.0%} of views",
                  flush=True)
    loc = model.locator
    opt = torch.optim.AdamW([p for p in loc.parameters()], 1e-3, weight_decay=1e-4)
    X, Y = data["fit"]
    for step in range(args.steps):
        idx = torch.randint(0, len(X), (256,))
        loss = nn.functional.cross_entropy(loc.scores(X[idx].cuda())[0].flatten(0, 1), Y[idx].cuda().flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    X, Y = data["test"]
    with torch.no_grad():
        k = torch.cat([loc.scores(X[b:b + 512].cuda())[0].argmax(-1).cpu() for b in range(0, len(X), 512)])
    ball, picked = Y < GRID * GRID, k < GRID * GRID
    near = (k % GRID - Y % GRID).abs().le(1) & (k // GRID - Y // GRID).abs().le(1)
    print(f"test views with the ball: patch exact {(k == Y)[ball].float().mean():.1%}, within one patch "
          f"{(near & picked)[ball].float().mean():.1%}, missed {(~picked)[ball].float().mean():.1%}; "
          f"views without: false pick {picked[~ball].float().mean():.1%}", flush=True)
    out = {k: v for k, v in ck.items() if k != "ema_state_dict"}
    out.update({"ball": False, "locator": True, "ema_state_dict": model.state_dict(), "located_from": args.ckpt})
    torch.save(out, args.out)


if __name__ == "__main__":
    main()
