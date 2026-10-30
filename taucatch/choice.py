"""Stage 2 of selective catching (plan §5.10): instructions and the VLA's inputs.

The event tracks give the candidates. The VLA sees:
  view 0: the RGB frame (center square, 224 px) with a numbered ring at each candidate and its
          predicted path to the catch window drawn in;
  view 1: a 2 x 2 mosaic of full-resolution crops around the candidates, with the same numbers;
  state:  speed and time to the catch window per candidate (zeros before release);
and answers with the number of the named ball, or "none" when it is not among the candidates
yet (candidates are the balls in flight: a ball has a track only after its release).

Instruction kinds need different things: color and type (appearance), thrower side and shirt
(relation to the scene), speed and arrival (the tracks' facts), landing side (the drawn paths),
negation. Test instructions use templates never seen in training, and two colors (purple, pink)
are never named in training.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_CANDIDATES = 4
NONE = MAX_CANDIDATES          # answer index for "the named ball is not among the candidates (yet)"
VIEW = 224
REF_RADIUS = 0.03              # m: crops are sized by distance with this radius, so ball sizes still differ
SHIRTS = {"red": (0.8, 0.15, 0.12), "green": (0.15, 0.55, 0.2), "white": (0.9, 0.9, 0.88),
          "black": (0.08, 0.08, 0.08), "yellow": (0.9, 0.8, 0.1), "blue": (0.15, 0.25, 0.75)}
HELD_OUT_COLORS = ("purple", "pink")
TYPE_WORD = {"tennis": "tennis", "pingpong": "ping-pong"}

TEMPLATES = {                       # kind: (training templates, test templates)
    "color": (["catch the {c} ball", "get the {c} one", "catch the ball that is {c}"],
              ["grab the {c} ball", "go for the {c} one"]),
    "type": (["catch the {t} ball", "get the {t} ball"], ["grab the {t} ball"]),
    "side_person": (["catch the ball from the person on your {s}", "catch the ball thrown from your {s}"],
                    ["take the ball the person on your {s} threw"]),
    "side_launcher": (["catch the ball from the launcher on your {s}", "catch the ball shot from your {s}"],
                      ["take the ball the launcher on your {s} fired"]),
    "shirt_person": (["catch the ball thrown by the person in {c}", "catch the ball from the person wearing {c}"],
                     ["grab the ball the person in {c} threw"]),
    "shirt_launcher": (["catch the ball from the {c} launcher", "catch the ball shot by the {c} launcher"],
                       ["grab the ball the {c} launcher fired"]),
    "speed": (["catch the {sp} ball", "get the {sp} one"], ["go for the {sp} ball"]),
    "arrival": (["catch the ball that arrives {ar}", "catch the ball that gets here {ar}"],
                ["grab whichever ball reaches you {ar}"]),
    "landing": (["catch the ball coming to your {l}", "catch the ball that will reach your {l} side"],
                ["grab the ball heading to your {l}"]),
    "height": (["catch the {h} ball", "catch the ball that comes in {h}"], ["grab the ball arriving {h}"]),
    "negation": (["do not catch the {c} ball", "catch the ball that is not {c}"], ["leave the {c} ball alone"]),
}
SPEED_WORD = {"fastest": ("fastest", "quickest"), "slowest": ("slowest", "slowest")}


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _unique_max(values, margin):
    """Index of the largest value if it beats the runner-up by margin, else None."""
    order = np.argsort(values)[::-1]
    return int(order[0]) if len(values) > 1 and values[order[0]] - values[order[1]] >= margin else None


def instruction_candidates(facts, rng, split):
    """All unambiguous (kind, text, named ball) options for paired action data.

    facts: "thrower" ("human" or "launcher") and "balls", per ball: color, type, side_y (release
    point y; +y is the robot's left), shirt (thrower's shirt or launcher color), released, speed,
    arrival (s), aim (y, z). split "test" uses the test templates and names only the held-out colors."""
    balls, who = facts["balls"], "person" if facts["thrower"] == "human" else "launcher"
    n, test = len(balls), split == "test"

    def say(kind, **kw):
        tpls = TEMPLATES[kind][1 if test else 0]
        return tpls[rng.integers(len(tpls))].format(**kw)

    def nameable(c):
        return c in HELD_OUT_COLORS if test else c not in HELD_OUT_COLORS

    cands = []                                     # (kind, text, named ball)
    for i, b in enumerate(balls):
        if nameable(b["color"]):
            cands.append(("color", say("color", c=b["color"]), i))
        if [x["type"] for x in balls].count(b["type"]) == 1:
            cands.append(("type", say("type", t=TYPE_WORD[b["type"]]), i))
        cands.append(("shirt", say("shirt_" + who, c=b["shirt"]), i))
        if n == 2 and nameable(b["color"]):
            cands.append(("negation", say("negation", c=b["color"]), 1 - i))
    names = ["left", "right"] if n == 2 else ["left", "middle", "right"]
    for rank, i in enumerate(np.argsort([-b["side_y"] for b in balls])):
        text = say("side_" + who, s=names[rank]).replace("on your middle", "in the middle")
        cands.append(("side", text.replace("from your middle", "from the middle"), int(i)))
    if all(b["released"] for b in balls):
        sp = np.array([b["speed"] for b in balls])
        ar = np.array([b["arrival"] for b in balls])
        aim = np.array([b["aim"] for b in balls])
        for word, vals, margin, kind, key in (
                ("fastest", sp, 1.0, "speed", "sp"), ("slowest", -sp, 1.0, "speed", "sp"),
                ("first", -ar, 0.015, "arrival", "ar"), ("last", ar, 0.015, "arrival", "ar"),
                ("left", aim[:, 0], 0.15, "landing", "l"), ("right", -aim[:, 0], 0.15, "landing", "l"),
                ("highest", aim[:, 1], 0.12, "height", "h"), ("lowest", -aim[:, 1], 0.12, "height", "h")):
            i = _unique_max(vals, margin)
            if i is not None:
                w = SPEED_WORD[word][1 if test else 0] if kind == "speed" else word
                cands.append((kind, say(kind, **{key: w}), i))
    return cands


def instructions(facts, rng, split, k=3):
    """Up to k (text, named ball, kind), kinds drawn without repeats."""
    cands = instruction_candidates(facts, rng, split)
    out = []
    for kind in rng.permutation(sorted({c[0] for c in cands}))[:k]:
        same = [c for c in cands if c[0] == kind]
        _, text, i = same[rng.integers(len(same))]
        out.append((text, i, str(kind)))
    return out


def views(frame, cands, rng=None):
    """VLA inputs from the 640 x 480 RGB frame. cands: list (in mark order) of dicts with uv (px),
    r_px (REF_RADIUS at the candidate's distance, in px), path_uv (list of px points or None).
    Returns overview, mosaic (224 x 224 x 3 uint8)."""
    h, w = frame.shape[:2]
    x0 = (w - h) // 2
    scale = VIEW / h
    over = Image.fromarray(frame[:, x0:x0 + h]).resize((VIEW, VIEW), Image.BILINEAR)
    dr = ImageDraw.Draw(over)
    font = _font(11)
    to_view = lambda uv: ((uv[0] - x0) * scale, uv[1] * scale)
    for k, c in enumerate(cands):
        if c.get("path_uv"):
            pts = [to_view(p) for p in c["path_uv"]]
            for a, b in zip(pts[:-1:2], pts[1::2]):
                dr.line([a, b], fill=(255, 255, 255), width=1)
            ex, ey = pts[-1]
            dr.line([(ex - 3, ey - 3), (ex + 3, ey + 3)], fill=(255, 255, 255), width=1)
            dr.line([(ex - 3, ey + 3), (ex + 3, ey - 3)], fill=(255, 255, 255), width=1)
        u, v = to_view(c["uv"])
        rr = max(5.0, c["r_px"] * scale + 3)
        dr.ellipse([u - rr, v - rr, u + rr, v + rr], outline=(255, 255, 255), width=1)
        dr.text((u + rr + 1, v - rr - 9), str(k + 1), fill=(255, 255, 255), font=font, stroke_width=2,
                stroke_fill=(0, 0, 0))
    mosaic = Image.new("RGB", (VIEW, VIEW), (0, 0, 0))
    cell = VIEW // 2
    for k, c in enumerate(cands[:MAX_CANDIDATES]):
        half = int(np.clip(6 * c["r_px"], 12, 48))
        u, v = int(round(c["uv"][0])), int(round(c["uv"][1]))
        pad = np.pad(frame, ((half, half), (half, half), (0, 0)))
        crop = Image.fromarray(pad[v:v + 2 * half, u:u + 2 * half]).resize((cell, cell), Image.BILINEAR)
        ImageDraw.Draw(crop).text((4, 2), str(k + 1), fill=(255, 255, 255), font=_font(16), stroke_width=2,
                                  stroke_fill=(0, 0, 0))
        mosaic.paste(crop, ((k % 2) * cell, (k // 2) * cell))
    return np.asarray(over), np.asarray(mosaic)


def state_vector(cands):
    """Speed / 20 and time to the catch window / 0.3 per candidate (zeros if not yet released)."""
    s = np.zeros(2 * MAX_CANDIDATES, np.float32)
    for k, c in enumerate(cands[:MAX_CANDIDATES]):
        if c["released"]:
            s[k] = c["speed"] / 20.0
            s[MAX_CANDIDATES + k] = c["time_to_window"] / 0.3
    return s
