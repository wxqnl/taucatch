"""Ball-sized objects told apart by what they are, not by colour: each comes with a look-alike of the same mean colour.

Sphere geoms keep the catch physics of the plain ball. Each kind is a procedural cube texture. Instructions name the kind.
"""
import numpy as np

SIDE = 128                       # texels per cube face


def _directions():
    """Unit directions of the texels, faces in MuJoCo's order +x -x +y -y +z -z stacked vertically [6 S, S, 3]."""
    u, v = np.meshgrid((np.arange(SIDE) + 0.5) / SIDE * 2 - 1, (np.arange(SIDE) + 0.5) / SIDE * 2 - 1)
    one = np.ones_like(u)
    faces = [(one, -v, -u), (-one, -v, u), (u, one, v), (u, -one, -v), (u, -v, one), (-u, -v, -one)]
    d = np.concatenate([np.stack(f, -1) for f in faces], 0)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _seam(d, width):
    """Tennis-ball curve on the sphere: distance to it below width."""
    phi = np.arctan2(d[..., 1], d[..., 0])
    return np.abs(d[..., 2] - 0.55 * np.cos(2 * phi)) < width


def _speckle(d, seed, amount):
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(24, 3)) * 14
    phase = rng.uniform(0, 2 * np.pi, 24)
    return amount * np.sin(d @ k.T + phase).mean(-1, keepdims=True) * 2


def _icosahedron():
    g = (1 + 5 ** 0.5) / 2
    v = np.array([(s * 1, t * g, 0) for s in (-1, 1) for t in (-1, 1)] + [(0, s * 1, t * g) for s in (-1, 1) for t in (-1, 1)]
                 + [(t * g, 0, s * 1) for s in (-1, 1) for t in (-1, 1)], float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def basketball(d):
    c = np.ones(d.shape) * (0.86, 0.40, 0.09)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lines = (np.abs(z) < 0.035) | (np.abs(x) < 0.035) | (np.abs(np.abs(y) - 0.62 * (1 - 0.5 * z * z)) < 0.035)
    c[lines] = (0.05, 0.04, 0.04)
    return c


def tangerine(d):
    return np.clip((0.90, 0.46, 0.08) + _speckle(d, 1, 0.08), 0, 1)


def tennis(d):
    c = np.ones(d.shape) * (0.74, 0.84, 0.16)
    c[_seam(d, 0.07)] = (0.96, 0.96, 0.92)
    return c


def lime(d):
    return np.clip((0.68, 0.82, 0.20) + _speckle(d, 2, 0.07), 0, 1)


def soccer(d):
    c = np.ones(d.shape) * (0.95, 0.95, 0.93)
    near = (d @ _icosahedron().T).max(-1)
    c[near > 0.94] = (0.06, 0.06, 0.07)
    return c


def baseball(d):
    c = np.ones(d.shape) * (0.95, 0.94, 0.90)
    phi = np.arctan2(d[..., 1], d[..., 0])
    stitches = _seam(d, 0.06) & (np.sin(phi * 34) > 0)
    c[stitches] = (0.80, 0.10, 0.10)
    return c


def watermelon(d):
    c = np.ones(d.shape) * (0.26, 0.60, 0.22)
    phi = np.arctan2(d[..., 1], d[..., 0])
    c[np.sin(9 * phi + 1.5 * np.sin(5 * d[..., 2])) > 0.15] = (0.06, 0.26, 0.09)
    return c


def lettuce_ball(d):
    """Solid green ball of the same mean colour as the watermelon's stripes and rind together."""
    return np.clip((0.17, 0.44, 0.16) + _speckle(d, 3, 0.03), 0, 1)


KINDS = {                        # word: (texture, look-alike word)
    "basketball": (basketball, "tangerine"), "tangerine": (tangerine, "basketball"),
    "tennis ball": (tennis, "lime"), "lime": (lime, "tennis ball"),
    "soccer ball": (soccer, "baseball"), "baseball": (baseball, "soccer ball"),
    "watermelon": (watermelon, "green ball"), "green ball": (lettuce_ball, "watermelon"),
}
TONES = {"red": (0.85, 0.14, 0.12), "green": (0.14, 0.62, 0.22), "blue": (0.16, 0.30, 0.88), "yellow": (0.88, 0.76, 0.12)}


def _fibonacci(n):
    i = np.arange(n) + 0.5
    z, phi = 1 - 2 * i / n, np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


FIGURES = {                      # adjective: where the dark tone lies on the sphere, about half of it
    "striped": lambda d: np.sin(np.arcsin(np.clip(d[..., 2], -1, 1)) * 11) > 0,
    "dotted": lambda d: (d @ _fibonacci(26).T).max(-1) > 0.955,
    "checkered": lambda d: (np.sin(5 * np.arctan2(d[..., 1], d[..., 0])) * np.sin(np.arcsin(np.clip(d[..., 2], -1, 1)) * 6.5)) > 0,
    "wavy": lambda d: np.sin(7 * np.arctan2(d[..., 1], d[..., 0]) + 3.0 * np.sin(6 * d[..., 2])) > 0,
    "spiral": lambda d: np.sin(4 * np.arctan2(d[..., 1], d[..., 0]) + 11 * d[..., 2]) > 0,
}
PATTERNS = [f"{tone} {figure} ball" for tone in TONES for figure in FIGURES]
UNSEEN_PATTERNS = ("red dotted ball", "green wavy ball", "blue checkered ball", "yellow striped ball")     # never trained
PHRASES = ("catch the {}", "grab the {}", "get the {}", "catch that {}")
HELD_OUT_PHRASES = ("take the {}",)


def textures():
    """Texel colours per kind, each pair of look-alikes brought to one mean colour: mean colour and brightness tell
    them apart no better than a coin."""
    d = _directions()
    out = {word: np.clip(paint(d), 0, 1) for word, (paint, _) in KINDS.items()}
    for _ in range(8):
        for word, (_, twin) in KINDS.items():
            target = (out[word].reshape(-1, 3).mean(0) + out[twin].reshape(-1, 3).mean(0)) / 2
            out[word] = np.clip(out[word] * target / out[word].reshape(-1, 3).mean(0), 0, 1)
    return out


def pattern_textures():
    """Two tones of one colour on every figure, each brought to the colour's mean: colour says the tone, not the figure."""
    d = _directions()
    out = {}
    for tone, c in TONES.items():
        c = np.asarray(c)
        for figure, dark in FIGURES.items():
            t = np.ones(d.shape) * np.clip(c * 1.25, 0, 1)
            t[dark(d)] = c * 0.45
            out[f"{tone} {figure} ball"] = np.clip(t * (0.85 * c) / t.reshape(-1, 3).mean(0), 0, 1)
    return out


def mean_colours():
    return {word: t.reshape(-1, 3).mean(0) for word, t in textures().items()}


def all_textures():
    return {**textures(), **pattern_textures()}


def expose(image, gain, rng=None, noise=3.0):
    """The camera's take of a scene: exposure and white balance of this toss, sensor noise."""
    x = image.astype(np.float32) * gain
    if rng is not None:
        x += rng.normal(0, noise, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def look(rng):
    """Exposure and white balance of one toss, and the tints of three object instances."""
    gain = rng.uniform(0.65, 1.15) * rng.uniform(0.92, 1.08, 3)
    return gain, rng.uniform(0.82, 1.0, (3, 1)) * rng.uniform(0.94, 1.06, (3, 3))


def add_kinds(spec):
    """Materials `kind_0`... in the order of all_textures."""
    import mujoco
    for i, (word, texels) in enumerate(all_textures().items()):
        rgb = (texels * 255).astype(np.uint8)
        tex = spec.add_texture(name=f"kind_{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE, width=SIDE, height=6 * SIDE, nchannel=3)
        tex.data = rgb.tobytes()
        mat = spec.add_material(name=f"kind_{i}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"kind_{i}"
        mat.rgba = (1, 1, 1, 1)
        mat.specular, mat.shininess = 0.2, 0.2
