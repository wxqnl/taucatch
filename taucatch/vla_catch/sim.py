"""OpenArm v2 catching scene: toss sampling, task-space control and the catch check.

The scene is Enactic's `pedestal/catch_scene.xml` (third_party/openarm_mujoco, Apache-2.0).
The right arm catches; the left arm hangs. Shadows, reflections and haze are switched off
so the event cameras see only motion.

Selective catching (distractors > 0): further balls of other colours fly at the same time on their own
ballistic paths (mocap bodies, no contacts), an RGB camera beside each event camera sees the colours,
and the instruction names the ball to catch. The `ball` body is always the one named.
"""
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from . import objects

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "third_party/openarm_mujoco/v2/pedestal/catch_scene.xml"
GRAVITY = np.array([0.0, 0.0, -9.81])
GRASP_OFFSET = np.array([0.0, 0.0, -0.15])     # grasp centre in the control-point frame
ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
FINGER_OPEN, FINGER_CLOSED = -0.7854, 0.4
BALL_RADIUS = 0.035


def flat_flight(to_target, speed):
    """Shortest flight time over this displacement at this release speed; the slowest throw if no flight reaches it."""
    t = np.arange(0.05, 1.2, 0.001)
    v = np.linalg.norm((to_target[None] - 0.5 * GRAVITY[None] * t[:, None] ** 2) / t[:, None], axis=1)
    reach = np.flatnonzero(v <= speed)
    return float(t[reach[0] if reach.size else np.argmin(v)])


@dataclass
class Toss:
    launch: np.ndarray      # launch position (m)
    velocity: np.ndarray    # launch velocity (m/s)
    target: np.ndarray      # grasp-centre point the ball passes through
    flight: float           # time from release to target (s)
    hold: float             # time the ball is held before release (s); the arm settles meanwhile
    others: tuple = ()      # distractors: (launch, velocity, hold) each
    colors: tuple = ()      # colour names: the named ball first, then the distractors
    instruction: str = "catch the ball"


def ballistic(p, v, t):
    t = np.asarray(t, dtype=float)[..., None]
    return p + v * t + 0.5 * GRAVITY * t * t


AIM_X = float(os.environ.get("VLA_AIM_X", 0.85))   # m along the throw the cameras look at; 0.85 sees to 3.5 m, 1.3 to 5 m
EVENT_CAMERAS = {           # name: (position, look-at point); behind the arm, ~90 deg apart, see hand and launch area
    "ev_left": ((-0.25, -1.25, 1.55), (AIM_X, -0.2, 0.95)),
    "ev_right": ((-0.25, 0.85, 1.55), (AIM_X, -0.2, 0.95)),
}
EVENT_FOVY = 62
RGB_OFFSET = 0.03           # m: each RGB camera sits beside its event camera, along the image x axis
RGB_CAMERAS = {"rgb_left": "ev_left", "rgb_right": "ev_right"}
COLORS = {                  # rgb of the ball colours an instruction can name
    "red": (0.90, 0.12, 0.10), "green": (0.10, 0.70, 0.20), "blue": (0.15, 0.30, 0.95),
    "yellow": (0.95, 0.85, 0.10), "purple": (0.60, 0.20, 0.80), "white": (0.95, 0.95, 0.95),
    "orange": (0.95, 0.55, 0.10),
}
PHRASES = ("catch the {} ball", "catch the {} one", "grab the {} ball", "get the {} ball")
HELD_OUT_PHRASES = ("take the {} ball",)
RPM = 2 * np.pi / 60
MOTORS = {                  # Damiao: no-load rad/s at 24 V and 48 V, peak Nm
    "8009": (160 * RPM, 320 * RPM, 40.0), "4340": (52 * RPM, 100 * RPM, 27.0), "4310": (200 * RPM, 400 * RPM, 7.0)}
SENSOR = 896                # px per side of the RGB sensor: the 224 px view is this image shrunk 4 times
TEXTURED = [False]          # kinds scenes: materials of objects.py in the model (use_kinds before building a sim)


def use_kinds():
    TEXTURED[0] = True


def rgb_pose(event_camera):
    pos, target = EVENT_CAMERAS[event_camera]
    x = look_at(pos, target)[:3]
    return tuple(np.asarray(pos) + RGB_OFFSET * x), tuple(np.asarray(target) + RGB_OFFSET * x)


def look_at(pos, target, up=(0, 0, 1)):
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.concatenate([x, np.cross(z, x)])


def add_selective(spec, distractors):
    """Distractor balls (mocap, no contacts) and the RGB cameras."""
    for i in range(distractors):
        body = spec.worldbody.add_body(name=f"distractor{i}", pos=[0, 0, -1], mocap=True)
        body.add_geom(name=f"distractor{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0],
                      rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
    for name, event_camera in RGB_CAMERAS.items():
        pos, target = rgb_pose(event_camera)
        cam = spec.worldbody.add_camera(name=name, pos=pos, fovy=EVENT_FOVY)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = look_at(pos, target)
    if TEXTURED[0]:
        objects.add_kinds(spec)
        spec.visual.global_.offwidth = spec.visual.global_.offheight = SENSOR


def load_model(gravcomp=False, distractors=0):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    if gravcomp:           # the arm controller cancels gravity, as the real OpenArm stack does
        for body in spec.bodies:
            if body.name.startswith("openarm_right"):
                body.gravcomp = 1.0
    for name, (pos, target) in EVENT_CAMERAS.items():
        cam = spec.worldbody.add_camera(name=name, pos=pos, fovy=EVENT_FOVY)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = look_at(pos, target)
    if distractors or TEXTURED[0]:
        add_selective(spec, distractors)
    return spec.compile()


class CatchSim:
    """OpenArm v2 right arm; subclasses swap the arm by overriding the class attributes and `load`."""
    SITE = "right_ee_control_point"
    JOINTS = ARM_JOINTS
    ACTUATORS = [f"right_joint{i}_ctrl" for i in range(1, 8)]
    FINGER_ACT, FINGER_JOINT = "right_finger1_ctrl", "openarm_right_finger_joint1"
    FINGER_OPEN, FINGER_CLOSED = FINGER_OPEN, FINGER_CLOSED
    GRASP_OFFSET = GRASP_OFFSET
    SHOULDER, REACH = np.array([0.0, -0.092, 0.697]), 0.66
    JOINT_SPEED = (8, 8, 10, 10, 12, 12, 12)
    IK_DAMPING = 1e-4
    EXPERT_BLEND = 1.0            # fingers turn to face the incoming ball
    EXPERT_LEAD = 0.04            # close this long before the ball reaches the grasp centre (~90 ms to contact)
    READY_KEY = "catch_ready"

    drag = 0.0                 # 1/m: ball deceleration per speed squared
    launch_x = (1.15, 1.65)    # m: where tosses start
    launch_z = (0.75, 1.20)    # m: release height
    throw_speed = None         # (low, high) m/s: flat throws at this release speed in place of a drawn flight time
    ball_quat = (1.0, 0.0, 0.0, 0.0)
    rig_home = None            # camera poses as calibrated
    MOTOR_LAYOUT = ("8009", "8009", "4340", "4340", "4310", "4310", "4310")
    no_load = None             # rad/s per joint once limit_motors holds the arm to its motors' torque-speed line
    gain = np.ones(3)          # kinds scenes: the RGB camera's exposure and white balance of this toss

    def __init__(self, gravcomp=True, grasp_offset=None, distractors=0):
        self.grasp_offset = np.asarray(self.GRASP_OFFSET if grasp_offset is None else grasp_offset, dtype=float)
        self.distractors = distractors
        m = self.load(gravcomp)
        m.vis.quality.shadowsize = 0
        m.vis.quality.offsamples = 0     # MSAA resolve is not repeatable: static edges would fire events
        m.mat_reflectance[:] = 0
        m.vis.rgba.haze[:] = 0
        m.light_castshadow[:] = 0
        self.m, self.d = m, mujoco.MjData(m)
        name = lambda kind, n: mujoco.mj_name2id(m, kind, n)
        self.site = name(mujoco.mjtObj.mjOBJ_SITE, self.SITE)
        self.ball = name(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_q = m.jnt_qposadr[m.body_jntadr[self.ball]]
        self.ball_v = m.jnt_dofadr[m.body_jntadr[self.ball]]
        self.ball_geom = name(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.mocap = [m.body_mocapid[name(mujoco.mjtObj.mjOBJ_BODY, f"distractor{i}")] for i in range(distractors)]
        self.distractor_geoms = [name(mujoco.mjtObj.mjOBJ_GEOM, f"distractor{i}_geom") for i in range(distractors)]
        self.default_rgba = m.geom_rgba[self.ball_geom].copy()
        joints = [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.vadr = np.array([m.jnt_dofadr[j] for j in joints])
        self.qlo, self.qhi = m.jnt_range[joints].T
        self.arm_act = np.array([name(mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self.ACTUATORS])
        self.finger_act = None if self.FINGER_ACT is None else name(mujoco.mjtObj.mjOBJ_ACTUATOR, self.FINGER_ACT)
        self.finger_q = None if self.FINGER_JOINT is None else \
            m.jnt_qposadr[name(mujoco.mjtObj.mjOBJ_JOINT, self.FINGER_JOINT)]
        self.ready_key = name(mujoco.mjtObj.mjOBJ_KEY, self.READY_KEY)
        self.speed = np.asarray(self.JOINT_SPEED, dtype=float)
        self.ee_body = m.site_bodyid[self.site]
        self.reset_arm()
        self.ready_grasp = self.grasp_point().copy()
        self.ready_rot = self.site_rot().copy()

    def load(self, gravcomp):
        return load_model(gravcomp, self.distractors)

    # ----- state -------------------------------------------------------------------------
    def reset_arm(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.ready_key)
        self.d.qvel[:] = 0
        self.q_ref = self.d.qpos[self.qadr].copy()
        self.set_finger(False)
        mujoco.mj_forward(self.m, self.d)

    def site_rot(self):
        return self.d.site_xmat[self.site].reshape(3, 3)

    def grasp_point(self):
        return self.d.site_xpos[self.site] + self.site_rot() @ self.grasp_offset

    def grasp_velocity(self):
        jacp = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, self.grasp_point(), self.ee_body)
        return jacp @ self.d.qvel

    def ball_state(self):
        q, v = self.ball_q, self.ball_v
        return self.d.qpos[q:q + 3].copy(), self.d.qvel[v:v + 3].copy()

    def finger(self):
        return 0.0 if self.finger_q is None else float(self.d.qpos[self.finger_q])

    def finger_command(self):
        return 0.0 if self.finger_act is None else float(self.d.ctrl[self.finger_act])

    def state8(self):
        """Proprioception given to the policy: grasp-centre position, velocity, finger angle, command."""
        return np.concatenate([self.grasp_point(), self.grasp_velocity(),
                               [self.finger(), self.finger_command()]]).astype(np.float32)

    # ----- toss --------------------------------------------------------------------------
    def sample_toss(self, rng, box=((-0.06, 0.10), (-0.18, 0.18), (-0.14, 0.14)), flight=(0.45, 0.75)):
        offset = np.array([rng.uniform(*b) for b in box])
        target = self.ready_grasp + offset
        launch = np.array([rng.uniform(*self.launch_x), rng.uniform(-0.75, 0.25), rng.uniform(*self.launch_z)])
        t = rng.uniform(*flight)
        if self.throw_speed is not None:
            t = flat_flight(target - launch, rng.uniform(*self.throw_speed))
        velocity = (target - launch - 0.5 * GRAVITY * t * t) / t
        return Toss(launch, velocity, target, t, rng.uniform(0.30, 0.40))

    def sample_scene(self, rng, held_out=False):
        """Selective toss: the named ball as sample_toss, the distractors released within 0.12 s of it on
        their own paths through the catch region (pass points at least 15 cm from the named ball's)."""
        toss = self.sample_toss(rng)
        names = rng.choice(list(COLORS), self.distractors + 1, replace=False)
        others = []
        for _ in range(self.distractors):
            for _ in range(100):
                d = self.sample_toss(rng, box=((-0.10, 0.15), (-0.30, 0.30), (-0.22, 0.22)))
                if np.linalg.norm(d.target - toss.target) > 0.15:
                    break
            others.append((d.launch, d.velocity, max(0.2, toss.hold + rng.uniform(-0.12, 0.12))))
        phrase = rng.choice(HELD_OUT_PHRASES if held_out else PHRASES)
        return Toss(toss.launch, toss.velocity, toss.target, toss.flight, toss.hold, tuple(others),
                    tuple(str(c) for c in names), phrase.format(names[0]))

    def sample_kinds(self, rng, held_out=False):
        """Selective toss as sample_scene, the objects named by kind: the named one's look-alike always flies with it."""
        toss = self.sample_scene(rng, held_out)
        words = list(objects.KINDS)
        named = str(rng.choice(words))
        twin = objects.KINDS[named][1]
        rest = [w for w in words if w not in (named, twin)]
        others = [twin] + [str(w) for w in rng.choice(rest, self.distractors - 1, replace=False)]
        phrase = str(rng.choice(objects.HELD_OUT_PHRASES if held_out else objects.PHRASES))
        return replace(toss, colors=(named, *others), instruction=phrase.format(named))

    def sample_patterns(self, rng, held_out=False, unseen=False):
        """Selective toss as sample_scene, named by colour and figure: one other object shares the colour, one the figure.
        unseen: the named object is one of the combinations never trained on; otherwise none of them appears."""
        toss = self.sample_scene(rng, held_out)
        hidden = objects.UNSEEN_PATTERNS
        pool = list(hidden) if unseen else [w for w in objects.PATTERNS if w not in hidden]
        named = str(rng.choice(pool))
        tone, figure = named.split()[:2]
        same_tone = [w for w in objects.PATTERNS if w.startswith(tone + " ") and w != named and (unseen or w not in hidden)]
        same_figure = [w for w in objects.PATTERNS if f" {figure} " in w and w != named and (unseen or w not in hidden)]
        others = (str(rng.choice(same_tone)), str(rng.choice(same_figure)))[:self.distractors]
        phrase = str(rng.choice(objects.HELD_OUT_PHRASES if held_out else objects.PHRASES))
        return replace(toss, colors=(named, *others), instruction=phrase.format(named))

    def paint(self, toss):
        """Ball colours of this toss (the default colour without a named one), or the objects of a kinds scene,
        each turned to a random side."""
        self.ball_quat = (1.0, 0.0, 0.0, 0.0)
        if toss.colors and toss.colors[0] not in COLORS:
            rng = np.random.default_rng(int(abs(toss.launch[0]) * 1e9))
            words = list(objects.all_textures())
            turn = lambda: tuple(q / np.linalg.norm(q) for q in [rng.normal(size=4)])[0]
            self.gain, tints = objects.look(rng)
            for geom, word, tint in zip([self.ball_geom] + self.distractor_geoms, toss.colors, tints):
                self.m.geom_matid[geom] = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MATERIAL, f"kind_{words.index(word)}")
                self.m.geom_rgba[geom] = (*np.minimum(tint, 1.0), 1.0)
            self.ball_quat = turn()
            for i in self.mocap:
                self.d.mocap_quat[i] = turn()
            return
        rgba = lambda c: (*COLORS[c], 1.0)
        self.m.geom_rgba[self.ball_geom] = rgba(toss.colors[0]) if toss.colors else self.default_rgba
        for geom, c in zip(self.distractor_geoms, toss.colors[1:]):
            self.m.geom_rgba[geom] = rgba(c)

    def move_distractors(self, toss, t):
        """Distractors on their ballistic paths at time t, resting on the floor after landing."""
        for i, (launch, velocity, hold) in zip(self.mocap, toss.others):
            p = ballistic(np.asarray(launch), np.asarray(velocity), max(t - hold, 0.0))
            p[2] = max(p[2], BALL_RADIUS)
            self.d.mocap_pos[i] = p
        for i in self.mocap[len(toss.others):]:
            self.d.mocap_pos[i] = (0.0, 0.0, -1.0)

    def place_ball(self, p, v=np.zeros(3)):
        q, dv = self.ball_q, self.ball_v
        self.d.qpos[q:q + 3] = p
        self.d.qpos[q + 3:q + 7] = self.ball_quat
        self.d.qvel[dv:dv + 3] = v
        self.d.qvel[dv + 3:dv + 6] = 0

    # ----- control -----------------------------------------------------------------------
    def track(self, target, rot_target, dt, gain=1.0):
        """One damped-least-squares IK step on the joint reference, rate limited per joint."""
        m, d = self.m, self.d
        saved = d.qpos[self.qadr].copy()
        d.qpos[self.qadr] = self.q_ref
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        point = self.grasp_point()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, point, self.ee_body)
        J = np.vstack([jacp[:, self.vadr], 0.3 * jacr[:, self.vadr]])
        R = self.site_rot()
        err_r = 0.5 * (np.cross(R[:, 0], rot_target[:, 0]) + np.cross(R[:, 1], rot_target[:, 1])
                       + np.cross(R[:, 2], rot_target[:, 2]))
        err = np.concatenate([target - point, 0.3 * err_r]) * gain
        dq = J.T @ np.linalg.solve(J @ J.T + self.IK_DAMPING * np.eye(6), err)
        dq = np.clip(dq, -self.speed * dt, self.speed * dt)
        self.q_ref = np.clip(self.q_ref + dq, self.qlo, self.qhi)
        d.qpos[self.qadr] = saved
        mujoco.mj_forward(m, d)
        d.ctrl[self.arm_act] = self.q_ref

    def set_throws(self, speed, start):
        """Flat throws at this release speed from this far ahead of the ready point (m): one value, or low and high."""
        x = self.ready_grasp[0]
        margin = 0.1 if len(start) == 1 else 0.0
        self.throw_speed = (speed[0], speed[-1])
        self.launch_x = (x + start[0] - margin, x + start[-1] + margin)

    def limit_motors(self, volts):
        """The motors' datasheet line: peak torque at rest, none at the no-load speed, braking beyond it."""
        motors = [MOTORS[k] for k in self.MOTOR_LAYOUT]
        self.no_load = np.array([m[0 if volts == 24 else 1] for m in motors])
        self.peak = np.array([m[2] for m in motors])
        self.speed = np.minimum(self.speed, 0.95 * self.no_load)

    def step_physics(self):
        if self.no_load is not None:
            w, act = self.d.qvel[self.vadr], np.array(self.arm_act)
            self.m.actuator_forcerange[act, 1] = np.clip(self.peak * (1 - w / self.no_load), -self.peak, self.peak)
            self.m.actuator_forcerange[act, 0] = np.clip(-self.peak * (1 + w / self.no_load), -self.peak, self.peak)
        if self.drag:              # air drag on the ball: deceleration drag * speed ** 2
            _, v = self.ball_state()
            self.d.xfrc_applied[self.ball, :3] = -self.drag * self.m.body_mass[self.ball] * np.linalg.norm(v) * v
        mujoco.mj_step(self.m, self.d)

    def miscalibrate(self, cm, deg, seed=0):
        """Move each camera rig away from its calibration by cm and deg, each in a fixed random direction."""
        rng = np.random.default_rng(seed)
        if self.rig_home is None:
            self.rig_home = self.m.cam_pos.copy(), self.m.cam_quat.copy()
        self.m.cam_pos[:], self.m.cam_quat[:] = self.rig_home
        for event, rgb in zip(EVENT_CAMERAS, RGB_CAMERAS):
            shift, axis = rng.normal(size=3), rng.normal(size=3)
            shift, axis = shift / np.linalg.norm(shift), axis / np.linalg.norm(axis)
            turn, out = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(turn, axis, np.radians(deg))
            for name in (event, rgb):
                i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if i < 0:
                    continue
                self.m.cam_pos[i] += shift * cm / 100
                mujoco.mju_mulQuat(out, self.m.cam_quat[i], turn)
                self.m.cam_quat[i] = out

    def set_finger(self, close):
        if self.finger_act is not None:
            self.d.ctrl[self.finger_act] = self.FINGER_CLOSED if close else self.FINGER_OPEN

    def caught(self):
        """Ball held: centre within 4.5 cm of the grasp centre and moving with the hand."""
        p, v = self.ball_state()
        near = np.linalg.norm(p - self.grasp_point()) < 0.045
        slow = np.linalg.norm(v - self.grasp_velocity()) < 0.5
        return bool(near and slow and p[2] > 0.2)


def align_rotation(base, approach):
    """Rotate `base` minimally so that its -z axis (the finger direction) points along `approach`."""
    a = -base[:, 2]
    b = approach / np.linalg.norm(approach)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-8:
        return base.copy()
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + s * K + (1 - c) * K @ K
    return rot @ base
