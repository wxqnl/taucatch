"""Builds the catching scene: FR3 arm + Shadow hand on a pedestal, a ball, cameras.

World frame: the robot base sits at (0, 0, BASE_Z) and faces +x; throws come
from x = 3 to 4 m toward the catch window around x = WINDOW_X.
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .ball import BALLS

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"

BASE_Z = 0.75
WINDOW_X = 0.70
TIMESTEP = 0.0005

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
# Panda limits (conservative for FR3): joint speed (rad/s), Cartesian speed and acceleration.
ARM_QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
CART_V_MAX = 1.7
CART_A_MAX = 13.0
# Arm profiles: multipliers on the Panda joint speed, acceleration/jerk and torque limits.
# "panda" is the real arm; "fast" is the idealized arm used for development and the demo
# (user decision 2026-10-20: idealize first, real limits later; E5 sweeps between them).
# "agile" is a high-speed arm for the long-reach demo (hand up to about 4 m/s and 4 g, the range of
# fast industrial arms); from rest, "fast" moves the palm about 30 cm in 300 ms, "agile" about 55 cm.
ARM_PROFILES = {
    "panda": dict(vel=1.0, acc=1.0, torque=1.0),
    "fast": dict(vel=1.5, acc=3.0, torque=2.0),
    "agile": dict(vel=3.0, acc=8.0, torque=5.0),
}
ARM_HOME = np.array([0.0, 0.3, 0.0, -1.95, 0.0, 3.85, 0.0])   # IK seed for the home pose (scripts/home_pose.py)

# Idealized hand (user decision 2026-10-20): light forearm, fast and strong fingers.
FOREARM_MASS = 0.05
WRIST_MOUNT = (0.03, 0.0, -0.08)   # wrist in the (rotated) forearm frame: 3 cm out, palm centred
FINGER_KP, FINGER_KV, FINGER_FORCE = 8.0, 0.0, 8.0
WRIST_KP, WRIST_FORCE = 60.0, 30.0


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """Quaternion of a MuJoCo camera at `pos` looking at `target` (camera looks along -z)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    z = -f
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


@dataclass
class SceneIds:
    arm_qpos: np.ndarray    # qpos addresses of the 7 arm joints
    arm_dof: np.ndarray     # dof addresses of the 7 arm joints
    arm_act: np.ndarray     # actuator ids of the 7 arm motors
    hand_act: np.ndarray    # actuator ids of the hand (wrist first)
    hand_qpos: np.ndarray   # qpos addresses of the 24 hand joints
    ball_body: int
    ball_qpos: int
    ball_dof: int
    ball_geom: int
    palm_site: int
    hand_geoms: np.ndarray  # geom ids that belong to the hand
    mocap: dict             # thrower part name -> mocap index
    balls: list             # BallIds of every ball (the first is ball_body / ball_qpos / ...)


@dataclass
class BallIds:
    body: int
    qpos: int
    dof: int
    geom: int


def ball_body_name(k):
    """Body name of ball k; the first ball keeps the plain name."""
    return "ball" if k == 0 else f"ball{k}"


def part_name(name, k):
    """Thrower part `name` ("thr_...") of thrower set k."""
    return name if k == 0 else name.replace("thr_", f"thr{k}_", 1)


# Thrower figure and launcher (plan §5.3): mocap bodies without collisions, placed per throw.
# name: (geom type, size, rgba). Capsules lie along the body z axis.
THROWER_PARTS = {
    "thr_upper": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.045, 0.15, 0), (0.85, 0.68, 0.58, 1)),
    "thr_fore": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.038, 0.13, 0), (0.85, 0.68, 0.58, 1)),
    "thr_hand": (mujoco.mjtGeom.mjGEOM_SPHERE, (0.045, 0, 0), (0.85, 0.68, 0.58, 1)),
    "thr_torso": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.15, 0.24, 0), (0.22, 0.3, 0.5, 1)),
    "thr_head": (mujoco.mjtGeom.mjGEOM_SPHERE, (0.11, 0, 0), (0.85, 0.68, 0.58, 1)),
    "thr_leg_l": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.07, 0.43, 0), (0.2, 0.2, 0.25, 1)),
    "thr_leg_r": (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.07, 0.43, 0), (0.2, 0.2, 0.25, 1)),
    "thr_barrel": (mujoco.mjtGeom.mjGEOM_CYLINDER, (0.05, 0.2, 0), (0.35, 0.35, 0.38, 1)),
    "thr_stand": (mujoco.mjtGeom.mjGEOM_BOX, (0.06, 0.06, 0.8), (0.3, 0.3, 0.33, 1)),   # reaches the floor
}
PARKED = (0.0, 0.0, -10.0)   # unused parts wait below the floor


def _add_thrower(spec, k=0):
    for name, (gtype, size, rgba) in THROWER_PARTS.items():
        name = part_name(name, k)
        body = spec.worldbody.add_body(name=name, pos=PARKED, mocap=True)
        body.add_geom(name=name + "_geom", type=gtype, size=list(size), rgba=list(rgba), contype=0, conaffinity=0,
                      density=0)


def _add_cameras(spec):
    world = spec.worldbody
    cams = {
        # stereo event cameras above and behind the robot, like eyes over a shoulder
        "ev_left": ((-0.6, 0.2, 1.9), (2.0, 0.0, 1.2), 55.0),
        "ev_right": ((-0.6, -0.2, 1.9), (2.0, 0.0, 1.2), 55.0),
        # RGB camera between them, for the VLA (plan §5.10)
        "rgb_head": ((-0.6, 0.0, 1.9), (2.0, 0.0, 1.2), 55.0),
        # third-person views for videos
        "side": ((2.2, -4.3, 1.7), (2.2, 0.0, 1.1), 50.0),      # robot to thrower
        "close": ((1.25, -0.75, 1.45), (0.72, 0.0, 1.25), 45.0),
        "reach": ((2.2, -0.8, 1.5), (0.8, 0.0, 1.22), 45.0),      # whole arm and the last metre of flight
        "front": ((4.6, 1.4, 1.8), (0.4, 0.0, 1.2), 40.0),
    }
    for name, (pos, target, fovy) in cams.items():
        world.add_camera(name=name, pos=pos, quat=look_at_quat(pos, target), fovy=fovy)


def build_model(ball_name="pingpong", arm="fast", balls=None):
    """`balls`: ball types of a volley (plan §5.10), one thrower set each; default one ball of `ball_name`."""
    balls = [ball_name] if balls is None else list(balls)
    ball = BALLS[balls[0]]          # sets the palm site: the ball centre when it rests on the palm
    torque = ARM_PROFILES[arm]["torque"]
    arm = mujoco.MjSpec.from_file(str(MENAGERIE / "franka_fr3" / "fr3.xml"))
    hand = mujoco.MjSpec.from_file(str(MENAGERIE / "shadow_hand" / "right_hand.xml"))

    # Hand mounted palm out: the palm normal is the flange axis and the fingers lie across the
    # flange, like a human wrist bent back to catch. The forearm housing is dropped (idealized).
    forearm = hand.body("rh_forearm")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float).flatten())
    forearm.quat = quat
    scale = FOREARM_MASS / forearm.mass
    forearm.mass = FOREARM_MASS
    forearm.inertia = np.asarray(forearm.inertia) * scale
    for geom in forearm.geoms:
        geom.contype, geom.conaffinity, geom.group = 0, 0, 5
    hand.body("rh_wrist").pos = list(WRIST_MOUNT)
    for act in hand.actuators:
        wrist = "WRJ" in act.name
        kp, force = (WRIST_KP, WRIST_FORCE) if wrist else (FINGER_KP, FINGER_FORCE)
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -FINGER_KV
        act.forcerange = [-force, force]
    hand.body("rh_palm").add_site(
        name="palm_center", pos=[0.005, -(0.012 + ball.radius), 0.08], size=[0.004], rgba=[0, 1, 0, 0.5]
    )

    # Arm: torque-controlled motors limited to the real joint torques.
    for act in arm.actuators:
        act.set_to_motor()
        jnt = arm.joint(act.target)
        jnt.actfrcrange = torque * np.asarray(jnt.actfrcrange)
        act.ctrlrange = jnt.actfrcrange
    arm.attach(hand, prefix="hand/", site=arm.site("attachment_site"))

    spec = arm
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]

    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.65, 0.75], rgb2=[0.1, 0.12, 0.15], width=512, height=3072)
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.42, 0.45, 0.5], rgb2=[0.33, 0.36, 0.4], width=300, height=300)
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[8, 8], texuniform=True, reflectance=0.1)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"

    world = spec.worldbody
    world.add_light(pos=[1.5, -1.0, 3.5], dir=[0, 0.2, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    world.add_light(pos=[3.0, 1.5, 3.0], dir=[-0.3, -0.3, -1], diffuse=[0.4, 0.4, 0.4], castshadow=False)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor_mat")
    world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, BASE_Z / 2],
                   size=[0.2, 0.2, BASE_Z / 2], rgba=[0.3, 0.3, 0.33, 1])
    spec.body("base").pos = [0, 0, BASE_Z]

    for k, name in enumerate(balls):
        b = BALLS[name]
        body = world.add_body(name=ball_body_name(k), pos=[3.5, 0.3 * k, 1.3])
        body.add_freejoint(name=ball_body_name(k) + "_free")
        body.explicitinertial = True
        body.ipos = [0, 0, 0]
        body.iquat = [1, 0, 0, 0]
        body.mass = b.mass
        body.inertia = [b.inertia] * 3
        # group 1: the event cameras draw the ball themselves (taucatch/events.py)
        body.add_geom(name=ball_body_name(k) + "_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[b.radius, 0, 0],
                      rgba=list(b.rgba), mass=0, density=0, condim=4, priority=2, group=1,
                      friction=[1.0, 0.005, 0.0001], solref=[0.004, 2.0], solimp=[0.9, 0.99, 0.001, 0.5, 2.0])
    _add_cameras(spec)
    for k in range(len(balls)):
        _add_thrower(spec, k)

    model = spec.compile()
    return model, scene_ids(model)


def scene_ids(model):
    arm_jids = [model.joint(n).id for n in ARM_JOINTS]
    hand_jids = [j for j in range(model.njnt) if model.joint(j).name.startswith("hand/")]
    hand_body_ids = {b for b in range(model.nbody) if model.body(b).name.startswith("hand/")}
    ball_jid = model.joint("ball_free").id
    balls = []
    k = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ball_body_name(k)) >= 0:
        jid = model.joint(ball_body_name(k) + "_free").id
        balls.append(BallIds(model.body(ball_body_name(k)).id, model.jnt_qposadr[jid], model.jnt_dofadr[jid],
                             model.geom(ball_body_name(k) + "_geom").id))
        k += 1
    return SceneIds(
        arm_qpos=np.array([model.jnt_qposadr[j] for j in arm_jids]),
        arm_dof=np.array([model.jnt_dofadr[j] for j in arm_jids]),
        arm_act=np.array([model.actuator(n).id for n in ARM_JOINTS]),
        hand_act=np.array([a for a in range(model.nu) if model.actuator(a).name.startswith("hand/")]),
        hand_qpos=np.array([model.jnt_qposadr[j] for j in hand_jids]),
        ball_body=model.body("ball").id,
        ball_qpos=model.jnt_qposadr[ball_jid],
        ball_dof=model.jnt_dofadr[ball_jid],
        ball_geom=model.geom("ball_geom").id,
        palm_site=model.site("hand/palm_center").id,
        hand_geoms=np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in hand_body_ids]),
        mocap={part_name(name, j): model.body_mocapid[model.body(part_name(name, j)).id]
               for j in range(len(balls)) for name in THROWER_PARTS},
        balls=balls,
    )
