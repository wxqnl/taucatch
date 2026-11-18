"""OpenArm v2 real-robot scene: a ping-pong ball from a fixed launcher, one DVS, two third-person RGB, a wrist RGB.

Everything is in the robot frame of sim.py: origin on the floor under the pedestal centre, x toward the launcher,
y to the robot's left, z up; the arms attach 0.697 m up and the right arm catches. The scene (docs/openarm_scene.md)
is what the remote site builds:

- Ball: 40 mm, 2.7 g, with air drag (Cd 0.5: 0.14 /m, a third of g at 5 m/s, 37 cm over a 0.65 s flight).
- Launcher: outlet 2.0 m ahead of the hand's ready point, in line with it, 0.80 m above the floor, no spin. The
  nominal ball leaves at 5.2 m/s, 45 deg up, peaks at 1.41 m and reaches the hand after 0.65 s.
- Hand: the v2 gripper (fingers and housing) replaced by an adapter and a palm facing the launcher with four fingers
  and a thumb (b601.add_hand), palm lined so a ping-pong ball barely rebounds; `catcher="gripper"` keeps the gripper.
- Cameras (tools/openarm_scene.py `cameras` and `tracker` chose them): the DVS on the robot's right, past the
  launcher, looking back at the robot; two third-person RGB on the left: rgb1 at the DVS's mirror image, rgb2 low
  behind the launcher looking along the flight; the wrist RGB on the palm's upper edge, looking where the palm
  faces.
- A black three-sided enclosure, its back 1 m behind the robot, which also stops missed balls.
"""
import mujoco
import numpy as np

from . import b601
from .expert import Expert
from .sim import GRAVITY, SCENE, CatchSim, Toss, look_at

BALL_RADIUS, BALL_MASS = 0.020, 0.0027
DRAG = 0.5 * 1.2 * 0.5 * np.pi * BALL_RADIUS ** 2 / BALL_MASS   # 1/m: rho 1.2, Cd 0.5
BALL_RGBA = (0.98, 0.45, 0.05, 1.0)                              # orange
READY_POINT = np.array([0.4128, -0.219, 0.9568])                  # the v2 ready grasp point (sim.CatchSim)
READY_ROT = np.array([[0.0, 0.519, -0.8548], [1.0, 0.0, 0.0], [0.0, -0.8548, -0.519]])
LAUNCHER = np.array([READY_POINT[0] + 2.0, READY_POINT[1], 0.80])  # outlet centre
FLIGHT = 0.65                                                      # s: nominal flight, outlet to ready point
TOSS_BOX = ((-0.05, 0.05), (-0.10, 0.10), (-0.10, 0.10))          # m: where tosses pass, around the ready point
TOSS_FLIGHT = (0.55, 0.75)                                         # s
CAMERAS = {    # name: (position, look-at, vertical fov deg, width, height)
    "dvs": ((2.60, -1.42, 1.60), (1.50, -0.219, 1.05), 62.0, 640, 480),     # right, past the launcher, looking back
    "rgb1": ((2.60, 0.98, 1.60), (1.50, -0.219, 1.05), 43.0, 1280, 720),    # the DVS's mirror image on the left
    "rgb2": ((3.10, 0.38, 1.10), (1.20, -0.219, 0.95), 43.0, 1280, 720),    # low behind the launcher, along the flight
}
WRIST_FOVY, WRIST_SIZE = 66.0, (960, 600)                          # the v2 wrist camera's lens and image
WRIST_POS = (-0.01, -0.075, 0.0)   # m, hand mount frame (x where the palm faces, -y up at the ready pose)
HAND_MOUNT = 0.075                                                 # m: palm plate this far along the site's -z
HAND_NAMES = [f"hand_{k}{link}" for k in range(5) for link in "ab"]
BACKDROP = ((-1.0, 2.6, -2.6, 2.2), 2.2)    # m: (back x, sides up to x, right side y, left side y), height
RIG_RGBA = {"dvs": (0.16, 0.47, 0.84, 1), "rgb1": (0.11, 0.69, 0.48, 1), "rgb2": (0.84, 0.27, 0.48, 1)}
GRIPPER_READY = [1.63955, 0.67234, -0.68604, 1.32130, -0.69509, 0.33258, 0.43927]


def fly(p, v, t_end, dt=0.001, drag=DRAG):
    """Positions and velocities every dt from 0 to t_end under gravity and drag."""
    n = int(round(t_end / dt)) + 1
    P, V = np.zeros((n, 3)), np.zeros((n, 3))
    p, v = np.array(p, float), np.array(v, float)
    for i in range(n):
        P[i], V[i] = p, v
        v = v + (GRAVITY - drag * np.linalg.norm(v) * v) * dt
        p = p + v * dt
    return P, V


def aim(launch, target, t, drag=DRAG):
    """Launch velocity that reaches target after t seconds under drag."""
    v = (target - launch - 0.5 * GRAVITY * t * t) / t
    for _ in range(30):
        err = target - fly(launch, v, t, drag=drag)[0][-1]
        if np.linalg.norm(err) < 1e-6:
            break
        v = v + err / t
    return v


def add_camera(body, name, pos, target, fovy, up=(0, 0, 1)):
    cam = body.add_camera(name=name, pos=list(pos), fovy=fovy)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = look_at(pos, target, up)
    return cam


def add_launcher(world, velocity):
    """Stand, body, hopper and barrel; the barrel ends at the outlet along the launch direction (visual only)."""
    vis = dict(contype=0, conaffinity=0)
    d = velocity / np.linalg.norm(velocity)
    back = LAUNCHER - 0.16 * d                       # barrel root inside the body
    world.add_geom(name="launcher_stand", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[back[0] + 0.05, back[1], 0.27],
                   size=[0.2, 0.2, 0.27], rgba=[0.45, 0.33, 0.22, 1], **vis)
    world.add_geom(name="launcher_body", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[back[0] + 0.07, back[1], 0.62],
                   size=[0.12, 0.10, 0.08], rgba=[0.12, 0.2, 0.42, 1], **vis)
    world.add_geom(name="launcher_hopper", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[back[0] + 0.12, back[1], 0.80],
                   size=[0.11, 0.10, 0], rgba=[0.75, 0.8, 0.88, 0.55], **vis)
    world.add_geom(name="launcher_barrel", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=list(LAUNCHER - 0.08 * d),
                   size=[0.028, 0.08, 0], quat=_quat_z_to(d), rgba=[0.15, 0.15, 0.17, 1], **vis)


def add_rig(world, name, pos, target, rgba):
    """A camera body on a pole, its lens along the view (visual only, so overviews show where the cameras stand)."""
    vis = dict(contype=0, conaffinity=0)
    x, y = look_at(pos, target)[:3], look_at(pos, target)[3:]
    R = np.stack([x, y, np.cross(x, y)], 1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    body = world.add_body(name=f"{name}_rig", pos=list(pos), quat=list(q))
    body.add_geom(name=f"{name}_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.035, 0.035], pos=[0, 0, 0.05],
                  rgba=list(rgba), **vis)
    body.add_geom(name=f"{name}_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.015, 0], pos=[0, 0, 0.0],
                  rgba=[0.1, 0.1, 0.1, 1], **vis)
    world.add_geom(name=f"{name}_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, (pos[2] - 0.05) / 2, 0],
                   pos=[pos[0], pos[1], (pos[2] - 0.05) / 2], rgba=[0.3, 0.3, 0.32, 1], **vis)


def _quat_z_to(d):
    """Quaternion turning +z onto d."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s, c = np.linalg.norm(axis), float(z @ d)
    if s < 1e-9:
        return [1.0, 0, 0, 0]
    a = np.arctan2(s, c)
    axis /= s
    return [np.cos(a / 2), *(np.sin(a / 2) * axis)]


def scene_spec(catcher="hand", gravcomp=True, trail=(), rigs=False):
    """The v2 catch scene with the real site's ball, launcher, cameras and backdrop.
    trail: ball marks to draw; rigs: draw the cameras on their poles."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    if gravcomp:
        for body in spec.bodies:
            if body.name.startswith("openarm_right"):
                body.gravcomp = 1.0
    ball = spec.geom("ball_geom")
    ball.size = [BALL_RADIUS, 0, 0]
    ball.mass = BALL_MASS
    ball.rgba = list(BALL_RGBA)
    world = spec.worldbody
    for name, (pos, target, fovy, w, h) in CAMERAS.items():
        add_camera(world, name, pos, target, fovy)
        if rigs:
            add_rig(world, name, pos, target, RIG_RGBA[name])
    add_launcher(world, aim(LAUNCHER, READY_POINT, FLIGHT))
    (x0, x1, y0, y1), height = BACKDROP
    black = dict(rgba=[0.05, 0.05, 0.06, 1], type=mujoco.mjtGeom.mjGEOM_BOX)
    world.add_geom(name="backdrop_back", pos=[x0, (y0 + y1) / 2, height / 2],
                   size=[0.01, (y1 - y0) / 2, height / 2], **black)
    for side, y in (("right", y0), ("left", y1)):
        world.add_geom(name=f"backdrop_{side}", pos=[(x0 + x1) / 2, y, height / 2],
                       size=[(x1 - x0) / 2, 0.01, height / 2], **black)
    for i, p in enumerate(trail):
        world.add_geom(name=f"trail{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=list(p), size=[BALL_RADIUS, 0, 0],
                       rgba=[*BALL_RGBA[:3], 0.55], contype=0, conaffinity=0)
    if catcher == "hand":
        add_hand(spec, gravcomp)
    return spec


def add_hand(spec, gravcomp):
    """Replace the right gripper fingers by a palm facing the throw, four fingers and a thumb, and a wrist camera."""
    for name in ("openarm_right_ee_inner_finger", "openarm_right_ee_outer_finger"):
        spec.delete(spec.body(name))
    for e in list(spec.equalities):
        if e.name == "openarm_right_ee_finger_joint_mimic":
            spec.delete(e)
    for a in list(spec.actuators):
        if a.name == "right_finger1_ctrl":
            spec.delete(a)
    for c in list(spec.cameras):
        if c.name == "camera_wrist_right":
            spec.delete(c)
    for key in list(spec.keys):
        spec.delete(key)
    ee = spec.body("openarm_right_ee_base_link")
    for g in list(ee.geoms):           # the gripper housing reaches 11 cm past the flange, beyond the palm: it goes too
        spec.delete(g)
    ee.add_geom(name="hand_adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.032, HAND_MOUNT / 2, 0],
                pos=[0, 0, -HAND_MOUNT / 2], rgba=[0.25, 0.25, 0.27, 1], contype=0, conaffinity=0)
    # mount frame: x along the site's -z (where the palm faces), z across the palm from the thumb to the fingers
    mount = ee.add_body(name="hand_mount", pos=[0, 0, -HAND_MOUNT], quat=[0.7071068, 0, 0.7071068, 0])
    b601.add_hand(mount)
    # wrist camera on the palm's upper edge (mount -y is up at the ready pose), looking where the palm faces
    add_camera(mount, "camera_wrist_right", WRIST_POS, (0.6, WRIST_POS[1] + 0.06, 0.0), WRIST_FOVY, up=(0, -1, 0))
    kp, kv, force = b601.HAND_GAINS
    for j in HAND_NAMES:
        spec.add_actuator(name=f"{j}_ctrl", target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -kv] + [0] * 7,
                          forcelimited=True, forcerange=[-force, force])
    for body in spec.bodies:
        if body.name.startswith("hand_"):
            for other in ("openarm_right_ee_base_link", "openarm_right_link6"):
                spec.add_exclude(bodyname1=body.name, bodyname2=other)
            if gravcomp:
                body.gravcomp = 1.0


class PingPongSim(CatchSim):
    """OpenArm v2 catching ping-pong balls from the launcher, with the hand (default) or the v2 gripper."""
    EXPERT_BLEND = 0.5
    EXPERT_LEAD = 0.05            # the hand closes in ~100 ms; 30 ms early drops to 58%, 80 ms to 1%
    hand_act = None
    hand_goal = hand_now = 0.0

    def __init__(self, catcher="hand", bounce=0.5, trail=(), rigs=False, **kw):
        self.catcher, self.trail, self.rigs = catcher, trail, rigs
        if catcher == "hand":
            self.FINGER_ACT = self.FINGER_JOINT = None
            self.GRASP_OFFSET = np.array([0.0, 0.0, -(HAND_MOUNT + b601.CUP_START + b601.CUP_WALL + BALL_RADIUS)])
        else:
            self.EXPERT_BLEND, self.EXPERT_LEAD = CatchSim.EXPERT_BLEND, CatchSim.EXPERT_LEAD
        super().__init__(**kw)
        self.drag = DRAG
        self.launch_x = (LAUNCHER[0], LAUNCHER[0])
        self.set_bounce(bounce)

    def load(self, gravcomp):
        spec = scene_spec(self.catcher, gravcomp, self.trail, self.rigs)
        if self.catcher != "hand":
            return spec.compile()
        m = spec.compile()
        q, ctrl = self._solve_ready(m)
        key = spec.add_key(name=self.READY_KEY)
        key.qpos, key.ctrl = q, ctrl
        return spec.compile()

    def _solve_ready(self, m):
        """Joint pose with the grasp centre at READY_POINT and the site at READY_ROT, from the gripper's pose."""
        d = mujoco.MjData(m)
        site = m.site(self.SITE).id
        joints = [m.joint(j).id for j in self.JOINTS]
        qadr = [m.jnt_qposadr[j] for j in joints]
        vadr = [m.jnt_dofadr[j] for j in joints]
        lo, hi = m.jnt_range[joints].T
        ball = m.jnt_qposadr[m.body_jntadr[m.body("ball").id]]
        d.qpos[ball:ball + 7] = [*LAUNCHER, 1, 0, 0, 0]
        d.qpos[qadr] = GRIPPER_READY
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(3000):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[site].reshape(3, 3)
            point = d.site_xpos[site] + R @ self.GRASP_OFFSET
            err_r = 0.5 * sum(np.cross(R[:, i], READY_ROT[:, i]) for i in range(3))
            err = np.concatenate([READY_POINT - point, err_r])
            mujoco.mj_jac(m, d, jacp, jacr, point, m.site_bodyid[site])
            J = np.vstack([jacp[:, vadr], jacr[:, vadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq, lo, hi)
        if np.linalg.norm(err) > 1e-4:
            raise RuntimeError(f"no ready pose for the hand (error {np.linalg.norm(err):.2g})")
        ctrl = np.zeros(m.nu)
        for i, name in enumerate(self.ACTUATORS):
            ctrl[m.actuator(name).id] = d.qpos[qadr[i]]
        return d.qpos.copy(), ctrl

    def set_bounce(self, damping):
        """Ball contact as a spring-damper of this damping ratio: on the hand, or (gripper) on all it touches."""
        names = ("hand_",) if self.catcher == "hand" else ("finger_",)
        ids = [i for i in range(self.m.ngeom) if (self.m.geom(i).name or "").startswith(names)]
        g = self.m.geom("ball_geom").id
        self.m.geom_solref[ids + [g]] = (0.004, damping)
        if self.catcher != "hand":          # the gripper's collision class has priority: let the ball's solref win
            self.m.geom_priority[g] = 5

    # ----- tosses from the launcher ------------------------------------------------------
    def sample_toss(self, rng, box=TOSS_BOX, flight=TOSS_FLIGHT):
        target = self.ready_grasp + np.array([rng.uniform(*b) for b in box])
        t = rng.uniform(*flight)
        return Toss(LAUNCHER.copy(), aim(LAUNCHER, target, t, self.drag), target, t, rng.uniform(0.30, 0.40))

    # ----- the hand ----------------------------------------------------------------------
    def set_finger(self, close):
        if self.catcher != "hand":
            return super().set_finger(close)
        self.hand_goal = b601.HAND_CLOSED if close else 0.0

    def reset_arm(self):
        self.hand_now = self.hand_goal = 0.0
        super().reset_arm()

    def step_physics(self):
        if self.catcher == "hand":
            if self.hand_act is None:
                self.hand_act = [self.m.actuator(f"{j}_ctrl").id for j in HAND_NAMES]
            step = b601.HAND_RATE * self.m.opt.timestep
            self.hand_now += np.clip(self.hand_goal - self.hand_now, -step, step)
            self.d.ctrl[self.hand_act] = self.hand_now
        super().step_physics()

    def caught(self):
        if self.catcher != "hand":
            return super().caught()
        p, v = self.ball_state()
        axis = -self.site_rot()[:, 2]
        off = p - self.grasp_point()
        along = off @ axis
        across = np.linalg.norm(off - along * axis)
        slow = np.linalg.norm(v - self.grasp_velocity()) < 0.5
        return bool(slow and -0.02 < along < 0.08 and across < 0.06 and p[2] > 0.2)


class DragExpert(Expert):
    """The true-state expert with the ball's drag in its intercept."""

    def intercept(self, p, v):
        sim = self.sim
        P, V = fly(p, v, self.taus[-1], dt=0.005, drag=sim.drag)
        pts, vel = P[:len(self.taus)], V[:len(self.taus)]
        g = sim.grasp_point()
        ok = (pts[:, 2] > 0.45) & (np.linalg.norm(pts - sim.SHOULDER, axis=1) < self.reach) & (pts[:, 0] > 0.12)
        if not ok.any():
            return None
        dist = np.linalg.norm(pts - g, axis=1)
        reachable = self.hand_speed * np.maximum(self.taus - 0.05, 0.0)
        cost = np.linalg.norm(pts - sim.ready_grasp, axis=1) + 5.0 * np.maximum(dist - reachable, 0.0)
        cost[~ok] = np.inf
        i = int(np.argmin(cost))
        return self.taus[i], pts[i], vel[i]
