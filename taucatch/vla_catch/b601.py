"""Seeed reBot Arm B601-DM in the catching scene.

The arm comes from Seeed's URDF (github.com/Seeed-Projects/reBotArm_control_py, urdf/DM), fetched into
vendor/rebot_b601 by tools/fetch_b601.sh; the repository carries no license, so its meshes stay out of
this one. The stock parallel gripper opens 57 mm, less than the 70 mm ball, so the B601 catches with a
cup held at the gripper (110 mm opening, 80 mm deep). The pedestal puts the cup's ready point where the
OpenArm hand waits, so tosses, cameras and the policy's task-space actions are shared by both arms.

Motors follow the DM variant: DM4340 (27 Nm) on joints 1-3, DM4310 (7 Nm) on joints 4-6, with the
OpenArm v2 stiffness of the same motors and more damping for the lighter links.
"""
from pathlib import Path

import mujoco
import numpy as np

from .sim import TEXTURED, EVENT_CAMERAS, EVENT_FOVY, ROOT, CatchSim, add_selective, align_rotation, look_at

URDF = ROOT / "vendor/rebot_b601/DM/urdf/ReBot_Arm_DM.urdf"
BASE = (0.0, -0.219, 0.55)                     # pedestal top
CUP_RADIUS, CUP_DEPTH, CUP_WALL = 0.055, 0.08, 0.004
CUP_MOUTH = CUP_RADIUS                         # m: mouth radius; larger makes the cup a funnel
CATCHER = "cup"                                # "cup", or "hand": palm, four fingers and a thumb held half open
HAND_HALF = (0.05, 0.045)                      # m: palm half width across the fingers, half height from fingers to thumb
FINGER = 0.045                                 # m: each of the two finger links
FINGER_YS = (-0.0345, -0.0115, 0.0115, 0.0345)
FINGER_HALF, THUMB_HALF = 0.0105, 0.016        # m: half widths
HAND_MASS = 0.54                               # kg, palm and fingers
HAND_OPEN = 0.26                               # rad: fingers bent back from straight while waiting
HAND_CLOSED = 1.0                              # rad: each finger joint when shut
HAND_RATE = np.radians(570.0)                  # rad/s per joint
HAND_GAINS = (5.0, 0.05, 1.0)                  # kp, kv, Nm
CUP_START = 0.005                              # cup bottom, ahead of the fingertips along end_link x
GAINS = {1: (190, 6, 27), 2: (190, 6, 27), 3: (190, 6, 27), 4: (30, 2, 7), 5: (30, 2, 7), 6: (30, 2, 7)}   # kp, kv, Nm
ARMATURE = {1: 0.03, 2: 0.03, 3: 0.03, 4: 0.002, 5: 0.002, 6: 0.002}


def mujoco_urdf():
    """The URDF with a MuJoCo compiler tag and mesh paths relative to the meshes folder."""
    out = URDF.with_name("ReBot_Arm_DM_mj.urdf")
    if not out.exists():
        text = URDF.read_text(encoding="utf-8-sig")
        tag = ('<mujoco><compiler meshdir="../meshes" strippath="false" balanceinertia="true" '
               'discardvisual="false" fusestatic="false"/></mujoco>')
        text = text.replace('<robot name="ReBot_Arm_DM">', f'<robot name="ReBot_Arm_DM">\n  {tag}', 1)
        out.write_text(text.replace('filename="../meshes/', 'filename="'))
    return out


def scene_spec(gravcomp=True):
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.add_texture(type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.55, 0.58], rgb2=[0.2, 0.2, 0.2], width=512, height=3072)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.38, 0.38, 0.4], rgb2=[0.3, 0.3, 0.32], markrgb=[0.6, 0.6, 0.6], width=300, height=300)
    mat = spec.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    world = spec.worldbody
    world.add_light(pos=[0, 0, 2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="groundplane")
    world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[BASE[0], BASE[1], BASE[2] / 2],
                   size=[0.09, 0.09, BASE[2] / 2], rgba=[0.25, 0.25, 0.27, 1])
    ball = world.add_body(name="ball", pos=[1.05, -0.25, 0.95])
    ball.add_freejoint()
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.035, 0, 0],
                  rgba=[0.95, 0.55, 0.1, 1], mass=0.05, friction=[1, 0.05, 0.01])
    for name, (pos, target) in EVENT_CAMERAS.items():
        cam = world.add_camera(name=name, pos=pos, fovy=EVENT_FOVY)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = look_at(pos, target)

    arm = mujoco.MjSpec.from_file(str(mujoco_urdf()))
    for j in arm.joints:
        i = int(j.name[-1]) if j.name.startswith("joint") else None
        if i:
            j.damping = [1.0 if i <= 3 else 0.9, 0, 0]
            j.armature, j.frictionloss = ARMATURE[i], (0.2 if i <= 3 else 0.04)
    end = arm.body("end_link")
    for body in ("finger_left_link", "finger_right_link"):     # fingers stay open and hold the cup
        for g in arm.body(body).geoms:
            g.contype = g.conaffinity = 0
    if CATCHER == "hand":
        add_hand(end)
    cup = end.add_body(name="cup", pos=[CUP_START, 0, 0])
    if CATCHER == "cup":
        add_cup(cup)
    # Site convention of CatchSim: -z along the cup axis (end_link +x), y along the finger opening.
    end.add_site(name="b601_ee_control_point", quat=[0.7071068, 0, -0.7071068, 0])
    frame = world.add_frame(pos=list(BASE))
    frame.attach_body(arm.body("base_link"), "b601_", "")
    for i, (kp, kv, force) in GAINS.items():
        spec.add_actuator(name=f"b601_joint{i}_ctrl", target=f"b601_joint{i}", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -kv] + [0] * 7,
                          forcelimited=True, forcerange=[-force, force])
    for f, q in (("finger_left", 0.0285), ("finger_right", -0.0285)):
        spec.add_actuator(name=f"b601_{f}_ctrl", target=f"b601_{f}", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[200] + [0] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -200, -2] + [0] * 7,
                          ctrlrange=sorted([0.0, q]), ctrllimited=True)
    if CATCHER == "hand":
        kp, kv, force = HAND_GAINS
        for j in hand_joints():
            spec.add_actuator(name=f"{j}_ctrl", target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                              biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -kv] + [0] * 7,
                              forcelimited=True, forcerange=[-force, force])
    links = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "end_link"]
    for a, b in zip(links, links[1:]):        # the URDF collision meshes of neighbouring links overlap
        spec.add_exclude(bodyname1=f"b601_{a}", bodyname2=f"b601_{b}")
    if gravcomp:
        for body in spec.bodies:
            if body.name.startswith("b601_") or body.name == "cup":
                body.gravcomp = 1.0
    return spec


def add_cup(cup):
    cup.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS + CUP_WALL, CUP_WALL / 2, 0],
                 quat=[0.7071068, 0, 0.7071068, 0], pos=[CUP_WALL / 2, 0, 0], mass=0.03, rgba=[0.15, 0.55, 0.85, 1])
    n = 16
    half = (CUP_MOUTH + CUP_WALL) * np.tan(np.pi / n)
    flare, mid = np.arctan2(CUP_MOUTH - CUP_RADIUS, CUP_DEPTH), (CUP_RADIUS + CUP_MOUTH + CUP_WALL) / 2
    for k in range(n):
        a = 2 * np.pi * k / n
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, [np.cos(a / 2), np.sin(a / 2), 0, 0], [np.cos(flare / 2), 0, 0, np.sin(flare / 2)])
        cup.add_geom(name=f"cup_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[np.hypot(CUP_DEPTH, CUP_MOUTH - CUP_RADIUS) / 2, CUP_WALL / 2, half],
                     pos=[CUP_DEPTH / 2, mid * np.cos(a), mid * np.sin(a)],
                     quat=quat, mass=0.005, rgba=[0.15, 0.55, 0.85, 1])


def hand_joints():
    return [f"b601_hand_{k}{link}" for k in range(5) for link in "ab"]


def add_hand(end):
    """Palm on the end link, facing the thrower; four fingers on one edge, a thumb on the other, bent back HAND_OPEN."""
    palm = end.add_body(name="hand_palm", pos=[CUP_START, 0, 0])
    width, height = HAND_HALF
    part = HAND_MASS / 22
    palm.add_geom(name="hand_palm_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUP_WALL / 2, width, height], pos=[CUP_WALL / 2, 0, 0],
                  mass=HAND_MASS - 10 * part, rgba=[0.8, 0.8, 0.82, 1])
    for k, (side, y, half) in enumerate([(1, y, FINGER_HALF) for y in FINGER_YS] + [(-1, 0.0, THUMB_HALF)]):
        bend = -side * HAND_OPEN / 2
        link = palm.add_body(name=f"hand_{k}a", pos=[CUP_WALL, y, side * height], quat=[np.cos(bend), 0, np.sin(bend), 0])
        for name in "ab":
            link.add_joint(name=f"hand_{k}{name}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, side, 0], range=[-0.1, 1.6],
                           armature=1e-4, damping=0.01)
            link.add_geom(name=f"hand_{k}{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[FINGER / 2, half, 0.006],
                          pos=[FINGER / 2, 0, 0], mass=part, rgba=[0.8, 0.8, 0.82, 1])
            if name == "a":
                link = link.add_body(name=f"hand_{k}b", pos=[FINGER, 0, 0])


class B601Sim(CatchSim):
    hand_act = None                # hand actuators, when the arm carries the hand
    hand_goal = hand_now = 0.0     # rad: where the finger joints are heading and where they are

    def caught(self):
        """Ball in the cup or the closed hand, moving with it."""
        p, v = self.ball_state()
        axis = -self.site_rot()[:, 2]
        off = p - self.grasp_point()
        along = off @ axis
        across = np.linalg.norm(off - along * axis)
        slow = np.linalg.norm(v - self.grasp_velocity()) < 0.5
        if CATCHER == "hand":
            return bool(slow and -0.02 < along < 0.08 and across < 0.06 and p[2] > 0.2)
        return bool(along < 0.045 and across < CUP_RADIUS - 0.03 and slow and p[2] > 0.2)

    def set_bounce(self, damping):
        """Ball on the catcher as a spring and damper of this damping ratio; 1 keeps no rebound at all."""
        ids = [i for i in range(self.m.ngeom) if (self.m.geom(i).name or "").startswith(("b601_cup_", "b601_hand_"))]
        self.m.geom_solref[ids + [self.m.geom("ball_geom").id]] = (0.004, damping)

    def set_finger(self, close):
        self.hand_goal = HAND_CLOSED if close else 0.0

    def reset_arm(self):
        self.hand_now = self.hand_goal = 0.0
        super().reset_arm()

    def step_physics(self):
        if CATCHER == "hand":
            if self.hand_act is None:
                self.hand_act = [self.m.actuator(f"{j}_ctrl").id for j in hand_joints()]
            step = HAND_RATE * self.m.opt.timestep
            self.hand_now += np.clip(self.hand_goal - self.hand_now, -step, step)
            self.d.ctrl[self.hand_act] = self.hand_now
        super().step_physics()

    SITE = "b601_ee_control_point"
    JOINTS = [f"b601_joint{i}" for i in range(1, 7)]
    ACTUATORS = [f"b601_joint{i}_ctrl" for i in range(1, 7)]
    FINGER_ACT = FINGER_JOINT = None
    GRASP_OFFSET = np.array([0.0, 0.0, -(CUP_START + CUP_WALL + 0.035)])   # ball resting on the cup bottom
    SHOULDER, REACH = np.array([BASE[0], BASE[1], BASE[2] + 0.14]), 0.62
    JOINT_SPEED = (8, 8, 8, 12, 12, 12)
    MOTOR_LAYOUT = ("4340", "4340", "4340", "4310", "4310", "4310")
    IK_DAMPING = 1e-2             # six joints and no redundancy: keep the IK well damped
    EXPERT_BLEND = 0.15           # the cup keeps facing the thrower
    READY_KEY = "catch_ready"
    READY_POINT = np.array([0.413, -0.219, 0.957])        # the OpenArm ready grasp point
    READY_APPROACH = np.array([0.855, 0.0, 0.519])       # cup axis toward the thrower, as the OpenArm fingers

    def load(self, gravcomp):
        spec = scene_spec(gravcomp)
        if self.distractors or TEXTURED[0]:
            add_selective(spec, self.distractors)
        m = spec.compile()
        q = self._solve_ready(m)
        key = spec.add_key(name=self.READY_KEY)
        key.qpos = q
        ctrl = np.zeros(m.nu)
        ctrl[:6] = q[:6]
        ctrl[6:8] = [0.0285, -0.0285]
        key.ctrl = ctrl
        return spec.compile()

    def _solve_ready(self, m):
        """Joint pose with the cup centre at READY_POINT and its axis along READY_APPROACH."""
        d = mujoco.MjData(m)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.SITE)
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.JOINTS]
        qadr = [m.jnt_qposadr[j] for j in joints]
        vadr = [m.jnt_dofadr[j] for j in joints]
        lo, hi = m.jnt_range[joints].T
        ball = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")]]
        d.qpos[ball:ball + 7] = [1.05, -0.25, 0.95, 1, 0, 0, 0]
        d.qpos[qadr] = [0.0, -1.2, -1.4, 0.0, 0.0, 0.0]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(2000):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[site].reshape(3, 3)
            point = d.site_xpos[site] + R @ self.GRASP_OFFSET
            target = align_rotation(R, self.READY_APPROACH)
            err_r = 0.5 * sum(np.cross(R[:, i], target[:, i]) for i in range(3))
            err = np.concatenate([self.READY_POINT - point, err_r])
            mujoco.mj_jac(m, d, jacp, jacr, point, m.site_bodyid[site])
            J = np.vstack([jacp[:, vadr], jacr[:, vadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq, lo, hi)
            if np.linalg.norm(err) < 1e-5:
                break
        return d.qpos.copy()
