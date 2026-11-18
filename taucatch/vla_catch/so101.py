"""SO-101 in the catching scene: a cup on the wrist of the LeRobot SO-101 follower arm.

The arm comes from TheRobotStudio's SO-ARM100 model (Simulation/SO101, Apache-2.0), fetched into vendor/so101 by
tools/fetch_so101.sh; the variant with the wrist camera mount is used by default (35 g more at the wrist). A pedestal
puts the cup's ready point where the OpenArm hand waits, so tosses, cameras and the policy's task-space actions are
shared with the other arms.

The arm has five joints and three of them pitch about parallel axes. A cup on the wrist-roll axis would keep its axis
in the vertical plane set by shoulder_pan (its yaw follows where the cup is), so the cup sits on the side of the
gripper, square to the roll axis: wrist_flex and wrist_roll then aim it in any direction, and the five joints set the
cup's position and axis (see MOUNTS). The IK aims the cup axis only: the cup is round, its roll does not matter.

Servo model (STS3215, the 7.4 V version, C001). The servo is a position controller with a P(D) loop inside; there is
no torque mode and no gravity compensation. Per joint:

    drive  = clamp(kp * (goal - q) - kv * dq, -stall, +stall)      PWM saturates at the stall torque
    joint  = drive - back_emf * dq - friction                      back EMF: no-load speed = stall / back_emf

stall and kp scale with the supply (PWM duty times volts); back_emf does not. stall and the no-load speed at 7.4 V are
the distributor figures (19.5 kg cm, 52 rpm); the armature (reflected rotor inertia) and friction are the upstream
model's identified values. kp, kv and SETPOINT_ACCEL (the servo's own acceleration ramp, register 41) are guesses
until tools/so101_step.py has measured the real arm: SERVO holds them and load_servo() reads a fitted set.
"""
import json
import os

import mujoco
import numpy as np

from .sim import TEXTURED, EVENT_CAMERAS, EVENT_FOVY, ROOT, CatchSim, add_selective, look_at

MODELS = {True: ROOT / "vendor/so101/so101_new_calib_camera.xml", False: ROOT / "vendor/so101/so101_new_calib.xml"}
WRIST_CAMERA = True                     # the arm carries the wrist camera and its mount
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
READY_POINT = np.array([0.413, -0.219, 0.957])          # the OpenArm ready grasp point
READY_APPROACH = np.array([0.855, 0.0, 0.519])          # cup axis toward the thrower, as the OpenArm fingers
BASE = None                             # pedestal top (x, y, z); None: the cup mount's own placement in MOUNTS
CUP_RADIUS, CUP_DEPTH, CUP_WALL = 0.055, 0.08, 0.004    # m: the B601 cup's 110 mm opening, 80 mm deep
CUP_MOUTH = CUP_RADIUS                  # m: mouth radius; larger makes the cup a funnel
CUP_MASS = 0.06                         # kg: printed cup and the clip on the jaws
CUP_SOLREF = (0.004, 1.0)               # ball on cup: stiff and critically damped, a cup lined so the ball does not
                                        # bounce. MuJoCo's default (0.02 s) lets a 3.5 m/s ball sink 2.7 cm into the
                                        # 4 mm bottom and onto the jaws behind it, which then throw it out or pin it.
CUP_MOUNT = os.environ.get("SO101_CUP_MOUNT", "side")
# Cup mounts on the jaws' tip: the cup axis tilted this far (deg) from the gripper axis (the wrist-roll axis; the jaws
# point along the gripper body's -z) toward the moving jaw's side (+x), and the pedestal (m behind and below the
# ready point) from tools/so101_feasibility.py --scan-base. The expert's catch rate, 200 tosses at 2-4 m/s:
#   axial (0 deg): wrist_roll only spins the round cup, so the cup axis stays in the shoulder_pan plane and its yaw
#     follows where the cup is. 85% at its best pedestal; the misses are balls the cup cannot face or reach.
#   side (90 deg): wrist_roll swings the axis round the wrist, so flex and roll aim it anywhere and position and axis
#     take all five joints. 99.5%, and >= 95% over +-4 cm of pedestal placement. The cup faces the moving jaw's side:
#     on the -x side the ready pose needs wrist_flex past its -95 deg limit (no pedestal placement works), so the real
#     cup bolts to the gripper body where the moving jaw was.
#   Tilts in between (30-60 deg) caught 77-86%.
MOUNTS = {"axial": (0.0, (0.34, 0.38)), "side": (90.0, (0.30, 0.16))}
JAW_TIP = np.array([0.0, 0.0, -0.104])  # the fixed jaw's tip on the gripper axis: the cup axis passes through it
JAW_BOX = ((-0.035, 0.030), (-0.105, 0.001))    # the fixed jaw's x and z extent (m): the cup bottom clears it
BALL_RADIUS = 0.035
RPM = 2 * np.pi / 60
SUPPLY = float(os.environ.get("SO101_SUPPLY", 5.0))     # V: the standard kit runs the 7.4 V servos from 5 V
SERVO = {
    "stall_74": 1.91,          # Nm at 7.4 V
    "no_load_74": 52 * RPM,    # rad/s at 7.4 V
    "sat_deg": 3.0,            # error (deg) at which the P loop reaches the stall torque: kp = stall / sat
    "zeta": 0.6,               # damping ratio of the servo loop at the ready pose's inertia: sets kv
    "armature": 0.028,         # kg m^2
    "friction": 0.052,         # Nm
    "setpoint_accel": None,    # rad/s^2: the servo ramps its goal at this rate (None: no ramp)
    "send_every": 0.01,        # s: joint goals reach the servos at 100 Hz (Feetech bus, sync write)
    "sag_comp": True,          # goals offset by gravity torque / kp, as the real controller will do
    "absorb": 0.15,            # s: on the catch signal the goals follow the measured pose (the arm gives with the
                               # ball instead of springing back and throwing it out), then hold where it ended
}


def base_position():
    if BASE is not None:
        return np.asarray(BASE, float)
    behind, below = MOUNTS[CUP_MOUNT][1]
    return READY_POINT - np.array([behind, 0.0, below])


def load_servo(path=None):
    """Servo parameters fitted by tools/so101_step.py fit (a JSON with SERVO's keys); SO101_SERVO names the file."""
    path = path or os.environ.get("SO101_SERVO")
    if path:
        SERVO.update(json.loads(open(path).read())["servo"])
    return SERVO


def servo_constants(volts=None):
    """stall (Nm), back EMF (Nm s/rad), kp (Nm/rad), no-load speed (rad/s) at this supply."""
    volts = SUPPLY if volts is None else volts
    stall = SERVO["stall_74"] * volts / 7.4
    back_emf = SERVO["stall_74"] / SERVO["no_load_74"]
    kp = stall / np.radians(SERVO["sat_deg"])
    return stall, back_emf, kp, stall / back_emf


def scene_spec(distractors=0):
    spec = mujoco.MjSpec.from_file(str(MODELS[WRIST_CAMERA]))
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
    base = base_position()
    world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[base[0], base[1], base[2] / 2],
                   size=[0.07, 0.07, base[2] / 2], rgba=[0.25, 0.25, 0.27, 1])
    ball = world.add_body(name="ball", pos=[1.05, -0.25, 0.95])
    ball.add_freejoint()
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0],
                  rgba=[0.95, 0.55, 0.1, 1], mass=0.05, friction=[1, 0.05, 0.01], solref=CUP_SOLREF)
    for name, (pos, target) in EVENT_CAMERAS.items():
        cam = world.add_camera(name=name, pos=pos, fovy=EVENT_FOVY)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = look_at(pos, target)
    spec.body("base").pos = list(base)
    # The jaws stay shut under the cup clip; the camera hangs off the wrist: none of them touch the ball.
    for body in ("moving_jaw_so101_v1", "wrist_camera_mount", "wrist_camera"):
        if WRIST_CAMERA or not body.startswith("wrist_camera"):
            for g in spec.body(body).geoms:
                g.contype = g.conaffinity = 0
    gripper = spec.body("gripper")
    pos, quat = add_cup(gripper)
    # Site convention of CatchSim: -z along the cup axis, out of the opening; here the site sits on the cup bottom.
    gripper.add_site(name="so101_ee_control_point", pos=pos, quat=quat)
    if distractors or TEXTURED[0]:
        add_selective(spec, distractors)
    return spec


def add_cup(gripper):
    """The cup of CUP_MOUNT on the gripper body. Returns the inner bottom centre and the site quaternion there."""
    tilt = np.radians(MOUNTS[CUP_MOUNT][0])
    opening = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    theta = np.pi / 2 - tilt                      # about y: the cup's +x onto the opening
    turn = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, turn)
    R = R.reshape(3, 3)
    origin = JAW_TIP
    corners = np.array([[x, 0.0, z] for x in JAW_BOX[0] for z in JAW_BOX[1]])
    start = max((corners - origin) @ opening) + 0.001

    def place(p, q):                              # cup frame (x along the depth) to the gripper frame
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, turn, np.asarray(q, float))
        return origin + R @ p, out

    n = 16
    bottom = 0.3 * CUP_MASS
    pos, quat = place(np.array([start + CUP_WALL / 2, 0, 0]), [0.7071068, 0, 0.7071068, 0])   # disc normal along x
    gripper.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[CUP_RADIUS + CUP_WALL, CUP_WALL / 2, 0],
                     pos=pos, quat=quat, mass=bottom, rgba=[0.15, 0.55, 0.85, 1], solref=CUP_SOLREF)
    half = (CUP_MOUTH + CUP_WALL) * np.tan(np.pi / n)
    flare, mid = np.arctan2(CUP_MOUTH - CUP_RADIUS, CUP_DEPTH), (CUP_RADIUS + CUP_MOUTH + CUP_WALL) / 2
    for k in range(n):
        a = 2 * np.pi * k / n
        local = np.zeros(4)
        mujoco.mju_mulQuat(local, [np.cos(a / 2), np.sin(a / 2), 0, 0], [np.cos(flare / 2), 0, 0, np.sin(flare / 2)])
        pos, quat = place(np.array([start + CUP_DEPTH / 2, mid * np.cos(a), mid * np.sin(a)]), local)
        gripper.add_geom(name=f"cup_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[np.hypot(CUP_DEPTH, CUP_MOUTH - CUP_RADIUS) / 2, CUP_WALL / 2, half],
                         pos=pos, quat=quat, mass=(CUP_MASS - bottom) / n, rgba=[0.15, 0.55, 0.85, 1],
                         solref=CUP_SOLREF)
    # The site: its -z along the cup's +x (the opening), i.e. the cup frame turned -90 deg about y.
    return place(np.array([start + CUP_WALL, 0, 0]), [0.7071068, 0, -0.7071068, 0])


class SO101Sim(CatchSim):
    SITE = "so101_ee_control_point"
    JOINTS = list(JOINTS)
    ACTUATORS = list(JOINTS)                       # the upstream position actuators, re-tuned by apply_servo
    FINGER_ACT = FINGER_JOINT = None
    GRASP_OFFSET = np.array([0.0, 0.0, -BALL_RADIUS])      # ball resting on the cup bottom (the site)
    SHOULDER, REACH = None, 0.42                   # shoulder: the shoulder_lift axis at the ready pose (set on load)
    JOINT_SPEED = (6.0,) * 5
    IK_DAMPING = 1e-2
    EXPERT_BLEND = 0.5                             # the cup turns halfway toward the incoming ball (0.15: 82%, 1: 99%)
    EXPERT_LEAD = 0.02                             # the catch signal (absorb) this long before the ball arrives
    READY_KEY = "catch_ready"
    READY_POINT, READY_APPROACH = READY_POINT, READY_APPROACH

    def __init__(self, gravcomp=False, grasp_offset=None, distractors=0):
        self.supply = SUPPLY
        super().__init__(False, grasp_offset, distractors)     # the servos cancel no gravity
        self.apply_servo()
        self.dg = mujoco.MjData(self.m)                        # scratch data for the gravity torques
        lift = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift")
        self.SHOULDER = self.d.xanchor[lift].copy()
        self.reset_arm()

    def load(self, gravcomp):
        spec = scene_spec(self.distractors)
        m = spec.compile()
        q = self._solve_ready(m)
        key = spec.add_key(name=self.READY_KEY)
        key.qpos = q
        ctrl = np.zeros(m.nu)
        for a in range(m.nu):
            ctrl[a] = q[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
        key.ctrl = ctrl
        return spec.compile()

    def apply_servo(self, volts=None):
        """The servo model of SERVO at this supply on the arm's actuators and joints."""
        self.supply = self.supply if volts is None else volts
        stall, back_emf, kp, no_load = servo_constants(self.supply)
        m = self.m
        inertia = self._ready_inertia()
        kv = np.maximum(2 * SERVO["zeta"] * np.sqrt(kp * inertia) - back_emf, 0.0)
        for i, a in enumerate(self.arm_act):
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, :3] = (0.0, -kp, -kv[i])
            m.actuator_forcerange[a] = (-stall, stall)
        m.dof_armature[self.vadr] = SERVO["armature"]
        m.dof_damping[self.vadr] = back_emf
        m.dof_frictionloss[self.vadr] = SERVO["friction"]
        self.kp, self.kv, self.stall, self.no_load_speed = kp, kv, stall, no_load
        self.speed = np.minimum(np.asarray(self.JOINT_SPEED, float), 0.95 * no_load)

    def limit_motors(self, volts):
        """The servos at this supply voltage (V)."""
        self.apply_servo(volts)

    def _ready_inertia(self):
        """Each arm joint's diagonal of the mass matrix at the ready pose, armature included."""
        m, d = self.m, mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(m, d, self.ready_key)
        m.dof_armature[self.vadr] = SERVO["armature"]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        if hasattr(d, "qM"):
            mujoco.mj_fullM(m, M, d.qM)
        else:                                  # MuJoCo >= 3.5: mj_fullM(m, d, dst)
            mujoco.mj_fullM(m, d, M)
        return M[self.vadr, self.vadr]

    def reset_arm(self):
        super().reset_arm()
        self.goal = self.q_ref.copy()          # the goal the servos last received
        self.setpoint = self.q_ref.copy()      # the servos' own ramped goal
        self.setpoint_v = np.zeros_like(self.q_ref)
        self.sent = -np.inf
        self.absorb_from = None                # time of the catch signal
        if hasattr(self, "dg"):
            self.goal = self.with_sag(self.q_ref)
            self.setpoint = self.goal.copy()
            self.d.ctrl[self.arm_act] = self.goal

    def with_sag(self, q):
        """Goal that holds q against gravity: q + gravity torque / kp (the P loop's droop taken off)."""
        if not SERVO["sag_comp"]:
            return q.copy()
        dg = self.dg
        dg.qpos[:] = self.d.qpos
        dg.qpos[self.qadr] = q
        dg.qvel[:] = 0
        mujoco.mj_forward(self.m, dg)
        return q + np.clip(dg.qfrc_bias[self.vadr] / self.kp, -0.2, 0.2)

    def track(self, target, rot_target, dt, gain=1.0):
        """One damped-least-squares IK step on the joint reference (position, and the cup axis only), rate limited
        per joint; the servos get the reference every send_every s."""
        m, d = self.m, self.d
        if self.absorb_from is not None and d.time - self.absorb_from < SERVO["absorb"]:
            self.q_ref = d.qpos[self.qadr].copy()          # give: the goal is wherever the arm is now
            if d.time - self.sent >= SERVO["send_every"] - 1e-9:
                self.goal = self.with_sag(self.q_ref)
                self.sent = d.time
            return
        saved = d.qpos[self.qadr].copy()
        d.qpos[self.qadr] = self.q_ref
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        point = self.grasp_point()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, point, self.ee_body)
        J = np.vstack([jacp[:, self.vadr], 0.3 * jacr[:, self.vadr]])
        err_r = np.cross(self.site_rot()[:, 2], rot_target[:, 2])
        err = np.concatenate([target - point, 0.3 * err_r]) * gain
        dq = J.T @ np.linalg.solve(J @ J.T + self.IK_DAMPING * np.eye(6), err)
        dq = np.clip(dq, -self.speed * dt, self.speed * dt)
        self.q_ref = np.clip(self.q_ref + dq, self.qlo, self.qhi)
        d.qpos[self.qadr] = saved
        mujoco.mj_forward(m, d)
        if d.time - self.sent >= SERVO["send_every"] - 1e-9:
            self.goal = self.with_sag(self.q_ref)
            self.sent = d.time

    def step_physics(self):
        a = SERVO["setpoint_accel"]
        if a is None:
            self.setpoint = self.goal
        else:                                  # the servo's goal ramp: accelerate toward the goal, stop on it
            dt = self.m.opt.timestep
            gap = self.goal - self.setpoint
            v_stop = np.sign(gap) * np.sqrt(2 * a * np.abs(gap))
            v_cap = np.clip(v_stop, -self.no_load_speed, self.no_load_speed)
            self.setpoint_v += np.clip(v_cap - self.setpoint_v, -a * dt, a * dt)
            self.setpoint = self.setpoint + self.setpoint_v * dt
        self.d.ctrl[self.arm_act] = self.setpoint
        super().step_physics()

    def set_finger(self, close):
        """A cup closes nothing: the catch signal starts the absorb phase instead."""
        if close and self.absorb_from is None and SERVO["absorb"]:
            self.absorb_from = self.d.time

    def set_bounce(self, damping):
        """Ball on the cup as a spring and damper of this damping ratio; 1 keeps no rebound at all."""
        ids = [i for i in range(self.m.ngeom) if (self.m.geom(i).name or "").startswith("cup_")]
        self.m.geom_solref[ids + [self.m.geom("ball_geom").id]] = (0.004, damping)

    def caught(self):
        """Ball inside the cup (centre below the mouth, within the wall at its depth) and moving with it. The cup is
        tilted, so a held ball leans on the low wall: up to CUP_RADIUS - BALL_RADIUS off the axis, more in a funnel."""
        p, v = self.ball_state()
        axis = -self.site_rot()[:, 2]
        off = p - self.grasp_point()
        along = off @ axis
        across = np.linalg.norm(off - along * axis)
        depth = np.clip((along + BALL_RADIUS) / CUP_DEPTH, 0.0, 1.0)
        wall = CUP_RADIUS + (CUP_MOUTH - CUP_RADIUS) * depth
        slow = np.linalg.norm(v - self.grasp_velocity()) < 0.5
        return bool(-0.01 < along < CUP_DEPTH - BALL_RADIUS and across < wall - BALL_RADIUS + 0.005
                    and slow and p[2] > 0.2)

    def _solve_ready(self, m):
        """Joint pose with the grasp centre at READY_POINT and the cup axis as near READY_APPROACH as the arm turns,
        from several starts, keeping away from the joint limits."""
        d = mujoco.MjData(m)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.SITE)
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.JOINTS]
        qadr = [m.jnt_qposadr[j] for j in joints]
        vadr = [m.jnt_dofadr[j] for j in joints]
        lo, hi = m.jnt_range[joints].T
        ball = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")]]
        inward = -self.READY_APPROACH / np.linalg.norm(self.READY_APPROACH)
        best, rng = None, np.random.default_rng(0)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for trial in range(40):
            d.qpos[:] = 0
            d.qpos[ball:ball + 7] = [1.05, -0.25, 0.95, 1, 0, 0, 0]
            d.qpos[qadr] = 0.0 if trial == 0 else lo + (hi - lo) * rng.uniform(0.1, 0.9, len(qadr))
            for _ in range(1500):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                R = d.site_xmat[site].reshape(3, 3)
                point = d.site_xpos[site] + R @ self.GRASP_OFFSET
                err = np.concatenate([self.READY_POINT - point, 0.3 * np.cross(R[:, 2], inward)])
                mujoco.mj_jac(m, d, jacp, jacr, point, m.site_bodyid[site])
                J = np.vstack([jacp[:, vadr], 0.3 * jacr[:, vadr]])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
                d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq, lo, hi)
            pos_err = np.linalg.norm(self.READY_POINT - point)
            margin = np.min(np.minimum(d.qpos[qadr] - lo, hi - d.qpos[qadr]) / (hi - lo))
            score = (pos_err < 1e-4, margin - np.linalg.norm(np.cross(R[:, 2], inward)))
            if best is None or score > best[0]:
                best = (score, d.qpos.copy(), pos_err)
        if best[2] > 1e-3:
            base = tuple(np.round(base_position(), 3))
            raise RuntimeError(f"no ready pose: the cup misses the ready point by {best[2] * 100:.1f} cm (base {base})")
        return best[1]
