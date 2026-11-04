"""OpenArm v1 in the catching scene.

The arm comes from Enactic's openarm_mujoco (v1/openarm_bimanual.xml, Apache-2.0, upstream commit in
third_party/openarm_mujoco/v1/UPSTREAM_COMMIT). Motors and gains are those of v2 (DM8009 / DM4340 /
DM4310); the gripper is a parallel slide pair whose inner faces open to 78 mm. The arm base sits where
the v2 base sits, and the ready pose puts the grasp centre and hand orientation where the v2 hand waits,
so tosses, cameras and the policy's task-space actions are shared with the other arms.
"""
import mujoco
import numpy as np

from .sim import TEXTURED, EVENT_CAMERAS, EVENT_FOVY, ROOT, CatchSim, add_selective, look_at

MODEL = ROOT / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"
BALL_RADIUS = 0.035


def scene_spec(gravcomp=True, ball_radius=BALL_RADIUS):
    spec = mujoco.MjSpec.from_file(str(MODEL))
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
    ball = world.add_body(name="ball", pos=[1.05, -0.25, 0.95])
    ball.add_freejoint()
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball_radius, 0, 0],
                  rgba=[0.95, 0.55, 0.1, 1], mass=0.05, friction=[1, 0.05, 0.01])
    for name, (pos, target) in EVENT_CAMERAS.items():
        cam = world.add_camera(name=name, pos=pos, fovy=EVENT_FOVY)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = look_at(pos, target)
    if gravcomp:           # the arm controller cancels gravity, as the real OpenArm stack does
        for body in spec.bodies:
            if body.name.startswith("openarm_right"):
                body.gravcomp = 1.0
    return spec


class OpenArmV1Sim(CatchSim):
    FINGER_OPEN, FINGER_CLOSED = 0.044, -0.005        # actuator range: pressed open, pressed shut
    GRASP_OFFSET = np.array([0.0, 0.0, -0.15])      # between the pads, ball near the palm
    EXPERT_LEAD = 0.0125          # the slide fingers close in a few ms; 40 ms early pinches the ball at the tips
    SHOULDER, REACH = np.array([0.0, -0.0925, 0.697]), 0.63
    READY_POINT = np.array([0.4128, -0.219, 0.9568])  # the v2 ready grasp point and hand orientation
    READY_ROT = np.array([[0.0, 0.519, -0.8548], [1.0, 0.0, 0.0], [0.0, -0.8548, -0.519]])
    BALL_RADIUS = BALL_RADIUS

    def load(self, gravcomp):
        spec = scene_spec(gravcomp, self.BALL_RADIUS)
        if self.distractors or TEXTURED[0]:
            add_selective(spec, self.distractors)
        m = spec.compile()
        q = self._solve_ready(m)
        key = spec.add_key(name=self.READY_KEY)
        key.qpos = q
        ctrl = np.zeros(m.nu)
        for i, name in enumerate(self.ACTUATORS):
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            ctrl[a] = q[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.JOINTS[i])]]
        ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.FINGER_ACT)] = self.FINGER_OPEN
        key.ctrl = ctrl
        return spec.compile()

    def _solve_ready(self, m):
        """Joint pose with the grasp centre at READY_POINT and the hand at READY_ROT, from several starts."""
        d = mujoco.MjData(m)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.SITE)
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.JOINTS]
        qadr = [m.jnt_qposadr[j] for j in joints]
        vadr = [m.jnt_dofadr[j] for j in joints]
        lo, hi = m.jnt_range[joints].T
        fingers = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_right_finger_joint{i}")]
                   for i in (1, 2)]
        ball = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")]]
        best, rng = None, np.random.default_rng(0)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for trial in range(40):
            d.qpos[:] = 0
            d.qpos[ball:ball + 7] = [1.05, -0.25, 0.95, 1, 0, 0, 0]
            d.qpos[fingers] = 0.04
            d.qpos[qadr] = lo + (hi - lo) * (0.5 if trial == 0 else rng.uniform(0.2, 0.8, len(qadr)))
            for _ in range(1500):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                R = d.site_xmat[site].reshape(3, 3)
                point = d.site_xpos[site] + R @ self.GRASP_OFFSET
                err_r = 0.5 * sum(np.cross(R[:, i], self.READY_ROT[:, i]) for i in range(3))
                err = np.concatenate([self.READY_POINT - point, err_r])
                mujoco.mj_jac(m, d, jacp, jacr, point, m.site_bodyid[site])
                J = np.vstack([jacp[:, vadr], jacr[:, vadr]])
                mid = (lo + hi) / 2
                null = np.eye(len(qadr)) - np.linalg.pinv(J) @ J      # stay away from the joint limits
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err) \
                    + 0.05 * null @ ((mid - d.qpos[qadr]) / (hi - lo))
                d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq, lo, hi)
            margin = np.min(np.minimum(d.qpos[qadr] - lo, hi - d.qpos[qadr]) / (hi - lo))
            if np.linalg.norm(err) < 1e-4 and (best is None or margin > best[0]):
                best = (margin, d.qpos.copy())
        if best is None:
            raise RuntimeError("no ready pose")
        return best[1]
