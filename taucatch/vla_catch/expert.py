"""True-state teacher: ballistic intercept, approach aligned with the incoming ball, timed closure."""
import numpy as np

from .sim import BALL_RADIUS, GRAVITY, align_rotation, ballistic


class EstimateSim:
    """The simulator as the expert sees it, with the ball state from a tracker."""

    def __init__(self, sim):
        self.sim, self.estimate = sim, None

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def ball_state(self):
        return self.estimate


class Expert:
    def __init__(self, sim, lead=None, blend=None, hand_speed=1.0, freeze=0.0):
        self.sim, self.reach = sim, sim.REACH
        self.lead = sim.EXPERT_LEAD if lead is None else lead
        self.blend = sim.EXPERT_BLEND if blend is None else blend
        self.hand_speed, self.freeze = hand_speed, freeze
        self.ready_approach = -sim.ready_rot[:, 2]
        self.taus = np.arange(0.0, 1.2, 0.005)
        self.closed = False
        self.time = 0.0

    def reset(self):
        self.closed = False
        self.locked = None          # catch time, fixed once it is less than `freeze` away

    def intercept(self, p, v):
        """Future time and point that need the lowest average hand speed inside the reach."""
        sim = self.sim
        pts = ballistic(p, v, self.taus)
        g = sim.grasp_point()
        ok = (pts[:, 2] > 0.45) & (np.linalg.norm(pts - sim.SHOULDER, axis=1) < self.reach) & (pts[:, 0] > 0.12)
        if not ok.any():
            return None
        dist = np.linalg.norm(pts - g, axis=1)
        reachable = self.hand_speed * np.maximum(self.taus - 0.05, 0.0)
        cost = np.linalg.norm(pts - sim.ready_grasp, axis=1) + 5.0 * np.maximum(dist - reachable, 0.0)
        cost[~ok] = np.inf
        i = int(np.argmin(cost))
        return self.taus[i], pts[i], v + GRAVITY * self.taus[i]

    def act(self, released, t=0.0):
        """Grasp-centre target, rotation target and finger command at time t."""
        sim = self.sim
        self.time = t
        if not released:
            return sim.ready_grasp.copy(), sim.ready_rot.copy(), False
        p, v = sim.ball_state()
        if self.closed:
            return p.copy(), sim.site_rot().copy(), True
        if self.locked is not None:
            tau = self.locked - self.time
            hit = (tau, ballistic(p, v, max(tau, 0.0)), v + GRAVITY * max(tau, 0.0))
        else:
            hit = self.intercept(p, v)
            if hit is not None and hit[0] < self.freeze:
                self.locked = self.time + hit[0]
        if hit is None:
            target, rot = sim.grasp_point(), sim.site_rot().copy()
        else:
            tau, target, vel = hit
            approach = (1 - self.blend) * self.ready_approach - self.blend * vel / np.linalg.norm(vel)
            rot = align_rotation(sim.ready_rot, approach)
        # Close `lead` seconds before the ball reaches the grasp centre, once the hand is there.
        if hit is not None and tau <= self.lead and np.linalg.norm(target - sim.grasp_point()) < 0.04:
            self.closed = True
        return target, rot, self.closed
