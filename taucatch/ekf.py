"""Physics filter baseline (plan §4, baseline 5): an EKF on the finder detections.

State: ball position, velocity and spin (9). Flight model: gravity, drag and spin lift with
the ball's coefficients (optionally scaled, as on a real robot where they are only known
roughly). Measurements: the finder's pixel positions in both cameras. The filter starts at
the first stereo detection after release (it is told the release time, a privilege).
"""
import numpy as np

from .ball import GRAVITY, aero_accel


class BallEKF:
    def __init__(self, ball, geoms, coef_scale=1.0, sigma_px=0.5, sigma_acc=None, sigma_spin=20.0):
        self.ball_name = ball.name
        self.params = ball.params.copy()
        self.params[2] *= coef_scale           # drag coefficient
        self.lift_scale = coef_scale
        self.geoms = geoms
        self.r_px = sigma_px ** 2
        self.sigma_acc = sigma_acc if sigma_acc is not None else (8.0 if ball.name == "pingpong" else 2.0)
        self.sigma_spin = sigma_spin           # rad/s per sqrt(s), random walk of the spin
        self.x = None

    def _accel(self, v, w):
        return aero_accel(v, w * self.lift_scale, self.params) + np.array([0.0, 0.0, -GRAVITY])

    def _f(self, x, dt):
        p, v, w = x[:3], x[3:6], x[6:9]
        a = self._accel(v, w)
        return np.concatenate([p + v * dt + 0.5 * a * dt * dt, v + a * dt, w])

    def _F(self, x, dt):
        F = np.eye(9)
        F[:3, 3:6] = dt * np.eye(3)
        eps = 1e-4
        a0 = self._accel(x[3:6], x[6:9])
        for j in range(6):                        # d accel / d (v, w), numerically
            xp = x.copy()
            xp[3 + j] += eps
            da = (self._accel(xp[3:6], xp[6:9]) - a0) / eps
            F[3:6, 3 + j] += dt * da
            F[0:3, 3 + j] += 0.5 * dt * dt * da
        return F

    def start(self, p, t):
        self.x = np.concatenate([p, [-12.0, 0.0, 0.0], np.zeros(3)])
        self.P = np.diag([0.05 ** 2] * 3 + [8.0 ** 2] * 3 + [150.0 ** 2] * 3)
        self.t = t

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = self._F(self.x, dt)
        self.x = self._f(self.x, dt)
        Q = np.zeros((9, 9))
        qa = self.sigma_acc ** 2 * dt
        Q[3:6, 3:6] = qa * np.eye(3)
        Q[0:3, 0:3] = qa * dt * dt / 3 * np.eye(3)
        Q[0:3, 3:6] = Q[3:6, 0:3] = qa * dt / 2 * np.eye(3)
        Q[6:9, 6:9] = self.sigma_spin ** 2 * dt * np.eye(3)
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, det, score_min=0.5):
        """det (2, 3): [u, v, score] per camera."""
        for c, g in enumerate(self.geoms):
            if det[c, 2] <= score_min:
                continue
            pc = (self.x[:3] - g.pos) @ g.rot
            z = -pc[2]
            if z < 0.1:
                continue
            h = np.array([g.cx + g.f * pc[0] / z, g.cy - g.f * pc[1] / z])
            # d(u, v) / d(camera coords) times d(camera coords) / d(world position)
            J = np.array([[g.f / z, 0.0, g.f * pc[0] / z ** 2], [0.0, -g.f / z, -g.f * pc[1] / z ** 2]]) @ g.rot.T
            H = np.zeros((2, 9))
            H[:, :3] = J
            y = det[c, :2] - h
            S = H @ self.P @ H.T + self.r_px * np.eye(2)
            if y @ np.linalg.solve(S, y) > 50.0:        # gate outliers (false detections)
                continue
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(9) - K @ H) @ self.P

    def innovation_cost(self, det, score_min=0.5):
        """Predictive negative log likelihood before incorporating the current pixels.

        Include covariance volume: a more uncertain model should not win just because its
        normalized residual is smaller. No ball label or simulator state enters this score.
        """
        cost, seen = 0.0, 0
        for c, g in enumerate(self.geoms):
            if det[c, 2] <= score_min:
                continue
            pc = (self.x[:3] - g.pos) @ g.rot
            z = -pc[2]
            if z <= 0.1:
                continue
            expected = np.array([g.cx + g.f * pc[0] / z, g.cy - g.f * pc[1] / z])
            J = np.array([[g.f / z, 0, g.f * pc[0] / z ** 2],
                          [0, -g.f / z, -g.f * pc[1] / z ** 2]]) @ g.rot.T
            residual = det[c, :2] - expected
            cov = J @ self.P[:3, :3] @ J.T + self.r_px * np.eye(2)
            sign, logdet = np.linalg.slogdet(cov)
            if sign > 0:
                cost += 0.5 * (residual @ np.linalg.solve(cov, residual) + logdet)
                seen += 1
        return cost if seen else None


class BallEKFBank:
    """Parallel ball physics hypotheses, selected using recent pixel innovations.

    The chosen model provides the same state/covariance interface as BallEKF. Both filters
    keep receiving measurements, so changing the choice never discards a filter's history.
    """

    def __init__(self, balls, geoms, coef_scale=1.0):
        # Compare physics under the same noise prior. Type-specific process noise would
        # reward the narrower tennis covariance even on ping-pong measurements at 1 kHz.
        self.filters = [BallEKF(b, geoms, coef_scale, sigma_acc=2.0) for b in balls]
        if not self.filters:
            raise ValueError("At least one ball hypothesis is required")
        self.costs = np.zeros(len(self.filters))
        self.selected = 0

    def __getattr__(self, name):
        if name in {"x", "P", "params", "lift_scale", "ball_name"}:
            return getattr(self.filters[self.selected], name)
        raise AttributeError(name)

    def start(self, p, t):
        self.costs[:] = 0
        self.selected = 0
        for f in self.filters:
            f.start(p, t)

    def predict(self, t):
        for f in self.filters:
            f.predict(t)

    def update(self, det, score_min=0.5):
        costs = [f.innovation_cost(det, score_min) for f in self.filters]
        if all(c is not None for c in costs):
            self.costs = 0.98 * self.costs + np.asarray(costs)
            self.costs -= self.costs.min()
            self.selected = int(np.argmin(self.costs))
        for f in self.filters:
            f.update(det, score_min)
