"""Unchanged native sensor tensors plus timed, causal event-estimate delivery."""
import time

from .action_observation import ActionObservation
from .multiscale_v1 import DeliveryQueue, tracker_forecast


class TimedRig:
    def __init__(self, rig):
        self.rig = rig
        self.elapsed_ms = 0.

    def __getattr__(self, name):
        return getattr(self.rig, name)

    def step(self, t):
        start = time.perf_counter()
        result = self.rig.step(t)
        self.elapsed_ms += 1000*(time.perf_counter()-start)
        return result


class MultiscaleObservation(ActionObservation):
    def __init__(self, sim, seed, period=.004):
        super().__init__(sim, seed=seed)
        self.tracker.rig = TimedRig(self.tracker.rig)
        self.period = period
        self.queue = DeliveryQueue()
        self.accumulated_ms = 0.
        self.next_estimate = 0.
        self.timings = []
        self.estimates = []

    def step(self, t):
        self.tracker.rig.elapsed_ms = 0.
        start = time.perf_counter()
        self.tracker.step(self.sim, t)
        tracker_ms = 1000*(time.perf_counter()-start)
        render_ms = self.tracker.rig.elapsed_ms
        compute_ms = max(0., tracker_ms-render_ms)
        start = time.perf_counter()
        self.rgb.step(self.sim, t)
        rgb_ms = 1000*(time.perf_counter()-start)
        self.timings.append(dict(t=t, tracker_compute_ms=compute_ms,
                                 event_simulation_ms=render_ms, rgb_simulation_ms=rgb_ms))
        if t < 0:
            self.accumulated_ms = 0.
            return
        self.accumulated_ms += compute_ms
        if t+1e-9 >= self.next_estimate:
            start = time.perf_counter()
            forecast = tracker_forecast(self.tracker, t)
            self.accumulated_ms += 1000*(time.perf_counter()-start)
            ready = self.queue.submit(forecast, t, self.accumulated_ms/1000)
            self.estimates.append(dict(t=t, available_t=ready, compute_ms=self.accumulated_ms,
                                       valid=forecast is not None,
                                       track_id=None if forecast is None else forecast.track_id,
                                       measured_t=None if forecast is None else forecast.measured_t))
            self.accumulated_ms = 0.
            self.next_estimate = t+self.period

