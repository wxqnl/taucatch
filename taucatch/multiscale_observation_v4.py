"""Timed latest-event processing; scoring labels are inaccessible to the controller."""
import time
import numpy as np
from dataclasses import asdict

from .action_observation import ActionObservation
from .multiscale_event_tracker_v4 import LatestEventTracker
from .multiscale_observation_v1 import TimedRig
from .multiscale_v1 import DeliveryQueue,tracker_forecast
from .rgb import RGBCam


class LatestObservation(ActionObservation):
    def __init__(self,sim,seed,period=.004):
        self.sim=sim
        self.tracker=LatestEventTracker(sim,'outputs/models/finder_v2.pt','ekf',oracle=False,
                                       forecast_period=period)
        self.tracker.stop_after_contact=False
        self.rgb=RGBCam(sim);self.rng=np.random.default_rng(seed)
        self.cache_time=None;self.cache=None
        self.tracker.rig._render_background(0)
        self.tracker.rig=TimedRig(self.tracker.rig)
        self.period=period;self.queue=DeliveryQueue();self.accumulated_ms=0.
        self.timings=[];self.estimates=[]

    def step(self,t):
        self.tracker.processing_allowed=t+1e-9>=self.queue.busy_until
        self.tracker.rig.elapsed_ms=0.
        start=time.perf_counter();self.tracker.step(self.sim,t)
        tracker_ms=1000*(time.perf_counter()-start)
        render_ms=self.tracker.rig.elapsed_ms
        compute_ms=max(0.,tracker_ms-render_ms)
        start=time.perf_counter();self.rgb.step(self.sim,t)
        rgb_ms=1000*(time.perf_counter()-start)
        self.timings.append(dict(t=t,tracker_compute_ms=compute_ms,event_simulation_ms=render_ms,
            rgb_simulation_ms=rgb_ms,processed=self.tracker.processed,
            busy_skip=not self.tracker.processing_allowed))
        if t<0:self.accumulated_ms=0.;return
        self.accumulated_ms+=compute_ms
        if self.tracker.processed:
            start=time.perf_counter();forecast=tracker_forecast(self.tracker,t)
            self.accumulated_ms+=1000*(time.perf_counter()-start)
            ready=self.queue.submit(forecast,t,self.accumulated_ms/1000)
            self.estimates.append(dict(t=t,available_t=ready,compute_ms=self.accumulated_ms,
                valid=forecast is not None,track_id=None if forecast is None else forecast.track_id,
                measured_t=None if forecast is None else forecast.measured_t,
                forecast=None if forecast is None else asdict(forecast)))
            self.accumulated_ms=0.
