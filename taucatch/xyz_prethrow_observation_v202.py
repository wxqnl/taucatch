"""Fixed six-ms finder slots, all measured work charged and overruns retained."""
import time
from dataclasses import asdict
import numpy as np

from .multiscale_graph_observation_v16 import GraphObservation
from .multiscale_v1 import tracker_forecast
from .multiscale_observation_v5 import forecast_audit
from .xyz_prethrow_v198 import bill_slot


class SlottedObservation(GraphObservation):
    slot = .006

    def step(self,t):
        tick=round(t/.001)
        self.tracker.processing_allowed=(t>=0 and tick%6==0 and t+1e-9>=self.queue.busy_until)
        self.tracker.rig.elapsed_ms=0.
        start=time.perf_counter();self.tracker.step(self.sim,t)
        compute=max(0.,time.perf_counter()-start-self.tracker.rig.elapsed_ms/1000)
        start=time.perf_counter();self.rgb.step(self.sim,t)
        render_ms=1000*(time.perf_counter()-start)
        self.timings.append(dict(t=t,tracker_compute_ms=1000*compute,
            rgb_simulation_ms=render_ms,processed=self.tracker.processed,
            busy_skip=not self.tracker.processing_allowed,segments=dict(self.tracker.last_segments)))
        if t<0:self.accumulated_ms=0.;return
        self.accumulated_ms+=1000*compute
        if self.tracker.processed:
            start=time.perf_counter();forecast=tracker_forecast(self.tracker,t)
            self.accumulated_ms+=1000*(time.perf_counter()-start)
            charged,overrun=bill_slot(self.accumulated_ms/1000,self.slot)
            ready=self.queue.submit(forecast,t,charged)
            self.estimates.append(dict(t=t,available_t=ready,actual_compute_ms=self.accumulated_ms,
                compute_ms=1000*charged,padding_ms=1000*charged-self.accumulated_ms,overrun=overrun,
                forecast=None if forecast is None else forecast_audit(asdict(forecast))))
            self.accumulated_ms=0.
