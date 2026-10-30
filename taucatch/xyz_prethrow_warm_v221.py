"""Fixed six-ms finder slots, all measured work charged and overruns retained."""
import time
from dataclasses import asdict
import numpy as np

from .xyz_prethrow_strict_v214 import StrictSlottedObservation as GraphObservation
from .multiscale_v1 import tracker_forecast
from .multiscale_observation_v5 import forecast_audit
from .xyz_prethrow_v198 import bill_slot


class WarmSlottedObservation(GraphObservation):
    slot = .008

    def prepare_frontend(self):
        before=(len(self.tracker.tracks),tuple(len(x)for x in self.tracker.hist),self.tracker.on)
        report=warm_cpu_frontend()
        after=(len(self.tracker.tracks),tuple(len(x)for x in self.tracker.hist),self.tracker.on)
        assert before==after
        report['tracker_state_unchanged']=True
        return report

    def step(self,t):
        tick=round(t/.001)
        self.tracker.processing_allowed=(t>=0 and tick%8==0 and t+1e-9>=self.queue.busy_until)
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


def warm_cpu_frontend():
    from .finder import event_grid,decode_peaks
    import torch
    start=time.perf_counter_ns()
    x=np.arange(128,dtype=np.uint16)%640
    y=np.arange(128,dtype=np.uint16)%480
    ts=np.arange(128,dtype=np.int32)
    polarity=np.ones(128,dtype=np.int8)
    outputs=[]
    for _ in range(3):
        outputs.append(event_grid(x,y,ts,polarity,6000))
        decode_peaks(torch.zeros((2,1,60,80)),torch.zeros((2,2,60,80)))
    assert all(np.array_equal(outputs[0],x)for x in outputs[1:])
    return dict(started_ns=start,completed_ns=time.perf_counter_ns(),
        elapsed_ms=(time.perf_counter_ns()-start)/1e6,kernels=['event_grid_numba','decode_peaks_CPU'],
        synthetic_inputs=True,future_flight_inputs=False,model_actions=False,warm_grid_repeats_exact=True)
