"""Count event sensor reference generation as simulation time, including reset."""
import time
from .multiscale_observation_v1 import TimedRig
from .multiscale_observation_v5 import AuditedLatestObservation


class ReferenceTimedRig(TimedRig):
    def reset(self,t):
        start=time.perf_counter()
        try:return self.rig.reset(t)
        finally:self.elapsed_ms+=1000*(time.perf_counter()-start)


class ReferenceTimedObservation(AuditedLatestObservation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tracker.rig=ReferenceTimedRig(self.tracker.rig.rig)
