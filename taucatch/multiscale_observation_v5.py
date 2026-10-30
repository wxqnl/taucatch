"""Latest-event observation with a JSON-safe scoring-independent forecast audit."""
import numpy as np
from .multiscale_observation_v4 import LatestObservation


def forecast_audit(value):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in value.items()}


class AuditedLatestObservation(LatestObservation):
    def step(self,t):
        super().step(t)
        if self.tracker.processed and self.estimates[-1]['forecast'] is not None:
            self.estimates[-1]['forecast']=forecast_audit(self.estimates[-1]['forecast'])
