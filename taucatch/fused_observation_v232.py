"""Original strict eight-ms protocol with fused finder and saved executed peaks."""
import os
from pathlib import Path
import numpy as np

from .fused_event_tracker_v231 import FusedEventTracker
from .xyz_prethrow_warm_v221 import WarmSlottedObservation
from .xyz_prethrow_strict_v214 import StrictDeliveryQueue
from .multiscale_observation_v6 import ReferenceTimedRig
from .rgb import RGBCam
from .action_horizon_data import sha256


class FusedObservation(WarmSlottedObservation):
    def __init__(self,sim,seed,period=.004):
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        self.sim=sim
        self.tracker=FusedEventTracker(sim,'outputs/models/finder_v2.pt','ekf',oracle=False,forecast_period=period)
        self.tracker.stop_after_contact=False
        self.rgb=RGBCam(sim);self.rng=np.random.default_rng(seed)
        self.cache_time=None;self.cache=None
        self.tracker.rig._render_background(0)
        self.tracker.rig=ReferenceTimedRig(self.tracker.rig)
        self.period=period;self.queue=StrictDeliveryQueue();self.accumulated_ms=0.
        self.timings=[];self.estimates=[];self.numeric_root=None;self.seed=seed

    def close(self):
        if self.tracker.numeric_peaks:
            root=Path(self.numeric_root);root.mkdir(parents=True,exist_ok=True)
            path=root/f'{self.seed}_executed_peaks.npz';assert not path.exists()
            n=len(self.tracker.numeric_peaks)
            values=np.full((n,2,4,3),np.nan,dtype=np.float64)
            counts=np.zeros((n,2),dtype=np.int32)
            flags=np.zeros((n,2),dtype=bool)
            for i,(peaks,fallback) in enumerate(self.tracker.numeric_peaks):
                flags[i]=fallback
                for camera,rows in enumerate(peaks):
                    counts[i,camera]=len(rows);values[i,camera,:len(rows)]=rows
            np.savez_compressed(path,peaks=values,counts=counts,CPU_fallback=flags)
            self.tracker.peak_evidence=dict(path=str(path),sha256=sha256(path),batches=n)
        super().close()
