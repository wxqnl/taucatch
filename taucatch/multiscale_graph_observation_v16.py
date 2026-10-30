"""Native policy sensor contract with an independently audited GPU0 event frontend."""
import json,os
import numpy as np
from pathlib import Path
from .multiscale_graph_tracker_v16 import GraphEventTracker as NeuralEventTracker
from .multiscale_observation_v6 import ReferenceTimedObservation,ReferenceTimedRig
from .multiscale_v1 import DeliveryQueue
from .rgb import RGBCam
from .action_horizon_data import sha256


class GraphObservation(ReferenceTimedObservation):
    def __init__(self,sim,seed,period=.004):
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        self.sim=sim
        self.tracker=NeuralEventTracker(sim,'outputs/models/finder_v2.pt','ekf',oracle=False,forecast_period=period)
        self.tracker.stop_after_contact=False
        self.rgb=RGBCam(sim);self.rng=np.random.default_rng(seed)
        self.cache_time=None;self.cache=None
        self.tracker.rig._render_background(0)
        self.tracker.rig=ReferenceTimedRig(self.tracker.rig)
        self.period=period;self.queue=DeliveryQueue();self.accumulated_ms=0.
        self.timings=[];self.estimates=[];self.numeric_root=None;self.seed=seed

    def step(self,t):
        super().step(t)
        if self.timings:
            self.timings[-1]['segments']=dict(self.tracker.last_segments)

    def close(self):
        try:
            self.numeric_report=self.tracker.verify_raw()
            assert self.numeric_root is not None
            root=Path(self.numeric_root);root.mkdir(parents=True,exist_ok=True)
            path=root/f'{self.seed}.npz';assert not path.exists()
            np.savez_compressed(path,grids=np.stack([x[0] for x in self.tracker.numeric]),
                gpu_heat=np.stack([x[1] for x in self.tracker.numeric]),gpu_offset=np.stack([x[2] for x in self.tracker.numeric]))
            self.numeric_report['raw_inputs_outputs']=dict(path=str(path),sha256=sha256(path))
            (root/f'{self.seed}.json').write_text(json.dumps(self.numeric_report,indent=2)+'\n')
            assert self.numeric_report['accepted'],'Raw finder numeric gate failed; preserve evidence, no promotion'
        finally:super().close()
