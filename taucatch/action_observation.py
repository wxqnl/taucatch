"""Causal RGB/DVS/proprioception for the action policy. No semantic target or oracle binding."""
import numpy as np
from .action_chunk import robot_state
from .choice import views
from .finder import event_grid
from .rgb import RGBCam
from .volley_tracker import VolleyTracker


class ActionObservation:
    def __init__(self,sim,finder='outputs/models/finder_v2.pt',seed=0):
        self.sim=sim
        self.tracker=VolleyTracker(sim,finder,'ekf',oracle=False)
        self.tracker.stop_after_contact=False   # Contact identity is scoring truth, never an observation gate.
        self.rgb=RGBCam(sim);self.rng=np.random.default_rng(seed)
        self.cache_time=None;self.cache=None
        # OSMesa's first left-camera render differs; discard it before sensor references.
        self.tracker.rig._render_background(0)

    def reset(self):
        self.tracker.rig.set_config(self.tracker.rig.cfg)
        self.tracker.reset();self.rgb.reset();self.cache_time=None;self.cache=None

    def step(self,t):
        self.tracker.step(self.sim,t);self.rgb.step(self.sim,t)

    def sample(self,t):
        frame=self.rgb.latest(t)
        if frame is None:return None
        if self.cache_time!=frame[0]:
            from scripts.eval_volley_vla import track_candidates
            candidates=track_candidates(self.tracker,self.rgb.geom,frame[0],self.rng)
            # Detection/tracking only supplies target-free RGB crops; no analytic forecast is drawn.
            for c in candidates:c['path_uv']=None
            self.cache=views(frame[1],candidates);self.cache_time=frame[0]
        grids=[]
        for history in self.tracker.hist:
            if history:
                x,y,ts,p=(np.concatenate(a) for a in zip(*history))
                grid=event_grid(x,y,ts,p,round(t*1e6))
            else:grid=np.zeros((6,120,160),np.float32)
            grids.append(grid.reshape(6,60,2,80,2).mean((2,4)))
        return dict(over=self.cache[0],mosaic=self.cache[1],events=np.stack(grids).astype(np.float16),
                    state=robot_state(self.sim),observed_t=float(t),rgb_t=float(frame[0]))

    def close(self):
        self.rgb.renderer.close();self.tracker.rig.con.free();self.tracker.rig.gl.free()
