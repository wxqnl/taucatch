"""Batched event updates with a latest-observation schedule instead of a FIFO backlog."""
import numpy as np

from .finder import event_grid
from .multiscale_event_tracker_v2 import SparseEventTracker, sparse_peaks


class LatestEventTracker(SparseEventTracker):
    def reset(self, t_rec=-.01):
        super().reset(t_rec)
        self.next_process=0.;self.processing_allowed=True;self.processed=False

    def step(self, sim, t):
        self.processed=False
        if not self.on:
            if t>=self.t_rec-1e-9:
                self.rig.reset(t);self.on=True
            return
        evs=self.rig.step(t)
        for c,e in enumerate(evs):
            self.hist[c].append((e['x'],e['y'],np.round(e['t']*1e6).astype(np.int32),e['p']))
        if t<0 or t+1e-9<self.next_process or not self.processing_allowed:
            return
        grids=[]
        for history in self.hist:
            x,y,ts,p=(np.concatenate(a) for a in zip(*history))
            grids.append(event_grid(x,y,ts,p,round(t*1e6)))
        peaks=[sparse_peaks(g) for g in grids]
        for tr in self.tracks:tr.ekf.predict(t)
        dets=self._associate(peaks,t);self._spawn(peaks,t)
        self.tracks=[tr for tr in self.tracks if t-tr.last_seen<=self.max_miss]
        for tr in self.tracks:
            if tr.id in dets:tr.ekf.update(dets[tr.id],self.score_min)
        if self.first_det_t is None and self.tracks:self.first_det_t=t
        self._predict(dets,t)
        self.next_process=t+self.forecast_period;self.processed=True
        self.log.append(dict(t=t,peaks=peaks,events=evs,bound=None,
            tracks={tr.id:tr.ekf.x[:3].copy() for tr in self.tracks},
            candidates={tr.id:dict(hits=tr.hits,speed=float(np.linalg.norm(tr.ekf.x[3:6])),
                path=tr.outs[-1]['path']) for tr in self.tracks if tr.outs}))
