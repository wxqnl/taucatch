"""CPU event peaks for the single-ball fast loop, without per-ms CNN inference."""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from .finder import CELL, WINDOW_US, event_grid
from .volley_tracker import VolleyTracker


def sparse_peaks(grid, k=4):
    """Local event-mass centroids; only measured event counts/offsets are read."""
    mass = np.expm1(grid[0])+np.expm1(grid[3])
    density = gaussian_filter(mass, .7)
    maxima = (density == maximum_filter(density, size=5)) & (density > .7)
    flat = np.flatnonzero(maxima)
    if not len(flat):
        return np.empty((0, 3))
    ix = flat[np.argsort(density.ravel()[flat])[-k:][::-1]]
    result = []
    for index in ix:
        cy, cx = np.unravel_index(index, density.shape)
        y0, y1 = max(0, cy-2), min(mass.shape[0], cy+3)
        x0, x1 = max(0, cx-2), min(mass.shape[1], cx+3)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        weights = mass[y0:y1, x0:x1]
        n = weights.sum()
        if n < 4:
            continue
        on, off = np.expm1(grid[0, y0:y1, x0:x1]), np.expm1(grid[3, y0:y1, x0:x1])
        dx = (on*grid[1, y0:y1, x0:x1]+off*grid[4, y0:y1, x0:x1])*1.5
        dy = (on*grid[2, y0:y1, x0:x1]+off*grid[5, y0:y1, x0:x1])*1.5
        u = ((xx*CELL+1.5)*weights+dx).sum()/n
        v = ((yy*CELL+1.5)*weights+dy).sum()/n
        result.append((u, v, float(min(.999, .5+n/100))))
    return np.asarray(result, float).reshape(-1, 3)


class SparseEventTracker(VolleyTracker):
    def __init__(self, *args, forecast_period=.004, **kwargs):
        super().__init__(*args, **kwargs)
        self.finder = None
        self.forecast_period = forecast_period

    def reset(self, t_rec=-.01):
        super().reset(t_rec)
        self.next_forecast = 0.

    def _spawn(self, peaks, t):
        old = {tr.id for tr in self.tracks}
        super()._spawn(peaks, t)
        # Task prior: new incoming tracks originate in the launch workspace, not on the hand.
        self.tracks = [tr for tr in self.tracks if tr.id in old or tr.ekf.x[0] > 1.2]

    def step(self, sim, t):
        if not self.on:
            if t >= self.t_rec-1e-9:
                self.rig.reset(t); self.on = True
            return
        evs = self.rig.step(t); grids = []
        for c, e in enumerate(evs):
            self.hist[c].append((e['x'],e['y'],np.round(e['t']*1e6).astype(np.int32),e['p']))
            x,y,ts,p = (np.concatenate(a) for a in zip(*self.hist[c]))
            grids.append(event_grid(x,y,ts,p,round(t*1e6)))
        peaks = [sparse_peaks(g) for g in grids]
        for tr in self.tracks:
            tr.ekf.predict(t)
        dets = self._associate(peaks,t);self._spawn(peaks,t)
        self.tracks = [tr for tr in self.tracks if t-tr.last_seen <= self.max_miss]
        for tr in self.tracks:
            if tr.id in dets:
                tr.ekf.update(dets[tr.id],self.score_min)
        if self.first_det_t is None and t>=0 and self.tracks:
            self.first_det_t = t
        if t+1e-9 >= self.next_forecast:
            self._predict(dets,t);self.next_forecast = t+self.forecast_period
        self.log.append(dict(t=t,peaks=peaks,events=evs,bound=None,
            tracks={tr.id:tr.ekf.x[:3].copy() for tr in self.tracks},
            candidates={tr.id:dict(hits=tr.hits,speed=float(np.linalg.norm(tr.ekf.x[3:6])),
                path=tr.outs[-1]['path']) for tr in self.tracks if tr.outs}))
