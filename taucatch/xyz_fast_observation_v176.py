"""Fresh DVS/state without paying for an unused new RGB crop in the fast path."""
import numpy as np

from .action_chunk import robot_state
from .finder import event_grid


def fast_sample(observation, t):
    grids = []
    for history in observation.tracker.hist:
        if history:
            x, y, ts, polarity = (np.concatenate(a) for a in zip(*history))
            grid = event_grid(x, y, ts, polarity, round(t*1e6))
        else:
            grid = np.zeros((6, 120, 160), np.float32)
        grids.append(grid.reshape(6, 60, 2, 80, 2).mean((2, 4)))
    return dict(events=np.stack(grids).astype(np.float16), state=robot_state(observation.sim),
                observed_t=float(t))
