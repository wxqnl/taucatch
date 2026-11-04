"""Figure: the 20 ms event window against the trace frame (three time surfaces) on one throw."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from taucatch.vla_catch.data import load_tosses

toss = load_tosses(sorted(glob.glob("data_v2/bc_v1"))[0], lambda m: m["seed"] == 40)[0]
steps = [s for s in (3, 10, 18, 26) if s < len(toss.times)]
fig, axes = plt.subplots(2, len(steps), figsize=(2.6 * len(steps), 5.6), facecolor="black")
for col, s in enumerate(steps):
    t_ms = 1000 * (toss.times[s] - toss.meta["release"])
    for row, mode in enumerate(("window", "trace")):
        ax = axes[row, col]
        ax.imshow(toss.frames(s, mode=mode)[0])
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#444")
        if row == 0:
            ax.set_title(f"{t_ms:.0f} ms after release", color="white", fontsize=10)
axes[0, 0].set_ylabel("20 ms window\n(R+, G-, B newest)", color="white", fontsize=10)
axes[1, 0].set_ylabel("trace frame\n(τ 10 / 50 / 250 ms)", color="white", fontsize=10)
plt.tight_layout()
plt.savefig("docs/figures/spikevla_catch/trace_vs_window.png", dpi=130, facecolor="black")
