"""Render the supplied saved trajectory; does not run a policy or simulator."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "artifacts/examples/trajectory_1100060.csv").open() as stream:
        data = np.array([[float(v) for v in row.values()]
                         for row in csv.DictReader(stream)])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    lines = []
    for ax, (a, b), label in zip(axes, [(1, 3), (2, 3)],
                                ["Side view (X-Z)", "Front view (Y-Z)"]):
        ax.set_title(label)
        ax.set_xlabel("X (m)" if a == 1 else "Y (m)")
        ax.set_ylabel("Z (m)")
        ax.grid(alpha=.2)
        ax.plot(data[:, a], data[:, b], color="#d17b26", alpha=.25, lw=1)
        ax.plot(data[:, a+3], data[:, b+3], color="#197d93", alpha=.25, lw=1)
        ball, = ax.plot([], [], "o", color="#d17b26", ms=8, label="ball (scoring truth)")
        palm, = ax.plot([], [], "s", color="#197d93", ms=7, label="actual palm")
        ax.legend(fontsize=8, loc="best")
        ax.margins(.12)
        lines.append((ball, palm, a, b))
    title = fig.suptitle("Saved simulation", fontsize=11)

    def update(i):
        for ball, palm, a, b in lines:
            ball.set_data([data[i, a]], [data[i, b]])
            palm.set_data([data[i, a+3]], [data[i, b+3]])
        title.set_text(f"Saved simulation | seed 1100060 | 18 m/s | t={data[i, 0]*1000:.0f} ms")

    frames = np.unique(np.linspace(0, len(data)-1, 26).astype(int))
    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    animation.save(ROOT / "docs/figures/handoff_saved_trajectory.gif",
                   writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
