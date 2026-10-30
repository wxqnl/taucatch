"""Plot all six completed development pairs from the supplied original report."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "artifacts/evidence/generalization/completion.json").read_text())
    pairs = report["summary"]["pairs"]
    seeds = sorted(pairs, key=lambda s: (pairs[s]["speed_mps"], s))
    assert len(seeds) == 6 and report["physical_trials"] == 12
    values = np.array([[int(pairs[s][key]) for s in seeds]
                       for key in ["zero_held", "mean_held"]])
    fig, (ax, summary) = plt.subplots(1, 2, figsize=(11, 3.8),
                                      gridspec_kw={"width_ratios": [1.65, 1]},
                                      layout="constrained")
    ax.imshow(values, vmin=0, vmax=1, cmap=ListedColormap(["#f5d8d1", "#b4dfce"]), aspect="auto")
    for i in range(2):
        for j in range(6):
            ax.text(j, i, "Held" if values[i, j] else "Miss", ha="center", va="center",
                    color="#183743", fontsize=12)
    ax.set_yticks([0, 1], ["DVS fast only", "+ SpikeVLA"])
    ax.set_xticks(range(6), [f"{s}\n{pairs[s]['speed_mps']:g} m/s" for s in seeds], fontsize=9)
    ax.set_title("All completed independent development geometries")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    groups = [[i for i, s in enumerate(seeds) if pairs[s]["speed_mps"] == 14],
              [i for i, s in enumerate(seeds) if pairs[s]["speed_mps"] == 18], list(range(6))]
    x = np.arange(3)
    for row, offset, color, label in [(0, -.19, "#51929d", "DVS fast only"),
                                      (1, .19, "#8b79b3", "+ SpikeVLA")]:
        rates = [values[row, ids].sum() / len(ids) * 100 for ids in groups]
        bars = summary.bar(x+offset, rates, width=.36, color=color, label=label)
        for bar, ids in zip(bars, groups):
            summary.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                         f"{values[row, ids].sum()}/{len(ids)}", ha="center", fontsize=10)
    summary.set_xticks(x, ["14 m/s", "18 m/s", "All"])
    summary.set_ylim(0, 126)
    summary.set_yticks([0, 25, 50, 75, 100])
    summary.set_ylabel("Held rate (%)")
    summary.set_title("Paired outcomes; small sample")
    summary.legend(loc="upper right", fontsize=8)
    summary.spines[["top", "right"]].set_visible(False)
    fig.suptitle("TauCatch | fast only 4/6 vs. SpikeVLA 3/6 | no generalization gain established",
                 fontsize=13)
    fig.savefig(ROOT / "docs/figures/handoff_generalization.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
