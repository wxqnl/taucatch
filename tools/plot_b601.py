"""Figures for the B601 8 m/s catching docs (README, docs/spikevla_catch.md, docs/real_robot.md).

Numbers come from the judge JSONs under runs_v2/ (seeds 1000000-1000299, 8 m/s from 4 m) and, for the replay and the
start-delay checks, from the logged outputs of tools/replay_measurements.py and the start-delay check (outputs/checks/vla_catch/velff_check.py).

    python tools/plot_b601.py docs/figures/b601
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch, Wedge

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/figures/b601"
OUT.mkdir(parents=True, exist_ok=True)
for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK JP"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [f, "DejaVu Sans"]
        break
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bfc8d2", "axes.labelcolor": "#435063", "xtick.color": "#7a8595",
                     "ytick.color": "#435063", "axes.grid": True, "grid.color": "#e6eaef", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
BLUE, ORANGE, AQUA, INK, MUTED, SOFT = "#2a78d6", "#eb6834", "#1baf7a", "#111a24", "#7a8595", "#b9c3cf"
SUB = "#435063"


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=13, fontweight="bold", color=INK, pad=24 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=SUB, va="bottom")


def caught(*names):
    out = {}
    for n in names:
        for t in json.loads((ROOT / n).read_text())["tosses"]:
            out[t["seed"]] = bool(t["caught"])
    return out


def rate(d, seeds):
    return 100 * np.mean([d[s] for s in seeds])


R1 = "runs_v2/saim_r1_b601/ema_3000."
runs = {"q4": caught(R1 + "v8d4slice_q4.json", R1 + "v8d4slice_q4_s2.json"),
        "q20": caught(R1 + "v8d4slice.json", R1 + "v8d4slice_s2.json"),
        "plan": caught(R1 + "v8d4sliceplan_q4_300.json"),
        "base": caught("runs_v2/fast/base300_8_4.json")}
seeds = sorted(set.intersection(*(set(v) for v in runs.values())))
expert = 100 * np.mean([t["expert"] for t in json.loads((ROOT / "runs_v2/fast/base300_8_4.json").read_text())["tosses"]])

# 1. headline: the same 300 tosses
rows = [("真实球状态专家（测试时拿不到）", expert, SOFT),
        ("专家规划器 + SpikeVLA 的球轨迹", rate(runs["plan"], seeds), AQUA),
        ("SpikeVLA 直接出动作，每 4 ms 查询", rate(runs["q4"], seeds), BLUE),
        ("运控 base（自带 250 Hz 事件跟踪）", rate(runs["base"], seeds), ORANGE),
        ("SpikeVLA 直接出动作，每 20 ms 查询", rate(runs["q20"], seeds), "#9cc0ec")]
fig, ax = plt.subplots(figsize=(7.4, 3.3))
y = np.arange(len(rows))[::-1]
for yi, (name, v, c) in zip(y, rows):
    ax.barh(yi, v, height=0.56, color=c, zorder=2)
    ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=10.5, color=INK, fontweight="bold" if c == BLUE else None)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlim(50, 102)
ax.set_xlabel("接住率（%）")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
title(ax, "B601 五指手接 8 m/s 球：SpikeVLA 追平运控 base",
      f"同一批 {len(seeds)} 球，4 m 外平抛，事件相机是唯一传感器，策略延迟 20 ms")
fig.savefig(OUT / "headline.png")
plt.close(fig)

# 2. progress of the SpikeVLA head at 8 m/s from 4 m
steps = [("起点\n动作块定合手", 14, 100), ("合手看手的\n当前位置", 61, 100), ("每 4 ms 事件片\n测一次球", 76, 100),
         ("DAgger\n一轮", 81, 100), ("每 4 ms\n查询一次模型", rate(runs["q4"], seeds), len(seeds))]
fig, ax = plt.subplots(figsize=(7.4, 3.5))
x = np.arange(len(steps))
ax.bar(x, [s[1] for s in steps], width=0.56, color=BLUE, zorder=2)
for xi, (_, v, n) in zip(x, steps):
    ax.text(xi, v + 1.5, f"{v:.0f}%" if n == 100 else f"{v:.1f}%", ha="center", fontsize=10.5, color=INK)
for v, c, name in ((rate(runs["base"], seeds), ORANGE, "运控 base"), (rate(runs["plan"], seeds), AQUA, "规划器 + SpikeVLA 轨迹")):
    ax.axhline(v, color=c, lw=1.6, ls=(0, (5, 3)), zorder=1)
    ax.text(-0.42, v + 1.2, f"{name} {v:.1f}%", fontsize=9.5, color=SUB, va="bottom")
ax.set_xticks(x, [s[0] for s in steps], fontsize=9.5)
ax.set_ylim(0, 100)
ax.set_ylabel("接住率（%）")
ax.grid(axis="x", visible=False)
title(ax, "SpikeVLA 在 8 m/s 上的四步", "前四步各 100 球，最后一步 300 球；每步只改一处")
fig.savefig(OUT / "progress.png")
plt.close(fig)

# 3. time budget: the expert on the true state, its start held back (velff_check.py, 100 tosses)
delay, ok = [0, 85, 150, 200, 250], [100, 95, 72, 38, 23]
fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.plot(delay, ok, color=INK, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.4, zorder=3)
for d, v in zip(delay, ok):
    ax.text(d + 5, v + 3, f"{v}%", fontsize=10, color=INK)
ax.axvspan(0, 85, color=AQUA, alpha=0.12, lw=0)
ax.text(42, 12, "SpikeVLA 的球轨迹\n在 85 ms 内成立", ha="center", fontsize=9.5, color=SUB)
ax.set_xlim(-8, 265)
ax.set_ylim(0, 110)
ax.set_xlabel("专家起步时刻（出手后 ms）")
ax.set_ylabel("接住率（%）")
title(ax, "时间预算：晚 150 ms 起步就丢掉 1/4", "真实球状态专家，飞行约 530 ms，100 球")
fig.savefig(OUT / "time_budget.png")
plt.close(fig)

# 4. what limits the closed loop: real measurements replayed to the planner (replay_measurements.py, 100 tosses)
rows = [("真值，出手即用", 96, SOFT), ("真值，85 ms 起用", 92, SOFT),
        ("实测逐片测量（现状）", 90, BLUE), ("深度误差 × 0.4（侧向相机）", 92, BLUE), ("去掉系统偏差", 92, BLUE),
        ("448 px 传感器", 88, BLUE),
        ("估计每 20 ms 刷新，延迟 20 ms", 85, ORANGE), ("估计每 20 ms 刷新，延迟 40 ms", 82, ORANGE)]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
y = np.arange(len(rows))[::-1]
for yi, (name, v, c) in zip(y, rows):
    ax.barh(yi, v, height=0.56, color=c, zorder=2)
    ax.text(v + 0.5, yi, f"{v}%", va="center", fontsize=10, color=INK)
ax.set_yticks(y, [r[0] for r in rows], fontsize=10)
ax.set_xlim(70, 100)
ax.set_xlabel("专家规划器接住率（%）")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=SOFT, label="真值参照"), Patch(color=BLUE, label="换测量"), Patch(color=ORANGE, label="换刷新节奏")],
          loc="lower right", frameon=False, fontsize=9.5)
title(ax, "测量已经够准，丢分在刷新节奏", "把实测的每片测量原样回放给规划器，同一批 100 球")
fig.savefig(OUT / "bottleneck.png")
plt.close(fig)


# 5. rig layout in the robot base frame (sim constants: b601.BASE, sim.EVENT_CAMERAS with VLA_AIM_X=1.3)
base = np.array([0.0, -0.219, 0.55])
cams = {"左相机": np.array([-0.25, -1.25, 1.55]), "右相机": np.array([-0.25, 0.85, 1.55])}
aim = np.array([1.3, -0.2, 0.95])
ready = np.array([0.413, -0.219, 0.957])
launch = np.array([4.45, -0.03, 0.77])
fig, (top, side) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (top, side):
    ax.set_aspect("equal")
    ax.grid(True)
rel = lambda p: p - base
# top view (x forward, y left)
top.add_patch(FancyBboxPatch((-0.08, -0.08), 0.16, 0.16, boxstyle="round,pad=0.01", fc=INK, ec="none", zorder=3))
top.text(-0.05, 0.2, "B601 底座（原点）", ha="center", va="bottom", fontsize=9, color=INK)
for name, c in cams.items():
    p, d = rel(c), rel(aim) - rel(c)
    ang = np.degrees(np.arctan2(d[1], d[0]))
    top.add_patch(Wedge(p[:2], 5.2, ang - 31, ang + 31, fc=BLUE, alpha=0.06, ec="none"))
    top.plot([p[0], rel(aim)[0]], [p[1], rel(aim)[1]], color=BLUE, lw=0.8, ls=(0, (2, 2)), alpha=0.7)
    top.plot(*p[:2], marker="s", ms=9, color=BLUE, zorder=4)
    top.text(p[0] + 0.12, p[1] + (0.2 if p[1] > 0 else -0.2), f"{name} ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})",
             fontsize=9, color=INK, ha="left", va="center")
r, a, l = rel(ready), rel(aim), rel(launch)
top.plot(*r[:2], marker="o", ms=8, color=AQUA, zorder=4)
top.annotate(f"手的待命点\n({r[0]:.2f}, {r[1]:+.2f}, {r[2]:.2f})", r[:2], (r[0] - 0.35, r[1] - 0.62), fontsize=9, color=INK, va="top",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
top.plot(*a[:2], marker="+", ms=11, mew=2, color=BLUE, zorder=4)
top.annotate("两台相机的瞄准点\n视场 62°", a[:2], (a[0] + 0.35, a[1] - 0.62), fontsize=9, color=SUB, va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
top.add_patch(FancyBboxPatch((l[0] - 0.12, -0.53), 0.24, 1.0, boxstyle="round,pad=0.02", fc=ORANGE, alpha=0.15, ec="none"))
top.text(l[0], 0.62, "出手区\n离待命点约 4 m", ha="center", va="bottom", fontsize=9, color=INK)
top.annotate("", r[:2] + [0.08, 0], (l[0] - 0.14, l[1]), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8))
top.text(2.9, 0.22, "球 8 m/s，约 0.53 s", ha="center", fontsize=9.5, color=INK)
top.set_xlim(-0.7, 4.9)
top.set_ylim(-1.55, 1.55)
top.set_xlabel("x（m，朝抛球方向）")
top.set_ylabel("y（m，向左）")
title(top, "俯视", "机器人底座坐标系，括号内为 (x, y, z) m")
# side view (x forward, z up), ball arc of the example toss
t = np.linspace(0, 0.547, 60)
v0 = np.array([-7.37, -0.53, 3.04])
arc = launch + np.outer(t, v0) + 0.5 * np.outer(t ** 2, [0, 0, -9.81])
side.plot(arc[:, 0] - base[0], arc[:, 2] - base[2], color=ORANGE, lw=1.8, ls=(0, (4, 3)))
side.add_patch(FancyBboxPatch((-0.08, -0.55), 0.16, 0.55, boxstyle="round,pad=0.01", fc=INK, ec="none"))
side.text(0.12, -0.45, "0.55 m 立柱", fontsize=9, color=SUB)
side.plot(r[0], r[2], marker="o", ms=8, color=AQUA, zorder=4)
side.plot(rel(cams["左相机"])[0], rel(cams["左相机"])[2], marker="s", ms=9, color=BLUE)
side.text(rel(cams["左相机"])[0] + 0.1, rel(cams["左相机"])[2] + 0.08, "相机 z = 1.00", fontsize=9, color=INK)
side.text(l[0], l[2] - 0.12, "出手高 0.2–0.65 m", fontsize=9, color=INK, ha="center", va="top")
side.axhline(-0.55, color=MUTED, lw=1)
side.set_xlim(-0.7, 4.9)
side.set_ylim(-0.65, 1.4)
side.set_xlabel("x（m）")
side.set_ylabel("z（m，底座上表面为 0）")
title(side, "侧视", "一次 8 m/s 抛球，飞行 0.55 s")
fig.savefig(OUT / "rig_layout.png")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
