"""Figures for taucatch/docs/leader_brief.md (numbers from docs/results.md, artifacts/evidence, archived research_plan.md)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK JP"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [f, "DejaVu Sans"]
        break
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bfc8d2", "axes.labelcolor": "#435063", "xtick.color": "#7a8595",
                     "ytick.color": "#7a8595", "axes.grid": True, "grid.color": "#e6eaef", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.15})
BLUE, ORANGE, AQUA, INK, MUTED, GOOD, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#111a24", "#7a8595", "#0ca30c", "#d03b3b"


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color="#435063", va="bottom")


# 1. events vs true-state ceiling
fig, ax = plt.subplots(figsize=(7.2, 3.6))
cats = ["乒乓球 12 m/s", "乒乓球 16 m/s\n+ 20 转/秒", "网球 15 m/s", "网球 20 m/s"]
ev, tr = [94, 86, 88, 70], [92, 86, 86, 75]
x = range(len(cats))
ax.bar(x, ev, width=0.38, color=BLUE, label="事件相机快环", zorder=2)
for i, (e, t) in enumerate(zip(ev, tr)):
    ax.plot([i - 0.32, i + 0.32], [t, t], color=INK, lw=2.2, solid_capstyle="round", zorder=3)
    ax.text(i + 0.36, e, f"{e}%", va="center", fontsize=10, color=INK)
ax.plot([], [], color=INK, lw=2.2, label="真值上限（真实球状态）")
ax.set_xticks(list(x), cats)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="lower left", fontsize=9.5, ncol=2, bbox_to_anchor=(0, 1.0))
ax.set_title("从事件流接球，已接近真实球状态的上限", loc="left", fontsize=13, fontweight="bold", color=INK, pad=30)
fig.savefig(OUT / "events_vs_truth.png")
plt.close(fig)

# 2. prediction error over time
fig, ax = plt.subplots(figsize=(7.2, 3.6))
t = [0, 10, 20, 50, 100, 150]
for name, ys, c in [("单一时间常数 10 ms", [30.4, 17.1, 16.2, 9.9, 4.3, 2.2], ORANGE),
                    ("LSTM", [29.1, 18.5, 14.8, 8.6, 4.2, 2.1], AQUA),
                    ("TauCatch（时间常数 1–300 ms 分布）", [20.2, 16.3, 15.1, 7.7, 3.1, 1.0], BLUE)]:
    ax.plot(t, ys, color=c, lw=2, marker="o", ms=5, mec="white", mew=1.2, label=name, zorder=3)
ax.annotate("出手时 20 cm\n（其他约 30 cm）", (0, 20.2), (14, 25), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("150 ms 时 1.0 cm", (150, 1.0), (118, 8), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlabel("出手后的时间 (ms)")
ax.set_ylabel("落点预测误差 (cm)")
ax.set_xticks([0, 50, 100, 150])
ax.set_ylim(0, 33)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
title(ax, "多时间常数神经元，比任何单一时间常数预测得都准", "乒乓球，中位误差，300 次留出投球")
fig.savefig(OUT / "tau_error.png")
plt.close(fig)

# 3. long reach
fig, ax = plt.subplots(figsize=(7.2, 3.8))
d = [25, 32, 40, 45]
ax.plot(d, [83, 92, 84, 27], color=AQUA, lw=2, marker="o", ms=5, mec="white", mew=1.2, label="快速臂 · 真实球状态")
ax.plot(d[1:], [55, 45, 19], color=BLUE, lw=2, marker="o", ms=5, mec="white", mew=1.2, label="快速臂 · 事件流")
ax.plot(d, [44, 22, 9, 0], color=ORANGE, lw=2, marker="o", ms=5, mec="white", mew=1.2, label="普通臂 · 真实球状态")
ax.annotate("", (40.25, 45.5), (40.25, 83.5), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
ax.text(39.7, 64, "差 39 个百分点", ha="right", va="center", fontsize=10, color=INK)
ax.set_xticks(d, [f"{v} cm" for v in d])
ax.set_xlabel("手需要移动的距离")
ax.set_ylim(0, 102)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.legend(frameon=False, fontsize=9.5, loc="lower left")
title(ax, "手要移动得远时，机械臂速度和预测质量都成为瓶颈", "乒乓球 16 m/s，20 转/秒，发球机，每点 64 次")
fig.savefig(OUT / "long_reach.png")
plt.close(fig)

# 4. one recorded catch: timeline
fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.set_xlim(-45, 215)
ax.set_ylim(-0.6, 3.3)
ax.grid(False)
for s in ("left",):
    ax.spines[s].set_visible(False)
ax.set_yticks([2.6, 1.6, 0.6], ["球", "快环", "SpikeVLA"])
ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=INK)
ax.axvline(0, color="#bfc8d2", lw=1)
ax.axvline(139, color=INK, lw=1.4)
ax.text(1.5, 3.15, "出手", fontsize=9.5, color=MUTED)
ax.text(141, 3.15, "接触 139 ms", fontsize=9.5, color=INK, fontweight="bold")
ax.barh(2.6, 139, left=0, height=0.22, color="#bfc8d2")
ax.text(69, 2.85, "飞行 139 ms，18 m/s", ha="center", fontsize=9.5, color="#435063")
ax.plot([141, 212], [2.6, 2.6], color=GOOD, lw=2)
ax.text(212, 2.3, "保持 ≥ 0.5 s →", ha="right", fontsize=9.5, color="#006300")
ax.barh(1.6, 139 - 44, left=44, height=0.22, color=BLUE)
for k in range(48, 139, 4):
    ax.plot([k, k], [1.52, 1.68], color="white", lw=0.8)
ax.text(42, 1.6, "机械臂起动 44 ms ", ha="right", va="center", fontsize=9.5, color="#435063")
ax.text(91, 1.88, "每 4 ms 重算接球点 · 手掌移动 14.5 cm", ha="center", fontsize=9.5, color="#435063")
ax.barh(0.6, 38, left=-42, height=0.22, color="#fde3d7", edgecolor=ORANGE, lw=0.8)
ax.text(-23, 0.88, "出手前编码", ha="center", fontsize=9.5, color="#435063")
for i, (o, a) in enumerate([(20, 43.4), (46, 64.7), (67, 88.9), (91, 113.0)]):
    y = 0.68 if i % 2 == 0 else 0.48
    ax.plot([o, a], [y, y], color=ORANGE, lw=2.5, solid_capstyle="round")
    ax.plot(o, y, "o", ms=6, mfc="white", mec=ORANGE, mew=1.6)
    ax.plot(a, y, "o", ms=7, color=ORANGE, mec="white", mew=1.2)
ax.text(-42, 0.05, "○ 观测   ● 修正可执行（约 20 ms 后），每块覆盖之后 192 ms", fontsize=9.5, color="#435063")
ax.set_xticks([0, 50, 100, 150, 200], ["0 ms", "50", "100", "150", "200"])
title(ax, "一次真实记录的接球（训练外投球 1100060，SpikeVLA 开启，接住）")
fig.savefig(OUT / "catch_timeline.png")
plt.close(fig)

# 5. latency
fig, ax = plt.subplots(figsize=(7.6, 3.2))
rows = [("选球，指令已缓存\n服务端计算，36 次", 6.4, 6.35, 6.55, True),
        ("融合动作，请求往返\n飞行中，48 次", 11.5, 10.3, 14.7, True),
        ("观测到修正可执行\n同上 48 次", 21.6, 17.8, 27.2, True)]
for i, (n, m, a, b, ok) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([a, b], [y, y], color=ORANGE, lw=2.5, solid_capstyle="round")
    ax.plot(m, y, "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=3)
    ax.text(b * 1.12, y, f"{m} ms" if ok else f"✕ {m} ms", va="center", fontsize=10,
            color=INK if ok else CRIT, fontweight="bold" if not ok else "normal")
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
ax.set_xscale("log")
ax.set_xlim(4, 300)
ax.set_xticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200 ms"])
ax.minorticks_off()
for v, lab in [(16, "16 ms 动作段"), (192, "192 ms 修正有效期")]:
    ax.axvline(v, color="#435063", lw=1.1)
    ax.text(v * 1.05, len(rows) - 0.45, lab, fontsize=9.5, color="#435063")
ax.set_ylim(-0.6, len(rows) - 0.2)
ax.grid(axis="y", visible=False)
title(ax, "SpikeVLA 时延：飞行中请求按时交付", "点为中位数，横线为范围；对数刻度")
fig.savefig(OUT / "latency.png")
plt.close(fig)

# 6. arm reach vs ball speed
fig, ax = plt.subplots(figsize=(7.4, 3.8))
v = [8, 10, 12, 15, 18, 20, 25]
fl = [399, 313, 259, 207, 172, 155, 124]
ax.axvspan(8, 12, color=BLUE, alpha=0.08, lw=0)
ax.text(10, 112, "首个真机目标", ha="center", fontsize=9.5, color="#435063")
for name, ys, c in [("仿真快速臂 / 高速工业臂（约 4 g）", [111, 76, 55, 34, 23, 18, 11], AQUA),
                    ("仿真普通臂（约 13 m/s²）", [42, 28, 19, 11, 8, 6, 4], ORANGE),
                    ("FR3（官方笛卡尔加速度上限 9 m/s²）", [32, 19, 13, 8, 5, 4, 2], BLUE)]:
    ax.plot(v, ys, color=c, lw=2, marker="o", ms=5, mec="white", mew=1.2, label=name)
ax.set_xticks(v, [f"{a} m/s\n{b} ms" for a, b in zip(v, fl)], fontsize=9)
ax.set_xlabel("网球速度 / 3 m 飞行时间")
ax.set_ylabel("手能移动的距离 (cm)")
ax.set_ylim(0, 120)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
title(ax, "一次飞行内，不同机械臂的手能移动多远（理想估算）",
      "从静止直线移动、对称加减速，可用时间 = 飞行时间 − 20 ms；用于比较档位")
fig.savefig(OUT / "arm_reach.png")
plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
