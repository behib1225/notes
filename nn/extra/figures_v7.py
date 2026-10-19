"""figures_v7.py -- figure regenerated for deck v7 (deterministic, no randomness).

loss_landscape.png: left, plain gradient descent on a 1-D toy loss with three valleys of nearly
equal depth (0.00, 0.02, 0.04) from three starts; right, gradient descent on J(u, v) = u^2 - v^2 + v^4/2
started next to the saddle at the origin (loss vs step).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in fm.findSystemFonts():
    if "Carlito" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Carlito", "font.size": 11})
TEAL, INK, MUTED, GREY, ORANGE, CURVE = "#0F7C82", "#17202E", "#5B6675", "#5B6675", "#C8571B", "#C9CED6"

# ---------- left: 1-D toy loss, valleys at w = -pi, 0, pi with nearly equal depth ----------
def f(w):
    tilt = 0.02 * (w + np.pi) / np.pi * (w > -np.pi)  # 0.00, 0.02, 0.04 at the three minima
    wall = 0.6 * np.maximum(0, np.abs(w) - 3.5) ** 2
    return 0.5 * (1 - np.cos(2 * w)) + tilt + wall

def df(w, h=1e-6):
    return (f(w + h) - f(w - h)) / (2 * h)

def gd(w0, eta, steps):
    ws = [w0]
    for _ in range(steps):
        ws.append(ws[-1] - eta * df(ws[-1]))
    return np.array(ws)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.89, 2.75), dpi=300,
                               gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.12})
w = np.linspace(-4.25, 4.25, 800)
ax1.plot(w, f(w), color=CURVE, lw=5, solid_capstyle="round", zorder=1)
runs = [(-4.15, TEAL, "start A"), (1.45, INK, "start B"), (4.15, GREY, "start C")]
for w0, col, lab in runs:
    path = gd(w0, 0.08, 40)
    ax1.scatter(path[1:-1], f(path[1:-1]), s=14, color=col, zorder=2)
    ax1.scatter([path[-1]], [f(path[-1])], s=55, color=col, edgecolor="white", lw=1.2, zorder=3)
    ax1.scatter([w0], [f(w0)], s=30, color=col, zorder=3)
    ax1.annotate(lab, (w0, f(w0)), xytext=(0, 9), textcoords="offset points", ha="center",
                 color=col, fontsize=11.5)
    print(f"{lab}: start {w0:+.2f} -> end w = {path[-1]:+.3f}, loss = {f(path[-1]):.3f}")
ax1.set_ylim(-0.08, 1.75)
ax1.set_title("Three starts, three valleys, similar loss", fontsize=13, color=INK)
ax1.set_xlabel("one weight", color=MUTED); ax1.set_ylabel("loss", color=MUTED)

# ---------- right: gradient descent next to a saddle ----------
def J(u, v): return u ** 2 - v ** 2 + v ** 4 / 2
u, v, eta, losses = 1.5, 1e-7, 0.05, []
for _ in range(260):
    losses.append(J(u, v))
    u, v = u - eta * 2 * u, v - eta * (-2 * v + 2 * v ** 3)
ax2.plot(losses, color=TEAL, lw=3)
k = 90
ax2.annotate("flat region:\nprogress stalls", (k, losses[k]), xytext=(k + 15, losses[k] + 1.0),
             color=ORANGE, fontsize=11.5, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
ax2.set_title("Near a saddle: plateau, then drop", fontsize=13, color=INK)
ax2.set_xlabel("step", color=MUTED); ax2.set_ylabel("loss", color=MUTED)

for ax in (ax1, ax2):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
fig.savefig("loss_landscape.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
print("final saddle-run loss", losses[-1])
