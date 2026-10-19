"""new_figures.py: figure for the slide "A network's loss is not one bowl".
Deterministic (no randomness): run `python new_figures.py` to regenerate loss_landscape.png.
Left : plain gradient descent on a 1-D non-convex loss from three starts -> three different valleys.
Right: plain gradient descent near a saddle point of J(u, v) = u^2 - v^2 + v^4/2:
       the loss plateaus while the escape direction grows slowly, then drops.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for p in font_manager.findSystemFonts():
    if "Lato" in p:
        font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = ["Lato", "DejaVu Sans"]

TEAL, NAVY, GREY, ORANGE, AXIS = "#0F7C82", "#17202E", "#8A94A3", "#C8571B", "#5B6675"

# ---------- left: 1-D non-convex loss ----------
def J(x):  return 0.06 * x**4 - 0.55 * x**2 + 0.5 * np.sin(3 * x) + 0.15 * x + 1.6
def dJ(x): return 0.24 * x**3 - 1.10 * x + 1.5 * np.cos(3 * x) + 0.15

def gd(x0, eta=0.05, steps=100):
    xs = [x0]
    for _ in range(steps):
        xs.append(xs[-1] - eta * dJ(xs[-1]))
    return np.array(xs)

starts = [(-3.3, TEAL, "start A"), (0.35, NAVY, "start B"), (3.4, AXIS, "start C")]

# ---------- right: saddle ----------
def Js(u, v): return u**2 - v**2 + 0.5 * v**4 + 0.5
u, v, eta, hist = 1.5, 1e-4, 0.05, []
for t in range(160):
    hist.append(Js(u, v))
    u, v = u - eta * 2 * u, v - eta * (-2 * v + 2 * v**3)
hist = np.array(hist)

fig, axs = plt.subplots(1, 2, figsize=(8.89, 2.75), dpi=300,
                        gridspec_kw={"width_ratios": [1.35, 1]})
ax = axs[0]
xx = np.linspace(-3.6, 3.7, 600)
ax.plot(xx, J(xx), color=GREY, lw=5, alpha=0.45, solid_capstyle="round")
for x0, c, lab in starts:
    xs = gd(x0)
    ax.plot(xs, J(xs), "o", color=c, ms=3.2, alpha=0.9)
    ax.plot(xs[-1], J(xs[-1]), "o", color=c, ms=8, mec="white", mew=1.5)
    ax.annotate(lab, (x0, J(x0)), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=10, color=c)
ax.set_title("Three starts, three different valleys", fontsize=12, color=NAVY)
ax.set_xlabel("one weight", fontsize=10, color=AXIS)
ax.set_ylabel("loss", fontsize=10, color=AXIS)

ax = axs[1]
ax.plot(np.arange(len(hist)), hist, color=TEAL, lw=2.5)
ax.annotate("flat region:\nprogress stalls", (45, hist[45]), xytext=(55, 1.25),
            fontsize=10, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
ax.set_title("Near a saddle: plateau, then drop", fontsize=12, color=NAVY)
ax.set_xlabel("step", fontsize=10, color=AXIS)
ax.set_ylabel("loss", fontsize=10, color=AXIS)

for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
fig.tight_layout(w_pad=2.5)
fig.savefig("loss_landscape.png", dpi=300, facecolor="white")
for x0, _, lab in starts:
    xs = gd(x0)
    print(f"{lab}: x0 = {x0:+.2f} -> x = {xs[-1]:+.3f}, loss = {J(xs[-1]):.3f}")
print(f"saddle run: loss {hist[0]:.2f} at step 0, {hist[60]:.3f} at step 60, {hist[-1]:.3f} at step {len(hist)-1}")
