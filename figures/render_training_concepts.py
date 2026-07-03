#!/usr/bin/env python3
"""Render focal loss curve and prototype calibration concept."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
    "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
fig.patch.set_facecolor("#0d1117")

# ── Left: Focal Loss curve ─────────────────────────────────────────────
pt = np.linspace(0.01, 0.99, 200)
for gamma, color, ls in [(0, "#8b949e", ":"), (1, "#58a6ff", "-."), (2, "#f778ba", "-"), (5, "#f0883e", "--")]:
    loss = -(1 - pt) ** gamma * np.log(pt)
    ax1.plot(pt, loss, color=color, lw=1.8, ls=ls, label=f"γ = {gamma:.0f}")

ax1.set_xlabel("Predicted probability for true class  $p_t$")
ax1.set_ylabel("Loss contribution")
ax1.set_title("Focal Loss  (γ = 2 used in AdaRelBot)", fontsize=12, fontweight="bold", color="#e6edf3", pad=10)
ax1.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax1.set_ylim(0, 4.5)
ax1.grid(True, alpha=0.2, color="#30363d")

# Highlight γ=2 region
ax1.axvspan(0.6, 0.99, alpha=0.08, color="#f778ba")
ax1.annotate("Easy examples\nheavily down-weighted", xy=(0.8, 0.15), fontsize=7,
             color="#8b949e", ha="center",
             arrowprops=dict(arrowstyle="->", color="#8b949e", lw=0.8))

# ── Right: Dual-head blending illustration ──────────────────────────────
x = np.linspace(-3, 3, 80)
y = np.linspace(-3, 3, 80)
X, Y = np.meshgrid(x, y)
Z_mlp = np.tanh(X + 0.3 * Y) * 0.8
Z_proto = -np.tanh(-X * 0.7 + Y * 0.5) * 0.7

# Prototypes
proto_bot = np.array([1.5, 1.0])
proto_human = np.array([-1.5, -1.0])

# Show some sample nodes
rng = np.random.default_rng(42)
n_samples = 30
samples_bot = rng.normal(loc=[1.2, 0.8], scale=0.4, size=(15, 2))
samples_human = rng.normal(loc=[-1.2, -0.8], scale=0.4, size=(15, 2))

ax2.scatter(samples_human[:, 0], samples_human[:, 1], c="#58a6ff", s=50, alpha=0.8,
            edgecolors="#0d1117", linewidth=0.8, label="Human", zorder=4)
ax2.scatter(samples_bot[:, 0], samples_bot[:, 1], c="#f778ba", s=50, alpha=0.8,
            edgecolors="#0d1117", linewidth=0.8, label="Bot", zorder=4)

# Plot prototypes
ax2.scatter(*proto_human, c="#58a6ff", s=180, marker="D", edgecolors="white", linewidth=1.5,
            zorder=5)
ax2.scatter(*proto_bot, c="#f778ba", s=180, marker="D", edgecolors="white", linewidth=1.5,
            zorder=5)
ax2.annotate("Human\nprototype", proto_human, xytext=(-2.6, -2.0),
             fontsize=8, color="#58a6ff", ha="center",
             arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.0))
ax2.annotate("Bot\nprototype", proto_bot, xytext=(2.5, 2.0),
             fontsize=8, color="#f778ba", ha="center",
             arrowprops=dict(arrowstyle="->", color="#f778ba", lw=1.0))

# Decision boundary (simplified)
ax2.axline((-1.5, -1.0), (0.5, 1.0), color="#8b949e", lw=1.2, ls="--", alpha=0.6)
ax2.text(1.2, -1.2, "Decision\nboundary", fontsize=7, color="#8b949e", ha="center", alpha=0.7)

ax2.set_xlabel("Embedding dim 1")
ax2.set_ylabel("Embedding dim 2")
ax2.set_title("Prototype-Based Classification", fontsize=12, fontweight="bold", color="#e6edf3", pad=10)
ax2.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
           loc="lower left")
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.15, color="#30363d")

fig.tight_layout()
fig.savefig("figures/training_concepts.svg", facecolor="#0d1117")
print("Saved figures/training_concepts.svg")
