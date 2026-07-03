#!/usr/bin/env python3
"""Render heterophily graph illustration (right panel)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_graph(ax, title, seed, edge_configs, subtitle=""):
    rng = np.random.default_rng(seed)
    n = 14
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    idx = rng.permutation(n)
    half = n // 2

    for edge_i, edge_j, style, label, lw, alpha in edge_configs:
        xi, yi = xs[edge_i], ys[edge_i]
        xj, yj = xs[edge_j], ys[edge_j]
        dx, dy = xj - xi, yj - yi
        mag = np.hypot(dx, dy)
        if mag > 0:
            dx /= mag
            dy /= mag
        shrink = 0.13
        ax.annotate("", xy=(xj - dx * shrink, yj - dy * shrink),
                    xytext=(xi + dx * shrink, yi + dy * shrink),
                    arrowprops=dict(arrowstyle="->", color="#8b949e",
                                   lw=lw, alpha=alpha, linestyle=style))
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        ax.text(mx + 0.08, my + 0.08, label, fontsize=8, color="#c9d1d9",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="#21262d", ec="#30363d", alpha=0.9))

    for i in range(n):
        cidx = idx[i]
        color = "#58a6ff" if cidx < half else "#f778ba"
        ax.scatter(xs[i], ys[i], s=260, c=color, edgecolors="#0d1117", linewidth=1.5, zorder=5)
        label = "H" if cidx < half else "B"
        ax.text(xs[i], ys[i], label, fontsize=9, color="#0d1117",
                ha="center", va="center", fontweight="bold", zorder=6)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color="#e6edf3")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    hp = mpatches.Patch(color="#58a6ff", label="Human")
    bp = mpatches.Patch(color="#f778ba", label="Bot")
    ax.legend(handles=[hp, bp], loc="lower right", fontsize=8,
              frameon=True, fancybox=True, framealpha=0.9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    if subtitle:
        ax.text(0, -1.38, subtitle, ha="center", fontsize=9,
                style="italic", color="#8b949e")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
})

fig, ax = plt.subplots(figsize=(5.2, 5.2))
fig.patch.set_facecolor("#0d1117")

draw_graph(ax, "Heterophily (real-world)", seed=123, edge_configs=[
    (0, 7, "--", "B↔H", 1.4, 0.7),
    (1, 8, "--", "B↔H", 1.4, 0.7),
    (2, 9, "--", "B↔H", 1.4, 0.7),
    (3, 10, "--", "B↔H", 1.4, 0.7),
    (0, 6, "--", "B↔H", 1.4, 0.7),
    (4, 11, "--", "B↔H", 1.4, 0.7),
    (5, 12, "--", "B↔H", 1.4, 0.7),
], subtitle="Bots hide among humans — GNNs get confused")

fig.tight_layout()
fig.savefig("figures/heterophily_graph.svg", facecolor="#0d1117")
print("Saved figures/heterophily_graph.svg")
