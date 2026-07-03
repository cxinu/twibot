#!/usr/bin/env python3
"""Render homophily vs heterophily conceptual illustration."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_graph(ax, title, seed, node_colors, edge_configs, subtitle=""):
    """Draw a small graph with color-coded nodes and labeled edges."""
    rng = np.random.default_rng(seed)
    n = 14

    # Fixed circular layout radius
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    # Shuffle which nodes are which color
    idx = rng.permutation(n)
    half = n // 2

    for i, (edge_i, edge_j, style, label, lw, alpha) in enumerate(edge_configs):
        xi, yi = xs[edge_i], ys[edge_i]
        xj, yj = xs[edge_j], ys[edge_j]
        dx = xj - xi
        dy = yj - yi
        mag = np.hypot(dx, dy)
        if mag > 0:
            dx /= mag
            dy /= mag
        shrink = 0.13
        xi_s = xi + dx * shrink
        yi_s = yi + dy * shrink
        xj_e = xj - dx * shrink
        yj_e = yj - dy * shrink

        ax.annotate("", xy=(xj_e, yj_e), xytext=(xi_s, yi_s),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                   lw=lw, alpha=alpha, linestyle=style))

        # Midpoint label
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        offset = 0.08
        ax.text(mx + offset, my + offset, label, fontsize=7, color="#444444",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    for i in range(n):
        cidx = idx[i]
        color = "#2E86AB" if cidx < half else "#A23B72"
        ax.scatter(xs[i], ys[i], s=220, c=color, edgecolors="white", linewidth=1.2,
                   zorder=5)
        label = "H" if cidx < half else "B"
        ax.text(xs[i], ys[i], label, fontsize=8, color="white",
                ha="center", va="center", fontweight="bold", zorder=6)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    hp = mpatches.Patch(color="#2E86AB", label="Human")
    bp = mpatches.Patch(color="#A23B72", label="Bot")
    ax.legend(handles=[hp, bp], loc="lower right", fontsize=8,
              frameon=True, fancybox=True, framealpha=0.85, edgecolor="#cccccc")

    if subtitle:
        ax.text(0, -1.32, subtitle, ha="center", fontsize=9,
                style="italic", color="#555555")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.8))

# ── Homophily graph ────────────────────────────────────────────────────
# Humans connect to humans; bots connect to bots
draw_graph(
    ax1, "Homophily (ideal)",
    seed=42,
    node_colors=None,
    edge_configs=[
        # Same-type edges
        (0, 1, "-", "H↔H", 1.5, 0.7),
        (1, 2, "-", "H↔H", 1.5, 0.7),
        (6, 7, "-", "B↔B", 1.5, 0.7),
        (7, 8, "-", "B↔B", 1.5, 0.7),
        (3, 4, "-", "H↔H", 1.5, 0.7),
        (9, 10, "-", "B↔B", 1.5, 0.7),
    ],
    subtitle="Birds of a feather — GNNs thrive here",
)

# ── Heterophily graph ──────────────────────────────────────────────────
# Bots connect to humans (as they do in real social networks)
draw_graph(
    ax2, "Heterophily (real-world)",
    seed=123,
    node_colors=None,
    edge_configs=[
        # Cross-type edges: bots following humans
        (0, 7, "--", "B↔H", 1.2, 0.6),
        (1, 8, "--", "B↔H", 1.2, 0.6),
        (2, 9, "--", "B↔H", 1.2, 0.6),
        (3, 10, "--", "B↔H", 1.2, 0.6),
        (0, 6, "--", "B↔H", 1.2, 0.6),
        (4, 11, "--", "B↔H", 1.2, 0.6),
        (5, 12, "--", "B↔H", 1.2, 0.6),
    ],
    subtitle="Bots hide among humans — GNNs get confused",
)

fig.tight_layout()
fig.savefig("figures/heterophily.svg")
print("Saved figures/heterophily.svg")
