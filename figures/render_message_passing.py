#!/usr/bin/env python3
"""Render message-passing comparison: with vs without edge features."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_message_passing(ax, title, with_edges, subtitle):
    """Draw a 5-node star graph illustrating message passing."""
    r = 1.2
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    # Center node
    cx, cy = 0, 0
    neighbor_xs = r * np.cos(angles)
    neighbor_ys = r * np.sin(angles)

    # Node types: center is target, neighbors are mix
    colors = ["#f778ba"] + ["#58a6ff"] * 3 + ["#f778ba"]
    labels = ["B"] + ["H"] * 3 + ["B"]

    # Draw edges
    for i in range(5):
        nx, ny = neighbor_xs[i], neighbor_ys[i]
        alpha = 0.3 if (with_edges and i == 2) else 0.8
        lw = 1.2 if (with_edges and i == 2) else 2.0
        style = "--" if (with_edges and i == 2) else "-"
        ax.annotate("", xy=(cx, cy), xytext=(nx * 0.75, ny * 0.75),
                    arrowprops=dict(arrowstyle="->", color="#8b949e",
                                   lw=lw, alpha=alpha, linestyle=style))

        # Edge feature annotation
        if with_edges:
            mx, my = nx * 0.4, ny * 0.4
            sim = "low" if i == 2 else "high"
            fc = "#3d1f2e" if sim == "low" else "#1f2e3d"
            ax.text(mx, my, f"cos≈0.1", fontsize=6, color="#f778ba" if sim == "low" else "#58a6ff",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc=fc, ec="none", alpha=0.85))
        else:
            mx, my = nx * 0.4, ny * 0.4
            ax.text(mx, my, "?", fontsize=8, color="#8b949e", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#21262d", ec="none", alpha=0.85))

    # Draw center node
    ax.scatter([cx], [cy], s=320, c="#f778ba", edgecolors="#0d1117", linewidth=1.8, zorder=5)
    ax.text(cx, cy, "B?", fontsize=10, color="#0d1117", ha="center", va="center", fontweight="bold")

    # Draw neighbor nodes
    for i in range(5):
        ax.scatter([neighbor_xs[i]], [neighbor_ys[i]], s=220, c=colors[i],
                   edgecolors="#0d1117", linewidth=1.2, zorder=4)
        ax.text(neighbor_xs[i], neighbor_ys[i], labels[i], fontsize=8, color="#0d1117",
                ha="center", va="center", fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#e6edf3")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, -1.52, subtitle, ha="center", fontsize=8.5, style="italic", color="#8b949e")


plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.5))
fig.patch.set_facecolor("#0d1117")

draw_message_passing(ax1, "Without Edge Features", with_edges=False,
                     subtitle="GNN sees all neighbors equally — can't tell friend from bot")

draw_message_passing(ax2, "With Edge Features (AdaRelBot)", with_edges=True,
                     subtitle="GNN down-weights dissimilar edges — signal stays clean")

fig.tight_layout()
fig.savefig("figures/message_passing.svg", facecolor="#0d1117")
print("Saved figures/message_passing.svg")
