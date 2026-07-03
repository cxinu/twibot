#!/usr/bin/env python3
"""Render results comparison bar chart for WRITEUP.md (dark mode)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

MODELS = ["Lee", "RoBERTa", "GAT", "SATAR", "BotRGCN", "BotMoE", "RGT", "AdaRelBot\n(Ensemble)"]
ACC = [75.73, 74.97, 77.32, 61.70, 83.21, 84.22, 85.20, 86.56]
F1 = [79.37, 72.80, 80.51, 71.95, 87.68, 86.89, 86.88, 88.30]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
})

FIG_W, FIG_H = 8.5, 4.8
x = np.arange(len(MODELS))
WIDTH = 0.30

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("#0d1117")

bars_acc = ax.bar(x - WIDTH/2, ACC, WIDTH, color="#58a6ff", edgecolor="#0d1117", linewidth=0.6, label="Accuracy (%)")
bars_f1  = ax.bar(x + WIDTH/2, F1,  WIDTH, color="#f778ba", edgecolor="#0d1117", linewidth=0.6, label="F1 Score (%)")

for bar in bars_acc:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#58a6ff")
for bar in bars_f1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#f778ba")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=9, color="#c9d1d9")
ax.set_ylabel("Score (%)", fontsize=11, color="#c9d1d9")
ax.set_title("TwiBot-20 Benchmark Results", fontsize=13, fontweight="bold", pad=12, color="#e6edf3")
ax.set_ylim(55, 95)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

# Divider before AdaRelBot
ax.axvline(x=6.5, color="#f0883e", linewidth=1.4, linestyle="--", alpha=0.7)
ax.text(6.55, 93, "Ours", fontsize=8, color="#f0883e", fontweight="bold", fontstyle="italic")

ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.15, linestyle=":", linewidth=0.5, color="#30363d")

fig.tight_layout()
fig.savefig("figures/results.svg", facecolor="#0d1117")
print("Saved figures/results.svg")
