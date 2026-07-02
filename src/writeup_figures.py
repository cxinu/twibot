#!/usr/bin/env python3
"""Generate conceptual / educational figures for the research writeup.

These figures are meant to explain *why* the problem exists and *how* the fix
works, not just to report final numbers. They are rendered into
results/figures/ for inclusion in WRITEUP.md, RESEARCH.md and README.md.
"""

import os
import warnings


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 11

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_DIR = "data/twibot-20"

os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR_HUMAN = "#457b9d"
COLOR_BOT = "#e63946"
COLOR_NEUTRAL = "#adb5bd"
COLOR_ACCENT = "#2a9d8f"


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ── Load real data ──────────────────────────────────────────────────
print("Loading TwiBot-20 graph ...")
graph = torch.load(os.path.join(DATA_DIR, "twibot_graph.pt"),
                   map_location="cpu", weights_only=False)
y = graph.y.long()
labeled_mask = y >= 0
y_labeled = y[labeled_mask].numpy()


def per_node_homophily(y_tensor, edge_index_tensor):
    lab = y_tensor >= 0
    src, dst = edge_index_tensor
    mask = lab[src] & lab[dst]
    s, d = src[mask], dst[mask]
    same = (y_tensor[s] == y_tensor[d]).float()
    n = y_tensor.size(0)
    total = torch.zeros(n)
    same_sum = torch.zeros(n)
    ones = torch.ones_like(same)
    total.index_add_(0, s, ones)
    total.index_add_(0, d, ones)
    same_sum.index_add_(0, s, same)
    same_sum.index_add_(0, d, same)
    return (same_sum / total.clamp(min=1)).numpy()


h_follow = per_node_homophily(y, graph.edge_index_follow)[labeled_mask.numpy()]
h_following = per_node_homophily(y, graph.edge_index_following)[labeled_mask.numpy()]
h_combined = per_node_homophily(
    y, torch.cat([graph.edge_index_follow, graph.edge_index_following], dim=1)
)[labeled_mask.numpy()]

# Degree data for isolated-node figure
ef = graph.edge_index_follow.numpy()
eg = graph.edge_index_following.numpy()
adj = [set() for _ in range(graph.num_nodes)]
for u, v in zip(ef[0], ef[1]):
    adj[int(u)].add(int(v))
for u, v in zip(eg[0], eg[1]):
    adj[int(u)].add(int(v))
unique_deg = np.array([len(s) for s in adj])
unique_deg_labeled = unique_deg[labeled_mask.numpy()]

# ── Figure 1: the isolated-node hypothesis vs reality ───────────────
print("Figure 1: isolated-node hypothesis")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: cartoon of isolated node
ax = axes[0]
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4])
G.add_edges_from([(0, 1), (0, 2)])
pos = nx.spring_layout(G, seed=42, k=1.2)
colors = [COLOR_HUMAN, COLOR_BOT, COLOR_BOT, COLOR_NEUTRAL, COLOR_NEUTRAL]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, ax=ax,
                       edgecolors="white", linewidths=2)
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, ax=ax)
ax.set_title("Hypothesis: isolated nodes are hard\n(very few neighbors)")
ax.axis("off")

# Right: real degree distribution
ax = axes[1]
ax.hist(unique_deg_labeled, bins=range(0, 80, 2), color=COLOR_ACCENT,
        edgecolor="white", alpha=0.85)
ax.axvline(np.median(unique_deg_labeled), color="#264653", linestyle="--",
           linewidth=2, label=f"median = {np.median(unique_deg_labeled):.0f}")
ax.set_xlabel("Unique neighbors per labeled node")
ax.set_ylabel("Number of nodes")
ax.set_title("Reality: TwiBot-20 has almost no isolated nodes\n(crawler cap ≈ 10 per direction)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

save(fig, "fig_01_isolated_hypothesis.png")

# ── Figure 2: homophily vs heterophily cartoon ──────────────────────
print("Figure 2: homophily vs heterophily")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))


def draw_small_graph(ax, edges, node_labels, title, homo_text):
    G = nx.Graph()
    G.add_nodes_from(range(len(node_labels)))
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)
    colors = [COLOR_HUMAN if label == 0 else COLOR_BOT for label in node_labels]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, ax=ax,
                           edgecolors="white", linewidths=2)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax)
    ax.set_title(f"{title}\n{homo_text}")
    ax.axis("off")


# Homophilic
axes[0].text(0.5, -0.12, "Edges mostly connect same-label nodes",
             transform=axes[0].transAxes, ha="center", fontsize=10,
             style="italic", color="#333")
draw_small_graph(axes[0], [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
                 [0, 0, 0, 1, 1, 1], "Homophily", "homophily ≈ 1.0")

# Heterophilic
axes[1].text(0.5, -0.12, "Edges mostly connect opposite-label nodes",
             transform=axes[1].transAxes, ha="center", fontsize=10,
             style="italic", color="#333")
draw_small_graph(axes[1], [(0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 3)],
                 [0, 0, 0, 1, 1, 1], "Heterophily", "homophily ≈ 0.0")

save(fig, "fig_02_homophily_concept.png")

# ── Figure 3: the smoothing problem under heterophily ───────────────
print("Figure 3: smoothing problem")
fig, ax = plt.subplots(figsize=(8, 6))

# Central node and 5 bot neighbors
G = nx.Graph()
G.add_nodes_from(range(6))
G.add_edges_from([(0, i) for i in range(1, 6)])
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
pos = {0: (0, 0)}
for i in range(1, 6):
    pos[i] = (0.8 * np.cos(angles[i]), 0.8 * np.sin(angles[i]))

colors = [COLOR_HUMAN] + [COLOR_BOT] * 5
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1200, ax=ax,
                       edgecolors="white", linewidths=2.5)
nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, ax=ax)

# Annotation
ax.annotate("human node", xy=pos[0], xytext=(0.25, 0.15),
            fontsize=11, ha="left",
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("bot neighbors", xy=pos[2], xytext=(0.55, 0.55),
            fontsize=11, ha="left",
            arrowprops=dict(arrowstyle="->", color="#333"))

# Big curved arrow showing pull
ax.annotate("", xy=(0.35, -0.35), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=COLOR_BOT, lw=3,
                           connectionstyle="arc3,rad=.3"))
ax.text(0.42, -0.42, "mean aggregation\npulls human toward bots", fontsize=10,
        color=COLOR_BOT, ha="center", va="top")

ax.set_title("Why mean aggregation fails under heterophily")
ax.axis("off")
save(fig, "fig_03_smoothing_problem.png")

# ── Figure 4: TwiBot-20 combined homophily distribution ─────────────
print("Figure 4: homophily distribution")
fig, ax = plt.subplots(figsize=(9, 5))

bot_h = h_combined[y_labeled == 1]
hum_h = h_combined[y_labeled == 0]

ax.hist(hum_h, bins=30, alpha=0.7, label=f"Human (n={len(hum_h):,})",
        color=COLOR_HUMAN, density=True)
ax.hist(bot_h, bins=30, alpha=0.7, label=f"Bot (n={len(bot_h):,})",
        color=COLOR_BOT, density=True)
ax.axvline(0.5, color="#264653", linestyle="--", linewidth=2,
           label="homophily = 0.5")
ax.set_xlabel("Combined homophily")
ax.set_ylabel("Density")
ax.set_title("TwiBot-20 is heterophilic: most nodes have < 50% same-label neighbors")
ax.legend(loc="upper center")
ax.grid(axis="y", alpha=0.3)
save(fig, "fig_04_homophily_distribution.png")

# ── Figure 5: relation-level disagreement ───────────────────────────
print("Figure 5: relation disagreement")
fig, ax = plt.subplots(figsize=(7, 7))

valid = (h_follow >= 0) & (h_following >= 0)
hf = h_follow[valid]
hg = h_following[valid]
labels = y_labeled[valid]

ax.scatter(hf[labels == 0], hg[labels == 0], alpha=0.35, s=15,
           color=COLOR_HUMAN, label="Human")
ax.scatter(hf[labels == 1], hg[labels == 1], alpha=0.35, s=15,
           color=COLOR_BOT, label="Bot")
ax.axhline(0.5, color="#264653", linestyle="--", linewidth=1)
ax.axvline(0.5, color="#264653", linestyle="--", linewidth=1)
ax.set_xlabel("Follow-relation homophily")
ax.set_ylabel("Following-relation homophily")
ax.set_title("The two relations disagree\n(Pearson r = −0.15)")
ax.legend(loc="upper right")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
save(fig, "fig_05_relation_disagreement.png")

# ── Figure 6: the soft-contrast gate formula ────────────────────────
print("Figure 6: soft-contrast gate")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

formula = r"$m_r(v) = \mathrm{low}_r(v) + \beta_r(v) \cdot (W_r h_v - \mathrm{low}_r(v))$"
ax.text(0.5, 0.72, formula, fontsize=22, ha="center", va="center")

# Annotate terms with arrows
ax.annotate("mean of neighbors", xy=(0.30, 0.72), xytext=(0.18, 0.88),
            fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("learned gate", xy=(0.52, 0.72), xytext=(0.52, 0.88),
            fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("transformed ego", xy=(0.74, 0.72), xytext=(0.86, 0.88),
            fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="->", color="#333"))

ax.text(0.5, 0.42, r"$\beta_r(v) = \mathrm{MLP}_r([W_r h_v \; \mathrm{concat} \; \mathrm{low}_r(v)])$",
        fontsize=16, ha="center", va="center", color="#333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#adb5bd"))

ax.text(0.5, 0.16, "If β ≈ 0 → use neighborhood mean (standard RGCN)\n"
        "If β ≈ 1 → ignore neighbors and trust the ego node",
        fontsize=12, ha="center", va="center", style="italic", color="#555")

ax.set_title("Soft-contrast adaptive aggregation")
save(fig, "fig_06_soft_contrast_gate.png")

# ── Figure 7: results comparison ────────────────────────────────────
print("Figure 7: results")
results_df = pd.read_csv(os.path.join(RESULTS_DIR, "tables",
                                      "heterophily_fix_results.csv"))
overall = results_df[results_df["metric"] == "overall"]
buckets = results_df[(results_df["relation"] == "combined") &
                     (results_df["metric"].isna())]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: overall F1 (binary) and MCC
ax = axes[0]
variants = ["BotRGCN", "GatedBotRGCN-global", "SoftContrastBotRGCN-global"]
rows = overall[overall["variant"].isin(variants)].set_index("variant").reindex(variants)
x = np.arange(len(variants))
width = 0.3

f1_vals = rows["f1_binary"].values
f1_errs = rows["f1_binary_std"].fillna(0).values
mcc_vals = rows["mcc"].values
mcc_errs = rows["mcc_std"].fillna(0).values

bars1 = ax.bar(x - width / 2, f1_vals, width, yerr=f1_errs,
               color=[COLOR_NEUTRAL, COLOR_ACCENT, COLOR_BOT],
               capsize=5, edgecolor="white", linewidth=2, label="F1 (bot class)")
bars2 = ax.bar(x + width / 2, mcc_vals, width, yerr=mcc_errs,
               color=[COLOR_NEUTRAL, COLOR_ACCENT, COLOR_BOT],
               capsize=5, edgecolor="white", linewidth=2, alpha=0.5, label="MCC")
ax.set_xticks(x)
ax.set_xticklabels(["BotRGCN\n(baseline)", "Gated\n(global)", "SoftContrast\n(global)"],
                   fontsize=10)
ax.set_ylabel("Score")
ax.set_ylim(0.60, 0.87)
ax.set_title("Overall test performance")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)
for bar, val in zip(bars1, f1_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# Right: per-bucket F1 (binary)
ax = axes[1]
pivot = buckets.pivot(index="bucket", columns="variant", values="f1_binary")
pivot = pivot[[c for c in variants if c in pivot.columns]]
pivot = pivot.reindex(["0", "0.01-0.25", "0.26-0.50", "0.51+"])
pivot.plot(kind="bar", ax=ax, color=[COLOR_NEUTRAL, COLOR_ACCENT, COLOR_BOT],
           edgecolor="white", linewidth=1.5)
ax.set_xlabel("Combined homophily bucket")
ax.set_ylabel("F1 (bot class)")
ax.set_title("F1 stratified by heterophily")
ax.legend(["BotRGCN", "Gated (global)", "SoftContrast (global)"],
          loc="lower right", fontsize=9)
ax.set_xticklabels(pivot.index, rotation=0)
ax.grid(axis="y", alpha=0.3)

save(fig, "fig_07_results.png")

print("\nAll conceptual figures generated.")
