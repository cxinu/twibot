import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

OUTPUT_DIR = "results/tables"
FIGURE_DIR = "results/figures"

COLORS_CONFIG = {
    "Baseline-Majority": "#264653",
    "Baseline-LogReg": "#264653",
    "RF-Profile": "#2A9D8F",
    "RF-Profile+Tweet": "#2A9D8F",
    "RF-Profile+Tweet+Topology": "#6A994E",
    "RF-Profile+Tweet+Topology+NeighbourAttr": "#F4A261",
    "RF-All": "#9D4EDD",
    "RF-All-minus-LabelProp": "#F4A261",
    "MLP-Profile": "#2A9D8F",
    "SAGE-Profile": "#6A994E",
    "MLP-All": "#2A9D8F",
    "SAGE-All": "#6A994E",
    "RelSAGE-All": "#E9C46A",
    "DomainRelSAGE-All": "#E63946",
    "HeteroSAGE-Profile": "#7B2D8E",
    "HeteroSAGE-All": "#D62828",
}

GROUP_COLORS = {
    "Baseline": "#264653",
    "Profile": "#2A9D8F",
    "Tweet": "#E9C46A",
    "Topology": "#6A994E",
    "NeighbourAttr": "#F4A261",
    "LabelProp": "#9D4EDD",
    "MLP": "#2A9D8F",
    "SAGE": "#6A994E",
    "RelSAGE": "#E9C46A",
    "DomainRelSAGE": "#E63946",
    "HeteroSAGE": "#D62828",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("Loading results...")
    baselines_df = pd.read_csv(os.path.join(OUTPUT_DIR, "baselines.csv"))
    rf_df = pd.read_csv(os.path.join(OUTPUT_DIR, "rf_ablation.csv"))
    gnn_df = pd.read_csv(os.path.join(OUTPUT_DIR, "gnn_results.csv"))

    # Build master results table
    master_rows = []

    # Baselines
    for _, row in baselines_df.iterrows():
        master_rows.append({
            "config": row["config"],
            "group": "Baseline",
            "f1_macro": row["f1_macro"],
            "auc": row["auc"],
            "precision": row["precision"],
            "recall": row["recall"],
        })

    # RF (only top-level configs, not per-domain)
    rf_main = rf_df[~rf_df["config"].str.contains("_", na=False)]
    for _, row in rf_main.iterrows():
        master_rows.append({
            "config": row["config"],
            "group": "RF",
            "f1_macro": row["f1_macro"],
            "auc": row["auc"],
            "precision": row["precision"],
            "recall": row["recall"],
        })

    # GNN (only top-level configs, not per-domain)
    gnn_main = gnn_df[~gnn_df["config"].str.contains("_", na=False)]
    for _, row in gnn_main.iterrows():
        f1_str = f"{row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f}"
        auc_str = f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}"
        master_rows.append({
            "config": row["config"],
            "group": "GNN",
            "f1_macro": row["f1_macro_mean"],
            "f1_macro_str": f1_str,
            "auc": row["auc_mean"],
            "auc_str": auc_str,
            "precision": row["precision_mean"],
            "recall": row["recall_mean"],
        })

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(os.path.join(OUTPUT_DIR, "master_results.csv"), index=False)
    print(f"Saved master_results.csv")
    print(master_df[["config", "f1_macro", "auc"]].to_string(index=False))

    # Main comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar colors
    bar_colors = []
    for _, row in master_df.iterrows():
        cfg = row["config"]
        bar_colors.append(COLORS_CONFIG.get(cfg, "#264653"))

    ax1 = axes[0]
    configs = master_df["config"].values
    f1_vals = master_df["f1_macro"].values
    auc_vals = master_df["auc"].values

    bars1 = ax1.bar(range(len(configs)), f1_vals, color=bar_colors, width=0.7)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("F1 Macro")
    ax1.set_title("Overall F1 Macro by Config")
    ax1.set_ylim(0, 1)

    # Add value labels
    for bar, val in zip(bars1, f1_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax2 = axes[1]
    bars2 = ax2.bar(range(len(configs)), auc_vals, color=bar_colors, width=0.7)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("AUC")
    ax2.set_title("Overall AUC by Config")
    ax2.set_ylim(0, 1)

    for bar, val in zip(bars2, auc_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig_main_comparison.png"), dpi=150)
    plt.close()
    print("Saved fig_main_comparison.png")

    # Bucket comparison figure
    bucket_path = os.path.join(OUTPUT_DIR, "hetero_bucket_comparison.csv")
    if os.path.exists(bucket_path):
        print("\nPlotting bucket comparison...")
        bdf = pd.read_csv(bucket_path)
        bdf_plot = bdf[bdf["bucket"] != "Overall"]

        def parse_mean(s):
            return float(s.split('±')[0])
        def parse_std(s):
            return float(s.split('±')[1])

        buckets = bdf_plot["bucket"].values
        sage_m = [parse_mean(v) for v in bdf_plot["sage_f1"]]
        sage_s = [parse_std(v) for v in bdf_plot["sage_f1"]]
        hetero_m = [parse_mean(v) for v in bdf_plot["hetero_f1"]]
        hetero_s = [parse_std(v) for v in bdf_plot["hetero_f1"]]

        x = np.arange(len(buckets))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, sage_m, width, yerr=sage_s,
               label="SAGE-All", color="#6A994E", capsize=3)
        ax.bar(x + width / 2, hetero_m, width, yerr=hetero_s,
               label="HeteroSAGE-All", color="#D62828", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, fontsize=10)
        ax.set_ylabel("F1 Macro", fontsize=11)
        ax.set_title("SAGE-All vs HeteroSAGE-All by Homophily Bucket", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0.75, 0.90)
        for i, (_, row) in enumerate(bdf_plot.iterrows()):
            y_max = max(sage_m[i] + sage_s[i], hetero_m[i] + hetero_s[i])
            ax.text(i, y_max + 0.008, f"p={row['mcnemar_p']:.3f}",
                    ha="center", fontsize=8, fontstyle="italic")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "fig_bucket_comparison.png"), dpi=150)
        plt.close()
        print("  Saved fig_bucket_comparison.png")
    else:
        print("\n  Skipping bucket comparison plot: hetero_bucket_comparison.csv not found")

    # MLP-All vs HeteroSAGE-All bucket comparison figure
    mlp_bucket_path = os.path.join(OUTPUT_DIR, "mlp_vs_hetero_bucket_comparison.csv")
    if os.path.exists(mlp_bucket_path):
        print("\nPlotting MLP vs HeteroSAGE bucket comparison...")
        mdf = pd.read_csv(mlp_bucket_path)
        mdf_plot = mdf[mdf["bucket"] != "Overall"]

        buckets = mdf_plot["bucket"].values
        mlp_m = [parse_mean(v) for v in mdf_plot["mlp_f1"]]
        mlp_s = [parse_std(v) for v in mdf_plot["mlp_f1"]]
        hetero_m = [parse_mean(v) for v in mdf_plot["hetero_f1"]]
        hetero_s = [parse_std(v) for v in mdf_plot["hetero_f1"]]

        x = np.arange(len(buckets))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, mlp_m, width, yerr=mlp_s,
               label="MLP-All", color="#2A9D8F", capsize=3)
        ax.bar(x + width / 2, hetero_m, width, yerr=hetero_s,
               label="HeteroSAGE-All", color="#D62828", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, fontsize=10)
        ax.set_ylabel("F1 Macro", fontsize=11)
        ax.set_title("MLP-All vs HeteroSAGE-All by Homophily Bucket", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0.75, 0.90)
        for i, (_, row) in enumerate(mdf_plot.iterrows()):
            y_max = max(mlp_m[i] + mlp_s[i], hetero_m[i] + hetero_s[i])
            ax.text(i, y_max + 0.008, f"p={row['mcnemar_p']:.3f}",
                    ha="center", fontsize=8, fontstyle="italic")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "fig_mlp_bucket_comparison.png"), dpi=150)
        plt.close()
        print("  Saved fig_mlp_bucket_comparison.png")
    else:
        print("\n  Skipping MLP bucket comparison: mlp_vs_hetero_bucket_comparison.csv not found")

    # Per-domain scatter overlay with GNN results
    print("\nPer-domain F1 breakdowns:")
    rf_per_domain = rf_df[rf_df["config"].str.contains("_", na=False)]
    print(rf_per_domain[["config", "f1_macro", "bot_rate"]].to_string(index=False))

    gnn_per_domain = gnn_df[gnn_df["config"].str.contains("_", na=False)]
    print(gnn_per_domain[["config", "f1_macro_mean"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in evaluate.py: {e}", file=sys.stderr)
        raise
