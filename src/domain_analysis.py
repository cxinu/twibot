import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
from scipy.stats import chi2_contingency
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

FIGURE_DIR = "results/figures"
OUTPUT_DIR = "results/tables"
FEATURE_DIR = "dataset"
COLORS = {
    "profile": "#2A9D8F", "tweet": "#E9C46A", "topology": "#6A994E",
    "neighbour_attr": "#F4A261", "label_prop": "#9D4EDD", "all": "#264653",
}
DOMAIN_COLORS = {"politics": "#E63946", "business": "#457B9D", "entertainment": "#2A9D8F", "sports": "#E9C46A"}


def compute_metrics(y_true, y_pred, y_prob):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return f1_macro, auc, prec, rec


def mcnemar_test(y_true, pred1, pred2):
    """Paired McNemar's test for two classifiers."""
    correct1 = pred1 == y_true
    correct2 = pred2 == y_true
    n01 = (correct1 & ~correct2).sum()
    n10 = (~correct1 & correct2).sum()
    if n01 == 0 and n10 == 0:
        return 1.0, 0.0
    if n01 + n10 < 10:
        return 1.0, 0.0
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    from scipy.stats import chi2
    p_val = 1.0 - chi2.cdf(statistic, 1)
    return float(p_val), float(statistic)


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet("dataset/twibot_df.parquet")
    with open(os.path.join(FEATURE_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    feats = {}
    for group in ["profile", "tweet", "topology", "neighbour_attr", "label_prop"]:
        feats[group] = np.load(os.path.join(FEATURE_DIR, f"features_{group}.npy"))

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    y_train = df["label"].values[train_mask].astype(int)
    y_test = df["label"].values[test_mask].astype(int)
    domain_test = df["domain"].values[test_mask]
    domains = sorted(set(domain_test))

    # ── 6A: Per-domain RF-All feature importance (2×2 grid) ──
    print("\n6A: Per-domain feature importances...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    for ax_idx, domain in enumerate(domains):
        dom_mask_train = df["domain"].values[train_mask] == domain
        dom_mask_test = domain_test == domain
        X_train_dom = np.concatenate(
            [feats[g][train_mask][dom_mask_train] for g in ["profile", "tweet", "topology", "neighbour_attr", "label_prop"]], axis=1
        )
        X_test_dom = np.concatenate(
            [feats[g][test_mask][dom_mask_test] for g in ["profile", "tweet", "topology", "neighbour_attr", "label_prop"]], axis=1
        )
        y_train_dom = y_train[dom_mask_train]
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
        if len(np.unique(y_train_dom)) >= 2:
            clf.fit(X_train_dom, y_train_dom)
        all_feat_names = []
        for g in ["profile", "tweet", "topology", "neighbour_attr", "label_prop"]:
            all_feat_names.extend(feature_names[g])
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            importances = np.ones(len(all_feat_names)) / len(all_feat_names)
        top_k = min(10, len(all_feat_names))
        top_idx = np.argsort(importances)[::-1][:top_k]
        top_names = [all_feat_names[i] for i in top_idx]
        top_imps = importances[top_idx]
        feat_to_group = {}
        for g in ["profile", "tweet", "topology", "neighbour_attr", "label_prop"]:
            for fn in feature_names[g]:
                feat_to_group[fn] = g
        bar_colors = [COLORS.get(feat_to_group.get(n, "all"), "#264653") for n in top_names]
        ax = axes_flat[ax_idx]
        ax.barh(range(top_k), top_imps, color=bar_colors[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"{domain.capitalize()} (n_test={(dom_mask_test).sum()}, n_train={len(y_train_dom)})")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[g], label=g) for g in ["profile", "tweet", "topology", "neighbour_attr", "label_prop"]]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(FIGURE_DIR, "fig_domain_feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved fig_domain_feature_importance.png")

    # ── 6B: Neighbourhood contribution by domain, decomposed by mechanism ──
    print("\n6B: Decomposed neighbourhood contribution by domain...")
    config_groups_6b = {
        "profile_tweet": ["profile", "tweet"],
        "profile_tweet_topology": ["profile", "tweet", "topology"],
        "profile_tweet_topology_neighbour": ["profile", "tweet", "topology", "neighbour_attr"],
        "rf_all": ["profile", "tweet", "topology", "neighbour_attr", "label_prop"],
        "rf_all_minus_labelprop": ["profile", "tweet", "topology", "neighbour_attr"],
    }

    domain_breakdown = []
    for domain in domains:
        mask_test = domain_test == domain
        y_d = y_test[mask_test]
        if mask_test.sum() == 0:
            continue
        base_rate = y_d.mean()
        scores = {}
        for cfg_name, groups in config_groups_6b.items():
            X_train = np.concatenate([feats[g][train_mask] for g in groups], axis=1)
            X_test_d = np.concatenate([feats[g][test_mask][mask_test] for g in groups], axis=1)
            clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test_d)
            y_prob = clf.predict_proba(X_test_d)[:, 1]
            f1_macro, auc, prec, rec = compute_metrics(y_d, y_pred, y_prob)
            scores[cfg_name] = {"f1": f1_macro, "auc": auc, "prec": prec, "rec": rec}

        delta_topology = scores["profile_tweet_topology"]["f1"] - scores["profile_tweet"]["f1"]
        delta_attr = scores["profile_tweet_topology_neighbour"]["f1"] - scores["profile_tweet_topology"]["f1"]
        delta_labelprop = scores["rf_all"]["f1"] - scores["rf_all_minus_labelprop"]["f1"]
        domain_breakdown.append({
            "domain": domain,
            "base_rate": round(base_rate, 4),
            "f1_profile_tweet": round(scores["profile_tweet"]["f1"], 4),
            "f1_plus_topology": round(scores["profile_tweet_topology"]["f1"], 4),
            "f1_plus_neighbour": round(scores["profile_tweet_topology_neighbour"]["f1"], 4),
            "f1_rf_all": round(scores["rf_all"]["f1"], 4),
            "f1_rf_all_minus_labelprop": round(scores["rf_all_minus_labelprop"]["f1"], 4),
            "delta_F1_topology": round(delta_topology, 4),
            "delta_F1_attr": round(delta_attr, 4),
            "delta_F1_labelprop": round(delta_labelprop, 4),
        })
        print(f"  {domain}: base_rate={base_rate:.3f}, Δtopo={delta_topology:+.4f}, Δattr={delta_attr:+.4f}, Δlp={delta_labelprop:+.4f}")

    pd.DataFrame(domain_breakdown).to_csv(os.path.join(OUTPUT_DIR, "domain_decomposed_contributions.csv"), index=False)
    print("  Saved domain_decomposed_contributions.csv")

    # ── 6C: Significance testing (McNemar) ──
    print("\n6C: Significance testing (McNemar)...")
    sig_results = []

    def train_and_predict(groups):
        X_tr = np.concatenate([feats[g][train_mask] for g in groups], axis=1)
        X_te = np.concatenate([feats[g][test_mask] for g in groups], axis=1)
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_train)
        return clf.predict(X_te)

    def train_and_predict_per_domain(groups, domain):
        dom_train = df["domain"].values[train_mask] == domain
        dom_test = domain_test == domain
        X_tr = np.concatenate([feats[g][train_mask][dom_train] for g in groups], axis=1)
        X_te = np.concatenate([feats[g][test_mask][dom_test] for g in groups], axis=1)
        y_tr = y_train[dom_train]
        if len(np.unique(y_tr)) < 2:
            return np.zeros(len(y_test[dom_test]), dtype=int)
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        return clf.predict(X_te)

    pred_profile = train_and_predict(["profile"])
    pred_profile_tweet = train_and_predict(["profile", "tweet"])
    pred_profile_tweet_topo = train_and_predict(["profile", "tweet", "topology"])
    pred_profile_tweet_topo_attr = train_and_predict(["profile", "tweet", "topology", "neighbour_attr"])
    pred_all = train_and_predict(["profile", "tweet", "topology", "neighbour_attr", "label_prop"])
    pred_nolp = pred_profile_tweet_topo_attr  # same as All-minus-LabelProp

    # Sequential ladder comparisons (each isolates one mechanism)
    comparisons = [
        ("RF-Profile vs RF-Profile+Tweet", pred_profile, pred_profile_tweet),
        ("RF-Profile+Tweet vs RF-Profile+Tweet+Topology", pred_profile_tweet, pred_profile_tweet_topo),
        ("RF-Profile+Tweet+Topology vs RF-Profile+Tweet+Topology+NeighbourAttr",
         pred_profile_tweet_topo, pred_profile_tweet_topo_attr),
        ("RF-Profile+Tweet+Topology+NeighbourAttr vs RF-All",
         pred_profile_tweet_topo_attr, pred_all),
        ("RF-Profile vs RF-All", pred_profile, pred_all),
        ("RF-All vs RF-All-minus-LabelProp", pred_all, pred_nolp),
    ]

    # Per-domain sequential tests for the neighbourhood-relevant steps
    for domain in sorted(set(domain_test)):
        mask = domain_test == domain
        if mask.sum() < 30:
            continue
        y_d = y_test[mask]
        d_profile = train_and_predict_per_domain(["profile"], domain)
        d_profile_tweet = train_and_predict_per_domain(["profile", "tweet"], domain)
        d_profile_tweet_topo = train_and_predict_per_domain(["profile", "tweet", "topology"], domain)
        d_profile_tweet_topo_attr = train_and_predict_per_domain(["profile", "tweet", "topology", "neighbour_attr"], domain)
        d_all = train_and_predict_per_domain(["profile", "tweet", "topology", "neighbour_attr", "label_prop"], domain)

        # Only add comparisons that have different predictions
        comps = [
            (f"RF-Profile+Tweet vs +Topology ({domain})", d_profile_tweet, d_profile_tweet_topo),
            (f"+Topology vs +NeighbourAttr ({domain})", d_profile_tweet_topo, d_profile_tweet_topo_attr),
            (f"RF-All vs RF-All-minus-LabelProp ({domain})", d_all, d_profile_tweet_topo_attr),
        ]
        for cname, p1, p2 in comps:
            if len(set(p1.tolist() + p2.tolist())) < 2:
                continue  # degenerate — skip
            p_val, stat = mcnemar_test(y_d, p1, p2)
            sig = ""
            if p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            print(f"  {cname}: stat={stat:.2f}, p={p_val:.4f} {sig}")
            sig_results.append({
                "comparison": cname,
                "statistic": round(stat, 2),
                "p_value": round(p_val, 4),
                "significant": sig,
            })

    # GNN predictions - load from gnn_results.csv (metrics only, no stored predictions)
    # We compare RF-based models here; GNN comparisons are discussed in the report
    gnn_results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "gnn_results.csv"))

    for comp_name, pred1, pred2 in comparisons:
        if len(pred1) != len(pred2) or len(pred1) == 0:
            continue
        p_val, stat = mcnemar_test(y_test, pred1, pred2)
        sig = ""
        if p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        print(f"  {comp_name}: stat={stat:.2f}, p={p_val:.4f} {sig}")
        sig_results.append({
            "comparison": comp_name,
            "statistic": round(stat, 2),
            "p_value": round(p_val, 4),
            "significant": sig,
        })

    pd.DataFrame(sig_results).to_csv(os.path.join(OUTPUT_DIR, "significance_tests.csv"), index=False)
    print("  Saved significance_tests.csv")

    # ── 6D: Global vs Per-Domain vs Domain-Conditioned ──
    print("\n6D: Global vs Per-Domain vs Domain-Conditioned...")
    # Global RF-All
    rf_all_global = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    rf_all_global.fit(X_train_all, y_train)
    y_pred_global = rf_all_global.predict(X_test_all)
    y_prob_global = rf_all_global.predict_proba(X_test_all)[:, 1]
    f1_global, auc_global, _, _ = compute_metrics(y_test, y_pred_global, y_prob_global)

    comparison_rows = []
    for domain in domains:
        mask_test = domain_test == domain
        y_d = y_test[mask_test]
        base_rate = y_d.mean() if mask_test.sum() > 0 else 0
        # Per-domain RF-All
        dom_mask_train = df["domain"].values[train_mask] == domain
        X_train_d = X_train_all[dom_mask_train]
        y_train_d = y_train[dom_mask_train]
        X_test_d = X_test_all[mask_test]
        if len(y_train_d) > 0 and mask_test.sum() > 0:
            rf_domain = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
            rf_domain.fit(X_train_d, y_train_d)
            y_pred_d = rf_domain.predict(X_test_d)
            y_prob_d = rf_domain.predict_proba(X_test_d)[:, 1]
            f1_d, auc_d, _, _ = compute_metrics(y_d, y_pred_d, y_prob_d)
        else:
            f1_d, auc_d = 0, 0
        comparison_rows.append({
            "domain": domain,
            "n_test": mask_test.sum(),
            "bot_rate": round(base_rate, 4),
            "global_f1": round(f1_global, 4),
            "per_domain_f1": round(f1_d, 4),
        })

    # Add DomainRelSAGE F1 from existing results
    gnn_df = pd.read_csv(os.path.join(OUTPUT_DIR, "gnn_results.csv"))
    for _, row in gnn_df.iterrows():
        cfg = row["config"]
        if cfg.startswith("DomainRelSAGE-All_") or cfg.startswith("RelSAGE-All_"):
            parts = cfg.split("_", 1)
            if len(parts) == 2:
                config_name, domain = parts[0], parts[1]
                for cr in comparison_rows:
                    if cr["domain"] == domain:
                        if "DomainRelSAGE-All" in config_name:
                            cr["domain_conditioned_f1"] = row["f1_macro_mean"]
                        elif "RelSAGE-All" in config_name:
                            cr["domain_rel_f1"] = row["f1_macro_mean"]

    pd.DataFrame(comparison_rows).to_csv(os.path.join(OUTPUT_DIR, "global_vs_domain.csv"), index=False)
    print("  Saved global_vs_domain.csv")

    # ── 6E: Bot behavioural profile per domain (heatmap) ──
    print("\n6E: Bot behavioural profile per domain (heatmap)...")
    profile_cols = ["followers_count", "friends_count", "statuses_count", "favourites_count",
                    "verified", "default_profile", "has_description", "screen_name_length",
                    "description_length", "account_age_days"]
    tweet_cols = ["tweet_count", "avg_tweet_length", "hashtag_count", "url_count",
                  "mention_count", "retweet_ratio", "avg_retweet_count", "avg_favorite_count"]

    # Compute per-domain, per-class means
    heatmap_data = []
    for domain in domains:
        for label_name, label_val in [("Human", 0), ("Bot", 1)]:
            mask = (df["domain"] == domain) & (df["label"] == label_val)
            if mask.sum() == 0:
                continue
            row_data = {"domain": domain, "class": label_name}
            for col in profile_cols:
                vals = df[col][mask].replace([np.inf, -np.inf], 0).fillna(0).values
                row_data[col] = np.log1p(np.median(vals))
            tweet_retweet_ratio = (df["retweet_count"][mask] / df["tweet_count"][mask].clip(1)).median()
            row_data["retweet_ratio"] = np.log1p(tweet_retweet_ratio) if tweet_retweet_ratio > 0 else 0
            for col in tweet_cols:
                if col == "retweet_ratio":
                    continue
                vals = df[col][mask].replace([np.inf, -np.inf], 0).fillna(0).values
                if col in ["hashtag_count", "url_count", "mention_count", "retweet_count",
                           "num_numeric", "num_special_chars", "avg_retweet_count", "avg_favorite_count"]:
                    row_data[col] = np.log1p(np.median(vals))
                else:
                    row_data[col] = np.median(vals)
            heatmap_data.append(row_data)

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot_cols = [c for c in profile_cols + ["retweet_ratio"] + tweet_cols if c != "retweet_ratio" and c in heatmap_df.columns]

    if len(heatmap_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for idx, label_name in enumerate(["Human", "Bot"]):
            sub = heatmap_df[heatmap_df["class"] == label_name].set_index("domain")
            sub = sub[heatmap_pivot_cols]
            sns.heatmap(sub, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[idx], cbar_kws={"shrink": 0.8})
            axes[idx].set_title(f"{label_name} - Profile by Domain")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "fig_bot_behavioural_profile.png"), dpi=150)
        plt.close()
        print("  Saved fig_bot_behavioural_profile.png")

    # Save aggregated heatmap of bot-human differences
    if len(heatmap_df) > 0:
        diff_data = []
        for domain in domains:
            bot_row = heatmap_df[(heatmap_df["domain"] == domain) & (heatmap_df["class"] == "Bot")]
            hum_row = heatmap_df[(heatmap_df["domain"] == domain) & (heatmap_df["class"] == "Human")]
            if len(bot_row) > 0 and len(hum_row) > 0:
                diff = {}
                for col in heatmap_pivot_cols:
                    diff[col] = bot_row[col].values[0] - hum_row[col].values[0]
                diff["domain"] = domain
                diff_data.append(diff)
        if diff_data:
            diff_df = pd.DataFrame(diff_data).set_index("domain")
            fig, ax = plt.subplots(figsize=(14, 5))
            sns.heatmap(diff_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title("Bot-Human Difference by Domain (log1p median)")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURE_DIR, "fig_bot_human_diff_heatmap.png"), dpi=150)
            plt.close()
            print("  Saved fig_bot_human_diff_heatmap.png")

    print("\nDomain analysis complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in domain_analysis.py: {e}", file=sys.stderr)
        raise
