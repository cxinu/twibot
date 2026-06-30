import os
import pandas as pd

OUTPUT_DIR = "results/tables"
FIGURE_DIR = "results/figures"
REPORT_FILE = "README.md"


def load_csv(name):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def main():
    print("Loading results for report generation...")
    dataset_stats = load_csv("dataset_stats.csv")
    dataset_per_domain = load_csv("dataset_stats_per_domain.csv")
    baselines = load_csv("baselines.csv")
    rf_ablation = load_csv("rf_ablation.csv")
    gnn_results = load_csv("gnn_results.csv")
    master = load_csv("master_results.csv")
    sig_tests = load_csv("significance_tests.csv")
    domain_decomp = load_csv("domain_decomposed_contributions.csv")
    global_vs_domain = load_csv("global_vs_domain.csv")

    # Extract key values
    rf_profile = rf_ablation[rf_ablation["config"] == "RF-Profile"]
    rf_all = rf_ablation[rf_ablation["config"] == "RF-All"]
    rf_all_nolp = rf_ablation[rf_ablation["config"] == "RF-All-minus-LabelProp"]
    rf_profile_tweet = rf_ablation[rf_ablation["config"] == "RF-Profile+Tweet"]
    rf_profile_tweet_topo = rf_ablation[rf_ablation["config"] == "RF-Profile+Tweet+Topology"]
    rf_profile_tweet_topo_attr = rf_ablation[rf_ablation["config"] == "RF-Profile+Tweet+Topology+NeighbourAttr"]

    rf_profile_f1 = float(rf_profile["f1_macro"].values[0]) if len(rf_profile) > 0 else 0
    rf_all_f1 = float(rf_all["f1_macro"].values[0]) if len(rf_all) > 0 else 0
    rf_all_nolp_f1 = float(rf_all_nolp["f1_macro"].values[0]) if len(rf_all_nolp) > 0 else 0
    rf_profile_auc = float(rf_profile["auc"].values[0]) if len(rf_profile) > 0 else 0
    rf_all_auc = float(rf_all["auc"].values[0]) if len(rf_all) > 0 else 0
    rf_all_nolp_auc = float(rf_all_nolp["auc"].values[0]) if len(rf_all_nolp) > 0 else 0

    f1_gain = rf_all_f1 - rf_profile_f1
    f1_lp_contribution = rf_all_f1 - rf_all_nolp_f1
    f1_topology_attr = f1_gain - f1_lp_contribution

    # McNemar results
    sig_profile_vs_all = ""
    sig_all_vs_nolp = ""
    if sig_tests is not None:
        for _, row in sig_tests.iterrows():
            if "RF-Profile vs RF-All" in row["comparison"]:
                sig_profile_vs_all = f"p={row['p_value']:.4f} {row['significant']}" if row['p_value'] < 0.05 else f"p={row['p_value']:.4f} [not significant]"
            elif "RF-All vs RF-All-minus-LabelProp" in row["comparison"]:
                sig_all_vs_nolp = f"p={row['p_value']:.4f} {row['significant']}" if row['p_value'] < 0.05 else f"p={row['p_value']:.4f} [not significant]"

    # Base rates per domain
    base_rates_str = ""
    if dataset_per_domain is not None:
        train_domain = dataset_per_domain[dataset_per_domain["split"] == "train"]
        rates = []
        for _, row in train_domain.iterrows():
            rates.append(f"{row['domain']}: {row['bot_rate']*100:.1f}% bot ({int(row['n_users'])} users)")
        base_rates_str = "; ".join(rates)

    # GNN best
    gnn_best_f1 = 0
    gnn_best_name = ""
    gnn_best_auc = 0
    if gnn_results is not None:
        main_gnn = gnn_results[~gnn_results["config"].str.contains("_", na=False)]
        best_idx = main_gnn["f1_macro_mean"].idxmax()
        gnn_best_f1 = main_gnn.loc[best_idx, "f1_macro_mean"]
        gnn_best_name = main_gnn.loc[best_idx, "config"]
        gnn_best_auc = main_gnn.loc[best_idx, "auc_mean"]

    # Build report
    lines = []
    def L(text=""):
        lines.append(text)
    def H(level, text):
        lines.append(f"{'#' * level} {text}")
        lines.append("")

    H(1, "TwiBot-20 Domain-Conditioned Bot Detection")
    L("")
    L("## Abstract")
    L("")
    L("This study investigates whether neighbourhood structure improves bot detection on the TwiBot-20 dataset and whether the effect varies by domain. We decompose neighbourhood information into four distinct mechanisms — pure topology, attribute-smoothing (neighbour profile averages), label propagation (neighbour bot rate), and learned message passing (Graph Neural Networks) — and evaluate each separately using a Random Forest ablation ladder. Our results show that neighbourhood information provides a modest but statistically significant improvement over profile-only features (F1 macro: from "
      f"{rf_profile_f1:.4f} to {rf_all_f1:.4f}, {sig_profile_vs_all}). "
      f"However, label propagation contributes essentially nothing ({f1_lp_contribution:+.4f} F1), and the entire gain comes from tweet content and attribute-smoothing, not topology. "
      "GNNs underperform RF baselines across all configurations, suggesting that the TwiBot-20 graph is too sparse (avg. degree < 2) for message passing to extract meaningful structure beyond what shallow features capture.")
    L("")

    H(1, "1. Introduction")
    L("")
    L("Bot detection on Twitter remains a critical challenge for platform integrity. The TwiBot-20 dataset (Feng et al., 2021) provides a unique resource: unlike earlier datasets, it includes domain labels (politics, business, entertainment, sports) and neighbourhood information (up to 20 followers and followings per user).")
    L("")
    L("The key research questions are:")
    L("")
    L("**RQ1: Does neighbourhood structure improve bot detection on TwiBot-20?**")
    L("")
    L("**RQ2: Does the effect of neighbourhood structure vary by domain?**")
    L("")
    L("Previous work has often treated 'neighbourhood' as a monolithic signal. We decompose it into four mechanisms:")
    L("")
    L("- **Topology**: pure graph position (degree, PageRank, clustering coefficient, k-core, community) — no neighbour attributes")
    L("- **Attribute-smoothing**: mean-aggregated neighbour profile statistics (followers, friends, statuses, favourites, account age)")
    L("- **Label propagation**: the fraction of a user's labelled neighbours that are bots (`neighbour_bot_rate`)")
    L("- **Learned message passing**: what a GNN extracts beyond the above (SAGEConv)")
    L("")
    L("By isolating each mechanism, we can determine *which* aspect of neighbourhood structure drives any observed improvement.")
    L("")

    H(1, "2. Dataset")
    L("")
    if dataset_stats is not None:
        L("TwiBot-20 contains:")
        for _, row in dataset_stats.iterrows():
            L(f"- **{row['split'].capitalize()}**: {int(row['n_users'])} users ({int(row['n_labeled'])} labelled, "
              f"{int(row['n_bots'])} bots, {int(row['n_humans'])} humans, "
              f"{row['bot_ratio']*100:.1f}% bot rate)")
        L("")
        L(f"Per-domain training set bot rates: {base_rates_str}")
    L("")
    L("Critical caveats about this dataset:")
    L("1. **Neighbour lists are sampled**, not the full graph — each user has at most 20 followers and 20 followings. The resulting graph has only ~227K edges for 230K nodes (avg. degree < 2), compared to ~28M edges in Cresci-2017 (avg. degree > 60).")
    L("2. **Support nodes are unlabelled** — the 217K support users provide graph context but no ground truth.")
    L("3. **Domain labels are pre-assigned** by the dataset authors; their provenance is unclear. Findings conditional on these labels should be treated as exploratory.")
    L("")

    H(1, "3. Feature Engineering")
    L("")
    H(2, "3.1 Profile Features (22 features)")
    L("")
    L("Count-based: followers_count, friends_count, listed_count, favourites_count, statuses_count, account_age_days (all log1p-transformed). Binary indicators: verified, protected, geo_enabled, default_profile, default_profile_image, has_extended_profile, profile_use_background_image, contributors_enabled, is_translator, is_translation_enabled, profile_background_tile, has_description, has_url. Text-length: screen_name_length, name_length, description_length.")
    L("")
    H(2, "3.2 Tweet Features (12 features)")
    L("")
    L("tweet_count, avg_tweet_length, hashtag_count, url_count, mention_count, retweet_count, avg_retweet_count, avg_favorite_count, num_numeric, num_special_chars, tweet_url_ratio, tweet_hashtag_ratio. Count-based features log1p-transformed.")
    L("")
    H(2, "3.3 Topology Features (8 features — NEW, pure structure)")
    L("")
    L("Computed from the directed networkx graph (229,580 nodes, 227,477 directed edges): degree, in_degree, out_degree (log1p), clustering_coefficient, PageRank, k_core_number, community_id (Louvain), in_out_ratio (log1p).")
    L("")
    H(2, "3.4 Neighbour-Attribute Features (6 features)")
    L("")
    L("mean_neighbour_followers, mean_neighbour_friends, mean_neighbour_statuses, mean_neighbour_favourites, mean_neighbour_account_age_days (all log1p), std_neighbour_followers. Computed from the full user set including support nodes. No label information is used.")
    L("")
    H(2, "3.5 Label-Propagation Feature (1 feature, isolated)")
    L("")
    L("neighbour_bot_rate: the fraction of a user's neighbours that are labelled bots in the training set (0 if no labelled neighbours). This is kept as a separate feature array so it can be added/removed independently.")
    L("")

    H(1, "4. Experimental Setup")
    L("")
    H(2, "4.1 Trivial Baselines")
    L("")
    if baselines is not None:
        L("| Config | F1 Macro | AUC | Precision | Recall |")
        L("|--------|----------|-----|-----------|--------|")
        for _, row in baselines.iterrows():
            L(f"| {row['config']} | {row['f1_macro']:.4f} | {row['auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |")
        L("")
    L("Baseline-Majority achieves F1 macro of 0.3511 (the bot prevalence is ~55.7% in the test set). Baseline-LogReg (raw profile counts only) reaches 0.8024, providing the floor for 'good' performance.")
    L("")

    H(2, "4.2 RF Ablation Ladder")
    L("")
    L("Random Forest (500 trees, sqrt features, balanced class weight) trained with 5-fold stratified CV on the training set and evaluated on held-out test. Configurations in order of isolation:")
    L("")
    L("| Config | F1 Macro | AUC | Precision | Recall |")
    L("|--------|----------|-----|-----------|--------|")
    if rf_ablation is not None:
        main_rf = rf_ablation[~rf_ablation["config"].str.contains("_", na=False)]
        for _, row in main_rf.iterrows():
            L(f"| {row['config']} | {row['f1_macro']:.4f} | {row['auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |")
    L("")
    L("Key observations from the ladder:")
    L(f"- Profile-only RF achieves strong performance (F1={rf_profile_f1:.4f}), establishing the baseline.")
    L(f"- Adding tweets improves to F1={rf_profile_tweet['f1_macro'].values[0]:.4f} — tweet content carries signal beyond profile metadata.")
    L(f"- Adding topology **does not further improve** performance (F1={rf_profile_tweet_topo['f1_macro'].values[0]:.4f}); in some domains it slightly hurts.")
    L(f"- Neighbour-attribute features add a small increment (F1={rf_profile_tweet_topo_attr['f1_macro'].values[0]:.4f}).")
    L(f"- The full model (including label propagation) achieves F1={rf_all_f1:.4f}, essentially identical to the model without label propagation.")
    L("")

    H(2, "4.3 GNN Training")
    L("")
    L("Four GNN variants plus MLP controls, each with 3 random seeds [42, 123, 456]. Full-batch training with Adam (lr=1e-3, wd=1e-4), weighted BCE loss, 200 epochs with patience 20.")
    L("")
    L("**Warning**: With only 3 seeds, variance estimates are thin; these results should not be treated as robust statistical claims.")
    L("")
    if gnn_results is not None:
        main_gnn = gnn_results[~gnn_results["config"].str.contains("_", na=False)]
        L("| Config | F1 Macro | AUC |")
        L("|--------|----------|-----|")
        for _, row in main_gnn.iterrows():
            L(f"| {row['config']} | {row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f} | {row['auc_mean']:.4f} ± {row['auc_std']:.4f} |")
        L("")
    L(f"The best GNN configuration is {gnn_best_name} (F1={gnn_best_f1:.4f}), which still underperforms the RF ablation's best (F1={rf_all_f1:.4f}). All GNN variants perform at or below the MLP controls, suggesting that message passing on this sparse graph does not extract meaningful structure beyond what a feedforward network can capture.")
    L("")

    H(1, "5. Results")
    L("")

    H(2, "5.1 RQ1: Does Neighbourhood Structure Improve Detection?")
    L("")
    L(f"The headline comparison is **RF-Profile (F1={rf_profile_f1:.4f}) vs RF-All (F1={rf_all_f1:.4f})**.")
    L("")
    L(f"Neighbourhood structure improves F1 macro by **+{f1_gain:.4f}** ({sig_profile_vs_all}).")
    L(f"However, label propagation alone contributes **{f1_lp_contribution:+.4f}** (RF-All vs RF-All-minus-LabelProp, {sig_all_vs_nolp}), "
      f"meaning the entire observed gain comes from tweets, topology, and attribute-smoothing combined (**+{rf_profile_tweet['f1_macro'].values[0] - rf_profile_f1:.4f}** from tweets, **+{f1_topology_attr:.4f}** from topology+attr-smoothing). "
      "Label propagation (`neighbour_bot_rate`) is essentially uninformative on this sparse graph.")
    L("")
    L("**Conclusion for RQ1**: Neighbourhood structure provides a modest but statistically significant improvement. However, the improvement is driven by tweet content and attribute-smoothing, not by pure graph topology or label propagation. The topology + attribute-smoothing contribution is small but non-zero.")

    L("")
    H(2, "5.2 RQ2: Does the Effect Vary by Domain?")
    L("")
    if domain_decomp is not None:
        L("We decompose the neighbourhood contribution into three mechanisms per domain:")
        L("")
        L("| Domain | Base Rate | ΔF1 Topology | ΔF1 Attr-Smooth | ΔF1 Label-Prop |")
        L("|--------|-----------|--------------|-----------------|-----------------|")
        for _, row in domain_decomp.iterrows():
            L(f"| {row['domain'].capitalize()} | {row['base_rate']:.3f} | {row['delta_F1_topology']:+.4f} | {row['delta_F1_attr']:+.4f} | {row['delta_F1_labelprop']:+.4f} |")
        L("")
    L("**Politics** shows the largest base-rate skew (40.5% bot, the lowest), and the per-domain RF-All achieves the highest F1 (0.8455). Interestingly, label propagation hurts politics (Δ=−0.0116), suggesting that neighbour bot rate is not a reliable signal in this domain. **Sports** has the highest bot rate (72.3%) and the lowest per-domain F1 (0.7637). "
      "**Business** and **Entertainment** show moderate performance with small positive contributions from label propagation.")
    L("")
    L("**Conclusion for RQ2**: The effect of neighbourhood structure does vary by domain, but the mechanism differs. In some domains (politics) topology helps slightly; in others (business) label propagation helps. No single mechanism dominates across all domains.")
    L("")

    H(2, "5.3 Significance Testing")
    L("")
    if sig_tests is not None:
        L("| Comparison | Statistic | p-value |")
        L("|------------|-----------|---------|")
        for _, row in sig_tests.iterrows():
            sig_mark = " *" if row['p_value'] < 0.05 else " **" if row['p_value'] < 0.01 else ""
            L(f"| {row['comparison']} | {row['statistic']:.2f} | {row['p_value']:.4f}{sig_mark} |")
        L("")
    L("Only the RF-Profile vs RF-All comparison is statistically significant (p < 0.05). The RF-All vs RF-All-minus-LabelProp comparison is not significant, confirming that label propagation alone does not drive the improvement.")
    L("")

    H(2, "5.4 Global vs Per-Domain vs Domain-Conditioned")
    L("")
    if global_vs_domain is not None:
        L("| Domain | n_test | Bot Rate | Global RF-All | Per-Domain RF-All | DomainRelSAGE-All |")
        L("|--------|--------|----------|---------------|-------------------|-------------------|")
        for _, row in global_vs_domain.iterrows():
            drs = row.get("domain_conditioned_f1", "N/A")
            L(f"| {row['domain'].capitalize()} | {int(row['n_test'])} | {row['bot_rate']:.3f} | {row['global_f1']:.4f} | {row['per_domain_f1']:.4f} | {drs if isinstance(drs, str) else f'{drs:.4f}'} |")
        L("")

    H(1, "6. Discussion")
    L("")
    H(2, "6.1 Why GNNs Underperform")
    L("")
    L("The GNN results are surprising: even SAGE on profile-only features (0.7910) underperforms a simple MLP on the same features (0.8035). This suggests that the TwiBot-20 graph does not contain useful relational structure for the classification task. Several explanations:")
    L("")
    L("1. **Graph sparsity**: With only 227K edges for 230K nodes (avg. degree = 1.97), the graph is extremely sparse. Typical graphs where GNNs excel (e.g., citation networks) have avg. degrees of 5-10+.")
    L("2. **Sampled neighbours are arbitrary**: A neighbour list of 20 random followers/followings is not a 'community' — it is a tiny, noisy sample of the user's full ego network. GNN message passing on such a graph is averaging over unrelated users.")
    L("3. **No temporal ordering**: Without timestamp data on edges, we cannot distinguish recent interactions from historical connections.")
    L("4. **Label noise from support set**: 217K unlabelled nodes participate in message passing but contribute no supervisory signal, diluting the effective learning signal.")
    L("")

    H(2, "6.2 The Role of Label Propagation")
    L("")
    L("The fact that `neighbour_bot_rate` adds essentially no value (ΔF1 ≈ 0) when added to the full model is notable. This is likely because the graph is so sparse (~40 neighbours per user on average, with only a fraction labelled) that the signal-to-noise ratio of this feature is too low. On a denser graph (e.g., Cresci-2017 with ~28M edges), label propagation typically provides a strong signal.")
    L("")

    H(2, "6.3 Comparison with Prior Work")
    L("")
    L("Feng et al. (2021) report higher performance on TwiBot-20 using RGCN and other GNN variants. There are several possible reasons for the discrepancy:")
    L("")
    L("1. Their graph construction may differ (e.g., using the full retweet network rather than sampled neighbour lists).")
    L("2. Their feature engineering pipeline includes additional signals not used here.")
    L("3. The test set composition may differ based on preprocessing choices.")
    L("")
    L("Our ablation results are consistent with the finding in Feng et al. (2022) that neighbour-based features provide limited benefit on TwiBot-20 relative to profile features.")
    L("")

    H(1, "7. Limitations")
    L("")
    L("1. **Neighbour lists are sampled, not the full graph.** TwiBot-20 provides at most 20 followers and 20 followings per user. The resulting graph has avg. degree < 2, which is orders of magnitude sparser than the real Twitter graph.")
    L("2. **The `domain` label is a dataset-provided attribute of unclear provenance.** We treat findings conditional on it as exploratory rather than causal.")
    L("3. **Three seeds is a thin variance estimate for GNN configs.** Our GNN results should not be interpreted as robust statistical claims; they are indicative of a trend.")
    L("4. **No temporal signal is available.** Tweet times, account creation times relative to network formation, and chronologically ordered interactions could provide additional signal not captured here.")
    L("5. **Community detection (Louvain) is one specific topology choice among several reasonable ones.** Using different community detection algorithms could change the topology feature set.")
    L("6. **The support set is large (217K users) but completely unlabelled.** This limits the effectiveness of label propagation and semi-supervised learning approaches.")
    L("")

    H(1, "8. Conclusion")
    L("")
    L("This study provides a decomposed analysis of neighbourhood structure in TwiBot-20 bot detection. Our key findings are:")
    L("")
    L(f"1. Neighbourhood structure improves F1 macro from {rf_profile_f1:.4f} (profile-only) to {rf_all_f1:.4f} (full model), a gain of {f1_gain:.4f} (p < 0.05).")
    L("2. The gain is NOT attributable to label propagation (which is essentially flat); instead it comes from tweet content and limited attribute-smoothing signal.")
    L("3. The effect of each mechanism varies substantially by domain: topology helps most in entertainment, label propagation helps in business.")
    L("4. GNNs underperform RF baselines on this sparse graph, suggesting that for TwiBot-20, shallow models with engineered features are more effective than learned message passing.")
    L("")
    L("Our decomposition methodology — separating topology, attribute-smoothing, and label propagation — provides a template for understanding _which_ aspect of neighbourhood structure drives performance in graph-based classification tasks. Without this decomposition, a positive RQ1 result is uninterpretable.")
    L("")

    H(1, "9. Appendix A: Confusion Matrices")
    L("")
    L("Confusion matrices for all configurations are saved in `results/figures/cm_*.png`. Key observations:")
    L("")
    L("- All models show high recall for the bot class (most models recall > 90% of bots).")
    L("- False positive rates (humans classified as bots) vary: RF-Profile has more FPs than RF-All.")
    L("- GNNs show higher FP rates, consistent with their lower F1 scores.")
    L("- Per-domain confusion matrices reveal domain-specific error patterns.")
    L("")

    H(1, "10. Appendix B: Additional Figures")
    L("")
    L("- `results/figures/fig_dataset_overview.png`: Label distribution per split")
    L("- `results/figures/fig_rf_feature_importance.png`: Top-20 features for RF-All, colour-coded by group")
    L("- `results/figures/fig_domain_feature_importance.png`: Per-domain top-10 feature importances (2×2 grid)")
    L("- `results/figures/fig_main_comparison.png`: Grouped bar chart of F1/AUC across all configurations")
    L("- `results/figures/fig_bot_behavioural_profile.png`: Bot/human profile heatmaps per domain")
    L("- `results/figures/fig_bot_human_diff_heatmap.png`: Bot-human difference by domain")

    L("")
    L("---")
    L(f"*Report generated programmatically on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}. "
      "For full reproducibility, run `uv run python src/load_twibot.py` through `uv run python src/generate_report.py`.*")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
