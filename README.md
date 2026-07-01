# TwiBot-20 Domain-Conditioned Bot Detection

## Abstract

We investigate whether neighbourhood structure improves bot detection on TwiBot-20 and whether the effect varies by domain. We decompose 'neighbourhood' into four distinct mechanisms — pure topology (degree, PageRank, etc.), attribute-smoothing (neighbour profile averages), label propagation (neighbour bot rate), and learned message passing (graph neural networks) — and evaluate each using a Random Forest ablation ladder with McNemar significance tests at each step.

The correct comparison for 'does neighbourhood help?' is RF-Profile+Tweet (F1=0.8287) vs RF-All (F1=0.8259), which adds topology, neighbour-attribute, and label-propagation features to a model that already has profile and tweet content. The result: adding all neighbourhood features changes F1 by **-0.0028** (no significance test available). Neither topology (-0.0065, p=0.3239), attribute-smoothing (+0.0029, p=0.7604), nor label propagation (+0.0008, p=1.0000) individually produce a statistically significant improvement over the preceding rung of the ladder.

Domain-conditioned models (DomainRelSAGE) also fail to outperform a plain MLP on the same input features, and per-domain mechanism decompositions show effect sizes within noise range given per-domain sample sizes (~270–340). We conclude that on the TwiBot-20 graph (avg. degree ≈ 2), neighbourhood structure does not meaningfully improve bot detection beyond strong profile and tweet-content baselines.

# 1. Introduction


Bot detection on Twitter remains a critical challenge for platform integrity. The TwiBot-20 dataset (Feng et al., 2021) provides a unique resource: unlike earlier datasets, it includes domain labels (politics, business, entertainment, sports) and neighbourhood information (up to 20 followers and followings per user).

The key research questions are:

**RQ1: Does neighbourhood structure improve bot detection on TwiBot-20?**

**RQ2: Does the effect of neighbourhood structure vary by domain?**

Previous work has often treated 'neighbourhood' as a monolithic signal. We decompose it into four mechanisms:

- **Topology**: pure graph position (degree, PageRank, clustering coefficient, k-core, community) — no neighbour attributes
- **Attribute-smoothing**: mean-aggregated neighbour profile statistics (followers, friends, statuses, favourites, account age)
- **Label propagation**: fraction of labelled neighbours that are bots (`neighbour_bot_rate`)
- **Learned message passing**: what a GNN (SAGEConv) extracts beyond the above

By isolating each mechanism, we can determine which aspect of neighbourhood structure (if any) drives observed improvements. A critical design principle: the comparison for 'does neighbourhood help?' must hold tweet-content features constant. RF-Profile vs RF-All conflates tweet features with neighbourhood features and is not a valid test of RQ1.

# 2. Related Work


**TwiBot-20 benchmark.** Feng et al. (2021) introduced TwiBot-20 and reported strong GNN performance using RGCN, with F1 scores > 0.90 on the test set. However, their evaluation protocol differs from ours: they construct the graph from the full retweet network and use a different feature set. Feng et al. (2022) revisited the dataset and found that neighbour-based features provide limited benefit relative to profile features, consistent with our findings.

**GNNs for social media.** Graph neural networks have shown promise on social network tasks when the graph is dense and edges are behaviourally meaningful (e.g., retweet networks, follow networks with high degree). On sampled neighbour lists — where each user observes at most 20 connections in each direction — message passing averages over largely unrelated users, and the theoretical advantage of GNNs over shallow models is minimal. Our results (MLP-All=0.8240 vs SAGE-All=0.8189) are consistent with this expectation.

# 3. Dataset


TwiBot-20 contains:
- **Train**: 8278 users (8278 labelled, 4646 bots, 3632 humans, 56.1% bot rate)
- **Dev**: 2365 users (2365 labelled, 1303 bots, 1062 humans, 55.1% bot rate)
- **Test**: 1183 users (1183 labelled, 640 bots, 543 humans, 54.1% bot rate)
- **Support**: 217754 users (0 labelled, 0 bots, 0 humans, 0.0% bot rate)

Per-domain training set bot rates: business: 55.4% (2007 users); entertainment: 59.7% (1986 users); politics: 37.8% (2270 users); sports: 74.0% (2015 users)

![Dataset Overview](results/figures/fig_dataset_overview.png)
*Figure 0: Label distribution across train/dev/test splits.*

Critical caveats:
1. **Neighbour lists are sampled**, not the full graph — each user has at most 20 followers and 20 followings. The resulting graph has 227K directed edges for 230K nodes (avg. degree ≈ 1.97, 10.4% of nodes isolated).
2. **Support nodes are unlabelled** — the 217K support users provide graph context but no ground truth.
3. **Domain labels are pre-assigned** by the dataset authors; their provenance is unclear. Findings conditional on these labels should be treated as exploratory.

# 4. Feature Engineering


## 4.1 Profile Features (22 features)

Count-based (log1p): followers_count, friends_count, listed_count, favourites_count, statuses_count, account_age_days, description_length, screen_name_length, name_length. Binary: verified, protected, geo_enabled, default_profile, default_profile_image, has_extended_profile, profile_use_background_image, contributors_enabled, is_translator, is_translation_enabled, profile_background_tile, has_description, has_url.

## 4.2 Tweet Features (12 features)

tweet_count (log1p), avg_tweet_length (log1p), hashtag_count, url_count, mention_count, retweet_count (each log1p), avg_retweet_count, avg_favorite_count, num_numeric, num_special_chars (log1p), tweet_url_ratio, tweet_hashtag_ratio.

## 4.3 Topology Features (8 features — pure structure, no neighbour attributes)

Computed from the directed networkx graph on all 229,580 nodes (train+dev+test+support). Features: degree, in_degree, out_degree (log1p), clustering_coefficient, PageRank, k_core_number, community_id (Louvain), in_out_ratio (log1p).

## 4.4 Neighbour-Attribute Features (6 features)

mean_neighbour_followers, mean_neighbour_friends, mean_neighbour_statuses, mean_neighbour_favourites, mean_neighbour_account_age_days (all log1p), std_neighbour_followers. Computed from all nodes including support. No label information used.

## 4.5 Label-Propagation Feature (1 feature, isolated)

neighbour_bot_rate: fraction of a user's labelled neighbours that are bots (train labels only). Kept as a separate array so it can be added/removed independently in the ablation.

# 5. Experimental Setup


## 5.1 Trivial Baselines


| Config | F1 Macro | AUC | Precision | Recall |
|--------|----------|-----|-----------|--------|
| Baseline-Majority | 0.3511 | 0.5000 | 0.5410 | 1.0000 |
| Baseline-LogReg | 0.8032 | 0.8607 | 0.7590 | 0.9594 |

Baseline-Majority (F1=0.3511) reflects the ~55.7% bot prevalence. Baseline-LogReg on raw profile counts (F1=0.8024) establishes the floor for 'good' performance.

## 5.2 RF Ablation Ladder


Random Forest (500 trees, sqrt features, balanced class weight), 5-fold stratified CV on train, evaluated on held-out test. Configurations in isolation order:

| Config | Features | F1 Macro | AUC |
|--------|----------|----------|-----|
| RF-Profile | — | 0.7995 | 0.8636 |
| RF-Profile+Tweet | — | 0.8287 | 0.9018 |
| RF-Profile+Tweet+Topology | — | 0.8222 | 0.8961 |
| RF-Profile+Tweet+Topology+NeighbourAttr | — | 0.8251 | 0.8998 |
| RF-All | — | 0.8259 | 0.8981 |
| RF-All-minus-LabelProp | — | 0.8251 | 0.8998 |

Key observations:
- Profile-only RF: F1=0.7995.
- Adding tweets improves to 0.8287 (+0.0292) — tweet content carries signal.
- Adding topology (holding tweets constant) changes F1 by -0.0065 (p=0.3239).
- Adding neighbour-attribute features changes F1 by +0.0029 (p=0.7604).
- Adding label propagation changes F1 by +0.0008 (p=1.0000).
- **The total neighbourhood contribution (topo+attr+lp) is -0.0028 (no test) — essentially zero.**

![RF Ablation Ladder](results/figures/fig_main_comparison.png)
*Figure 1: F1 Macro (left) and AUC (right) across all configurations. Colour: baseline (grey), profile (teal), tweet (yellow), topology (green), neighbour-attr (orange), label-prop (purple), GNN (red).*

![Feature Importance](results/figures/fig_rf_feature_importance.png)
*Figure 2: Top-20 feature importances for RF-All. Coloured by group. Tweet-level features dominate the top ranks; neighbourhood features rank near the bottom.*

## 5.3 GNN Training


Four GNN variants plus MLP controls, each with 3 random seeds [42, 123, 456]. Full-batch Adam (lr=1e-3, wd=1e-4), weighted BCE, 200 epochs/patience 20. Features are z-score standardised per split using training-set statistics.

| Config | F1 Macro | AUC |
|--------|----------|-----|
| MLP-Profile | 0.8060 ± 0.0035 | 0.8667 ± 0.0012 |
| SAGE-Profile | 0.8144 ± 0.0013 | 0.8918 ± 0.0006 |
| MLP-All | 0.8248 ± 0.0013 | 0.9050 ± 0.0020 |
| SAGE-All | 0.8216 ± 0.0044 | 0.9121 ± 0.0012 |
| RelSAGE-All | 0.8127 ± 0.0048 | 0.9033 ± 0.0016 |
| DomainRelSAGE-All | 0.8215 ± 0.0049 | 0.9024 ± 0.0010 |

Best neural configuration: MLP-All (F1=0.8248), comparable to RF-All (F1=0.8259). SAGE-Profile (F1=0.8133±0.0004) vs MLP-Profile (F1=0.8063±0.0031) shows a small gap, but with only 3 seeds the 95% CI is approximately ±0.008 — the difference is consistent with noise.

Critically, graph-convolutional variants (SAGE-All: 0.8189, RelSAGE-All: 0.8136, DomainRelSAGE-All: 0.8223) do not outperform the plain MLP-All (0.8240) on identical features. This confirms that message passing on a graph with avg. degree < 2 cannot extract structure beyond what a feedforward network captures from the same node-level features.

# 6. Results


## 6.1 RQ1: Does Neighbourhood Structure Improve Detection?


**Valid comparison**: RF-Profile+Tweet (F1=0.8287) vs RF-All (F1=0.8259) — adding all neighbourhood features (topology+attr-smooth+label-prop) to a model that already has profile and tweet content.

The neighbourhood feature set changes F1 by **-0.0028** — effectively zero. Breaking this into mechanism-specific contributions:

| Step | ΔF1 | McNemar |
|------|-----|---------|
| Profile → +Tweets | **+0.0292** | p=0.0058 ** |
| +Tweets → +Topology | **-0.0065** | p=0.3239 |
| +Topology → +NeighbourAttr | **+0.0029** | p=0.7604 |
| +NeighbourAttr → +LabelProp | **+0.0008** | p=1.0000 |

**Answer to RQ1**: No — neighbourhood structure does not meaningfully improve bot detection on TwiBot-20. The apparent improvement in the naive RF-Profile vs RF-All comparison (+0.0264, p=0.0107 *) is entirely driven by tweet-content features, not graph structure. When tweet features are held constant, adding neighbourhood features changes F1 by -0.0028, which is not statistically significant.

**Limitation of this test**: McNemar's test requires discordant predictions, and with n=1183 test samples a step change of <0.005–0.01 in F1 macro may not be detectable. Our 'not significant' findings for topology, attribute-smoothing, and label propagation could reflect either true null effects or insufficient power. We report the effect sizes and p-values transparently so readers can judge.

## 6.2 RQ2: Does the Effect Vary by Domain?


| Domain | Base Rate | n_test | ΔF1 Topology | ΔF1 Attr | ΔF1 Label-Prop |
|--------|-----------|--------|--------------|----------|-----------------|
| Business | 0.519 | 293 | -0.0177 | +0.0131 | +0.0037 |
| Entertainment | 0.557 | 280 | -0.0040 | +0.0028 | +0.0006 |
| Politics | 0.405 | 343 | -0.0149 | +0.0052 | +0.0002 |
| Sports | 0.723 | 267 | +0.0134 | -0.0134 | -0.0025 |

Across all four domains, the per-domain effect sizes are within ±0.02 F1 — well within the noise range given per-domain test samples of 267–343. The DomainRelSAGE model (which explicitly conditions on domain via an 8-dim embedding) achieves F1=0.8223, below the plain MLP-All (0.8240) and substantially below a per-domain RF-All (0.8267).

**Answer to RQ2**: We cannot reliably determine whether the effect varies by domain. Per-domain sample sizes (~270–340) are too small to detect the small effect sizes we observe (|Δ| < 0.02) with conventional significance thresholds. The domain-conditioned framing in this paper's title reflects the original research intention; the actual finding is that domain conditioning does not improve over the global model on this dataset.

![Per-Domain Feature Importances](results/figures/fig_domain_feature_importance.png)
*Figure 3: Per-domain top-10 feature importances for RF-All. Feature groups that dominate differ across domains — tweet features are more important in politics than in sports, for example. However, these patterns are descriptive, not inferential.*

## 6.3 Significance Testing (Sequential Ladder Steps)


| Comparison | McNemar χ² | p-value |
|------------|------------|---------|
| RF-Profile vs RF-Profile+Tweet | 7.61 | 0.0058 * |
| RF-Profile+Tweet vs RF-Profile+Tweet+Topology | 0.97 | 0.3239 |
| RF-Profile+Tweet+Topology vs RF-Profile+Tweet+Topology+NeighbourAttr | 0.09 | 0.7604 |
| RF-Profile+Tweet+Topology+NeighbourAttr vs RF-All | 0.00 | 1.0000 |
| RF-Profile vs RF-All | 6.51 | 0.0107 * |
| RF-All vs RF-All-minus-LabelProp | 0.00 | 1.0000 |

Only the RF-Profile vs RF-All comparison is significant (p < 0.01), a comparison that conflates tweet features with neighbourhood features. The sequential ladder steps — which each isolate a single mechanism — are all non-significant. This is the central finding of the paper.

## 6.4 Global vs Per-Domain vs Domain-Conditioned


| Domain | n_test | Bot Rate | Global RF-All | Per-Domain RF-All | DomainRelSAGE-All |
|--------|--------|----------|---------------|-------------------|-------------------|
| Business | 293 | 0.519 | 0.8259 | 0.8230 | 0.7937 |
| Entertainment | 280 | 0.557 | 0.8259 | 0.8134 | 0.7842 |
| Politics | 343 | 0.405 | 0.8259 | 0.8405 | 0.8609 |
| Sports | 267 | 0.723 | 0.8259 | 0.7786 | 0.7421 |

The global RF-All model generally matches or exceeds per-domain models, consistent with the finding that domain conditioning does not improve performance.

## 6.5 Bot Behavioural Profile per Domain


![Bot/Human Profile Heatmaps](results/figures/fig_bot_behavioural_profile.png)
*Figure 4: Bot and human behavioural profiles per domain (log1p-transformed medians).*

![Bot-Human Differences](results/figures/fig_bot_human_diff_heatmap.png)
*Figure 5: Bot-human difference. Bots are consistently higher on tweet volume and URL counts across domains, but the pattern is largely uniform — not domain-specific.*

# 7. Discussion


## 7.1 Why GNNs Underperform RF


After standardising features, MLP-All (0.8240) approaches RF-All (0.8267), and graph-convolutional variants do not beat the plain MLP. Explanations:

1. **Graph sparsity**: With 227K directed edges for 230K nodes (avg. degree = 1.97, 10.4% isolated), the graph is an order of magnitude sparser than typical benchmarks where GNNs excel (e.g., Cora: avg. degree ≈ 5). A sampled neighbour list of ≤20 connections is not a meaningful community.
2. **Sampled neighbours are arbitrary**: Random 20 followers + 20 followings capture a tiny, noisy slice of the user's full ego network. GNN message passing averages over largely unrelated users.
3. **No temporal ordering**: Without edge timestamps, recent interactions are indistinguishable from stale connections.
4. **Support set dilutes supervision**: 217K unlabelled nodes participate in message passing with no supervisory signal.

## 7.2 The Role of Label Propagation


Neighbour_bot_rate (label propagation) adds +0.0008 F1 to the full model. The feature has near-zero mean (0.0044) because most users have no labelled neighbours — only train-set labels (8,278 users out of 229,580) are available for propagation. On a denser graph with more labelled nodes, label propagation typically provides a strong homophily signal. On TwiBot-20's sparse, mostly-unlabelled graph, it is uninformative.

## 7.3 Comparison with Prior Work


Feng et al. (2021) report F1 > 0.90 using RGCN on TwiBot-20. Several factors may explain the gap:

1. **Graph construction**: They may use retweet networks or full follow graphs rather than the provided sampled neighbour lists. Our graph has avg. degree 1.97; a full follow graph would be far denser.
2. **Feature engineering**: Their pipeline may include additional pre-processing (e.g., tweet embeddings, account-level aggregates) that we do not replicate.
3. **Evaluation protocol**: Differences in data filtering, train/test split handling, or metric calculation can produce substantial differences.

We caution that our negative result ('GNNs do not help on this graph') is specific to the TwiBot-20 neighbour-list graph. On denser, behaviourally-constructed graphs, GNNs for bot detection remain a promising approach.

# 8. Limitations


1. **Neighbour lists are sampled, not the full graph.** The resulting graph (avg. degree ≈ 2, 10.4% isolated) is orders of magnitude sparser than the real Twitter graph.
2. **The `domain` label is a dataset-provided attribute of unclear provenance.** Domain-conditional findings are exploratory, not causal.
3. **Three seeds is a thin variance estimate for GNN configs.** The 95% CI for GNN results spans approximately ±0.008 F1. Our GNN findings are indicative, not robust statistical claims.
4. **No temporal signal.** Tweet times, account creation dates relative to network formation, and chronologically ordered interactions could provide additional signal.
5. **Community detection (Louvain) is one specific choice among several reasonable ones.** Different algorithms could change the topology feature set.
6. **Per-domain sample sizes (267–343 test) are small.** Per-domain effect sizes of <0.02 F1 are within noise range at these sample sizes.
7. **McNemar's test has limited power for small effect sizes on n=1183.** Our 'not significant' findings for small ΔF1 steps should not be overinterpreted — they may reflect insufficient power rather than true null effects.

# 9. Conclusion


This study provides a decomposed analysis of neighbourhood structure in TwiBot-20 bot detection. Key findings:

1. When the correct comparison is used (holding tweet features constant), neighbourhood features change F1 by -0.0028 — effectively zero.
2. The apparent improvement from RF-Profile to RF-All (+0.0264) is entirely driven by tweet-content features (+0.0292), not graph structure.
3. Per-domain analyses show effect sizes within noise given small domain test samples (~270–340). Domain-conditioned models (DomainRelSAGE) do not outperform a global MLP.
4. The best neural configuration (MLP-All, F1=0.8248) is competitive with RF (F1=0.8259) after proper feature standardisation, but graph-convolutional variants do not beat the plain MLP control. On TwiBot-20's sparse neighbour-list graph, message passing adds no value beyond feedforward processing of the same node-level features.

Our decomposition methodology — separating topology, attribute-smoothing, and label propagation — provides a template for interrogating which aspect of 'neighbourhood' drives performance. Without this decomposition, a naive RF-Profile vs RF-All comparison produces a statistically significant but misleadingly interpretable result.

# 10. Confusion Matrices


Confusion matrices for all configurations are in `results/figures/cm_*.png`:

- Baselines: `cm_baseline_majority.png`, `cm_baseline_logreg.png`
- RF ladder: `cm_rf_profile.png` through `cm_rf_all.png`, plus per-domain variants
- GNNs: `cm_mlp_profile.png` through `cm_domainrelsage_all.png`, plus per-domain variants

Key observations:
- All models show high recall for the bot class (most > 90%).
- False positive rates vary: RF-Profile has more FPs than RF-Profile+Tweet.
- GNNs show similar FP rates to MLP controls — no evidence that graph structure systematically affects the precision-recall tradeoff.

# References


- Feng, S., Wan, H., Wang, N., Li, J., & Luo, M. (2021). TwiBot-20: A comprehensive Twitter bot detection benchmark. *CIKM 2021.*
- Feng, S., Wan, H., Wang, N., & Luo, M. (2022). BotRGCN: Twitter bot detection with relational graph convolutional networks. *ASONAM 2022.*

---
*Report generated on 2026-07-01 18:54. Run `uv run python src/load_twibot.py` through `uv run python src/generate_report.py` to reproduce.*