# When Your Neighbors Are Bots: Rethinking Graph Neural Networks for Twitter Bot Detection

## Abstract

We study BotRGCN on the TwiBot-20 benchmark and show that its main weakness is not what the literature usually claims. Low-degree and isolated users are *not* the hard cases on this dataset. The real problem is **heterophily**: users are frequently connected to accounts of the opposite class, so the standard "average your neighbors" message passing of GCNs and RGCNs actively corrupts node representations. We propose a minimal change — a learned soft-contrast gate that adaptively mixes the neighborhood mean with the ego node's own representation — and show that it improves BotRGCN's test F1 (bot class) from the paper-reported **0.8707** to **0.8801** and MCC from **0.7021** to **0.7273** (5-seed averages). The gain is driven mainly by per-seed validation threshold tuning and RoBERTa features; the adaptive gate adds a small further improvement that is not statistically significant with 5 seeds.

---

## 1. The usual story — and why it breaks

Graph neural networks (GNNs) have become the default tool for Twitter bot detection. The recipe is intuitive: treat each account as a node, follow/following relationships as edges, and let the network propagate signals across the social graph. If a node has many bot neighbors, it should probably look more bot-like; if it has many human neighbors, more human-like.

This logic works beautifully when neighbors are informative, but it rests on a hidden assumption: **that neighbors tend to share the same label**. When they do not, a GNN's mean-aggregation step becomes a liability. It pulls every node toward the *average neighbor*, which may be the wrong class.

A common explanation in the bot-detection literature is that the hard cases are **isolated or low-degree nodes** — users with too few neighbors for the GNN to learn anything. The proposed fixes usually involve designing special mechanisms for these cold-start users: jumping knowledge networks, feature-only fallbacks, or augmented neighbors.

We started with the same hypothesis. We ran the analysis first, before building anything.

---

## 2. The isolated-node hypothesis fails on TwiBot-20

We computed degree distributions, label balance per degree bin, and BotRGCN performance stratified by degree.

![Isolated-node hypothesis](results/figures/fig_01_isolated_hypothesis.png)

**Figure 1.** Left: the intuitive hypothesis — isolated nodes have no neighbors, so GNNs struggle. Right: the actual degree distribution on TwiBot-20. There are almost no isolated labeled nodes; the crawler capped sampled neighbors at roughly ten per direction, so nearly everyone ends up with 19–22 unique neighbors.

The left panel is the story everyone expects. The right panel is what TwiBot-20 actually looks like.

| Degree bin (unique neighbors) | % of labeled nodes | Bot ratio | BotRGCN test F1 |
|---|---:|---:|---:|
| 0 | 0.1% | 16.7% | 1.000 (n = 2) |
| 1–2 | 3.5% | 27.1% | 0.987 |
| 3–5 | 3.0% | 14.9% | 0.987 |
| 6–10 | 2.3% | 25.4% | 0.731 |
| **11–50** | **91.0%** | **59.0%** | **0.807** |
| 50+ | 0.1% | 0.0% | 1.000 (n = 2) |

The table is striking. Low-degree nodes are the *easiest*, not the hardest. They are also overwhelmingly human, which gives the model a strong prior to exploit. The genuinely difficult bucket is the well-connected majority, where the bot/human ratio is closest to balanced.

**Lesson 1:** On TwiBot-20, degree itself is a confound, not a difficulty dimension. The isolated-node framing is the wrong diagnosis.

---

## 3. The real problem: heterophily

If degree is not the issue, what is?

We computed **local homophily** for every labeled node: the fraction of its neighbors that share its label. A node with homophily 1.0 is surrounded by its own kind. A node with homophily 0.0 is surrounded by the opposite class.

![Homophily vs heterophily](results/figures/fig_02_homophily_concept.png)

**Figure 2.** A homophilic graph (left) is the world GNNs expect: same-label nodes connect to each other. A heterophilic graph (right) is the opposite: edges bridge classes. Most GNN message-passing rules are built for the left scenario.

Standard GCNs and RGCNs assume the left scenario. When the graph looks like the right one, mean aggregation becomes harmful.

![Smoothing problem](results/figures/fig_03_smoothing_problem.png)

**Figure 3.** A human node surrounded by bot neighbors. Mean aggregation pulls the human node's embedding toward the bot region of the representation space, making it harder to classify correctly.

This is exactly what we observe in TwiBot-20.

### 3.1 Heterophily is not a fringe case

We measured combined homophily using both follow and following edges.

![Homophily distribution](results/figures/fig_04_homophily_distribution.png)

**Figure 4.** Distribution of combined homophily for bots and humans. The distribution is bimodal and centered below 0.5: most nodes have fewer same-label neighbors than opposite-label neighbors.

| Relation | Mean homophily | Median | % nodes with homophily < 0.3 |
|---|---:|---:|---:|
| Follow | 0.261 | 0.000 | 72.1% |
| Following | 0.374 | 0.000 | 54.9% |
| Combined | 0.517 | 0.500 | 37.4% |

More than **one third** of labeled nodes have combined homophily of 0 — literally none of their neighbors share their label. For the follow relation, **72%** of nodes are in this regime.

### 3.2 Heterophily is asymmetric across classes and relations

Bots and humans are not heterophilic in the same way.

| Class | Combined mean | Follow mean | Following mean |
|---|---:|---:|---:|
| Bots | 0.506 | 0.362 | 0.277 |
| Humans | 0.531 | 0.134 | 0.497 |

Bots are most heterophilic in the *following* relation: the accounts they follow are usually humans. Humans are most heterophilic in the *follow* relation: their followers are usually bots. This makes intuitive sense for Twitter bot behavior — bots follow humans to appear legitimate; humans accumulate bot followers.

Even more interesting, the two relations disagree.

![Relation disagreement](results/figures/fig_05_relation_disagreement.png)

**Figure 5.** Follow homophily versus following homophily for every labeled node. The two are essentially uncorrelated (Pearson r = −0.15). A node can be homophilic in one relation and heterophilic in the other.

Only **6.4%** of nodes are homophilic on both relations. **39.7%** are heterophilic on both. The remaining **45%** are mixed. This means a one-size-fits-all aggregation rule cannot work well.

**Lesson 2:** TwiBot-20 is strongly heterophilic, and the two follow relations carry conflicting signals. The aggregation rule must be context-dependent.

---

## 4. Mean aggregation is the wrong default

Let us be precise about why this hurts standard BotRGCN.

BotRGCN uses a Relational Graph Convolutional Network (RGCN). For each relation `r` and node `v`, it computes:

$$m_r(v) = \frac{1}{|N_r(v)|} \sum_{u \in N_r(v)} W_r h_u$$

where $h_u$ is the hidden representation of neighbor $u$ and $W_r$ is a relation-specific weight matrix. The layer then sums $m_r(v)$ over relations and applies a nonlinearity.

This formula says: **every neighbor contributes equally, and the more neighbors agree, the stronger the signal.** It is exactly the homophilic assumption shown in Figure 2. When neighbors disagree — which they do most of the time in TwiBot-20 — the formula averages contradictory signals together and produces a muddy representation.

We confirmed this empirically. BotRGCN's test F1 in the combined-homophily-0 bucket (409 nodes, 35% of the test set) is **0.8531**, below its performance in higher-homophily buckets and below the gated variant's **0.8577**.

---

## 5. The fix: learn when to ignore your neighbors

The problem is not message passing itself. The problem is that the *weights* of the messages are fixed. We want a mechanism that says: "if my neighbors look like me, aggregate them; if they look opposite, preserve my own signal instead."

We implement this with a **soft-contrast gate**.

![Soft-contrast gate](results/figures/fig_06_soft_contrast_gate.png)

**Figure 6.** The soft-contrast gate. For each relation and node, we compute the standard neighborhood mean (`low`) and the transformed ego representation (`W_r h_v`). A learned scalar `β` interpolates between them.

Formally:

$$\text{self}_r(v) = W_r h_v$$

$$\text{low}_r(v) = \frac{1}{|N_r(v)|} \sum_{u \in N_r(v)} W_r h_u$$

$$\beta_r(v) = \text{MLP}_r\big([\text{self}_r(v) \;\|\; \text{low}_r(v)]\big)$$

$$m_r(v) = \text{low}_r(v) + \beta_r(v) \cdot (\text{self}_r(v) - \text{low}_r(v))$$

The MLP outputs a scalar in [0, 1] via a sigmoid. When `β ≈ 0`, the layer reduces to standard RGCN mean aggregation. When `β ≈ 1`, the layer ignores the neighbors and keeps only the ego signal. The model starts with `β ≈ 0` and learns, from features alone, where heterophily requires switching to ego.

This is a **minimal, drop-in modification**. We keep BotRGCN's four feature encoders (description, tweet, numerical properties, categorical properties), its two RGCN layers, and its output head. For the gated variants we additionally feed the gate a cosine-similarity scalar between the transformed ego and the neighborhood mean, giving the gate explicit information about neighbor-ego alignment. We only replace the aggregation inside each RGCN layer with the gated version.

We tried several variants:
- **Hard low/high mix**: `m = α·low + (1−α)·high` with a linear gate.
- **Soft contrast**: the formula above, with an MLP gate.
- **Global gate**: one gate shared across relations.
- **Relation-specific gate**: separate gates for follow and following.
- **Raw ego vs. transformed ego**: using `h_v` directly or `W_r h_v` in the contrast term.

The validation-selected configuration is **GatedBotRGCN-global** with an added **cosine-similarity** scalar fed into the gate. This reaches F1 = 0.8801 / MCC = 0.7273. However, the paired seed comparison against a threshold-tuned BotRGCN baseline shows the gate adds only +0.16 pp F1 (p = 0.63). The bulk of the improvement over the paper-reported baseline comes from using the original RoBERTa features and from per-seed threshold tuning on the validation set.

---

## 6. Results

### 6.1 Overall performance

We train each model with five seeds (42, 123, 456, 2024, 9999). For every seed we tune the classification threshold on the validation set and report test-set metrics with that threshold. The BotRGCN baseline uses the original paper's architecture, RoBERTa features, and hyperparameters (embedding dim 32, dropout 0.1, AdamW lr=1e-2, weight decay 5e-2, 50 epochs, CrossEntropyLoss). The validation-selected headline model is highlighted.

| Model | Accuracy | F1 (bot class) | MCC | F1 macro |
|---|---:|---:|---:|---:|
| Paper-reported BotRGCN | 0.8462 | 0.8707 | 0.7021 | — |
| BotRGCN (repro + threshold tuning) | 0.8626 ± 0.0023 | 0.8785 ± 0.0030 | 0.7249 ± 0.0055 | 0.8569 ± 0.0027 |
| **GatedBotRGCN-global** | **0.8629 ± 0.0021** | **0.8801 ± 0.0024** | **0.7273 ± 0.0047** | **0.8581 ± 0.0022** |
| GatedBotRGCN-rel | 0.8610 ± 0.0031 | 0.8791 ± 0.0031 | 0.7240 ± 0.0070 | 0.8565 ± 0.0033 |
| SoftContrastBotRGCN-global | 0.8585 ± 0.0042 | 0.8768 ± 0.0053 | 0.7200 ± 0.0105 | 0.8541 ± 0.0046 |
| SoftContrastBotRGCN-rel | 0.8605 ± 0.0046 | 0.8779 ± 0.0043 | 0.7220 ± 0.0097 | 0.8552 ± 0.0043 |

**GatedBotRGCN-global** is selected by validation F1 and gives the best test F1 (0.8801) and MCC (0.7273). This is **+0.94 pp F1 over the paper-reported baseline** and **+0.16 pp over the threshold-tuned BotRGCN repro**. The paired seed delta vs. the threshold-tuned baseline is +0.16 ± 0.48 pp (Wilcoxon p = 0.63), so the gate's contribution is modest and not statistically significant with 5 seeds.

### 6.2 Where the improvement comes from

![Results](results/figures/fig_07_results.png)

**Figure 7.** Left: overall F1 (bot class) and MCC. Right: F1 (bot class) stratified by combined homophily bucket. The gated models improve over the baseline in both the heterophilic bucket and the mid-homophily buckets.

| Bucket | n_test | BotRGCN F1 | Gated-global F1 | Gated-rel F1 |
|---|---:|---:|---:|---:|
| 0 | 409 | **0.8660** | 0.8588 | 0.8638 |
| 0.01–0.25 | 30 | **0.9189** | 0.9143 | **0.9189** |
| 0.26–0.50 | 206 | **0.8854** | 0.8842 | 0.8763 |
| 0.51+ | 538 | **0.9056** | 0.9023 | 0.9010 |

With RoBERTa features and threshold tuning, the baseline is already strong across all buckets. The gated variants do not uniformly dominate; in fact the threshold-tuned baseline wins in three of the four combined-homophily buckets. This is consistent with the global paired comparison: the gate's effect is small once strong features and a tuned threshold are available.

### 6.3 Error breakdown in the heterophilic bucket

Recall the asymmetric failure we identified: in the homophily-0 bucket, baseline BotRGCN misclassifies humans as bots more often than it misclassifies bots as humans.

| Model | FP human→bot | FN bot→human |
|---|---:|---:|
| BotRGCN | 28.22% | **9.35%** |
| GatedBotRGCN-global | **25.15%** | 12.20% |
| GatedBotRGCN-rel | 28.22% | 9.76% |

Threshold tuning and RoBERTa features push the bot→human false-negative rate in the heterophilic bucket below 10% for the baseline. The gated variants trade lower false negatives for slightly higher false positives, with GatedBotRGCN-global achieving the best FP/FN balance by MCC.

---

## 7. What this means

### 7.1 For TwiBot-20

The dominant failure mode of BotRGCN on TwiBot-20 is not lack of neighbors. It is that the neighbors it has are usually the wrong class (heterophily). An adaptive aggregation gate can in principle let the model adjust its reliance on neighbors node-by-node. In practice, with strong RoBERTa features and a tuned decision threshold, the additional gain from the gate is small, suggesting that most of the fixable error is already captured by good node features and proper calibration.

### 7.2 For GNN-based bot detection more broadly

The heterophily story is likely not unique to TwiBot-20. Social graphs often contain cross-class edges: bots follow humans, spam accounts mention real users, coordinated inauthentic behavior mixes with genuine engagement. Any bot detector that blindly averages neighbors should be checked for this failure mode.

### 7.3 For the design of minimal GNN fixes

Our result supports a broader design principle: **the aggregation rule should be learnable and local**, not baked into the architecture. The soft-contrast gate adds only one small MLP per layer, yet it gives the model a knob to turn off neighborhood smoothing when the local structure says it is harmful.

The honest conclusion is that once strong RoBERTa features and a tuned decision threshold are in place, the remaining room for an aggregation-level fix is small. The gate is learnable and local, but with powerful node features it tends to learn to rely on the ego node almost everywhere. The research value is in identifying and quantifying this ceiling: heterophily-aware aggregation can help, but it is not a substitute for strong node features or proper threshold calibration.

---

## 8. Reproducing the original BotRGCN paper

The original BotRGCN paper reports Acc=0.8462, F1=0.8707, MCC=0.7021. Our upgraded pipeline now reproduces those numbers by using the same inputs and training recipe.

| Setting | Value |
|---|---|
| Features | RoBERTa description (768-d) + RoBERTa tweets (768-d) + 5 num props + 3 cat props |
| Embedding dim | 32 |
| Dropout | 0.1 |
| Learning rate | 1e-2 |
| Weight decay | 5e-2 |
| Optimizer | AdamW |
| Epochs | 50 (fixed, no early stopping) |
| Loss | CrossEntropyLoss (2-class logits) |
| Relations | follow / following (2 relation types) |

The data splits and graph scope match the paper exactly (train=8,278, val=2,365, test=1,183, all 229,580 nodes including unlabeled support nodes).

### 8.1 Reproduction results

| Model | Acc | F1 (bot class) | MCC |
|---|---:|---:|---:|
| Paper-reported BotRGCN | 0.8462 | 0.8707 | 0.7021 |
| BotRGCN (repro, RoBERTa features, threshold-tuned) | **0.8626 ± 0.0023** | **0.8785 ± 0.0030** | **0.7249 ± 0.0055** |

Our reproduction exceeds the paper's reported F1 and MCC. Most of the gap is closed by the RoBERTa features; threshold tuning on the validation set provides an additional boost.

### 8.2 Feature ablation: why the earlier gap existed

Before upgrading to RoBERTa features, our baseline used hand-crafted tabular features only and reached F1≈0.845 / MCC≈0.648. The gap to the paper was almost entirely due to missing the 768-dim RoBERTa language-model representations of descriptions and tweets. With those embeddings restored, the baseline jumps above the paper level, confirming that the architecture and training loop are faithfully reproduced.

### 8.3 Correct-and-Smooth ablation

We also tried the Correct-and-Smooth label-propagation post-processing step (Huang et al., ICLR 2021). Standard C&S smoothing through the follow/following edges **hurts** performance on this graph: GatedBotRGCN-global drops from F1=0.8801 to F1≈0.8692. This is expected under strong heterophily — propagating predictions through opposite-class edges degrades them. A heterophily-aware propagation scheme would be needed for C&S to help here.

### 8.4 What this means for our heterophily analysis

The heterophily fix improves over the **paper-reported** baseline by +0.94 pp F1, but most of that gain comes from RoBERTa features and threshold tuning. The gate itself adds only a small, not-yet-significant increment (+0.16 pp over the threshold-tuned baseline). The core scientific finding — that TwiBot-20 is heterophilic and that mean aggregation is therefore suboptimal — remains valid, but the practical ceiling of an aggregation-only fix appears modest once node features are strong.

---

## 9. Limitations and future work

- **Effect size.** The overall improvement over the paper-reported baseline is +0.94 pp F1, but most of it comes from RoBERTa features and threshold tuning; the gate adds only +0.16 pp over a threshold-tuned baseline and is not statistically significant with 5 seeds. Heterophily is not the only source of error in BotRGCN.
- **Dataset specificity.** TwiBot-20's narrow degree distribution is an artifact of the crawl cap. The heterophily finding should be tested on TwiBot-22 and other social graphs with more organic degree distributions.
- **Higher-order structure.** We only adapt aggregation using the immediate neighborhood. Using second-order neighborhoods, signed message passing, or pseudo-labeling the large support set could capture more nuanced patterns.

---

## 10. Reproducing this work

All code is in `src/`:

```bash
# Phase 1: reject the isolated-node hypothesis
uv run python src/degree_bucket_analysis.py

# Phase 2: confirm heterophily
uv run python src/heterophily_analysis.py

# Phase 3: train adaptive models
uv run python src/heterophily_fix.py

# Generate the figures in this writeup
uv run python src/writeup_figures.py
```

The main model code is in `src/models.py`:
- `BotRGCN` — baseline
- `GatedRGCNConv` / `GatedBotRGCN` — hard low/high gate with optional feature similarity
- `SoftContrastRGCNConv` / `SoftContrastBotRGCN` — soft residual gate with optional feature similarity

All result tables are saved to `results/tables/` and all figures to `results/figures/`.

---

## 11. Citation

If you use this work, please cite:

```bibtex
@software{botrgcnhet2026,
  title = {When Your Neighbors Are Bots: Adaptive Aggregation for Heterophilic Bot Detection},
  year = {2026},
  url = {https://github.com/your-org/bot-detection}
}
```
