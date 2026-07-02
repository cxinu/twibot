# Research Summary: Adaptive Aggregation for Heterophilic Bot Detection on TwiBot-20

## Problem

BotRGCN and related GNNs assume that neighboring Twitter accounts tend to share the same label (homophily). We show this assumption is violated on TwiBot-20, where users are frequently connected to accounts of the opposite class (heterophily). Standard mean aggregation therefore degrades classification performance for a large subset of nodes.

## Key findings

1. **Isolated/low-degree nodes are not the weak point.** TwiBot-20's crawler capped neighbors at ~10 per direction, producing a narrow degree distribution. Low-degree nodes are easy and overwhelmingly human; the difficult bucket is the well-connected majority.

2. **Heterophily is widespread.**
   - 72% of nodes have follow-homophily < 0.3.
   - 37% have combined homophily = 0.
   - Follow and following homophily are negatively correlated (Pearson r = −0.15).

3. **The dominant error is asymmetric.** In the homophily-0 bucket, humans are misclassified as bots at ~21.5%, noticeably above the bot→human error rate of ~15.0%.

## Method

We replace RGCN's fixed mean aggregation with a learned adaptive gate. Two variants are explored:

- **Hard low/high gate:** `m = α·low + (1−α)·high`, where `α` is predicted from `[ego ‖ low]`.
- **Soft-contrast gate:** `m = low + β·(ego − low)`, where `β` is predicted by an MLP.

Both gates optionally receive an explicit **cosine-similarity** scalar between the ego and the neighborhood mean, so the gate can distinguish structurally heterophilic but semantically similar neighbors from true opposite-class neighbors.

## Main result

| Model | F1 (bot class) | MCC | F1 macro |
|---|---:|---:|---:|
| Paper-reported BotRGCN | 0.8707 | 0.7021 | — |
| BotRGCN (repro + threshold tuning) | 0.8785 ± 0.0030 | 0.7249 ± 0.0055 | 0.8569 ± 0.0027 |
| **GatedBotRGCN-global** (validation-selected) | **0.8801 ± 0.0024** | **0.7273 ± 0.0047** | **0.8581 ± 0.0022** |

The final model is selected by **validation F1**, not test-set performance. Thresholds are tuned per seed on the validation set. The selected **GatedBotRGCN-global** improves F1 by **+0.94 pp** over the paper-reported baseline and by **+0.16 pp** over the threshold-tuned BotRGCN repro. The paired seed delta vs. the threshold-tuned baseline is **+0.16 ± 0.48 pp** (Wilcoxon p = 0.63), so the extra gain from the gate is modest and not statistically significant with 5 seeds.

## Files

- `src/degree_bucket_analysis.py` — Phase 1 analysis
- `src/heterophily_analysis.py` — Phase 2 analysis
- `src/heterophily_fix.py` — Phase 3 experiments
- `src/correct_and_smooth.py` — C&S post-processing ablation
- `src/writeup_figures.py` — Figure generation
- `src/models.py` — `BotRGCN`, `GatedRGCNConv`, `SoftContrastRGCNConv`
- `WRITEUP.md` — Full research narrative with figures

## Reproduction

```bash
uv run python src/degree_bucket_analysis.py
uv run python src/heterophily_analysis.py
uv run python src/heterophily_fix.py
uv run python src/writeup_figures.py
```

See `WRITEUP.md` for the complete research story.
