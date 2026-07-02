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

We replace RGCN's fixed mean aggregation with a learned **soft-contrast gate**:

$$m_r(v) = \mathrm{low}_r(v) + \beta_r(v) \cdot (W_r h_v - \mathrm{low}_r(v))$$

where `β` is computed by an MLP from the concatenation of the neighborhood mean and the transformed ego. The model starts at `β ≈ 0` (standard RGCN) and learns to trust the ego more in heterophilic neighborhoods.

## Main result

| Model | F1 (bot class) | MCC | F1 macro |
|---|---:|---:|---:|
| Paper-reported BotRGCN | 0.8707 | 0.7021 | — |
| BotRGCN (our repro) | 0.8708 ± 0.0017 | 0.7124 ± 0.0038 | 0.8557 ± 0.0019 |
| GatedBotRGCN-global | 0.8694 ± 0.0060 | 0.7106 ± 0.0086 | 0.8545 ± 0.0037 |
| **GatedBotRGCN-rel** | **0.8755 ± 0.0052** | **0.7199 ± 0.0090** | **0.8572 ± 0.0038** |
| SoftContrastBotRGCN-global | 0.8752 ± 0.0042 | 0.7163 ± 0.0085 | 0.8549 ± 0.0034 |
| SoftContrastBotRGCN-rel | 0.8683 ± 0.0053 | 0.7065 ± 0.0084 | 0.8521 ± 0.0035 |

The best method, **GatedBotRGCN-rel**, improves overall F1 (bot class) by **+0.47 pp** and MCC by **+0.75 pp** over the paper-matched BotRGCN baseline (5-seed averages). With the original RoBERTa features and training recipe our BotRGCN baseline reproduces the paper's reported F1 (0.8707) almost exactly.

## Files

- `src/degree_bucket_analysis.py` — Phase 1 analysis
- `src/heterophily_analysis.py` — Phase 2 analysis
- `src/heterophily_fix.py` — Phase 3 experiments
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
