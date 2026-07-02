# Research Summary: Adaptive Aggregation for Heterophilic Bot Detection on TwiBot-20

## Problem

BotRGCN and related GNNs assume that neighboring Twitter accounts tend to share the same label (homophily). We show this assumption is violated on TwiBot-20, where users are frequently connected to accounts of the opposite class (heterophily). Standard mean aggregation therefore degrades classification performance for a large subset of nodes.

## Key findings

1. **Isolated/low-degree nodes are not the weak point.** TwiBot-20's crawler capped neighbors at ~10 per direction, producing a narrow degree distribution. Low-degree nodes are easy and overwhelmingly human; the difficult bucket is the well-connected majority.

2. **Heterophily is widespread.**
   - 72% of nodes have follow-homophily < 0.3.
   - 37% have combined homophily = 0.
   - Follow and following homophily are negatively correlated (Pearson r = −0.15).

3. **The dominant error is asymmetric.** In the homophily-0 bucket, humans are misclassified as bots at ~27.6%, nearly double the bot→human error rate of ~15.5%.

## Method

We replace RGCN's fixed mean aggregation with a learned **soft-contrast gate**:

$$m_r(v) = \mathrm{low}_r(v) + \beta_r(v) \cdot (W_r h_v - \mathrm{low}_r(v))$$

where `β` is computed by an MLP from the concatenation of the neighborhood mean and the transformed ego. The model starts at `β ≈ 0` (standard RGCN) and learns to trust the ego more in heterophilic neighborhoods.

## Main result

| Model | F1 (bot class) | MCC | F1 macro |
|---|---:|---:|---:|
| BotRGCN | 0.8447 ± 0.0037 | 0.6484 ± 0.0051 | 0.8223 ± 0.0020 |
| GatedBotRGCN-rel | 0.8490 ± 0.0044 | **0.6577 ± 0.0073** | **0.8269 ± 0.0028** |
| **SoftContrastBotRGCN-global** | **0.8495 ± 0.0012** | 0.6560 ± 0.0032 | 0.8252 ± 0.0018 |

The best method improves overall F1 (bot class) by **+0.48 pp** and MCC by **+0.93 pp** over BotRGCN (5-seed averages). The improvement is concentrated in the majority of well-connected nodes rather than specifically in the most heterophilic bucket. The gate is global, not relation-specific: a single MLP outperforms separate follow/following gates by most metrics.

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
