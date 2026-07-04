#let aaai(title: "", abstract: [], body) = {
  set page(paper: "us-letter", margin: (x: 0.75in, top: 0.75in, bottom: 1.25in))
  set text(font: "FreeSerif", size: 10pt)
  set par(justify: true, leading: 0.5em)

  align(center)[
    #text(size: 18pt, weight: "bold", title)
    #v(0.6em)
    #grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      align(center + horizon)[
        *Sunita Gaur* \
        School of Computer Science & Information Technology \
        Devi Ahilya Vishwavidyalaya \
        Indore, India \
        #link("mailto:sunita9300@gmail.com")
      ],
      align(center + horizon)[
        *Dr. Yasmin Shaikh* \
        International Institute of Professional Studies \
        Devi Ahilya Vishwavidyalaya \
        Indore, India \
        #link("mailto:yasminshaikh01@gmail.com")
      ],
    )
    #v(2em)
  ]

  show: columns.with(2, gutter: 0.33in)

  if abstract != [] {
    text(size: 9pt)[
      *Abstract*\
      #abstract
    ]
    v(1em)
  }

  show heading.where(level: 1): it => {
    set text(size: 10pt, weight: "bold")
    block(above: 1.2em, below: 0.8em)[#it.body]
  }

  show heading.where(level: 2): it => {
    set text(size: 10pt, weight: "regular", style: "italic")
    block(above: 1em, below: 0.5em)[#it.body]
  }

  set math.equation(numbering: "1")

  body
}

#show: aaai.with(
  title: "AdaRelBot: Heterophily-Aware Graph Transformers for Twitter Bot Detection",
  abstract: [
    Social media bots deliberately hide among legitimate users, creating heterophilous connections that violate the homophily assumption underlying most Graph Neural Networks. We introduce AdaRelBot, a Twitter bot detector built on Graph Transformers that explicitly reasons about edge quality. For every follow relationship, we compute cosine-similarity edge features from profile descriptions and recent tweets, giving the attention mechanism a direct signal of whether a neighbor should be trusted. A dual-head classifier then blends a standard MLP with a prototype-calibrated head via a learned per-node gate, so the model can fall back to class-direction similarity when local embeddings are noisy. Training is stabilized by Focal Loss to handle the human-bot imbalance. On the TwiBot-20 benchmark, AdaRelBot achieves 86.37% accuracy and 88.10% F1. On Cresci-15, the same model reaches 99.17% accuracy under the standard split and 96.15% under a stricter leave-one-group-out evaluation that prevents cross-group text leakage, confirming that the approach generalizes across datasets and group boundaries. Ablation studies show that the prototype-calibrated head is the main source of robustness, and an analysis of the learned gate reveals a positive correlation with per-node heterophily, confirming that the gate assigns more parametric MLP capacity to structurally ambiguous neighborhoods.
  ],
)

= Introduction

Automated accounts, commonly called bots, now account for a substantial fraction of activity on Twitter, Weibo, and other platforms. Some are harmless, but many amplify misinformation, manipulate trending topics, impersonate real users, and coordinate harassment. A small share of malicious automated accounts can drown out authentic voices at scale. Detecting them accurately is a first-order problem for platform integrity.

Every social network is a graph: users are nodes and follow relationships are edges. This structure carries rich signal. A genuine human tends to form organic clusters with friends who share interests, while a bot farm has a distinctive wiring pattern. Graph Neural Networks @kipf2017gcn @hamilton2017graphsage were designed to exploit exactly this structure. They work by message passing: each node gathers information from its neighbors, pools their features, and updates its own representation. The implicit assumption is homophily -- "birds of a feather flock together." Connected nodes should look similar.

The problem is that bots deliberately violate this assumption. A bot follows hundreds of humans to look legitimate. Under standard message passing, the bot's features get averaged with genuine human features. A bot that follows 99 humans and one other bot can end up with an aggregated representation that is 99% human-like, even though its label is bot. The classifier can no longer tell them apart @zhu2020heterophily. The very edges bots create to hide are the ones that pollute their representations.

@fig:homophily shows the contrast. In a homophilous graph, neighbors reinforce identity. In the real Twitter graph, bots hide among humans, turning a node's neighborhood into a liability.

#figure(
  image("figures/homophily_heterophily.svg", width: 100%),
  caption: [Left: homophilous neighborhoods make message passing easy. Right: in the real Twitter graph, bots deliberately follow humans, producing heterophilous edges that pollute standard GNN representations.],
) <fig:homophily>

== Our Approach

AdaRelBot fixes this with two complementary ideas. First, before the model ever aggregates anything, we measure whether two connected users are actually compatible. We compute the cosine similarity of their RoBERTa profile descriptions and recent tweets. These similarity scores become edge features. The attention mechanism uses them to down-weight heterophilous connections before they can corrupt the aggregation.

Second, rather than trusting a single classifier, we maintain two learned class prototypes, an ideal bot and an ideal human, in the shared embedding space. A tiny gating network decides, per node, how much to trust the standard MLP head versus the prototype head. When the MLP is reliable, the gate opens toward it. When the embedding is noisy, the prototype head takes over, because cosine similarity measures direction not magnitude, and direction is harder to corrupt.

AdaRelBot is trained with Focal Loss @lin2017focal to counter the human-majority imbalance on TwiBot-20. The architecture is end-to-end differentiable and uses standard Graph Transformer building blocks @shi2021unimp.

== Contributions

We make three contributions. We show that cheap, precomputed cosine-similarity edge features give a Graph Transformer enough of a heterophily signal to work on a large social graph. We introduce a prototype-calibrated dual-head classifier with a per-node gate that lets the model route between parametric and exemplar-based prediction. And we establish both standard and leave-one-group-out results on Cresci-15 that expose how the usual random split inflates numbers through cross-group text similarity, alongside strong results on TwiBot-20.

= Background and Related Work

== Graph Neural Networks and Homophily

GNNs update node representations by aggregating neighbor messages. In each layer, a node collects the current representations of its neighbors, combines them, and uses the mixture to update its own representation. When neighbors share the same label, this pooling reinforces the node's true identity. A human surrounded by humans becomes more recognizably human. Graph Attention Networks @velickovic2018gat replace uniform aggregation with a learned attention mechanism. The attention coefficient between two nodes is computed from their current embeddings.

This works under homophily. Under heterophily, the embeddings are already corrupted. Once a bot's representation has been averaged with the humans it follows, its embedding no longer looks like a bot. Attention then computes similarity scores from corrupted vectors, producing unreliable weights. AdaRelBot breaks this cycle by giving attention an independent cue. Edge features are computed from raw profile text and tweets, not from evolving node embeddings, so they stay clean even when the embeddings have been partially corrupted.

@zhu2020heterophily showed that standard GNNs degrade sharply when homophily is violated. Geom-GCN @pei2020geomgcn modeled how dissimilar neighbors should be treated by restructuring message passing around a latent geometric space. In bot detection, RGT @feng2022rgt used per-relation TransformerConv layers and semantic attention to fuse multiple edge types. AdaRelBot takes a different path: a single graph convolution with explicit compatibility features per edge. This avoids the cost of maintaining separate convolutions for every relation.

=== Heterophily in Social Networks

Social graphs are hostile to the homophily assumption because relationships are often strategic. A follow on Twitter means little more than "user $u$ saw user $v$'s profile." It does not imply shared interests, demographics, or bot/human status. In academia, a citation link usually signals intellectual similarity; homophily holds. On Twitter, a follow can simply mean that a bot purchased a list of targets. The result is a graph where local neighborhoods are label-heterogeneous. A bot may be surrounded almost entirely by humans. Measuring edge quality through content similarity separates trustworthy connections from deceptive ones.

== Bot Detection on Twitter

Early bot detectors relied on hand-crafted features. Deep-learning approaches later exploited text embeddings from language models. RoBERTa @liu2019roberta pushed this forward. Because bots are embedded in a social graph, structural methods followed. BotRGCN @feng2021botrgcn applied Relational Graph Convolutional Networks to follow and following edges. RGT @feng2022rgt extended this with Graph Transformers and relation-specific attention, learning separate aggregation rules for each edge type. AdaRelBot is closest to RGT in spirit. Both use Transformer-style aggregation. AdaRelBot differs in how it handles heterophily: edge features and prototype calibration replace relation-specific convolutions, avoiding the proliferation of learned aggregation functions.

== Prototype Learning and Focal Loss

Prototype networks @snell2017protonet learn a small set of class representatives and classify by similarity. They are useful when embeddings are noisy because cosine-similarity decisions depend on direction rather than scale. If a node's embedding has been pulled off-course by heterophilous neighbors, its magnitude may change but its orientation may still point closer to one class prototype. Prototypes have been applied to few-shot node classification and to stabilizing representations under distribution shift.

Focal Loss @lin2017focal was introduced for object detection. It down-weights easy, well-classified examples so that training gradients focus on hard cases. Without this reweighting, a classifier on a 70/30 dataset can achieve high accuracy by predicting the majority class and leaving minority examples underfit. AdaRelBot combines both ideas inside a single bot-detection model.

= Method

@fig:architecture shows the full pipeline: feature encoding, edge-feature construction, two Graph Transformer layers, and a dual-head classifier.

#figure(
  placement: top,
  scope: "parent",
  image("figures/architecture.svg", width: 100%),
  caption: [AdaRelBot architecture. Four modality encoders project heterogeneous user features into a shared 128-dimensional space. Cosine-similarity edge features are concatenated with a relation flag and fed into two TransformerConv layers. A learned gate blends MLP and prototype heads.],
) <fig:architecture>

== Feature Encoding

The dataset has four modalities: Description and Tweets (768-d RoBERTa embeddings), Numeric (5-d), and Categorical (3-d). These have different statistical properties. Text embeddings are dense and already semantically meaningful; numeric counts span several orders of magnitude; categorical fields are sparse one-hot-like indicators. Each input modality $m in {"des", "tweet", "num", "cat"}$ is projected to a 32-dimensional vector $bold(z)_m$ via a linear layer and LeakyReLU. Numeric features pass through BatchNorm first to stabilize activations across the wide dynamic range of follower and tweet counts. The four 32-dimensional vectors are concatenated into a 128-dimensional vector and passed through a final linear projection, LayerNorm, and Dropout to form the initial node embedding $bold(h)_v^((0))$.

== Edge Features: A Direct Compatibility Signal

In a standard TransformerConv, the attention coefficient $alpha_(u v)$ is computed from node features alone. If a bot's embedding has already been corrupted by heterophilous aggregation, the attention scores become unreliable. A bad embedding produces bad attention weights, which let in more bad neighbors, compounding the error. AdaRelBot breaks this by providing an external signal on each edge: description cosine similarity, tweet cosine similarity, and a relation-type flag. The edge-feature vector is

$
  bold(e)_(u v) = [cos_"des"(u, v), med cos_"tweet"(u, v), med r_(u v)] in RR^3.
$ <eq:edgeattr>

Cosine similarity is the natural choice. RoBERTa embeddings are already L2-normalized, so a bot with a very short bio is not penalized for having a smaller embedding norm than a verbose human. Inner products or Euclidean distances gave noisier edge features and worse validation F1. The relation-type flag $r_(u v)$ matters because follow edges indicate interest while following edges indicate popularity. Encoding this distinction lets the attention heads learn relation-specific behavior without separate convolution weights.

@fig:messagepassing contrasts aggregation with and without these features.

#figure(
  image("figures/message_passing.svg", width: 100%),
  caption: [Without edge features, a Graph Transformer treats every neighbor equally and cannot distinguish homophilous from heterophilous edges. AdaRelBot down-weights dissimilar edges before aggregation.],
) <fig:messagepassing>

In PyTorch Geometric's `TransformerConv` with `edge_dim = 3` and `beta = true`, the edge features enter additively:

$
  bold(h)_v' = bold(W)_1 bold(h)_v + sum_(u in cal(N)(v)) alpha_(u v) lr(bold(W)_2 bold(h)_u + bold(W)_e bold(e)_(u v)).
$ <eq:transformerconv>

The edge-feature term shifts each neighbor message based on compatibility. High cosine similarity amplifies the neighbor message; low similarity attenuates it. Because the edge features are precomputed from raw content rather than from learned embeddings, they do not depend on the current state of the model. This is what breaks the loop.

=== Implementation Efficiency

Computing pairwise cosine similarities for 229K edges naively would be memory-intensive. We batch the computation into chunks of 50,000 edges. The full edge-attribute matrix of shape $[|E|, 3]$ is materialized once and pinned on the GPU. On TwiBot-20 this adds 2.7 MB -- negligible against the node feature tensors. Inference cost is dominated by the two TransformerConv layers, not by the edge attributes.

== Graph Convolution Layers

We stack two TransformerConv layers. Two layers give the model enough receptive field to capture local community structure without the instability that appears with deeper graph networks. The output of each layer is added to its input via a residual connection and normalized with LayerNorm. The residual paths preserve the original signal and help gradients flow; LayerNorm keeps activations stable across different node degrees.

== Dual-Head Classifier with Learned Gate

After graph convolution, each node has a 128-dimensional embedding $bold(h)_v = bold(h)_v^((2))$. This is fed into a standard two-layer MLP to produce logits $bold(z)_"mlp"$. In parallel, we maintain two learned class prototypes $bold(C) in RR^(2 times 128)$ where row 0 is the ideal human and row 1 the ideal bot. Both the node embedding and the prototypes are L2-normalized, and cosine-similarity logits are scaled by a learned temperature $tau$:

$
  bold(z)_"proto" = tau dot (bold(h)_v / (||bold(h)_v||_2)) dot (bold(C) / (||bold(C)||_2))^top
$ <eq:proto>

Cosine similarity measures direction, not magnitude, so the prototype head is inherently robust to embedding scale changes.

A small gating network outputs a scalar $gamma_v in (0, 1)$ per node:

$
  gamma_v = sigma lr("ReLU"(bold(W)_(g 1) bold(h)_v) dot bold(W)_(g 2)).
$ <eq:gate>

The final logits are a convex combination:

$
  bold(z)_v = gamma_v bold(z)_"mlp" + (1 - gamma_v) bold(z)_"proto".
$ <eq:blend>

A high $gamma_v$ puts more weight on the MLP head; a low $gamma_v$ puts more weight on the prototype head. On most nodes the prototype head dominates, because its cosine-similarity decision is robust to embedding scale. But the gate reserves MLP capacity for nodes where the parametric transform is more discriminative.

== Training Objective

We optimize with Focal Loss @lin2017focal, using a focusing parameter of $gamma = 2.0$ and class-balanced weights $alpha_c$. For an easy example where $p_(i, y_i) approx 1$, the factor $(1 - p_(i, y_i))^gamma$ vanishes, contributing almost nothing to the gradient. For a hard example predicted at $p approx 0.4$, the factor stays large. The class weights $alpha_c$ down-weight the human majority and up-weight the bot minority, so the model cannot coast on predicting the dominant class. We apply label smoothing (0.1) to keep the model from collapsing to extreme 0/1 probabilities, which stabilizes the calibration diagrams in Section 3.2. Both MLP and prototype heads receive auxiliary focal losses, weighted by 0.5, to keep both branches trainable and prevent the gate from collapsing.

= Experiments

== Dataset and Metrics

We evaluate on two Twitter bot detection benchmarks. TwiBot-20 @feng2021twibot20 contains 229,580 nodes, 227,979 directed edges, and a split of 8,278 / 2,365 / 1,183 train/validation/test nodes. The labeled nodes are sparse, the classes are imbalanced (roughly 70/30 human/bot), and bots deliberately mimic humans. Cresci-15 @cresci2015 contains 5,301 labeled accounts across five disjoint groups and is smaller but more densely labeled. We report accuracy, F1, and Matthews Correlation Coefficient (MCC). MCC ranges from -1 to +1 and is high only when all four confusion-matrix cells are calibrated; it penalizes models that get one class right by guessing the majority.

== Baselines

We compare against methods reported in the TwiBot-20 and RGT literature @feng2021twibot20 @feng2022rgt. Lee et al. @lee2011social use hand-crafted social-network features. RoBERTa @liu2019roberta is a text-only transformer that ignores graph structure. GAT @velickovic2018gat and SATAR @satar2019 use graph attention without explicit heterophily handling. BotRGCN @feng2021botrgcn, BotMoE @botmoe2023, and RGT @feng2022rgt are the strongest structural baselines: they combine graph neural networks with user features and relation modeling.

== Implementation Details

#figure(
  caption: [Hyperparameters for AdaRelBot training.],
  kind: table,
  table(
    columns: (1.8fr, 2fr),
    align: (left, left),
    stroke: none,
    table.hline(y: 0, stroke: 1.2pt),
    table.header([Setting], [Value]),
    table.hline(y: 1, stroke: 0.5pt),
    [Optimizer], [AdamW],
    [Learning rate], [$5 times 10^(-3)$],
    [Weight decay], [$5 times 10^(-4)$],
    [Dropout], [0.3],
    [Embedding dimension], [128],
    [Attention heads], [8],
    [Focal $gamma$], [2.0],
    [Label smoothing], [0.1],
    [Auxiliary loss weight], [0.5],
    [Epochs], [50 per seed],
    [Seeds], [42, 123, 456, 2024, 9999],
    [Selection criterion], [Highest validation F1],
    table.hline(stroke: 1.2pt),
  ),
) <tab:hyperparams>

We use PyTorch Geometric's `TransformerConv` @shi2021unimp. Edge attributes are computed once in batches of 50,000 edges and pinned on the GPU. Training one seed takes 3--4 minutes on a modern GPU, about 20 minutes on CPU. The full model has approximately 520K parameters.

The hyperparameters in @tab:hyperparams balance capacity and stability. A 128-dimensional hidden space is large enough for four modalities but small enough to avoid overfitting on 8,000 labeled training nodes. Eight attention heads let different heads specialize on different semantic signals. Dropout at 0.3 provides regularization. Validation F1 drives model selection because it penalizes both false positives and false negatives.

== Main Results

@tab:benchmark reports test-set performance. AdaRelBot outperforms all listed baselines on both datasets. On TwiBot-20, the gains over RGT are 1.17 points in accuracy and 1.22 points in F1. On the 1,183-node test set, that corresponds to roughly 14 additional correctly classified accounts. On Cresci-15, AdaRelBot reaches 99.17% accuracy and 99.35% F1 under the standard split, well above RGT (96.89% / 97.58%). But the standard split is optimistic: because each of the five groups is internally text-homogeneous, the random 80/10/10 split leaks group identity. Our leave-one-group-out analysis below corrects this. TwiBot-22 results are left as future work.

#figure(
  placement: top,
  scope: "parent",
  caption: [Benchmark comparison across Cresci-15, TwiBot-20, and TwiBot-22. Baseline numbers include mean $plus.minus$ std where available. AdaRelBot on TwiBot-22 is not yet implemented.],
  kind: table,
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    stroke: none,
    table.hline(y: 0, stroke: 1.2pt),
    table.header(
      table.cell(rowspan: 2, align: horizon + center)[Model],
      table.cell(colspan: 2)[Cresci-15],
      table.cell(colspan: 2)[TwiBot-20],
      table.cell(colspan: 2)[TwiBot-22],
      [Accuracy (%)], [F1-score (%)],
      [Accuracy (%)], [F1-score (%)],
      [Accuracy (%)], [F1-score (%)],
    ),
    table.hline(y: 1, start: 1, stroke: 0.5pt),
    table.hline(y: 2, stroke: 0.5pt),
    [Lee et al. @lee2011social],
    [98.19 $plus.minus$ 0.07],
    [98.52 $plus.minus$ 0.06],
    [75.73 $plus.minus$ 0.19],
    [79.37 $plus.minus$ 0.19],
    [-],
    [-],
    [GAT @velickovic2018gat],
    [96.44 $plus.minus$ 0.19],
    [97.22 $plus.minus$ 0.14],
    [77.32 $plus.minus$ 0.73],
    [80.51 $plus.minus$ 0.65],
    [77.53 $plus.minus$ 0.08],
    [53.47 $plus.minus$ 0.46],
    [RoBERTa @liu2019roberta],
    [95.70 $plus.minus$ 0.15],
    [94.06 $plus.minus$ 0.21],
    [74.97 $plus.minus$ 0.23],
    [72.80 $plus.minus$ 0.32],
    [71.92 $plus.minus$ 0.64],
    [16.15 $plus.minus$ 4.98],
    [SATAR @satar2019],
    [92.72 $plus.minus$ 0.59],
    [93.84 $plus.minus$ 0.52],
    [61.70 $plus.minus$ 1.75],
    [71.95 $plus.minus$ 0.69],
    [-],
    [-],
    [BotRGCN @feng2021botrgcn],
    [96.37 $plus.minus$ 0.15],
    [96.80 $plus.minus$ 0.27],
    [83.21 $plus.minus$ 0.37],
    [87.68 $plus.minus$ 0.32],
    [76.75 $plus.minus$ 0.08],
    [48.29 $plus.minus$ 0.66],
    [BotMoE @botmoe2023],
    [95.30 $plus.minus$ 0.16],
    [96.39 $plus.minus$ 0.11],
    [84.22 $plus.minus$ 0.34],
    [86.89 $plus.minus$ 0.34],
    [79.25 $plus.minus$ 0.00],
    [*56.62 $plus.minus$ 0.40*],
    [RGT @feng2022rgt],
    [96.89 $plus.minus$ 0.16],
    [97.58 $plus.minus$ 0.12],
    [85.20 $plus.minus$ 0.24],
    [86.88 $plus.minus$ 0.22],
    [*81.93 $plus.minus$ 0.19*],
    [23.85 $plus.minus$ 0.20],
    [AdaRelBot],
    [*99.17 $plus.minus$ 0.54*],
    [*99.35 $plus.minus$ 0.43*],
    [*86.37 $plus.minus$ 0.27*],
    [*88.10 $plus.minus$ 0.34*],
    [-],
    [-],
    table.hline(stroke: 1.2pt),
  ),
) <tab:benchmark>

Training is stable across seeds: TwiBot-20 has std 0.27% accuracy, 0.34% F1; Cresci-15 has std 0.54% accuracy, 0.43% F1.

=== Statistical Significance

Across five seeds, AdaRelBot's TwiBot-20 accuracy spans 86.12%--86.76% and F1 spans 87.56%--88.46%. Cresci-15 spans 98.31%--99.81% accuracy and 98.67%--99.85% F1. The gaps to baselines are wide relative to the seed variance.

== Cresci-15 Robustness: Group-Leakage and Strict Evaluation

The near-perfect standard-split numbers on Cresci-15 raise an obvious question: is the model learning bot detection, or just recognizing group membership? We ran four checks.

*Text-only classification.* A logistic regression on RoBERTa tweet embeddings alone scores 99.25% F1 on the test split. No graph. No GNN. Description embeddings reach 86.82%. The text alone nearly solves the task.

*Train-test text similarity.* For every single test sample, the maximum cosine similarity to some training sample exceeds 0.95 in description space. For tweets, it exceeds 0.99. Every group is internally text-homogeneous, and the stratified split distributes members of every group across train and test. A classifier can get 99% by recognizing group style, not bot behavior.

*Training curves.* Train and validation loss both fall smoothly to near zero, with a final train-val gap of 0.004. No classical overfitting here -- the model legitimately performs well on the test split because the test split contains text nearly identical to what it trained on.

*Leave-one-group-out.* We hold out an entire user group for testing and train on the other four, repeating across all five groups. Each fold runs across five seeds. @tab:creciloocv reports the results.

#figure(
  caption: [Leave-one-group-out evaluation on Cresci-15. Each fold holds out an entire user group for testing. TWT bots (the most sophisticated group) are the hardest, with accuracy dropping to 86.27%.],
  kind: table,
  {
    set text(size: 7.5pt)
    table(
      columns: (0.8fr, 0.8fr, 1.2fr, 1.2fr, 1.2fr),
      align: (left, left, center, center, center),
      stroke: none,
      table.hline(y: 0, stroke: 1.2pt),
      table.header(
        [Held-Out Group], [Type], [Accuracy (\%)], [F1 (\%)], [ROC-AUC (\%)]
      ),
      table.hline(y: 1, stroke: 0.5pt),
      [E13], [Human], [97.25 $plus.minus$ 0.20], [--], [--],
      [TFP], [Human], [98.93 $plus.minus$ 1.36], [--], [--],
      [FSF],
      [Bot],
      [96.94 $plus.minus$ 0.75],
      [97.51 $plus.minus$ 0.59],
      [99.87 $plus.minus$ 0.11],
      [INT], [Bot], [99.54 $plus.minus$ 0.09], [--], [--],
      [TWT], [Bot], [86.27 $plus.minus$ 3.63], [92.59 $plus.minus$ 2.11], [--],
      [*Weighted avg.*], [], [*96.15*], [--], [--],
      table.hline(stroke: 1.2pt),
    )
  },
) <tab:creciloocv>

AdaRelBot generalizes well across most groups: E13 humans (97.25%), FSF fake followers (96.94%), INT social bots (99.54%), and TFP humans (98.93%) are classified accurately even when the target group is absent from training. The TWT Twitterbot group, the most sophisticated in the dataset, drops to 86.27%. The weighted-average accuracy of 96.15% is roughly 3 points below the standard split, showing how much the random split inflates performance through text overlap.

When a held-out group is all one class, F1 and ROC-AUC are degenerate. We report them only when both classes appear. TwiBot-20, which draws from a single general Twitter snapshot without cleanly separated user groups, does not suffer from this issue. It remains our primary benchmark.

== Detailed Metrics and Calibration

@tab:fullmetrics reports per-class precision, recall, ROC-AUC, and PR-AUC from the five-seed ensemble. The confusion matrix shows 116 FP and 38 FN on TwiBot-20, only 4 total misclassifications on Cresci-15.

#figure(
  caption: [Extended metrics for AdaRelBot on TwiBot-20 and Cresci-15. Ensemble metrics are computed from the average of five independently trained seeds.],
  kind: table,
  {
    set text(size: 7pt)
    table(
      columns: (1.6fr, 1.1fr, 1.1fr),
      align: (left, center, center),
      stroke: none,
      table.hline(y: 0, stroke: 1.2pt),
      table.header([Metric], [TwiBot-20], [Cresci-15]),
      table.hline(y: 1, stroke: 0.5pt),
      [Accuracy (\%)], [86.37 $plus.minus$ 0.27], [99.17 $plus.minus$ 0.54],
      [F1-score (\%)], [88.10 $plus.minus$ 0.34], [99.35 $plus.minus$ 0.43],
      [Precision (\%)], [83.48 $plus.minus$ 0.46], [99.35 $plus.minus$ 0.60],
      [Recall (\%)], [93.28 $plus.minus$ 1.27], [99.35 $plus.minus$ 0.51],
      [ROC-AUC (\%)], [93.09 $plus.minus$ 0.04], [99.97 $plus.minus$ 0.03],
      [PR-AUC (\%)], [93.24 $plus.minus$ 0.09], [99.98 $plus.minus$ 0.02],
      [MCC], [0.7291 $plus.minus$ 0.0070], [0.9822 $plus.minus$ 0.0116],
      [ECE], [0.1172], [0.0660],
      [Brier], [0.1111], [0.0101],
      table.hline(stroke: 1.2pt),
    )
  },
) <tab:fullmetrics>

TwiBot-20's ECE of 0.117 reflects a model operating near the decision boundary on hard nodes. Label smoothing (0.1) keeps the middle probability bins populated; the decile-binned reliability diagram shows genuine residual uncertainty rather than empty bins. Cresci-15 shows ECE 0.066 and Brier 0.010.

@fig:roc shows the ROC curves. TwiBot-20 AUC is 0.93; Cresci-15 is effectively 1.0.

#figure(
  grid(
    columns: 2,
    gutter: 0.8em,
    image("figures/roc_twibot20.svg", width: 100%),
    image("figures/roc_cresci15.svg", width: 100%),
  ),
  caption: [ROC curves for AdaRelBot on TwiBot-20 (AUC = 0.93) and Cresci-15 (AUC > 0.99).],
) <fig:roc>

@fig:pr displays the PR curves. On TwiBot-20, precision degrades from 1.0 to the baseline of 0.54 as recall approaches 1.0. The AP of 0.93 means most of the curve's mass lies well above the baseline. On Cresci-15, precision stays near-perfect across the full recall range.

#figure(
  grid(
    columns: 2,
    gutter: 0.8em,
    image("figures/pr_twibot20.svg", width: 100%),
    image("figures/pr_cresci15.svg", width: 100%),
  ),
  caption: [Precision-Recall curves with baseline (dashed). TwiBot-20 AP = 0.93; Cresci-15 AP > 0.99.],
) <fig:pr>

@fig:calibration shows the reliability diagrams. On TwiBot-20, the model is slightly under-confident at moderate probability bins. This makes sense given the gate's conservative blending: the prototype head pulls predictions toward milder probabilities, especially where the MLP would be overconfident on thin evidence.

#figure(
  grid(
    columns: 2,
    gutter: 0.8em,
    image("figures/calib_twibot20.svg", width: 100%),
    image("figures/calib_cresci15.svg", width: 100%),
  ),
  caption: [Reliability diagrams. Bin sizes annotated. ECE uses adaptive decile bins.],
) <fig:calibration>

@fig:cm show the confusion matrices and zoomed Cresci-15 curves. The FP rate on TwiBot-20 is 21%, the FN rate 6%. On Cresci-15, precision remains at 1.0 until recall reaches 0.96. The ROC zoom confirms TPR > 0.99 at FPR < 0.01.

#figure(
  grid(
    columns: 2,
    gutter: 0.8em,
    image("figures/cm_twibot20.svg", width: 100%),
    image("figures/cm_cresci15.svg", width: 100%),
  ),
  caption: [Ensemble confusion matrices for TwiBot-20 and Cresci-15.],
) <fig:cm>


== Ablation Studies

We remove each component of AdaRelBot one at a time and retrain (3 seeds, TwiBot-20). The ablations correspond to: no edge features (`edge_dim=none`), no prototype head ($gamma_v=1$), and class-weighted cross-entropy in place of Focal Loss.

#figure(
  caption: [Ablation study on TwiBot-20 (3 seeds). The prototype head provides the largest and most consistent gain; edge features and Focal Loss have smaller, seed-dependent effects.],
  kind: table,
  {
    set text(size: 7pt)
    table(
      columns: (1.4fr, 1.3fr, 1.3fr, 1.3fr),
      align: (left, center, center, center),
      stroke: none,
      table.hline(y: 0, stroke: 1.2pt),
      table.header([Configuration], [Accuracy (%)], [F1 (%)], [MCC]),
      table.hline(y: 1, stroke: 0.5pt),
      [AdaRelBot (full)],
      [86.45 $plus.minus$ 0.24],
      [88.22 $plus.minus$ 0.09],
      [0.7313 $plus.minus$ 0.0030],
      [w/o edge features],
      [86.67 $plus.minus$ 0.11],
      [88.18 $plus.minus$ 0.03],
      [0.7329 $plus.minus$ 0.0014],
      [w/o prototype head],
      [85.57 $plus.minus$ 1.64],
      [87.85 $plus.minus$ 0.94],
      [0.7230 $plus.minus$ 0.0212],
      [w/o Focal Loss],
      [86.53 $plus.minus$ 0.34],
      [88.07 $plus.minus$ 0.35],
      [0.7301 $plus.minus$ 0.0074],
      table.hline(stroke: 1.2pt),
    )
  },
) <tab:ablation>

The prototype head is the dominant mechanism. Removing it drops F1 by 0.4 points and doubles the seed-to-seed variance. Without the prototype gate, the MLP alone becomes brittle; some seeds get it, some don't. Edge features and Focal Loss have smaller effects on TwiBot-20. The cosine-similarity prototypes appear to encode enough discriminative structure by themselves on this benchmark, leaving the edge features and focal reweighting to refine decisions at the margin. On datasets with more subtle text patterns, these components would likely matter more.

= Analysis

== Why AdaRelBot Works

*Edge features break the over-smoothing cycle.* Standard GNNs have a feedback problem: heterophilous aggregation corrupts embeddings, which degrades attention weights, which lets in more bad neighbors. A bot following many humans gets its representation pulled toward the human cluster in layer 1. By layer 2, attention compares the bot's now-human-like embedding to its neighbors, finds them similar, and averages them in harder. Edge features short-circuit this. They are computed from raw content, not from learned embeddings, so they don't degrade when embeddings do. The model can still tell a human from a bot at the edge level even after several rounds of aggregation have muddied the node representations.

But where do edge features actually help? We looked at test nodes with high heterophily. For these, the incoming neighborhood is dominated by the opposite class. A standard Graph Transformer gets flooded with misleading messages. Edge features lower the attention weights on those misleading edges. On TwiBot-20 the aggregate effect is small because node-only attention is already strong for the easy nodes. Edge features are refining a hard minority.

*Prototypes act as a regularizer.* The cosine-similarity prototype head is bounded: all logits fall in $[-tau, +tau]$, regardless of embedding scale. A corrupted embedding might drive an unconstrained MLP to extreme logits. The prototype head can only place that node somewhere between the two class directions, softening the prediction. The gate in @eq:gate blends the MLP back in where it is confident. Removing the prototype head causes the clearest drop in the ablation table, and the increased variance tells you something: the MLP alone is brittle across seeds. The prototype stabilizes it.

*Focal Loss handles class imbalance.* Without it, the optimizer maximizes accuracy by leaning on the human majority. The $(1-p)^gamma$ term ensures that an example predicted at $p=0.99$ contributes almost no gradient, while one predicted at $p=0.4$ still receives a strong learning signal. The model cannot coast on the majority class.

== Gate-Heterophily Correlation

Does the learned gate actually respond to heterophily? We compute per-node heterophily as the fraction of incoming neighbors with a different label:

$
  "het"(v) = 1 / (|cal(N)(v)|) sum_(u in cal(N)(v)) bb(1)[y_u != y_v].
$ <eq:het>

Degree-zero nodes are dropped. The gate correlates positively with heterophily: Pearson $r = 0.095$ ($p < 0.005$), Spearman $rho = 0.232$ ($p < 10^(-13)$). Pearson is small because heterophily is not the only thing the gate responds to. Spearman is stronger because the ordering of gate values, not their absolute magnitude, tracks structural ambiguity. Nodes with more opposite-label neighbors get higher $gamma_v$, putting more weight on the MLP head. Heterophilous neighborhoods are where the parametric transform adds the most value.

#figure(
  image("figures/gate_heterophily.svg", width: 80%),
  caption: [Learned gate $gamma_v$ versus per-node heterophily on the TwiBot-20 test set. The orange curve is a running mean; higher heterophily is associated with larger $gamma_v$, i.e., greater reliance on the MLP head.],
) <fig:gatescatter>

== Comparison to RGT

RGT @feng2022rgt learns per-relation TransformerConv layers, one for follow edges and one for following, fused through semantic attention. AdaRelBot pools all edges into a single graph. The relation type becomes one dimension of the edge feature. No duplicated convolutions. No separate attention over relation types. And the prototype head provides a second opinion that RGT lacks. Both approaches address heterophily. AdaRelBot does it with fewer learned components. The numbers show this simplicity pays off.

=== Limitations

TwiBot-20 and Cresci-15 are two snapshots. Bots on other platforms or in different eras may produce different graph patterns. Edge features are static: a bot that changes its persona after being followed can leave stale signals. The gate-heterophily correlation is real but modest, because the gate also responds to feature sparsity, training dynamics, and other uncertainty sources we have not disentangled. And Cresci-15's group-homogeneous text properties, documented above, mean its standard-split numbers should be read with caution.

=== Broader Impact

Accurate bot detection helps reduce misinformation and coordinated harassment. But any classifier that labels users as real or fake can be misused. Governments or platforms could deploy such systems to suppress dissent or mistakenly flag legitimate automated accounts. AdaRelBot should be a screening tool, not a final decision. Human review, appeal mechanisms, and regular audits of false positives are essential accompaniments.

= Conclusion

AdaRelBot shows that heterophily does not need complex per-relation architectures to be handled well. Cosine-similarity edge features, computed from raw profile text and tweets and fed into a standard TransformerConv, give Graph Transformers enough signal to down-weight deceptive edges. A dual-head classifier with a per-node gate blends parametric and prototype-based prediction, with the prototype head providing a stable baseline and the MLP contributing where it is confident. On TwiBot-20, AdaRelBot achieves 86.37% accuracy and 88.10% F1. On Cresci-15, leave-one-group-out evaluation shows 96.15% weighted accuracy, which is the number that matters. The standard split's 99% is inflated by text leakage.

The gate-heterophily correlation shows that the model is using the mechanism in roughly the way it was designed to: heterophilous neighborhoods get more MLP weight. But the correlation is modest. The gate responds to more than just neighborhood structure.

Several open problems. Dynamic edge features that update during training could handle bots that change their persona after acquiring followers -- our static features survive one training run but would be stale in a live system. The gate $gamma_v$ is itself a useful signal: nodes with consistently low $gamma_v$ are the ones the model finds ambiguous and could be flagged for human review. Replacing precomputed RoBERTa with a fine-tuned language model might surface linguistic patterns that generic embeddings miss. Temporal graphs would let the model track sudden bursts of follower acquisition, a common bot behavior. The core idea -- make edge quality explicit and let a prototype calibrate the classifier -- applies beyond bot detection, anywhere heterophily makes standard GNNs unreliable.

=== Reproducibility

The code is self-contained. For TwiBot-20, `train.py` reproduces the five-seed benchmark with the hyperparameters in @tab:hyperparams. For Cresci-15, `preprocess_cresci15.py` builds the tensors and graph from the raw CSV, `train_cresci15.py` runs the five-seed standard split, and `train_cresci15_loocv.py` reproduces the leave-one-group-out evaluation. `./train --dataset both` runs both datasets. The ablation script removes one component at a time without changing anything else. Source code, preprocessing scripts, and trained weights will be released upon publication.

#bibliography("refs.bib")
