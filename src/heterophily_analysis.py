#!/usr/bin/env python3
"""Heterophily mechanism analysis for TwiBot-20.

1. Per-node local homophily split by true label (bots vs. humans).
2. Stratify BotRGCN test performance by homophily buckets.
3. Confusion matrix for low-homophily test nodes.
4. Relation-level disagreement between follow-homophily and following-homophily.
"""

import json
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.optim import Adam
from torch_geometric.nn import RGCNConv

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "data/twibot-20"
GRAPH_PATH = os.path.join(DATA_DIR, "twibot_graph.pt")
PARQUET_PATH = os.path.join(DATA_DIR, "twibot_df.parquet")
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.json")
OUTPUT_DIR = "results/tables"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "heterophily_analysis.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WD = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Homophily buckets for stratified evaluation
HOMOPHILY_BINS = [
    (0.0, 0.0, "0"),
    (0.01, 0.25, "0.01-0.25"),
    (0.26, 0.50, "0.26-0.50"),
    (0.51, 1.00, "0.51+"),
]


def homo_bin_label(h, bins=HOMOPHILY_BINS):
    for lo, hi, label in bins:
        if lo <= h <= hi:
            return label
    return "other"


# ── Load data ───────────────────────────────────────────────────────
print("=" * 76)
print("Loading TwiBot-20 data ...")
print("=" * 76)

graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
df = pd.read_parquet(PARQUET_PATH)
with open(FEATURE_NAMES_PATH) as f:
    feature_names = json.load(f)

feats = {}
for group in ["profile", "tweet", "topology", "neighbour_attr"]:
    feats[group] = np.load(os.path.join(DATA_DIR, f"features_{group}.npy"))

N_NODES = graph.num_nodes
LABELED_MASK = graph.y >= 0
TRAIN_MASK = graph.train_mask
VAL_MASK = graph.val_mask
TEST_MASK = graph.test_mask
y = graph.y.long()

# ── Standardise features (train stats) ──────────────────────────────
def standardise(arr, mask):
    mu = arr[mask].mean(axis=0, keepdims=True)
    sd = arr[mask].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return (arr - mu) / sd


train_mask_np = TRAIN_MASK.numpy()
for group in ["profile", "tweet", "topology", "neighbour_attr"]:
    feats[group] = standardise(feats[group], train_mask_np)

x_profile = torch.tensor(feats["profile"], dtype=torch.float32)
x_tweet = torch.tensor(feats["tweet"], dtype=torch.float32)
x_topology = torch.tensor(feats["topology"], dtype=torch.float32)
x_neighbour = torch.tensor(feats["neighbour_attr"], dtype=torch.float32)

P_DIM = x_profile.size(1)
T_DIM = x_tweet.size(1)
TO_DIM = x_topology.size(1)
N_DIM = x_neighbour.size(1)


# ── Per-node homophily (fraction of neighbours sharing the node's label) ───────
def per_node_homophily(y_tensor, edge_index_tensor):
    """For every node: #same-label incident edges / #labeled incident edges."""
    labeled = y_tensor >= 0
    src, dst = edge_index_tensor
    mask = labeled[src] & labeled[dst]
    s, d = src[mask], dst[mask]
    same = (y_tensor[s] == y_tensor[d]).float()
    N = y_tensor.size(0)
    total = torch.zeros(N)
    same_sum = torch.zeros(N)
    ones = torch.ones_like(same)
    total.index_add_(0, s, ones)
    total.index_add_(0, d, ones)
    same_sum.index_add_(0, s, same)
    same_sum.index_add_(0, d, same)
    denom = total.clamp(min=1)
    return (same_sum / denom).numpy(), total.numpy()


print("Computing per-node homophily ...")
homo_follow, deg_follow = per_node_homophily(y, graph.edge_index_follow)
homo_following, deg_following = per_node_homophily(y, graph.edge_index_following)
combined_edge_index = torch.cat([graph.edge_index_follow, graph.edge_index_following], dim=1)
homo_combined, deg_combined = per_node_homophily(y, combined_edge_index)

labeled_idx = LABELED_MASK.nonzero(as_tuple=False).view(-1).numpy()
y_labeled = y[LABELED_MASK].numpy()
test_idx = TEST_MASK.nonzero(as_tuple=False).view(-1).numpy()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 1  —  Per-node local homophily by true label                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 1  —  Per-node local homophily by true label")
print("=" * 76)

bot_mask = y_labeled == 1
human_mask = y_labeled == 0


def label_stats(combined_arr, follow_arr, following_arr, deg_arr, label_name):
    valid = combined_arr >= 0
    n_valid = valid.sum()
    print(f"\n  {label_name} (n={len(combined_arr):,}, with ≥1 labeled neighbour={n_valid:,}):")
    print(f"    combined:  mean={combined_arr[valid].mean():.4f}, median={np.median(combined_arr[valid]):.4f}, "
          f"p25={np.percentile(combined_arr[valid], 25):.4f}, p75={np.percentile(combined_arr[valid], 75):.4f}")
    print(f"    follow:    mean={follow_arr[valid].mean():.4f}, median={np.median(follow_arr[valid]):.4f}")
    print(f"    following: mean={following_arr[valid].mean():.4f}, median={np.median(following_arr[valid]):.4f}")
    if deg_arr is not None:
        print(f"    avg degree (combined): {deg_arr[valid].mean():.1f}")


label_stats(homo_combined[labeled_idx], homo_follow[labeled_idx], homo_following[labeled_idx],
            deg_combined[labeled_idx], "All labeled nodes")
label_stats(homo_combined[labeled_idx][bot_mask], homo_follow[labeled_idx][bot_mask],
            homo_following[labeled_idx][bot_mask], deg_combined[labeled_idx][bot_mask], "Bots")
label_stats(homo_combined[labeled_idx][human_mask], homo_follow[labeled_idx][human_mask],
            homo_following[labeled_idx][human_mask], deg_combined[labeled_idx][human_mask], "Humans")

# Fraction of each class in heterophilic neighbourhoods
for rel, arr in [("combined", homo_combined), ("follow", homo_follow), ("following", homo_following)]:
    arr_l = arr[labeled_idx]
    valid = arr_l >= 0
    low_h = valid & (arr_l < 0.3)
    print(f"\n  {rel}: nodes with homophily < 0.3: {low_h.sum():,} / {valid.sum():,} "
          f"({low_h.sum() / valid.sum() * 100:.1f}%)")
    if low_h.sum() > 0:
        bot_frac_low = y_labeled[low_h].mean()
        print(f"    → bot fraction among low-homophily nodes: {bot_frac_low * 100:.1f}%")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 4  —  Relation-level disagreement (run before model for CSV layout)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 4  —  Relation-level disagreement")
print("=" * 76)

hf_l = homo_follow[labeled_idx]
hg_l = homo_following[labeled_idx]
valid_both = (hf_l >= 0) & (hg_l >= 0)
diff = np.abs(hf_l[valid_both] - hg_l[valid_both])

print("\n  |follow_homo - following_homo| for nodes with ≥1 neighbor in both relations:")
print(f"    mean={diff.mean():.4f}, median={np.median(diff):.4f}, "
      f"p25={np.percentile(diff, 25):.4f}, p75={np.percentile(diff, 75):.4f}, "
      f"max={diff.max():.4f}")

# Correlation
print(f"\n  Pearson correlation (follow vs. following homophily): "
      f"{np.corrcoef(hf_l[valid_both], hg_l[valid_both])[0, 1]:.4f}")

follow_high = hf_l[valid_both] > 0.5
following_low = hg_l[valid_both] < 0.5
follow_low = hf_l[valid_both] < 0.5
following_high = hg_l[valid_both] > 0.5

print(f"\n  Cross-relation patterns (n={valid_both.sum():,}):")
print(f"    Homophilic on follow ONLY     (>0.5 follow, <0.5 following): "
      f"{(follow_high & following_low).sum():,} "
      f"({(follow_high & following_low).sum() / valid_both.sum() * 100:.1f}%)")
print(f"    Homophilic on following ONLY  (<0.5 follow, >0.5 following): "
      f"{(follow_low & following_high).sum():,} "
      f"({(follow_low & following_high).sum() / valid_both.sum() * 100:.1f}%)")
print(f"    Homophilic on BOTH            (>0.5 both): "
      f"{(follow_high & following_high).sum():,} "
      f"({(follow_high & following_high).sum() / valid_both.sum() * 100:.1f}%)")
print(f"    Heterophilic on BOTH          (<0.5 both): "
      f"{(follow_low & following_low).sum():,} "
      f"({(follow_low & following_low).sum() / valid_both.sum() * 100:.1f}%)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 7 model  —  Train BotRGCN (same setup as degree-bucket baseline)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 2/3  —  BotRGCN baseline + per-homophily-bucket evaluation")
print("=" * 76)
print(f"  Device: {DEVICE}")
print(f"  Seeds:  {SEEDS}")


class BotRGCN(nn.Module):
    def __init__(self, profile_dim=P_DIM, tweet_dim=T_DIM,
                 topology_dim=TO_DIM, neighbour_dim=N_DIM,
                 embedding_dim=128, dropout=0.3):
        super().__init__()
        h = embedding_dim // 4
        self.dropout = dropout
        self.enc_profile = nn.Sequential(nn.Linear(profile_dim, h), nn.LeakyReLU())
        self.enc_tweet = nn.Sequential(nn.Linear(tweet_dim, h), nn.LeakyReLU())
        self.enc_topology = nn.Sequential(nn.Linear(topology_dim, h), nn.LeakyReLU())
        self.enc_neighbour = nn.Sequential(nn.Linear(neighbour_dim, h), nn.LeakyReLU())
        self.proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())
        self.rgcn = RGCNConv(embedding_dim, embedding_dim, num_relations=2)
        self.out1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())
        self.out2 = nn.Linear(embedding_dim, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        p = self.enc_profile(profile)
        t = self.enc_tweet(tweet)
        to = self.enc_topology(topology)
        n = self.enc_neighbour(neighbour)
        x = torch.cat([p, t, to, n], dim=1)
        x = self.proj(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.out1(x)
        x = self.out2(x)
        return torch.sigmoid(x).squeeze(-1)


def train_one_seed(model, data, train_mask, val_mask, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(DEVICE)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    d = data.clone().to(DEVICE)
    xp = x_profile.to(DEVICE)
    xt = x_tweet.to(DEVICE)
    xo = x_topology.to(DEVICE)
    xn = x_neighbour.to(DEVICE)
    tm = train_mask.to(DEVICE)
    vm = val_mask.to(DEVICE)

    y_train_labels = d.y[tm]
    n_pos = (y_train_labels == 1).sum()
    n_neg = (y_train_labels == 0).sum()
    pos_weight = n_neg / (n_pos + 1e-8)

    opt = Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(xp, xt, xo, xn, d.edge_index_rgcn, d.edge_type)
        train_pred = pred[tm]
        train_y = d.y[tm]
        weights = torch.where(train_y == 1, pos_weight, 1.0)
        loss = F.binary_cross_entropy(train_pred, train_y, weight=weights)
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(xp, xt, xo, xn, d.edge_index_rgcn, d.edge_type)
                val_y = d.y[vm]
                val_pred_m = val_pred[vm]
                val_weights = torch.where(val_y == 1, pos_weight, 1.0)
                val_loss = F.binary_cross_entropy(val_pred_m, val_y, weight=val_weights)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 5
                if patience_counter >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate(model, data, mask):
    model.eval()
    d = data.to(DEVICE)
    xp = x_profile.to(DEVICE)
    xt = x_tweet.to(DEVICE)
    xo = x_topology.to(DEVICE)
    xn = x_neighbour.to(DEVICE)
    m = mask.to(DEVICE)
    with torch.no_grad():
        pred = model(xp, xt, xo, xn, d.edge_index_rgcn, d.edge_type)
    y_prob = pred[m].cpu().numpy()
    y_true = d.y[m].cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    f1_b = f1_score(y_true, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return y_true, y_pred, y_prob, acc, f1_m, f1_b, auc


print("\nTraining BotRGCN ...")
all_test_y_true = []
all_test_y_pred = []
all_test_y_prob = []
seed_metrics = []

for s in SEEDS:
    print(f"  Seed {s} ...", end="", flush=True)
    model = BotRGCN()
    model = train_one_seed(model, graph, TRAIN_MASK, VAL_MASK, s)
    yt, yp, ypr, acc, f1m, f1b, auc = evaluate(model, graph, TEST_MASK)
    all_test_y_true.append(yt)
    all_test_y_pred.append(yp)
    all_test_y_prob.append(ypr)
    seed_metrics.append({"seed": s, "acc": acc, "f1_macro": f1m, "f1_binary": f1b, "auc": auc})
    print(f"  Acc={acc:.4f}  F1_macro={f1m:.4f}  AUC={auc:.4f}" if not np.isnan(auc)
          else f"  Acc={acc:.4f}  F1_macro={f1m:.4f}  AUC=nan")

# Ensemble predictions across seeds for stable per-node analysis
y_true_ens = all_test_y_true[0]
y_prob_ens = np.mean(all_test_y_prob, axis=0)
y_pred_ens = (y_prob_ens >= 0.5).astype(int)

print()
print("  ── Overall test performance (ensemble over seeds) ──")
print(f"  Accuracy:   {accuracy_score(y_true_ens, y_pred_ens):.4f}")
print(f"  F1 macro:   {f1_score(y_true_ens, y_pred_ens, average='macro'):.4f}")
print(f"  AUC:        {roc_auc_score(y_true_ens, y_prob_ens):.4f}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 2  —  Stratify test performance by homophily buckets                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 2  —  Test performance stratified by combined homophily")
print("=" * 76)

h_test = homo_combined[test_idx]
test_bin_labels = np.array([homo_bin_label(h) for h in h_test])

print(f"\n  {'Bucket':<12} {'N_test':>8} {'Acc':>8} {'F1_macro':>10} {'AUC':>8}")
print(f"  {'─'*52}")

task2_rows = []
for _, _, bname in HOMOPHILY_BINS:
    bmask = test_bin_labels == bname
    n_bin = bmask.sum()
    if n_bin == 0:
        print(f"  {bname:<12} {n_bin:>8}  {'—':>8}  {'—':>10}  {'—':>8}")
        continue
    yt_b = y_true_ens[bmask]
    yp_b = y_pred_ens[bmask]
    ypr_b = y_prob_ens[bmask]
    acc_b = accuracy_score(yt_b, yp_b)
    f1_b = f1_score(yt_b, yp_b, average="macro") if len(set(yt_b)) > 1 else float("nan")
    auc_b = roc_auc_score(yt_b, ypr_b) if len(set(yt_b)) > 1 else float("nan")
    print(f"  {bname:<12} {n_bin:>8,} {acc_b:>7.4f} {f1_b:>9.4f} {auc_b:>7.4f}")
    task2_rows.append({
        "bucket": bname, "n_test": n_bin,
        "acc": round(acc_b, 4), "f1_macro": round(f1_b, 4), "auc": round(auc_b, 4),
    })

# Also by follow and following relation separately
print("\n  By follow-relation homophily:")
print(f"  {'Bucket':<12} {'N_test':>8} {'Acc':>8} {'F1_macro':>10} {'AUC':>8}")
print(f"  {'─'*52}")
task2_follow_rows = []
hf_test = homo_follow[test_idx]
for _, _, bname in HOMOPHILY_BINS:
    labels = np.array([homo_bin_label(h) for h in hf_test])
    bmask = labels == bname
    n_bin = bmask.sum()
    if n_bin == 0:
        print(f"  {bname:<12} {n_bin:>8}  {'—':>8}  {'—':>10}  {'—':>8}")
        continue
    yt_b = y_true_ens[bmask]
    yp_b = y_pred_ens[bmask]
    ypr_b = y_prob_ens[bmask]
    acc_b = accuracy_score(yt_b, yp_b)
    f1_b = f1_score(yt_b, yp_b, average="macro") if len(set(yt_b)) > 1 else float("nan")
    auc_b = roc_auc_score(yt_b, ypr_b) if len(set(yt_b)) > 1 else float("nan")
    print(f"  {bname:<12} {n_bin:>8,} {acc_b:>7.4f} {f1_b:>9.4f} {auc_b:>7.4f}")
    task2_follow_rows.append({"bucket": bname, "n_test": n_bin, "acc": round(acc_b, 4),
                              "f1_macro": round(f1_b, 4), "auc": round(auc_b, 4)})

print("\n  By following-relation homophily:")
print(f"  {'Bucket':<12} {'N_test':>8} {'Acc':>8} {'F1_macro':>10} {'AUC':>8}")
print(f"  {'─'*52}")
task2_following_rows = []
hg_test = homo_following[test_idx]
for _, _, bname in HOMOPHILY_BINS:
    labels = np.array([homo_bin_label(h) for h in hg_test])
    bmask = labels == bname
    n_bin = bmask.sum()
    if n_bin == 0:
        print(f"  {bname:<12} {n_bin:>8}  {'—':>8}  {'—':>10}  {'—':>8}")
        continue
    yt_b = y_true_ens[bmask]
    yp_b = y_pred_ens[bmask]
    ypr_b = y_prob_ens[bmask]
    acc_b = accuracy_score(yt_b, yp_b)
    f1_b = f1_score(yt_b, yp_b, average="macro") if len(set(yt_b)) > 1 else float("nan")
    auc_b = roc_auc_score(yt_b, ypr_b) if len(set(yt_b)) > 1 else float("nan")
    print(f"  {bname:<12} {n_bin:>8,} {acc_b:>7.4f} {f1_b:>9.4f} {auc_b:>7.4f}")
    task2_following_rows.append({"bucket": bname, "n_test": n_bin, "acc": round(acc_b, 4),
                                 "f1_macro": round(f1_b, 4), "auc": round(auc_b, 4)})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 3  —  Confusion matrix for low-homophily test nodes                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 3  —  Confusion matrix for low-homophily test nodes")
print("=" * 76)

# Use combined homophily = 0 bucket as the primary low-homophily bucket
low_mask = test_bin_labels == "0"
n_low = low_mask.sum()
print(f"\n  Combined homophily = 0 bucket: n={n_low:,} test nodes")
if n_low > 0:
    yt_low = y_true_ens[low_mask]
    yp_low = y_pred_ens[low_mask]
    cm = confusion_matrix(yt_low, yp_low, labels=[0, 1])
    print("\n                 Pred Human  Pred Bot")
    print(f"  True Human        {cm[0, 0]:>5}     {cm[0, 1]:>5}")
    print(f"  True Bot          {cm[1, 0]:>5}     {cm[1, 1]:>5}")
    if cm[1, 0] + cm[0, 1] > 0:
        print("\n  Error breakdown:")
        print(f"    Bots predicted as human (FN): {cm[1, 0]:,} / {cm[1, :].sum():,} "
              f"({cm[1, 0] / cm[1, :].sum() * 100:.1f}% of bots in bucket)")
        print(f"    Humans predicted as bot (FP): {cm[0, 1]:,} / {cm[0, :].sum():,} "
              f"({cm[0, 1] / cm[0, :].sum() * 100:.1f}% of humans in bucket)")

# Also report for 0.01-0.25 bucket
low2_mask = test_bin_labels == "0.01-0.25"
n_low2 = low2_mask.sum()
print(f"\n  Combined homophily 0.01-0.25 bucket: n={n_low2:,} test nodes")
if n_low2 > 0:
    yt_low2 = y_true_ens[low2_mask]
    yp_low2 = y_pred_ens[low2_mask]
    cm2 = confusion_matrix(yt_low2, yp_low2, labels=[0, 1])
    print("\n                 Pred Human  Pred Bot")
    print(f"  True Human        {cm2[0, 0]:>5}     {cm2[0, 1]:>5}")
    print(f"  True Bot          {cm2[1, 0]:>5}     {cm2[1, 1]:>5}")
    if cm2[1, 0] + cm2[0, 1] > 0:
        print("\n  Error breakdown:")
        print(f"    Bots predicted as human (FN): {cm2[1, 0]:,} / {cm2[1, :].sum():,} "
              f"({cm2[1, 0] / cm2[1, :].sum() * 100:.1f}% of bots in bucket)")
        print(f"    Humans predicted as bot (FP): {cm2[0, 1]:,} / {cm2[0, :].sum():,} "
              f"({cm2[0, 1] / cm2[0, :].sum() * 100:.1f}% of humans in bucket)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Save results                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
rows = []
for _, _, bname in HOMOPHILY_BINS:
    r = {"bucket": bname}
    t2 = next((x for x in task2_rows if x["bucket"] == bname), {})
    r["n_test_combined"] = t2.get("n_test")
    r["acc_combined"] = t2.get("acc")
    r["f1_macro_combined"] = t2.get("f1_macro")
    r["auc_combined"] = t2.get("auc")
    tf = next((x for x in task2_follow_rows if x["bucket"] == bname), {})
    r["n_test_follow"] = tf.get("n_test")
    r["f1_macro_follow"] = tf.get("f1_macro")
    tg = next((x for x in task2_following_rows if x["bucket"] == bname), {})
    r["n_test_following"] = tg.get("n_test")
    r["f1_macro_following"] = tg.get("f1_macro")
    rows.append(r)

# Add per-label homophily summary rows
for label_name, mask in [("bots", bot_mask), ("humans", human_mask)]:
    rows.append({
        "bucket": f"{label_name}_summary",
        "mean_combined_homo": round(float(homo_combined[labeled_idx][mask][homo_combined[labeled_idx][mask] >= 0].mean()), 4),
        "median_combined_homo": round(float(np.median(homo_combined[labeled_idx][mask][homo_combined[labeled_idx][mask] >= 0])), 4),
        "mean_follow_homo": round(float(homo_follow[labeled_idx][mask][homo_follow[labeled_idx][mask] >= 0].mean()), 4),
        "mean_following_homo": round(float(homo_following[labeled_idx][mask][homo_following[labeled_idx][mask] >= 0].mean()), 4),
    })

# Add disagreement summary
rows.append({
    "bucket": "relation_disagreement",
    "mean_abs_diff": round(float(diff.mean()), 4),
    "median_abs_diff": round(float(np.median(diff)), 4),
    "pearson_corr": round(float(np.corrcoef(hf_l[valid_both], hg_l[valid_both])[0, 1]), 4),
    "pct_follow_only_homophilic": round(float((follow_high & following_low).sum() / valid_both.sum()), 4),
    "pct_following_only_homophilic": round(float((follow_low & following_high).sum() / valid_both.sum()), 4),
})

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_CSV}")

print()
print("=" * 76)
print("Heterophily analysis complete.")
print("=" * 76)
