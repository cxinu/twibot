#!/usr/bin/env python3
"""Degree-bucket analysis for TwiBot-20 dataset.

Tasks 1-6: dataset analysis (graph stats, degree distribution, label/split
  balance per bin, homophily by relation type, feature availability).
Task 7: BotRGCN baseline with per-degree-bucket evaluation.
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import Adam
from torch_geometric.nn import RGCNConv

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "data/twibot-20"
GRAPH_PATH = os.path.join(DATA_DIR, "twibot_graph.pt")
PARQUET_PATH = os.path.join(DATA_DIR, "twibot_df.parquet")
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.json")
OUTPUT_DIR = "results/tables"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "degree_bucket_analysis.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WD = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEGREE_BINS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 50, "11-50"),
    (51, None, "50+"),
]
BIN_LABELS = [label for _, _, label in DEGREE_BINS]


def bin_degree(d, bins):
    for lo, hi, label in bins:
        if hi is None:
            if d >= lo:
                return label
        elif lo <= d <= hi:
            return label
    return "unknown"


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
y_labeled = y[LABELED_MASK]

ef = graph.edge_index_follow.numpy()        # [2, E_follow]
eg = graph.edge_index_following.numpy()     # [2, E_following]
ei = graph.edge_index.numpy()               # [2, E_undirected]

# ── Precompute per-node degrees ─────────────────────────────────────
print("Computing degree features ...")

follow_in = np.bincount(ef[1], minlength=N_NODES).astype(np.int32)
follow_out = np.bincount(ef[0], minlength=N_NODES).astype(np.int32)
follow_total = follow_in + follow_out

following_in = np.bincount(eg[1], minlength=N_NODES).astype(np.int32)
following_out = np.bincount(eg[0], minlength=N_NODES).astype(np.int32)
following_total = following_in + following_out

adj = [set() for _ in range(N_NODES)]
for u, v in zip(ei[0], ei[1]):
    adj[int(u)].add(int(v))
unique_neighbors = np.array([len(s) for s in adj], dtype=np.int32)
del adj

labeled_idx = LABELED_MASK.nonzero(as_tuple=False).view(-1).numpy()
deg_labeled = unique_neighbors[labeled_idx]
bin_assignment = np.array([bin_degree(d, DEGREE_BINS) for d in deg_labeled])

# ── Helper: per-node homophily ──────────────────────────────────────
def compute_homophily(y_tensor, edge_index_tensor):
    """Per-node: fraction of incident edges whose other endpoint shares the same label."""
    labeled = y_tensor >= 0
    src, dst = edge_index_tensor
    mask = labeled[src] & labeled[dst]
    s, d = src[mask], dst[mask]
    same = (y_tensor[s] == y_tensor[d]).float()
    N = y_tensor.size(0)
    total = torch.zeros(N)
    same_sum = torch.zeros(N)
    total.index_add_(0, s, torch.ones_like(same))
    total.index_add_(0, d, torch.ones_like(same))
    same_sum.index_add_(0, s, same)
    same_sum.index_add_(0, d, same)
    denom = total.clamp(min=1)
    return (same_sum / denom).numpy()

homo_follow = compute_homophily(y, graph.edge_index_follow)
homo_following = compute_homophily(y, graph.edge_index_following)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 1  —  Basic Graph Stats                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 1  —  Basic Graph Stats")
print("=" * 76)

n_total = N_NODES
n_labeled = int(LABELED_MASK.sum())
n_train = int(TRAIN_MASK.sum())
n_val = int(VAL_MASK.sum())
n_test = int(TEST_MASK.sum())
n_support = n_total - n_labeled
n_bots = int((y_labeled == 1).sum())
n_humans = int((y_labeled == 0).sum())

print(f"  Total users:                {n_total:>10,}")
print(f"  Labeled users:              {n_labeled:>10,}")
print(f"    Bots:                     {n_bots:>10,}  ({n_bots/n_labeled*100:.1f}%)")
print(f"    Humans:                   {n_humans:>10,}  ({n_humans/n_labeled*100:.1f}%)")
print(f"    Support (unlabeled):      {n_support:>10,}")
print(f"  Split: train={n_train:,}  val={n_val:,}  test={n_test:,}")
print()
print(f"  Follow edges (followed → follower):      {ef.shape[1]:>10,}")
print(f"  Following edges (follower → followed):   {eg.shape[1]:>10,}")
print(f"  Total directed edges:                    {ef.shape[1] + eg.shape[1]:>10,}")
print(f"  Combined undirected (both directions):   {ei.shape[1]:>10,}")
print(f"  Unique undirected pairs:                 {ei.shape[1] // 2:>10,}")
print()
print("  Graph is DIRECTED — follower and following are stored as separate")
print("  edge_index tensors (follow, following) in the graph.pt file, each")
print("  representing a different relation type for RGCN.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 2  —  Degree Distribution (labeled nodes only)                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 2  —  Degree Distribution (labeled nodes, n=" + f"{n_labeled:,})")
print("=" * 76)


def percentiles_str(arr):
    ps = [0, 5, 25, 50, 75, 95, 100]
    vals = np.percentile(arr, ps)
    return ", ".join(f"p{p}={int(vals[i])}" for i, p in enumerate(ps))


def describe_deg(arr, label):
    print(f"\n  {label}:")
    print(f"    {percentiles_str(arr)}")
    print(f"    min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, "
          f"std={arr.std():.1f}")


deg_labeled_dict = {
    "Follow in-degree (#PPL this node is a follower of)": follow_in[labeled_idx],
    "Follow out-degree (#followers of this node)": follow_out[labeled_idx],
    "Follow total degree": follow_total[labeled_idx],
    "Following in-degree (#followers, from following edges)": following_in[labeled_idx],
    "Following out-degree (#PPL this node follows)": following_out[labeled_idx],
    "Following total degree": following_total[labeled_idx],
    "Combined unique neighbor count": deg_labeled,
}

for label, arr in deg_labeled_dict.items():
    describe_deg(arr, label)

# Bin histogram
print(f"\n  {'─'*72}")
print(f"  {'Degree bin (unique neighbors)':<28} {'Count':>8} {'%':>8}  {'Cum %':>8}")
print(f"  {'─'*72}")
cum = 0
bin_counts = {}
for _, _, bname in DEGREE_BINS:
    cnt = (bin_assignment == bname).sum()
    pct = cnt / n_labeled * 100
    cum += pct
    bin_counts[bname] = cnt
    print(f"  {bname:<28} {cnt:>8,} {pct:>7.1f}%  {cum:>7.1f}%")
print(f"  {'─'*72}")
print(f"  {'Total':<28} {n_labeled:>8,}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 3  —  Label Balance within Each Degree Bin                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 3  —  Label Balance within Each Degree Bin")
print("=" * 76)

print(f"\n  {'Bin':<12} {'Total':>7} {'Bots':>7} {'Humans':>7} {'Bot ratio':>10}  Signal")
print(f"  {'─'*54}")
task3_rows = []
for _, _, bname in DEGREE_BINS:
    mask = bin_assignment == bname
    bin_y = y_labeled[mask].numpy()
    n = len(bin_y)
    n_b = int((bin_y == 1).sum())
    n_h = int((bin_y == 0).sum())
    br = n_b / n if n > 0 else 0
    signal = ""
    if n >= 30:
        if br > 0.85:
            signal = "<<< strongly bot-dominated"
        elif br < 0.15:
            signal = "<<< strongly human-dominated"
        elif br > 0.65:
            signal = "< bot-skewed"
        elif br < 0.35:
            signal = "< human-skewed"
    task3_rows.append({"bin": bname, "n": n, "n_bots": n_b, "n_humans": n_h, "bot_ratio": round(br, 4)})
    print(f"  {bname:<12} {n:>7,} {n_b:>7,} {n_h:>7,} {br*100:>8.1f}%   {signal}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 4  —  Split Distribution across Degree Bins                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 4  —  Split Distribution across Degree Bins")
print("=" * 76)

train_bin_assignment = bin_assignment[TRAIN_MASK[LABELED_MASK].numpy()]
val_bin_assignment = bin_assignment[VAL_MASK[LABELED_MASK].numpy()]
test_bin_assignment = bin_assignment[TEST_MASK[LABELED_MASK].numpy()]

task4_rows = []
print(f"\n  {'Bin':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print(f"  {'─'*50}")
for _, _, bname in DEGREE_BINS:
    tc = (train_bin_assignment == bname).sum()
    vc = (val_bin_assignment == bname).sum()
    ssc = (test_bin_assignment == bname).sum()
    tot = tc + vc + ssc
    flag = ""
    if ssc > 0 and bname in ("0", "1-2") and ssc < 30:
        flag = "  ⚠ too few for statistical claim"
    task4_rows.append({"bin": bname, "train": tc, "val": vc, "test": ssc, "total": tot})
    print(f"  {bname:<12} {tc:>8,} {vc:>8,} {ssc:>8,} {tot:>8,}{flag}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 5  —  Homophily Ratio per Relation Type                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 5  —  Homophily Ratio per Relation Type")
print("=" * 76)

# Only consider nodes that have at least one labeled neighbor
valid_follow = homo_follow[labeled_idx]
valid_follow_mask = valid_follow >= 0
valid_following = homo_following[labeled_idx]
valid_following_mask = valid_following >= 0

print("\n  Overall (labeled nodes with degree ≥ 1 in that relation):")
print(f"    Follow homophily:    mean={valid_follow[valid_follow_mask].mean():.4f}  "
      f"median={np.median(valid_follow[valid_follow_mask]):.4f}  "
      f"(n={valid_follow_mask.sum():,})")
print(f"    Following homophily: mean={valid_following[valid_following_mask].mean():.4f}  "
      f"median={np.median(valid_following[valid_following_mask]):.4f}  "
      f"(n={valid_following_mask.sum():,})")

# Per degree bin
print("\n  Per degree bin (unique-neighbor bins):")
print(f"  {'Bin':<12} {'Follow homo':>14} {'N':>7}  {'Following homo':>14} {'N':>7}")
print(f"  {'─'*61}")
task5_rows = []
for _, _, bname in DEGREE_BINS:
    mask = bin_assignment == bname
    hf = homo_follow[labeled_idx][mask]
    hg = homo_following[labeled_idx][mask]
    vf = hf >= 0
    vg = hg >= 0
    mf = hf[vf].mean() if vf.sum() > 0 else float("nan")
    mg = hg[vg].mean() if vg.sum() > 0 else float("nan")
    nf = vf.sum()
    ng = vg.sum()
    task5_rows.append({"bin": bname, "follow_homo_mean": round(float(mf), 4) if not np.isnan(mf) else None,
                       "follow_homo_n": nf, "following_homo_mean": round(float(mg), 4) if not np.isnan(mg) else None,
                       "following_homo_n": ng})
    print(f"  {bname:<12} {mf:>14.4f} {nf:>7,}  {mg:>14.4f} {ng:>7,}")

if valid_follow_mask.sum() > 0:
    lo_homo = valid_follow[valid_follow_mask] < 0.3
    print(f"\n  Follow relation — nodes with homophily < 0.3: "
          f"{lo_homo.sum():,}/{valid_follow_mask.sum():,} "
          f"({lo_homo.sum()/valid_follow_mask.sum()*100:.1f}%)")
if valid_following_mask.sum() > 0:
    lo_homo = valid_following[valid_following_mask] < 0.3
    print(f"  Following relation — nodes with homophily < 0.3: "
          f"{lo_homo.sum():,}/{valid_following_mask.sum():,} "
          f"({lo_homo.sum()/valid_following_mask.sum()*100:.1f}%)")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 6  —  Feature Availability for Low-Degree Nodes                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 6  —  Feature Availability for Low-Degree Nodes")
print("=" * 76)

profile_feat_names = feature_names["profile"]
tweet_feat_names = feature_names["tweet"]

df_labeled = df.iloc[labeled_idx].copy()
df_labeled["bin"] = bin_assignment
df_labeled["deg"] = deg_labeled

for bname in ["0", "1-2"]:
    sub = df_labeled[df_labeled["bin"] == bname]
    n = len(sub)
    if n == 0:
        print(f"\n  Bin '{bname}': no nodes.")
        continue

    # Profile completeness: fraction of profile features with plausible non-zero values
    profile_cols = [c for c in profile_feat_names if c in sub.columns]
    n_profile_zero = (sub[profile_cols] == 0).sum(axis=1)
    profile_completeness = 1.0 - (n_profile_zero / len(profile_cols)).mean()

    has_tweets = (sub["tweet_count"] > 0)
    pct_has_tweets = has_tweets.mean() * 100
    avg_tweets = sub["tweet_count"].mean()

    print(f"\n  Bin '{bname}' (n={n:,}):")
    print(f"    Profile feature completeness (frac non-zero):  {profile_completeness:.1%}")
    print(f"    Nodes with ≥ 1 tweet:                          {has_tweets.sum():,} / {n:,}  ({pct_has_tweets:.1f}%)")
    print(f"    Mean tweet count:                               {avg_tweets:.1f}")
    print(f"    Mean tweet_count (among those with tweets):     {sub[has_tweets]['tweet_count'].mean():.1f}")

    # Check individual key features
    for feat in ["followers_count", "statuses_count", "tweet_count"]:
        if feat in sub.columns:
            nz = (sub[feat] > 0).sum()
            print(f"    {feat} > 0:  {nz:,}/{n:,}  ({nz/n*100:.1f}%)")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TASK 7  —  BotRGCN Baseline (per-degree-bucket evaluation)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("TASK 7  —  BotRGCN Baseline Model")
print("=" * 76)
print(f"  Device: {DEVICE}")
print(f"  Seeds:  {SEEDS}")
print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}, WD: {WD}")
print()

# ── Standardise features (train-set stats) ──────────────────────────
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

# ── Model ────────────────────────────────────────────────────────────
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
    pos_weight = (n_neg / (n_pos + 1e-8))

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


print("Training BotRGCN ...")
print()

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

print()
print("  ── Overall (mean ± std over seeds) ──")
accs = [m["acc"] for m in seed_metrics]
f1ms = [m["f1_macro"] for m in seed_metrics]
aucs = [m["auc"] for m in seed_metrics]
print(f"  Accuracy:   {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"  F1 macro:   {np.mean(f1ms):.4f} ± {np.std(f1ms):.4f}")
print(f"  AUC:        {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# ── Per-bucket evaluation ───────────────────────────────────────────
print()
print("  ── Per Degree-Bucket Evaluation (test set, macro F1 over seeds) ──")
print(f"  {'Bin':<12} {'N_test':>8} {'Acc':>8} {'F1_macro':>10} {'AUC':>8}")
print(f"  {'─'*52}")

test_bin_labels = test_bin_assignment
task7_rows = []
for _, _, bname in DEGREE_BINS:
    bmask = test_bin_labels == bname
    n_bin = bmask.sum()
    if n_bin == 0:
        print(f"  {bname:<12} {n_bin:>8}  {'—':>8}  {'—':>10}  {'—':>8}")
        continue

    bucket_f1s, bucket_aucs, bucket_accs = [], [], []
    for s_idx in range(len(SEEDS)):
        yt_s = all_test_y_true[s_idx][bmask]
        yp_s = all_test_y_pred[s_idx][bmask]
        ypr_s = all_test_y_prob[s_idx][bmask]
        if len(set(yt_s)) < 2:
            bucket_f1s.append(float("nan"))
            bucket_aucs.append(float("nan"))
        else:
            bucket_f1s.append(f1_score(yt_s, yp_s, average="macro"))
            bucket_aucs.append(roc_auc_score(yt_s, ypr_s))
        bucket_accs.append(accuracy_score(yt_s, yp_s))

    f1m_mean = np.nanmean(bucket_f1s)
    f1m_std = np.nanstd(bucket_f1s)
    auc_mean = np.nanmean(bucket_aucs)
    auc_std = np.nanstd(bucket_aucs)
    acc_mean = np.nanmean(bucket_accs)
    acc_std = np.nanstd(bucket_accs)

    flag = " ⚠ small" if n_bin < 30 else ""
    print(f"  {bname:<12} {n_bin:>8,} {acc_mean:>7.4f} {f1m_mean:>9.4f} {auc_mean:>7.4f}{flag}")
    task7_rows.append({
        "bin": bname, "n_test": n_bin,
        "acc_mean": round(acc_mean, 4), "acc_std": round(acc_std, 4),
        "f1_macro_mean": round(f1m_mean, 4), "f1_macro_std": round(f1m_std, 4),
        "auc_mean": round(auc_mean, 4), "auc_std": round(auc_std, 4),
    })

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Save results                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
rows = []
for _, _, bname in DEGREE_BINS:
    r = {"bin": bname}
    r["count"] = bin_counts.get(bname, 0)
    tr3 = next((x for x in task3_rows if x["bin"] == bname), {})
    r["n_bots"] = tr3.get("n_bots", 0)
    r["n_humans"] = tr3.get("n_humans", 0)
    r["bot_ratio"] = tr3.get("bot_ratio", 0)
    tr4 = next((x for x in task4_rows if x["bin"] == bname), {})
    r["train"] = tr4.get("train", 0)
    r["val"] = tr4.get("val", 0)
    r["test"] = tr4.get("test", 0)
    tr5 = next((x for x in task5_rows if x["bin"] == bname), {})
    r["follow_homo"] = tr5.get("follow_homo_mean")
    r["following_homo"] = tr5.get("following_homo_mean")
    tr7 = next((x for x in task7_rows if x["bin"] == bname), {})
    r["acc_mean"] = tr7.get("acc_mean")
    r["f1_macro_mean"] = tr7.get("f1_macro_mean")
    r["auc_mean"] = tr7.get("auc_mean")
    rows.append(r)

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_CSV}")

print()
print("=" * 76)
print("Analysis complete.")
print("=" * 76)
