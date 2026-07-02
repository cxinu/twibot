#!/usr/bin/env python3
"""Evaluate the per-relation gated RGCN fix against BotRGCN baseline.

Trains three variants on TwiBot-20:
  1. BotRGCN (plain baseline)
  2. GatedBotRGCN-global   (single shared low/high gate across relations)
  3. GatedBotRGCN-rel      (proposed: relation-specific gates)

Reports overall Acc/F1/AUC and per-homophily-bucket performance (combined,
follow-relation, following-relation), plus confusion-matrix breakdown for the
combined-homophily-0 bucket.
"""

import json
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.optim import Adam

from models import BotRGCN, GatedBotRGCN, SoftContrastBotRGCN

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "data/twibot-20"
GRAPH_PATH = os.path.join(DATA_DIR, "twibot_graph.pt")
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.json")
OUTPUT_DIR = "results/tables"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "heterophily_fix_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WD = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ── Per-node homophily (same definition as heterophily_analysis.py) ──
def per_node_homophily(y_tensor, edge_index_tensor):
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
    return (same_sum / denom).numpy()


print("Computing per-node homophily ...")
homo_follow = per_node_homophily(y, graph.edge_index_follow)
homo_following = per_node_homophily(y, graph.edge_index_following)
combined_edge_index = torch.cat([graph.edge_index_follow, graph.edge_index_following], dim=1)
homo_combined = per_node_homophily(y, combined_edge_index)

test_idx = TEST_MASK.nonzero(as_tuple=False).view(-1).numpy()
h_test_combined = homo_combined[test_idx]
h_test_follow = homo_follow[test_idx]
h_test_following = homo_following[test_idx]


# ── Training / evaluation utilities ─────────────────────────────────
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


def bucket_metrics(y_true, y_pred, y_prob, h_vals, relation_name, variant):
    bin_labels = np.array([homo_bin_label(h) for h in h_vals])
    rows = []
    for _, _, bname in HOMOPHILY_BINS:
        mask = bin_labels == bname
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        yt_b = y_true[mask]
        yp_b = y_pred[mask]
        ypr_b = y_prob[mask]
        acc_b = accuracy_score(yt_b, yp_b)
        f1_b = f1_score(yt_b, yp_b, average="macro") if len(set(yt_b)) > 1 else float("nan")
        auc_b = roc_auc_score(yt_b, ypr_b) if len(set(yt_b)) > 1 else float("nan")
        rows.append({
            "variant": variant,
            "relation": relation_name,
            "bucket": bname,
            "n_test": n_bin,
            "acc": round(acc_b, 4),
            "f1_macro": round(f1_b, 4),
            "auc": round(auc_b, 4),
        })
    return rows


def low_homo_confusion(y_true, y_pred, h_vals, variant):
    mask = np.array([homo_bin_label(h) for h in h_vals]) == "0"
    n = mask.sum()
    if n == 0:
        return None
    yt = y_true[mask]
    yp = y_pred[mask]
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return {
        "variant": variant,
        "bucket": "0",
        "n": n,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "fp_rate_human": round(cm[0, 1] / cm[0, :].sum(), 4) if cm[0, :].sum() > 0 else 0,
        "fn_rate_bot": round(cm[1, 0] / cm[1, :].sum(), 4) if cm[1, :].sum() > 0 else 0,
    }


# ── Model configs ───────────────────────────────────────────────────
configs = [
    {
        "name": "BotRGCN",
        "model_fn": lambda: BotRGCN(P_DIM, T_DIM, TO_DIM, N_DIM),
    },
    {
        "name": "GatedBotRGCN-global",
        "model_fn": lambda: GatedBotRGCN(P_DIM, T_DIM, TO_DIM, N_DIM, relation_specific=False),
    },
    {
        "name": "SoftContrastBotRGCN-global",
        "model_fn": lambda: SoftContrastBotRGCN(P_DIM, T_DIM, TO_DIM, N_DIM, relation_specific=False),
    },
    {
        "name": "SoftContrastBotRGCN-rel",
        "model_fn": lambda: SoftContrastBotRGCN(P_DIM, T_DIM, TO_DIM, N_DIM, relation_specific=True),
    },
]

all_results = []
all_confusion = []

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Train / evaluate each variant                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("Training and evaluating variants")
print("=" * 76)
print(f"  Device: {DEVICE}")
print(f"  Seeds:  {SEEDS}")
print()

for cfg in configs:
    name = cfg["name"]
    print(f"\n{'─'*76}")
    print(f"Variant: {name}")
    print(f"{'─'*76}")

    seed_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for s in SEEDS:
        print(f"  Seed {s} ...", end="", flush=True)
        model = cfg["model_fn"]()
        model = train_one_seed(model, graph, TRAIN_MASK, VAL_MASK, s)
        yt, yp, ypr, acc, f1m, f1b, auc = evaluate(model, graph, TEST_MASK)
        all_y_true.append(yt)
        all_y_pred.append(yp)
        all_y_prob.append(ypr)
        seed_metrics.append({"acc": acc, "f1_macro": f1m, "auc": auc})
        print(f"  Acc={acc:.4f}  F1_macro={f1m:.4f}  AUC={auc:.4f}" if not np.isnan(auc)
              else f"  Acc={acc:.4f}  F1_macro={f1m:.4f}  AUC=nan")

    # Aggregate over seeds
    acc_mean = np.mean([m["acc"] for m in seed_metrics])
    acc_std = np.std([m["acc"] for m in seed_metrics])
    f1_mean = np.mean([m["f1_macro"] for m in seed_metrics])
    f1_std = np.std([m["f1_macro"] for m in seed_metrics])
    auc_mean = np.mean([m["auc"] for m in seed_metrics])
    auc_std = np.std([m["auc"] for m in seed_metrics])

    print(f"\n  Overall: Acc={acc_mean:.4f}±{acc_std:.4f}  "
          f"F1_macro={f1_mean:.4f}±{f1_std:.4f}  AUC={auc_mean:.4f}±{auc_std:.4f}")

    all_results.append({
        "variant": name, "metric": "overall",
        "acc": round(acc_mean, 4), "acc_std": round(acc_std, 4),
        "f1_macro": round(f1_mean, 4), "f1_macro_std": round(f1_std, 4),
        "auc": round(auc_mean, 4), "auc_std": round(auc_std, 4),
    })

    # Ensemble predictions for per-bucket / confusion analysis
    y_true_ens = all_y_true[0]
    y_prob_ens = np.mean(all_y_prob, axis=0)
    y_pred_ens = (y_prob_ens >= 0.5).astype(int)

    # Per-bucket tables
    all_results.extend(bucket_metrics(y_true_ens, y_pred_ens, y_prob_ens,
                                      h_test_combined, "combined", name))
    all_results.extend(bucket_metrics(y_true_ens, y_pred_ens, y_prob_ens,
                                      h_test_follow, "follow", name))
    all_results.extend(bucket_metrics(y_true_ens, y_pred_ens, y_prob_ens,
                                      h_test_following, "following", name))

    # Confusion matrix for combined-homophily-0 bucket
    conf = low_homo_confusion(y_true_ens, y_pred_ens, h_test_combined, name)
    if conf is not None:
        all_confusion.append(conf)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Print summary tables                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("Summary: Overall performance")
print("=" * 76)
overall_df = pd.DataFrame([r for r in all_results if r.get("metric") == "overall"])
print(overall_df[["variant", "acc", "f1_macro", "auc"]].to_string(index=False))


def print_bucket_table(relation_name):
    print()
    print(f"  Per-bucket F1_macro ({relation_name})")
    sub = pd.DataFrame([r for r in all_results if r.get("relation") == relation_name])
    pivot = sub.pivot(index="bucket", columns="variant", values="f1_macro")
    pivot = pivot.reindex([b for _, _, b in HOMOPHILY_BINS])
    print(pivot.to_string())
    counts = sub.groupby("bucket")["n_test"].first().reindex([b for _, _, b in HOMOPHILY_BINS])
    print(f"  n_test per bucket: {counts.to_dict()}")


print_bucket_table("combined")
print_bucket_table("follow")
print_bucket_table("following")

print()
print("=" * 76)
print("Summary: Confusion matrix for combined-homophily-0 bucket")
print("=" * 76)
conf_df = pd.DataFrame(all_confusion)
print(conf_df[["variant", "n", "tn", "fp", "fn", "tp",
               "fp_rate_human", "fn_rate_bot"]].to_string(index=False))

# ── Save CSV ────────────────────────────────────────────────────────
pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_CSV}")

conf_csv = os.path.join(OUTPUT_DIR, "heterophily_fix_confusion.csv")
pd.DataFrame(all_confusion).to_csv(conf_csv, index=False)
print(f"Saved {conf_csv}")

print()
print("=" * 76)
print("Heterophily fix evaluation complete.")
print("=" * 76)
