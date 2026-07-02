#!/usr/bin/env python3
"""Evaluate the per-relation gated RGCN fix against BotRGCN baseline.

This version uses the original BotRGCN preprocessing:
    - RoBERTa description embeddings  (768-dim)
    - RoBERTa tweet embeddings        (768-dim)
    - 5 standardized numerical properties
    - 3 categorical properties

and the original BotRGCN training recipe:
    - CrossEntropyLoss
    - AdamW(lr=1e-2, weight_decay=5e-2)
    - 50 fixed epochs (no early stopping)

Reports overall Acc/F1/MCC and per-homophily-bucket performance (combined,
follow-relation, following-relation), plus confusion-matrix breakdown for the
combined-homophily-0 bucket.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from torch.optim import AdamW

from models import BotRGCN, GatedBotRGCN, SoftContrastBotRGCN

warnings.filterwarnings("ignore")

# ── Config (matches original train.py) ──────────────────────────────
DATA_DIR = "data/twibot-20"
GRAPH_PATH = os.path.join(DATA_DIR, "twibot_graph.pt")
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.json")
OUTPUT_DIR = "results/tables"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "heterophily_fix_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 2024, 9999]
EPOCHS = 50
LR = 1e-2
WD = 5e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Homophily buckets ───────────────────────────────────────────────
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
    _ = json.load(f)  # kept for compatibility

# Original paper features (already preprocessed by the authors).
x_des = torch.load(os.path.join(DATA_DIR, "des_tensor.pt"), map_location="cpu", weights_only=False).float()
x_tweet = torch.load(os.path.join(DATA_DIR, "tweets_tensor.pt"), map_location="cpu", weights_only=False).float()
x_num_prop = torch.load(os.path.join(DATA_DIR, "num_properties_tensor.pt"), map_location="cpu", weights_only=False).float()
x_cat_prop = torch.load(os.path.join(DATA_DIR, "cat_properties_tensor.pt"), map_location="cpu", weights_only=False).float()

N_NODES = graph.num_nodes
LABELED_MASK = graph.y >= 0
TRAIN_MASK = graph.train_mask
VAL_MASK = graph.val_mask
TEST_MASK = graph.test_mask
y = graph.y.long()

DES_DIM = x_des.size(1)
TWEET_DIM = x_tweet.size(1)
NUM_DIM = x_num_prop.size(1)
CAT_DIM = x_cat_prop.size(1)


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
    xdes = x_des.to(DEVICE)
    xtweet = x_tweet.to(DEVICE)
    xnum = x_num_prop.to(DEVICE)
    xcat = x_cat_prop.to(DEVICE)
    tm = train_mask.to(DEVICE)
    vm = val_mask.to(DEVICE)
    y_dev = y.to(DEVICE)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(xdes, xtweet, xnum, xcat, d.edge_index_rgcn, d.edge_type)
        train_logits = logits[tm]
        train_y = y_dev[tm]
        loss = loss_fn(train_logits, train_y)
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                logits = model(xdes, xtweet, xnum, xcat, d.edge_index_rgcn, d.edge_type)
                acc_train = (logits[tm].argmax(dim=1) == y_dev[tm]).float().mean().item()
                acc_val = (logits[vm].argmax(dim=1) == y_dev[vm]).float().mean().item()
            print(f"    Epoch {epoch+1:04d}: loss={loss.item():.4f}, "
                  f"train_acc={acc_train:.4f}, val_acc={acc_val:.4f}")

    model.eval()
    return model


def evaluate(model, data, mask):
    model.eval()
    d = data.to(DEVICE)
    xdes = x_des.to(DEVICE)
    xtweet = x_tweet.to(DEVICE)
    xnum = x_num_prop.to(DEVICE)
    xcat = x_cat_prop.to(DEVICE)
    m = mask.to(DEVICE)
    y_dev = y.to(DEVICE)
    with torch.no_grad():
        logits = model(xdes, xtweet, xnum, xcat, d.edge_index_rgcn, d.edge_type)
    y_prob = F.softmax(logits[m], dim=1)[:, 1].cpu().numpy()
    y_true = y_dev[m].cpu().numpy()
    y_pred = logits[m].argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1_binary = f1_score(y_true, y_pred, average="binary")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    return y_true, y_pred, y_prob, acc, f1_binary, f1_macro, mcc


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
        both_classes = len(set(yt_b)) > 1
        acc_b = accuracy_score(yt_b, yp_b)
        f1_binary_b = f1_score(yt_b, yp_b, average="binary")
        f1_macro_b = f1_score(yt_b, yp_b, average="macro") if both_classes else float("nan")
        mcc_b = matthews_corrcoef(yt_b, yp_b) if both_classes else float("nan")
        rows.append({
            "variant": variant,
            "relation": relation_name,
            "bucket": bname,
            "n_test": n_bin,
            "acc": round(acc_b, 4),
            "f1_binary": round(f1_binary_b, 4),
            "f1_macro": round(f1_macro_b, 4),
            "mcc": round(mcc_b, 4),
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
        "model_fn": lambda: BotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
        ),
    },
    {
        "name": "GatedBotRGCN-global",
        "model_fn": lambda: GatedBotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
            relation_specific=False,
        ),
    },
    {
        "name": "GatedBotRGCN-rel",
        "model_fn": lambda: GatedBotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
            relation_specific=True,
        ),
    },
    {
        "name": "SoftContrastBotRGCN-global",
        "model_fn": lambda: SoftContrastBotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
            relation_specific=False,
        ),
    },
    {
        "name": "SoftContrastBotRGCN-rel",
        "model_fn": lambda: SoftContrastBotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
            relation_specific=True,
        ),
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
print(f"  Epochs: {EPOCHS}, LR: {LR}, WD: {WD}")
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
        yt, yp, ypr, acc, f1_binary, f1_macro, mcc = evaluate(model, graph, TEST_MASK)
        all_y_true.append(yt)
        all_y_pred.append(yp)
        all_y_prob.append(ypr)
        seed_metrics.append({"acc": acc, "f1_binary": f1_binary,
                             "f1_macro": f1_macro, "mcc": mcc})
        print(f"  Acc={acc:.4f}  F1={f1_binary:.4f}  "
              f"F1_macro={f1_macro:.4f}  MCC={mcc:.4f}")

    # Aggregate over seeds
    acc_mean = np.mean([m["acc"] for m in seed_metrics])
    acc_std = np.std([m["acc"] for m in seed_metrics])
    f1_binary_mean = np.mean([m["f1_binary"] for m in seed_metrics])
    f1_binary_std = np.std([m["f1_binary"] for m in seed_metrics])
    f1_macro_mean = np.mean([m["f1_macro"] for m in seed_metrics])
    f1_macro_std = np.std([m["f1_macro"] for m in seed_metrics])
    mcc_mean = np.mean([m["mcc"] for m in seed_metrics])
    mcc_std = np.std([m["mcc"] for m in seed_metrics])

    print(f"\n  Overall: Acc={acc_mean:.4f}±{acc_std:.4f}  "
          f"F1={f1_binary_mean:.4f}±{f1_binary_std:.4f}  "
          f"F1_macro={f1_macro_mean:.4f}±{f1_macro_std:.4f}  "
          f"MCC={mcc_mean:.4f}±{mcc_std:.4f}")

    all_results.append({
        "variant": name, "metric": "overall",
        "acc": round(acc_mean, 4), "acc_std": round(acc_std, 4),
        "f1_binary": round(f1_binary_mean, 4), "f1_binary_std": round(f1_binary_std, 4),
        "f1_macro": round(f1_macro_mean, 4), "f1_macro_std": round(f1_macro_std, 4),
        "mcc": round(mcc_mean, 4), "mcc_std": round(mcc_std, 4),
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
overall_display = overall_df.rename(columns={"f1_binary": "f1", "f1_macro": "f1_macro"})
print(overall_display[["variant", "acc", "f1", "f1_macro", "mcc"]].to_string(index=False))


def print_bucket_table(relation_name, metric_name="f1_binary", display_name="F1"):
    print()
    print(f"  Per-bucket {display_name} ({relation_name})")
    sub = pd.DataFrame([r for r in all_results if r.get("relation") == relation_name])
    pivot = sub.pivot(index="bucket", columns="variant", values=metric_name)
    pivot = pivot.reindex([b for _, _, b in HOMOPHILY_BINS])
    print(pivot.to_string())
    counts = sub.groupby("bucket")["n_test"].first().reindex([b for _, _, b in HOMOPHILY_BINS])
    print(f"  n_test per bucket: {counts.to_dict()}")


print_bucket_table("combined", "f1_binary")
print_bucket_table("combined", "f1_macro")
print_bucket_table("follow", "f1_binary")
print_bucket_table("following", "f1_binary")

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
