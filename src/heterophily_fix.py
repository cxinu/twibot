#!/usr/bin/env python3
"""Evaluate gated RGCN fixes with RoBERTa features and proper methodology.

Key features:
    - Uses original BotRGCN features + hyperparameters.
    - Validation-set model selection (not test-set cherry-picking).
    - Per-seed threshold tuning on validation F1.
    - Paired seed deltas vs. baseline + Wilcoxon signed-rank test.
    - Gate activation diagnostics.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
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

THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.05), 2)


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

x_des = torch.load(os.path.join(DATA_DIR, "des_tensor.pt"), map_location="cpu", weights_only=False).float()
x_tweet = torch.load(os.path.join(DATA_DIR, "tweets_tensor.pt"), map_location="cpu", weights_only=False).float()
x_num_prop = torch.load(os.path.join(DATA_DIR, "num_properties_tensor.pt"), map_location="cpu", weights_only=False).float()
x_cat_prop = torch.load(os.path.join(DATA_DIR, "cat_properties_tensor.pt"), map_location="cpu", weights_only=False).float()

TRAIN_MASK = graph.train_mask
VAL_MASK = graph.val_mask
TEST_MASK = graph.test_mask
y = graph.y.long()

DES_DIM = x_des.size(1)
TWEET_DIM = x_tweet.size(1)
NUM_DIM = x_num_prop.size(1)
CAT_DIM = x_cat_prop.size(1)


# ── Per-node homophily ──────────────────────────────────────────────
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

test_idx_np = TEST_MASK.nonzero(as_tuple=False).view(-1).numpy()
h_test_combined = homo_combined[test_idx_np]
h_test_follow = homo_follow[test_idx_np]
h_test_following = homo_following[test_idx_np]


# ── Evaluation helpers ─────────────────────────────────────────────-
def metrics_at_threshold(y_prob, y_true, threshold):
    """y_prob: bot-class probabilities. threshold: scalar."""
    y_pred = (y_prob >= threshold).astype(int)
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return {"acc": accuracy_score(y_true, y_pred), "f1": 0.0, "mcc": 0.0}
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def find_best_threshold(y_prob, y_true):
    best = {"threshold": 0.5, "f1": -1.0}
    for thr in THRESHOLDS:
        m = metrics_at_threshold(y_prob, y_true, thr)
        if m["f1"] > best["f1"]:
            best = {"threshold": thr, "f1": m["f1"], **m}
    return best


def train_one_seed(model, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(DEVICE)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    xdes = x_des.to(DEVICE)
    xtweet = x_tweet.to(DEVICE)
    xnum = x_num_prop.to(DEVICE)
    xcat = x_cat_prop.to(DEVICE)
    edge_index = graph.edge_index_rgcn.to(DEVICE)
    edge_type = graph.edge_type.to(DEVICE)
    y_dev = y.to(DEVICE)
    tm = TRAIN_MASK.to(DEVICE)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(xdes, xtweet, xnum, xcat, edge_index, edge_type)
        loss = loss_fn(logits[tm], y_dev[tm])
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                logits = model(xdes, xtweet, xnum, xcat, edge_index, edge_type)
                acc_train = (logits[tm].argmax(dim=1) == y_dev[tm]).float().mean().item()
                acc_val = (logits[VAL_MASK.to(DEVICE)].argmax(dim=1) == y_dev[VAL_MASK.to(DEVICE)]).float().mean().item()
            print(f"    Epoch {epoch+1:04d}: loss={loss.item():.4f}, "
                  f"train_acc={acc_train:.4f}, val_acc={acc_val:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(xdes, xtweet, xnum, xcat, edge_index, edge_type)
    y_prob_full = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return model, y_prob_full


def extract_gate_stats(model, mask):
    """Return per-relation gate means for gated models, stratified by homophily."""
    if not hasattr(model, "forward_and_gates"):
        return None

    xdes = x_des.to(DEVICE)
    xtweet = x_tweet.to(DEVICE)
    xnum = x_num_prop.to(DEVICE)
    xcat = x_cat_prop.to(DEVICE)
    edge_index = graph.edge_index_rgcn.to(DEVICE)
    edge_type = graph.edge_type.to(DEVICE)

    model.eval()
    with torch.no_grad():
        _, gates_per_layer = model.forward_and_gates(
            xdes, xtweet, xnum, xcat, edge_index, edge_type
        )

    h_vals = homo_combined[test_idx_np]
    bin_labels = np.array([homo_bin_label(h) for h in h_vals])
    rows = []
    for r in range(2):
        gate_sum = torch.zeros(graph.num_nodes, 1, device=DEVICE)
        for layer_gates in gates_per_layer:
            gate_sum += layer_gates[r]
        gate_mean_per_node = (gate_sum / len(gates_per_layer)).squeeze(-1)[mask].cpu().numpy()

        for _, _, bname in HOMOPHILY_BINS:
            bmask = bin_labels == bname
            n_bin = bmask.sum()
            if n_bin == 0:
                continue
            rows.append({
                "relation": "follow" if r == 0 else "following",
                "bucket": bname,
                "gate_mean": round(float(gate_mean_per_node[bmask].mean()), 4),
                "gate_std": round(float(gate_mean_per_node[bmask].std()), 4),
                "n": int(n_bin),
            })
    return rows


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
            relation_specific=False, use_feature_similarity=True,
        ),
    },
    {
        "name": "GatedBotRGCN-rel",
        "model_fn": lambda: GatedBotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
            relation_specific=True, use_feature_similarity=True,
        ),
    },
    {
        "name": "SoftContrastBotRGCN-global",
        "model_fn": lambda: SoftContrastBotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
            relation_specific=False, use_feature_similarity=True,
        ),
    },
    {
        "name": "SoftContrastBotRGCN-rel",
        "model_fn": lambda: SoftContrastBotRGCN(
            des_size=DES_DIM, tweet_size=TWEET_DIM,
            num_prop_size=NUM_DIM, cat_prop_size=CAT_DIM,
            relation_specific=True, use_feature_similarity=True,
        ),
    },
]

all_results = []
all_confusion = []
all_seed_probs = {}
all_seed_thresholds = {}
all_seed_val_f1 = {}
all_seed_test_metrics = {}
all_gate_stats = {}

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
print("  Feature similarity in gates: True for gated variants")
print()

y_train_np = y[TRAIN_MASK].numpy()
y_val_np = y[VAL_MASK].numpy()
y_test_np = y[TEST_MASK].numpy()
train_idx_np = TRAIN_MASK.nonzero(as_tuple=False).view(-1).numpy()
val_idx_np = VAL_MASK.nonzero(as_tuple=False).view(-1).numpy()
test_idx_np = TEST_MASK.nonzero(as_tuple=False).view(-1).numpy()

for cfg in configs:
    name = cfg["name"]
    print(f"\n{'─'*76}")
    print(f"Variant: {name}")
    print(f"{'─'*76}")

    seed_probs = []
    seed_thresholds = []
    seed_val_f1 = []
    seed_test_metrics = []
    trained_models = []

    for s in SEEDS:
        print(f"  Seed {s} ...", end="", flush=True)
        model = cfg["model_fn"]()
        model, y_prob_full = train_one_seed(model, s)

        # Anchor training labels so the threshold search does not leak them.
        y_prob_full = y_prob_full.copy()
        y_prob_full[train_idx_np] = y[train_idx_np].float().numpy()

        # Tune threshold on validation F1.
        val_best = find_best_threshold(y_prob_full[val_idx_np], y_val_np)
        thr = val_best["threshold"]

        # Test metrics with tuned threshold.
        test_m = metrics_at_threshold(y_prob_full[test_idx_np], y_test_np, thr)

        seed_probs.append(y_prob_full)
        seed_thresholds.append(thr)
        seed_val_f1.append(val_best["f1"])
        seed_test_metrics.append(test_m)
        trained_models.append(model)
        print(f"  ValF1@thr={thr:.2f}={val_best['f1']:.4f}  "
              f"Test Acc={test_m['acc']:.4f}  F1={test_m['f1']:.4f}  MCC={test_m['mcc']:.4f}")

    # Aggregate over seeds.
    mean_val_f1 = np.mean(seed_val_f1)
    std_val_f1 = np.std(seed_val_f1)
    mean_test = {k: np.mean([m[k] for m in seed_test_metrics]) for k in seed_test_metrics[0]}
    std_test = {k: np.std([m[k] for m in seed_test_metrics]) for k in seed_test_metrics[0]}

    print(f"\n  Val F1 (threshold-tuned): {mean_val_f1:.4f} ± {std_val_f1:.4f}")
    print(f"  Test: Acc={mean_test['acc']:.4f}±{std_test['acc']:.4f}  "
          f"F1={mean_test['f1']:.4f}±{std_test['f1']:.4f}  "
          f"MCC={mean_test['mcc']:.4f}±{std_test['mcc']:.4f}")

    all_seed_probs[name] = seed_probs
    all_seed_thresholds[name] = seed_thresholds
    all_seed_val_f1[name] = seed_val_f1
    all_seed_test_metrics[name] = seed_test_metrics

    all_results.append({
        "variant": name, "metric": "overall",
        "acc": round(mean_test["acc"], 4), "acc_std": round(std_test["acc"], 4),
        "f1_binary": round(mean_test["f1"], 4), "f1_binary_std": round(std_test["f1"], 4),
        "mcc": round(mean_test["mcc"], 4), "mcc_std": round(std_test["mcc"], 4),
    })

    # Ensemble: average probabilities across seeds, then tune threshold on val.
    y_prob_ens = np.mean(seed_probs, axis=0)
    y_prob_ens[train_idx_np] = y[train_idx_np].float().numpy()
    ens_best = find_best_threshold(y_prob_ens[val_idx_np], y_val_np)
    ens_thr = ens_best["threshold"]
    y_pred_ens = (y_prob_ens[test_idx_np] >= ens_thr).astype(int)

    # Per-bucket tables.
    all_results.extend(bucket_metrics(y_test_np, y_pred_ens, y_prob_ens[test_idx_np],
                                      h_test_combined, "combined", name))
    all_results.extend(bucket_metrics(y_test_np, y_pred_ens, y_prob_ens[test_idx_np],
                                      h_test_follow, "follow", name))
    all_results.extend(bucket_metrics(y_test_np, y_pred_ens, y_prob_ens[test_idx_np],
                                      h_test_following, "following", name))

    # Confusion matrix for combined-homophily-0 bucket.
    conf = low_homo_confusion(y_test_np, y_pred_ens, h_test_combined, name)
    if conf is not None:
        all_confusion.append(conf)

    # Gate stats from the last trained model.
    gate_stats = extract_gate_stats(trained_models[-1], TEST_MASK)
    if gate_stats is not None:
        for row in gate_stats:
            row["variant"] = name
        all_gate_stats[name] = gate_stats


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Validation-based model selection                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("Validation-set model selection (threshold-tuned F1)")
print("=" * 76)
val_summary = []
for name in all_seed_val_f1:
    val_summary.append({
        "variant": name,
        "val_f1": np.mean(all_seed_val_f1[name]),
        "val_f1_std": np.std(all_seed_val_f1[name]),
        "test_f1": np.mean([m["f1"] for m in all_seed_test_metrics[name]]),
    })
val_df = pd.DataFrame(val_summary).sort_values("val_f1", ascending=False)
print(val_df.to_string(index=False))

selected_variant = val_df.iloc[0]["variant"]
print(f"\nSelected variant (by validation F1): {selected_variant}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Paired seed comparison against baseline                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("Paired seed deltas vs. BotRGCN baseline")
print("=" * 76)
baseline_test = all_seed_test_metrics["BotRGCN"]

for name in all_seed_test_metrics:
    if name == "BotRGCN":
        continue
    deltas = [all_seed_test_metrics[name][i]["f1"] - baseline_test[i]["f1"]
              for i in range(len(SEEDS))]
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas)
    try:
        _, pvalue = wilcoxon(deltas)
    except ValueError:
        pvalue = float("nan")
    print(f"{name:30s}: ΔF1 = {delta_mean:+.4f} ± {delta_std:.4f}  "
          f"Wilcoxon p = {pvalue:.4f}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Headline: selected variant only                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print(f"Headline test result: {selected_variant}")
print("=" * 76)
sel = all_seed_test_metrics[selected_variant]
for key in ["acc", "f1", "mcc"]:
    vals = [m[key] for m in sel]
    print(f"{key.upper():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"Thresholds per seed: {all_seed_thresholds[selected_variant]}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Gate activation diagnostics                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if all_gate_stats:
    print()
    print("=" * 76)
    print("Gate activation means by homophily bucket (last seed)")
    print("=" * 76)
    gate_rows = []
    for name, rows in all_gate_stats.items():
        gate_rows.extend(rows)
    gate_df = pd.DataFrame(gate_rows)
    if not gate_df.empty:
        for relation in ["follow", "following"]:
            print(f"\n{relation}")
            sub = gate_df[(gate_df["relation"] == relation) & (gate_df["variant"].isin([
                "GatedBotRGCN-global", "GatedBotRGCN-rel",
                "SoftContrastBotRGCN-global", "SoftContrastBotRGCN-rel"
            ]))]
            pivot = sub.pivot(index="bucket", columns="variant", values="gate_mean")
            pivot = pivot.reindex([b for _, _, b in HOMOPHILY_BINS])
            print(pivot.to_string())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Full results table (for reference)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print()
print("=" * 76)
print("Summary: All test performances (threshold-tuned)")
print("=" * 76)
overall_df = pd.DataFrame([r for r in all_results if r.get("metric") == "overall"])
overall_display = overall_df.rename(columns={"f1_binary": "f1"})
print(overall_display[["variant", "acc", "f1", "mcc"]].to_string(index=False))


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

if all_gate_stats:
    gate_csv = os.path.join(OUTPUT_DIR, "heterophily_fix_gates.csv")
    pd.DataFrame(gate_rows).to_csv(gate_csv, index=False)
    print(f"Saved {gate_csv}")

print()
print("=" * 76)
print("Heterophily fix evaluation complete.")
print("=" * 76)
