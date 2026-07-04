#!/usr/bin/env python3
"""Train AdaRelBot on Cresci-15.

Uses the same model and protocol as train.py (TwiBot-20) so results are
directly comparable. The only differences are the data directory and graph
file name.
"""

import os
import warnings
import copy
import csv

import numpy as np
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.metrics import (precision_score, recall_score,
                              roc_auc_score, average_precision_score,
                              confusion_matrix)

from model import AdaRelBot, compute_edge_attr, focal_loss

warnings.filterwarnings("ignore")

DATA_DIR = "data/cresci-15"
GRAPH_PATH = os.path.join(DATA_DIR, "cresci_graph.pt")
OUTPUT_DIR = "results/tables"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cresci15_5seeds.csv")
RESULTS_DIR = "results/metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 2024, 9999]
EPOCHS = 50
LR = 5e-3
WD = 5e-4
DROPOUT = 0.3
EMBEDDING_DIM = 128
NUM_HEADS = 8
GAMMA_FOCAL = 2.0
AUX_WEIGHT = 0.5
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_metrics(probs, labels):
    pred_v = torch.argmax(probs, dim=1)
    return (pred_v == labels).float().mean().item()


def F1_score(probs, labels):
    pred_v = torch.argmax(probs, dim=1)
    tp = (pred_v * labels).sum().float()
    fp = (pred_v * (1 - labels)).sum().float()
    fn = ((1 - pred_v) * labels).sum().float()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * (precision * recall) / (precision + recall)).item() if (precision + recall) > 0 else 0.0


def MCC_score(probs, labels):
    pred_v = torch.argmax(probs, dim=1)
    tp = (pred_v * labels).sum().float()
    tn = ((1 - pred_v) * (1 - labels)).sum().float()
    fp = (pred_v * (1 - labels)).sum().float()
    fn = ((1 - pred_v) * labels).sum().float()
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (numerator / denominator).item() if denominator > 0 else 0.0


def train_one_seed(model, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(DEVICE)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits, gamma, mlp_logits, proto_logits = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, return_heads=True)
        loss_blend = focal_loss(logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL, smooth=LABEL_SMOOTHING)
        loss_mlp = focal_loss(mlp_logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL, smooth=LABEL_SMOOTHING)
        loss_proto = focal_loss(proto_logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL, smooth=LABEL_SMOOTHING)
        loss = loss_blend + AUX_WEIGHT * (loss_mlp + loss_proto)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr)

            val_f1 = F1_score(logits_eval[val_mask], y[val_mask])
            val_acc = get_metrics(logits_eval[val_mask], y[val_mask])

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                train_acc = get_metrics(logits_eval[train_mask], y[train_mask])
                train_f1 = F1_score(logits_eval[train_mask], y[train_mask])
                print(f"  Seed {seed:5d} | {epoch+1:3d}/{EPOCHS} | "
                      f"loss={loss.item():.4f} | acc={train_acc:.4f} f1={train_f1:.4f} | val_f1={val_f1:.4f}")

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr)
        probs = F.softmax(logits, dim=1)

        test_probs = probs[test_mask].cpu().numpy()
        test_labels = y[test_mask].cpu().numpy()

        test_acc = get_metrics(probs[test_mask], y[test_mask])
        test_f1 = F1_score(probs[test_mask], y[test_mask])
        test_mcc = MCC_score(probs[test_mask], y[test_mask])

        preds = np.argmax(test_probs, axis=1)
        bot_probs = test_probs[:, 1]
        test_prec = precision_score(test_labels, preds, zero_division=0)
        test_rec = recall_score(test_labels, preds, zero_division=0)
        test_roc = roc_auc_score(test_labels, bot_probs)
        test_pr_auc = average_precision_score(test_labels, bot_probs)
        test_cm = confusion_matrix(test_labels, preds)

        metrics = {
            "seed": seed, "acc": test_acc, "f1": test_f1, "mcc": test_mcc,
            "prec": test_prec, "rec": test_rec,
            "roc_auc": test_roc, "pr_auc": test_pr_auc,
            "tn": int(test_cm[0, 0]), "fp": int(test_cm[0, 1]),
            "fn": int(test_cm[1, 0]), "tp": int(test_cm[1, 1]),
        }

    return metrics, probs


print("=" * 76)
print("AdaRelBot on Cresci-15 (proto+ea+focal)")
print("=" * 76)

graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
x_des = torch.load(f"{DATA_DIR}/des_tensor.pt", map_location="cpu", weights_only=False).float()
x_tweet = torch.load(f"{DATA_DIR}/tweets_tensor.pt", map_location="cpu", weights_only=False).float()
x_num = torch.load(f"{DATA_DIR}/num_properties_tensor.pt", map_location="cpu", weights_only=False).float()
x_cat = torch.load(f"{DATA_DIR}/cat_properties_tensor.pt", map_location="cpu", weights_only=False).float()

train_mask = graph.train_mask
val_mask = graph.val_mask
test_mask = graph.test_mask
y = graph.y.long()

n_train, n_val, n_test = train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()
print(f"  Nodes: {graph.num_nodes:,}  train={n_train}  val={n_val}  test={n_test}")

x_des = x_des.to(DEVICE)
x_tweet = x_tweet.to(DEVICE)
x_num = x_num.to(DEVICE)
x_cat = x_cat.to(DEVICE)
y = y.to(DEVICE)
train_mask = train_mask.to(DEVICE)
val_mask = val_mask.to(DEVICE)
test_mask = test_mask.to(DEVICE)

edge_index = torch.cat([graph.edge_index_follow, graph.edge_index_following], dim=1).to(DEVICE)
print(f"  Edges (follow+following): {edge_index.size(1):,}")

ea_follow = compute_edge_attr(x_des, x_tweet, graph.edge_index_follow.to(DEVICE), 0)
ea_following = compute_edge_attr(x_des, x_tweet, graph.edge_index_following.to(DEVICE), 1)
edge_attr = torch.cat([ea_follow, ea_following], dim=0).to(DEVICE)
del ea_follow, ea_following
torch.cuda.empty_cache()

n_bots = (y[train_mask] == 1).sum().float()
n_humans = (y[train_mask] == 0).sum().float()
alpha = torch.tensor([n_humans / (n_humans + n_bots),
                      n_bots / (n_humans + n_bots)], device=DEVICE)
print(f"  Focal alpha: [{alpha[0].item():.3f}, {alpha[1].item():.3f}]")

print(f"\nTraining ({len(SEEDS)} seeds, {EPOCHS} epochs)")
print(f"  LR={LR}  WD={WD}  Dropout={DROPOUT}  dim={EMBEDDING_DIM}")

all_probs = []
all_metrics = []
for seed in SEEDS:
    print(f"\n{'─'*50}\nSeed {seed}")
    model = AdaRelBot(
        des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
        embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout=DROPOUT,
    )
    m, probs = train_one_seed(model, seed)
    print(f"  → test: Acc={m['acc']:.4f} F1={m['f1']:.4f} Prec={m['prec']:.4f} Rec={m['rec']:.4f} "
          f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")
    all_metrics.append(m)
    all_probs.append(probs)

accs = [m["acc"] for m in all_metrics]
f1s  = [m["f1"] for m in all_metrics]
mccs = [m["mcc"] for m in all_metrics]
precs = [m["prec"] for m in all_metrics]
recs  = [m["rec"] for m in all_metrics]
rocs  = [m["roc_auc"] for m in all_metrics]
prs   = [m["pr_auc"] for m in all_metrics]

print("\n" + "=" * 76)
print("Summary (5 seeds)  Mean ± Std")
print("=" * 76)
for name, vals in [("Acc", accs), ("F1", f1s), ("MCC", mccs),
                    ("Prec", precs), ("Rec", recs),
                    ("ROC-AUC", rocs), ("PR-AUC", prs)]:
    print(f"  {name:>7}:  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

ens_probs = torch.mean(torch.stack(all_probs), dim=0)
ens_acc = get_metrics(ens_probs[test_mask], y[test_mask])
ens_f1 = F1_score(ens_probs[test_mask], y[test_mask])
ens_mcc = MCC_score(ens_probs[test_mask], y[test_mask])

ens_test_probs = ens_probs[test_mask].cpu().numpy()
ens_test_labels = y[test_mask].cpu().numpy()
ens_preds = np.argmax(ens_test_probs, axis=1)
ens_bot_probs = ens_test_probs[:, 1]
ens_prec = precision_score(ens_test_labels, ens_preds, zero_division=0)
ens_rec  = recall_score(ens_test_labels, ens_preds, zero_division=0)
ens_roc  = roc_auc_score(ens_test_labels, ens_bot_probs)
ens_pr   = average_precision_score(ens_test_labels, ens_bot_probs)

print(f"\n  Ensemble: Acc={ens_acc:.4f} F1={ens_f1:.4f} Prec={ens_prec:.4f} Rec={ens_rec:.4f} "
      f"ROC-AUC={ens_roc:.4f} PR-AUC={ens_pr:.4f} MCC={ens_mcc:.4f}")

# Save for plotting
np.savez(os.path.join(RESULTS_DIR, "cresci15_ensemble.npz"),
         probs=ens_test_probs, labels=ens_test_labels,
         acc=ens_acc, f1=ens_f1, prec=ens_prec, rec=ens_rec,
         roc_auc=ens_roc, pr_auc=ens_pr, mcc=ens_mcc)
np.savez(os.path.join(RESULTS_DIR, "cresci15_all_probs.npz"),
         probs=[p[test_mask].cpu().numpy() for p in all_probs],
         labels=ens_test_labels)

# Save per-seed results
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed", "acc", "f1", "mcc", "prec", "rec", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"])
    writer.writeheader()
    for row in all_metrics:
        writer.writerow(row)

print(f"\nSaved results to {OUTPUT_CSV}")
