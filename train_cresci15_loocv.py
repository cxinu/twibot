#!/usr/bin/env python3
"""Leave-one-group-out evaluation for Cresci-15.

For each of the 5 groups (E13, FSF, INT, TFP, TWT), train on the other 4
groups and test on the held-out group. This is a stricter test of
cross-group generalization and should reveal whether the model is
memorizing group-specific artifacts.
"""

import os
import warnings
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, average_precision_score,
                              confusion_matrix, matthews_corrcoef)

from model import AdaRelBot, compute_edge_attr, focal_loss

warnings.filterwarnings("ignore")

DATA_DIR = "data/cresci-15"
EXTRACT_DIR = "dataset/cresci-15/extracted"
GROUPS = ["E13", "FSF", "INT", "TFP", "TWT"]
GROUP_LABELS = {"E13": 0, "TFP": 0, "FSF": 1, "INT": 1, "TWT": 1}  # human=0, bot=1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456, 2024, 9999]
EPOCHS = 50
LR = 5e-3
WD = 5e-4
DROPOUT = 0.3
EMBEDDING_DIM = 128
NUM_HEADS = 8
GAMMA_FOCAL = 2.0
AUX_WEIGHT = 0.5


def build_masks(num_nodes, train_idx, val_idx, test_idx):
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val   = torch.zeros(num_nodes, dtype=torch.bool)
    test  = torch.zeros(num_nodes, dtype=torch.bool)
    train[train_idx] = True
    val[val_idx] = True
    test[test_idx] = True
    return train, val, test


def F1_score(probs, labels):
    pred_v = torch.argmax(probs, dim=1)
    tp = (pred_v * labels).sum().float()
    fp = (pred_v * (1 - labels)).sum().float()
    fn = ((1 - pred_v) * labels).sum().float()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * (precision * recall) / (precision + recall)).item() if (precision + recall) > 0 else 0.0


def get_metrics(probs, labels):
    pred_v = torch.argmax(probs, dim=1)
    return (pred_v == labels).float().mean().item()


def MCC_score(probs, labels):
    pred_v = torch.argmax(probs, dim=1)
    tp = (pred_v * labels).sum().float()
    tn = ((1 - pred_v) * (1 - labels)).sum().float()
    fp = (pred_v * (1 - labels)).sum().float()
    fn = ((1 - pred_v) * labels).sum().float()
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (numerator / denominator).item() if denominator > 0 else 0.0


def load_all_data():
    graph = torch.load(os.path.join(DATA_DIR, "cresci_graph.pt"), map_location="cpu", weights_only=False)
    x_des = torch.load(f"{DATA_DIR}/des_tensor.pt", map_location="cpu", weights_only=False).float()
    x_tweet = torch.load(f"{DATA_DIR}/tweets_tensor.pt", map_location="cpu", weights_only=False).float()
    x_num = torch.load(f"{DATA_DIR}/num_properties_tensor.pt", map_location="cpu", weights_only=False).float()
    x_cat = torch.load(f"{DATA_DIR}/cat_properties_tensor.pt", map_location="cpu", weights_only=False).float()
    return graph, x_des, x_tweet, x_num, x_cat


def get_group_indices():
    """Map each group to its node indices in the saved graph."""
    import pandas as pd
    frames = []
    for group in GROUPS:
        path = Path(EXTRACT_DIR) / group / "users.csv"
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        df = df.drop_duplicates(subset=["id"])
        df["group"] = group
        frames.append(df)
    users = pd.concat(frames, ignore_index=True)
    user_ids_in_order = users["id"].astype(int).tolist()

    group_indices = {}
    # Load node order from the graph; the saved graph preserves the order from preprocessing
    # Load from the raw CSV in extraction order
    extract_user_ids = []
    for group in GROUPS:
        path = Path(EXTRACT_DIR) / group / "users.csv"
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        df = df.drop_duplicates(subset=["id"])
        extract_user_ids.extend(df["id"].astype(int).tolist())

    # Build mapping: user_id -> index in graph (0..5299)
    uid_to_idx = {uid: i for i, uid in enumerate(extract_user_ids)}

    for group in GROUPS:
        path = Path(EXTRACT_DIR) / group / "users.csv"
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        df = df.drop_duplicates(subset=["id"])
        gids = df["id"].astype(int).tolist()
        group_indices[group] = [uid_to_idx[uid] for uid in gids]

    return group_indices


def train_one_seed(model, x_des, x_tweet, x_num, x_cat, edge_index, edge_attr,
                   y, train_mask, val_mask, test_mask, alpha, seed):
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
        logits, gamma, mlp_logits, proto_logits = model(
            x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, return_heads=True
        )
        loss_blend = focal_loss(logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL)
        loss_mlp = focal_loss(mlp_logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL)
        loss_proto = focal_loss(proto_logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL)
        loss = loss_blend + AUX_WEIGHT * (loss_mlp + loss_proto)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr)
            val_f1 = F1_score(logits_eval[val_mask], y[val_mask])
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr)
        probs = F.softmax(logits, dim=1)
        test_probs = probs[test_mask].cpu().numpy()
        test_labels = y[test_mask].cpu().numpy()
        preds = np.argmax(test_probs, axis=1)
        bot_probs = test_probs[:, 1]

        return {
            "acc": accuracy_score(test_labels, preds),
            "f1": f1_score(test_labels, preds, zero_division=0),
            "prec": precision_score(test_labels, preds, zero_division=0),
            "rec": recall_score(test_labels, preds, zero_division=0),
            "roc_auc": roc_auc_score(test_labels, bot_probs),
            "pr_auc": average_precision_score(test_labels, bot_probs),
            "mcc": matthews_corrcoef(test_labels, preds),
            "cm": confusion_matrix(test_labels, preds).tolist(),
        }


def run_leave_one_out():
    graph, x_des, x_tweet, x_num, x_cat = load_all_data()
    group_indices = get_group_indices()
    all_node_ids = sum(group_indices.values(), [])
    num_nodes = len(set(all_node_ids))
    print(f"Total unique nodes: {num_nodes}")

    all_results = {}

    for heldout_group in GROUPS:
        print(f"\n{'='*60}")
        print(f" Hold-out group: {heldout_group}")
        print(f"{'='*60}")

        train_groups = [g for g in GROUPS if g != heldout_group]
        test_indices = group_indices[heldout_group]

        # Use 20% of the train groups as validation
        train_indices_full = []
        for g in train_groups:
            train_indices_full.extend(group_indices[g])
        np.random.seed(42)
        np.random.shuffle(train_indices_full)
        n_val = int(0.2 * len(train_indices_full))
        val_indices = train_indices_full[:n_val]
        train_indices = train_indices_full[n_val:]

        print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")
        y_np = graph.y.numpy()
        for s in ["train", "val", "test"]:
            idx = eval(f"{s}_indices")
            bots = (y_np[idx] == 1).sum()
            humans = (y_np[idx] == 0).sum()
            print(f"    {s}: {humans} humans, {bots} bots")

        # Build masks
        train_mask, val_mask, test_mask = build_masks(num_nodes, train_indices, val_indices, test_indices)

        # Move data to GPU
        x_des_g = x_des.to(DEVICE)
        x_tweet_g = x_tweet.to(DEVICE)
        x_num_g = x_num.to(DEVICE)
        x_cat_g = x_cat.to(DEVICE)
        y_g = graph.y.long().to(DEVICE)
        train_mask_g = train_mask.to(DEVICE)
        val_mask_g = val_mask.to(DEVICE)
        test_mask_g = test_mask.to(DEVICE)

        edge_index = torch.cat([graph.edge_index_follow, graph.edge_index_following], dim=1).to(DEVICE)
        ea_follow = compute_edge_attr(x_des_g, x_tweet_g, graph.edge_index_follow.to(DEVICE), 0)
        ea_following = compute_edge_attr(x_des_g, x_tweet_g, graph.edge_index_following.to(DEVICE), 1)
        edge_attr = torch.cat([ea_follow, ea_following], dim=0).to(DEVICE)

        n_bots = (y_g[train_mask_g] == 1).sum().float()
        n_humans = (y_g[train_mask_g] == 0).sum().float()
        alpha = torch.tensor([n_humans / (n_humans + n_bots),
                              n_bots / (n_humans + n_bots)], device=DEVICE)

        group_seed_metrics = []
        for seed in SEEDS:
            model = AdaRelBot(
                des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout=DROPOUT,
            )
            m = train_one_seed(model, x_des_g, x_tweet_g, x_num_g, x_cat_g,
                               edge_index, edge_attr, y_g,
                               train_mask_g, val_mask_g, test_mask_g, alpha, seed)
            group_seed_metrics.append(m)
            print(f"  Seed {seed}: Acc={m['acc']:.4f} F1={m['f1']:.4f} "
                  f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f}")

        # Average across seeds
        accs = [m["acc"] for m in group_seed_metrics]
        f1s  = [m["f1"] for m in group_seed_metrics]
        mccs = [m["mcc"] for m in group_seed_metrics]
        precs = [m["prec"] for m in group_seed_metrics]
        recs  = [m["rec"] for m in group_seed_metrics]
        rocs  = [m["roc_auc"] for m in group_seed_metrics]
        prs   = [m["pr_auc"] for m in group_seed_metrics]

        print(f"  Mean±Std: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
              f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} "
              f"ROC-AUC={np.mean(rocs):.4f}±{np.std(rocs):.4f}")

        all_results[heldout_group] = {
            "acc": f"{np.mean(accs):.4f}±{np.std(accs):.4f}",
            "f1":  f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}",
            "mcc": f"{np.mean(mccs):.4f}±{np.std(mccs):.4f}",
            "prec": f"{np.mean(precs):.4f}±{np.std(precs):.4f}",
            "rec":  f"{np.mean(recs):.4f}±{np.std(recs):.4f}",
            "roc_auc": f"{np.mean(rocs):.4f}±{np.std(rocs):.4f}",
            "pr_auc":  f"{np.mean(prs):.4f}±{np.std(prs):.4f}",
        }

    print("\n" + "=" * 60)
    print("Leave-one-group-out Summary")
    print("=" * 60)
    for group, res in all_results.items():
        label = "HUMAN" if group in ("E13", "TFP") else "BOT"
        print(f"  {group} ({label}): Acc={res['acc']}  F1={res['f1']}  ROC-AUC={res['roc_auc']}")

    # Compute macro-average
    all_accs = []
    all_f1s = []
    all_rocs = []
    for res in all_results.values():
        acc_mean = float(res["acc"].split("±")[0])
        f1_mean = float(res["f1"].split("±")[0])
        roc_mean = float(res["roc_auc"].split("±")[0])
        all_accs.append(acc_mean)
        all_f1s.append(f1_mean)
        all_rocs.append(roc_mean)
    print(f"\n  Macro-average across 5 folds: "
          f"Acc={np.mean(all_accs):.4f}±{np.std(all_accs):.4f}  "
          f"F1={np.mean(all_f1s):.4f}±{np.std(all_f1s):.4f}  "
          f"ROC-AUC={np.mean(all_rocs):.4f}±{np.std(all_rocs):.4f}")

    # Save
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/cresci15_leave_one_out.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/tables/cresci15_leave_one_out.json")


if __name__ == "__main__":
    run_leave_one_out()
