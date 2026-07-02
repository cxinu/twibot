#!/usr/bin/env python3
"""Correct-and-Smooth (C&S) post-processing on top of a trained BotRGCN.

Trains the validation-selected base model (GatedBotRGCN-global) with the
original BotRGCN hyperparameters, then applies label-propagation smoothing
to the base predictions. Tunes the smoothing hyperparameters (alpha, steps)
on the validation set and reports test-set improvement.

Reference: Huang et al., "Combining Label Propagation and Simple Models
Out-performs Graph Neural Networks", ICLR 2021.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.optim import AdamW

from models import GatedBotRGCN

DATA_DIR = "data/twibot-20"
GRAPH_PATH = os.path.join(DATA_DIR, "twibot_graph.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456, 2024, 9999]
EPOCHS = 50
LR = 1e-2
WD = 5e-2


def load_data():
    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    x_des = torch.load(os.path.join(DATA_DIR, "des_tensor.pt"), map_location="cpu", weights_only=False).float()
    x_tweet = torch.load(os.path.join(DATA_DIR, "tweets_tensor.pt"), map_location="cpu", weights_only=False).float()
    x_num_prop = torch.load(os.path.join(DATA_DIR, "num_properties_tensor.pt"), map_location="cpu", weights_only=False).float()
    x_cat_prop = torch.load(os.path.join(DATA_DIR, "cat_properties_tensor.pt"), map_location="cpu", weights_only=False).float()
    return graph, x_des, x_tweet, x_num_prop, x_cat_prop


def train_base_model(graph, x_des, x_tweet, x_num_prop, x_cat_prop, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GatedBotRGCN(
        des_size=x_des.size(1), tweet_size=x_tweet.size(1),
        num_prop_size=x_num_prop.size(1), cat_prop_size=x_cat_prop.size(1),
        relation_specific=False,
    ).to(DEVICE)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    xdes = x_des.to(DEVICE)
    xtweet = x_tweet.to(DEVICE)
    xnum = x_num_prop.to(DEVICE)
    xcat = x_cat_prop.to(DEVICE)
    edge_index = graph.edge_index_rgcn.to(DEVICE)
    edge_type = graph.edge_type.to(DEVICE)
    y = graph.y.long().to(DEVICE)
    train_mask = graph.train_mask.to(DEVICE)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(xdes, xtweet, xnum, xcat, edge_index, edge_type)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(xdes, xtweet, xnum, xcat, edge_index, edge_type)
    return F.softmax(logits, dim=1).cpu()


def sparse_label_propagate(y_base, edge_index, alpha, steps):
    """PPR-style smoothing using edge_index scatter (no dense adjacency)."""
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(dst, minlength=y_base.size(0)).float().clamp(min=1).view(-1, 1)
    y = y_base.clone()
    for _ in range(steps):
        out = torch.zeros_like(y)
        out.index_add_(0, dst, y[src])
        out = out / deg
        y = (1 - alpha) * y_base + alpha * out
    return y


def evaluate_probs(y_prob, y_true):
    y_pred = y_prob.argmax(dim=1).cpu().numpy() if hasattr(y_prob, 'cpu') else y_prob.argmax(axis=1)
    y_true = y_true.cpu().numpy() if hasattr(y_true, 'cpu') else y_true
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, f1, mcc


def main():
    print("Loading data ...")
    graph, x_des, x_tweet, x_num_prop, x_cat_prop = load_data()
    y = graph.y.long()
    y_dev = y.to(DEVICE)
    train_idx = graph.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = graph.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = graph.test_mask.nonzero(as_tuple=False).view(-1)
    edge_index_dev = graph.edge_index_rgcn.to(DEVICE)

    print("Training base model per seed and tuning C&S on validation ...")
    results = []

    for seed in SEEDS:
        print(f"  Seed {seed} ...", end="", flush=True)
        y_prob = train_base_model(graph, x_des, x_tweet, x_num_prop, x_cat_prop, seed).to(DEVICE)

        # Base predictions: train nodes are anchored to true labels.
        y_base = y_prob.clone()
        y_base[train_idx] = F.one_hot(y_dev[train_idx], num_classes=2).float()

        # Base test metrics.
        base_acc, base_f1, base_mcc = evaluate_probs(y_base[test_idx], y_dev[test_idx])

        # Tune alpha/steps on validation F1.
        best_val_f1 = -1.0
        best_params = (0.5, 10)
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            for steps in [5, 10, 20]:
                y_smooth = sparse_label_propagate(y_base, edge_index_dev, alpha, steps)
                _, val_f1, _ = evaluate_probs(y_smooth[val_idx], y_dev[val_idx])
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_params = (alpha, steps)

        alpha, steps = best_params
        y_smooth = sparse_label_propagate(y_base, edge_index_dev, alpha, steps)
        cs_acc, cs_f1, cs_mcc = evaluate_probs(y_smooth[test_idx], y_dev[test_idx])

        results.append({
            "seed": seed,
            "base_acc": base_acc, "base_f1": base_f1, "base_mcc": base_mcc,
            "cs_acc": cs_acc, "cs_f1": cs_f1, "cs_mcc": cs_mcc,
            "delta_f1": cs_f1 - base_f1,
            "alpha": alpha, "steps": steps,
        })
        print(f"  base F1={base_f1:.4f}  C&S F1={cs_f1:.4f}  Δ={cs_f1-base_f1:+.4f}  "
              f"(α={alpha}, steps={steps})")

    print("\n" + "=" * 76)
    print("Summary: GatedBotRGCN-global vs. +Correct-and-Smooth")
    print("=" * 76)
    for key in ["acc", "f1", "mcc"]:
        base_vals = [r[f"base_{key}"] for r in results]
        cs_vals = [r[f"cs_{key}"] for r in results]
        delta_vals = [cs_vals[i] - base_vals[i] for i in range(len(SEEDS))]
        print(f"{key.upper():12s}: base {np.mean(base_vals):.4f} ± {np.std(base_vals):.4f}   "
              f"C&S {np.mean(cs_vals):.4f} ± {np.std(cs_vals):.4f}   "
              f"Δ {np.mean(delta_vals):+.4f} ± {np.std(delta_vals):.4f}")
    print(f"Chosen params: {[(r['alpha'], r['steps']) for r in results]}")


if __name__ == "__main__":
    main()
