#!/usr/bin/env python3
"""Ablation study on TwiBot-20 (single seed 42) with the fixed gate model."""

import os
import warnings
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from model import AdaRelBot, compute_edge_attr

warnings.filterwarnings("ignore")

DATA_DIR = "data/twibot-20"
GRAPH_PATH = os.path.join(DATA_DIR, "twibot_graph.pt")
SEEDS = [42, 123, 456]
EPOCHS = 50
LR = 5e-3
WD = 5e-4
DROPOUT = 0.3
EMBEDDING_DIM = 128
NUM_HEADS = 8
GAMMA_FOCAL = 2.0
AUX_WEIGHT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def focal_loss(inputs, targets, alpha, gamma):
    """Multiclass focal loss matching model.py."""
    ce = F.cross_entropy(inputs, targets, reduction="none")
    pt = torch.exp(-ce)
    at = alpha.gather(0, targets)
    return (at * (1 - pt) ** gamma * ce).mean()


def cross_entropy_loss(inputs, targets, alpha):
    """Class-weighted cross-entropy for the no-focal ablation."""
    return F.cross_entropy(inputs, targets, weight=alpha)


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


class AdaRelBotNoEdge(AdaRelBot):
    """Drop edge features from the convolution."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Recreate convolutions without edge_dim.
        from torch_geometric.nn import TransformerConv
        self.conv1 = TransformerConv(EMBEDDING_DIM, EMBEDDING_DIM // NUM_HEADS, heads=NUM_HEADS, dropout=DROPOUT, beta=True)
        self.conv2 = TransformerConv(EMBEDDING_DIM, EMBEDDING_DIM // NUM_HEADS, heads=NUM_HEADS, dropout=DROPOUT, beta=True)

    def forward(self, x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, return_heads=False):
        x = self.encode(x_des, x_tweet, x_num, x_cat)
        x_res = x
        x = self.conv1(x, edge_index)
        x = self.norm1(x + x_res)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_res = x
        x = self.conv2(x, edge_index)
        x = x + x_res
        mlp_logits = self.mlp_head(x)
        x_norm = F.normalize(x, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        proto_logits = (x_norm @ proto_norm.T) * self.proto_temp
        gamma = self.gate(x).sigmoid()
        logits = gamma * mlp_logits + (1.0 - gamma) * proto_logits
        if return_heads:
            return logits, gamma, mlp_logits, proto_logits
        return logits, gamma


class AdaRelBotMLPOnly(AdaRelBot):
    """Use only the MLP head (gamma fixed to 1)."""
    def forward(self, x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, return_heads=False):
        x = self.encode(x_des, x_tweet, x_num, x_cat)
        x_res = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x + x_res)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_res = x
        x = self.conv2(x, edge_index, edge_attr)
        x = x + x_res
        logits = self.mlp_head(x)
        gamma = torch.ones((x.size(0), 1), device=x.device)
        if return_heads:
            return logits, gamma, logits, logits
        return logits, gamma


def load_data():
    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    x_des = torch.load(f"{DATA_DIR}/des_tensor.pt", map_location="cpu", weights_only=False).float().to(DEVICE)
    x_tweet = torch.load(f"{DATA_DIR}/tweets_tensor.pt", map_location="cpu", weights_only=False).float().to(DEVICE)
    x_num = torch.load(f"{DATA_DIR}/num_properties_tensor.pt", map_location="cpu", weights_only=False).float().to(DEVICE)
    x_cat = torch.load(f"{DATA_DIR}/cat_properties_tensor.pt", map_location="cpu", weights_only=False).float().to(DEVICE)

    train_mask = graph.train_mask.to(DEVICE)
    val_mask = graph.val_mask.to(DEVICE)
    test_mask = graph.test_mask.to(DEVICE)
    y = graph.y.long().to(DEVICE)

    edge_index = torch.cat([graph.edge_index_follow, graph.edge_index_following], dim=1).to(DEVICE)
    ea_follow = compute_edge_attr(x_des, x_tweet, graph.edge_index_follow.to(DEVICE), 0)
    ea_following = compute_edge_attr(x_des, x_tweet, graph.edge_index_following.to(DEVICE), 1)
    edge_attr = torch.cat([ea_follow, ea_following], dim=0).to(DEVICE)

    n_bots = (y[train_mask] == 1).sum().float()
    n_humans = (y[train_mask] == 0).sum().float()
    alpha = torch.tensor([n_humans / (n_humans + n_bots),
                          n_bots / (n_humans + n_bots)], device=DEVICE)

    return x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, y, train_mask, val_mask, test_mask, alpha


def train(model, loss_fn, use_edge_attr=True, seed=42):
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
        if use_edge_attr:
            logits, gamma, mlp_logits, proto_logits = model(
                x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, return_heads=True
            )
        else:
            logits, gamma, mlp_logits, proto_logits = model(
                x_des, x_tweet, x_num, x_cat, edge_index, None, return_heads=True
            )

        if loss_fn == "focal":
            loss_blend = focal_loss(logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL)
            loss_mlp = focal_loss(mlp_logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL)
            loss_proto = focal_loss(proto_logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL)
            loss = loss_blend + AUX_WEIGHT * (loss_mlp + loss_proto)
        else:
            loss = cross_entropy_loss(logits[train_mask], y[train_mask], alpha=alpha)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            if use_edge_attr:
                logits_eval, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr)
            else:
                logits_eval, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, None)
            val_f1 = F1_score(logits_eval[val_mask], y[val_mask])
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        if use_edge_attr:
            logits, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr)
        else:
            logits, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, None)
        probs = F.softmax(logits, dim=1)
        test_acc = get_metrics(probs[test_mask], y[test_mask])
        test_f1 = F1_score(probs[test_mask], y[test_mask])
        test_mcc = MCC_score(probs[test_mask], y[test_mask])
    return test_acc, test_f1, test_mcc


if __name__ == "__main__":
    x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, y, train_mask, val_mask, test_mask, alpha = load_data()

    configs = [
        ("AdaRelBot (full)", AdaRelBot,
         {"des_size":768, "tweet_size":768, "num_prop_size":5, "cat_prop_size":3,
          "embedding_dim":EMBEDDING_DIM, "num_heads":NUM_HEADS, "dropout":DROPOUT},
         "focal", True),
        ("w/o edge features", AdaRelBotNoEdge,
         {"des_size":768, "tweet_size":768, "num_prop_size":5, "cat_prop_size":3,
          "embedding_dim":EMBEDDING_DIM, "num_heads":NUM_HEADS, "dropout":DROPOUT},
         "focal", False),
        ("w/o prototype head", AdaRelBotMLPOnly,
         {"des_size":768, "tweet_size":768, "num_prop_size":5, "cat_prop_size":3,
          "embedding_dim":EMBEDDING_DIM, "num_heads":NUM_HEADS, "dropout":DROPOUT},
         "focal", True),
        ("w/o Focal Loss", AdaRelBot,
         {"des_size":768, "tweet_size":768, "num_prop_size":5, "cat_prop_size":3,
          "embedding_dim":EMBEDDING_DIM, "num_heads":NUM_HEADS, "dropout":DROPOUT},
         "ce", True),
    ]

    print(f"{'Configuration':<22} {'Acc':>15} {'F1':>15} {'MCC':>15}")
    print("-" * 70)
    for name, cls, kwargs, loss_fn, use_edge in configs:
        metrics = []
        for seed in SEEDS:
            model = cls(**kwargs)
            acc, f1, mcc = train(model, loss_fn, use_edge, seed=seed)
            metrics.append((acc, f1, mcc))
        accs = np.array([m[0] for m in metrics])
        f1s = np.array([m[1] for m in metrics])
        mccs = np.array([m[2] for m in metrics])
        print(f"{name:<22} "
              f"{accs.mean()*100:6.2f}±{accs.std()*100:4.2f} "
              f"{f1s.mean()*100:6.2f}±{f1s.std()*100:4.2f} "
              f"{mccs.mean():6.4f}±{mccs.std():4.4f}")
