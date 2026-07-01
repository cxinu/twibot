import json
import os
import sys
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, RGCNConv
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_fscore_support, ConfusionMatrixDisplay
from torch.optim import Adam
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUT_DIR = "results/tables"
FIGURE_DIR = "results/figures"
FEATURE_DIR = "data/twibot-20"
SEEDS = [42, 123, 456]
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WD = 1e-4


class BotMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class BotSAGE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = SAGEConv(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.head(x)
        return torch.sigmoid(x).squeeze(-1)


class HeteroSAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        N = x.size(0)
        mean_neigh = torch.zeros_like(x)
        count = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        mean_neigh.index_add_(0, dst, x[src])
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype))
        mean_neigh = mean_neigh / count.clamp(min=1)
        return self.W1(x) + self.W2(x - mean_neigh)


class BotHeteroSAGE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = HeteroSAGEConv(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = HeteroSAGEConv(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.head(x)
        return torch.sigmoid(x).squeeze(-1)


class RelBotSAGE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1_follow = SAGEConv(in_dim, 128)
        self.bn1_follow = nn.BatchNorm1d(128)
        self.conv1_following = SAGEConv(in_dim, 128)
        self.bn1_following = nn.BatchNorm1d(128)
        self.conv2_follow = SAGEConv(128, 64)
        self.bn2_follow = nn.BatchNorm1d(64)
        self.conv2_following = SAGEConv(128, 64)
        self.bn2_following = nn.BatchNorm1d(64)
        self.merge = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index_follow, edge_index_following):
        # Follow branch
        xf = self.conv1_follow(x, edge_index_follow)
        xf = self.bn1_follow(xf)
        xf = F.relu(xf)
        xf = F.dropout(xf, p=0.4, training=self.training)
        xf = self.conv2_follow(xf, edge_index_follow)
        xf = self.bn2_follow(xf)
        xf = F.relu(xf)
        xf = F.dropout(xf, p=0.4, training=self.training)
        # Following branch
        xg = self.conv1_following(x, edge_index_following)
        xg = self.bn1_following(xg)
        xg = F.relu(xg)
        xg = F.dropout(xg, p=0.4, training=self.training)
        xg = self.conv2_following(xg, edge_index_following)
        xg = self.bn2_following(xg)
        xg = F.relu(xg)
        xg = F.dropout(xg, p=0.4, training=self.training)
        # Concatenate
        x_cat = torch.cat([xf, xg], dim=1)
        x_cat = self.merge(x_cat)
        return torch.sigmoid(self.head(x_cat)).squeeze(-1)


class DomainRelBotSAGE(nn.Module):
    def __init__(self, in_dim, n_domains=4, domain_dim=8):
        super().__init__()
        self.domain_emb = nn.Embedding(n_domains, domain_dim)
        total_dim = in_dim + domain_dim
        self.conv1_follow = SAGEConv(total_dim, 128)
        self.bn1_follow = nn.BatchNorm1d(128)
        self.conv1_following = SAGEConv(total_dim, 128)
        self.bn1_following = nn.BatchNorm1d(128)
        self.conv2_follow = SAGEConv(128, 64)
        self.bn2_follow = nn.BatchNorm1d(64)
        self.conv2_following = SAGEConv(128, 64)
        self.bn2_following = nn.BatchNorm1d(64)
        self.merge = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index_follow, edge_index_following, domain):
        domain_emb = self.domain_emb(domain)
        x = torch.cat([x, domain_emb], dim=1)
        xf = self.conv1_follow(x, edge_index_follow)
        xf = self.bn1_follow(xf)
        xf = F.relu(xf)
        xf = F.dropout(xf, p=0.4, training=self.training)
        xf = self.conv2_follow(xf, edge_index_follow)
        xf = self.bn2_follow(xf)
        xf = F.relu(xf)
        xf = F.dropout(xf, p=0.4, training=self.training)
        xg = self.conv1_following(x, edge_index_following)
        xg = self.bn1_following(xg)
        xg = F.relu(xg)
        xg = F.dropout(xg, p=0.4, training=self.training)
        xg = self.conv2_following(xg, edge_index_following)
        xg = self.bn2_following(xg)
        xg = F.relu(xg)
        xg = F.dropout(xg, p=0.4, training=self.training)
        x_cat = torch.cat([xf, xg], dim=1)
        x_cat = self.merge(x_cat)
        return torch.sigmoid(self.head(x_cat)).squeeze(-1)


class BotRGCN(nn.Module):
    def __init__(self, profile_dim=22, tweet_dim=12, topology_dim=8, neighbour_dim=6,
                 embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_profile = nn.Sequential(
            nn.Linear(profile_dim, embedding_dimension // 4),
            nn.LeakyReLU(),
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_dim, embedding_dimension // 4),
            nn.LeakyReLU(),
        )
        self.linear_relu_topology = nn.Sequential(
            nn.Linear(topology_dim, embedding_dimension // 4),
            nn.LeakyReLU(),
        )
        self.linear_relu_neighbour = nn.Sequential(
            nn.Linear(neighbour_dim, embedding_dimension // 4),
            nn.LeakyReLU(),
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        p = self.linear_relu_profile(profile)
        t = self.linear_relu_tweet(tweet)
        topo = self.linear_relu_topology(topology)
        n = self.linear_relu_neighbour(neighbour)
        x = torch.cat((p, t, topo, n), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return torch.sigmoid(x).squeeze(-1)


class BotRGCNProfile(nn.Module):
    def __init__(self, profile_dim=22, tweet_dim=12, topology_dim=8, neighbour_dim=6,
                 embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_profile = nn.Sequential(
            nn.Linear(profile_dim, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        x = self.linear_relu_profile(profile)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return torch.sigmoid(x).squeeze(-1)


class BotRGCNProfileTweet(nn.Module):
    def __init__(self, profile_dim=22, tweet_dim=12, topology_dim=8, neighbour_dim=6,
                 embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_profile = nn.Sequential(
            nn.Linear(profile_dim, embedding_dimension // 2),
            nn.LeakyReLU(),
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_dim, embedding_dimension // 2),
            nn.LeakyReLU(),
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        p = self.linear_relu_profile(profile)
        t = self.linear_relu_tweet(tweet)
        x = torch.cat((p, t), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return torch.sigmoid(x).squeeze(-1)


class BotRGCNTopoNeighbour(nn.Module):
    def __init__(self, profile_dim=22, tweet_dim=12, topology_dim=8, neighbour_dim=6,
                 embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_topology = nn.Sequential(
            nn.Linear(topology_dim, embedding_dimension // 2),
            nn.LeakyReLU(),
        )
        self.linear_relu_neighbour = nn.Sequential(
            nn.Linear(neighbour_dim, embedding_dimension // 2),
            nn.LeakyReLU(),
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        topo = self.linear_relu_topology(topology)
        n = self.linear_relu_neighbour(neighbour)
        x = torch.cat((topo, n), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return torch.sigmoid(x).squeeze(-1)


def train_model(model, data, train_mask, val_mask, seed, model_type, node_features=None,
                profile_feats=None, tweet_feats=None, topology_feats=None, neighbour_feats=None,
                device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WD)

    data = data.to(device)
    if node_features is not None:
        node_features = node_features.to(device)
    if profile_feats is not None:
        profile_feats = profile_feats.to(device)
    if tweet_feats is not None:
        tweet_feats = tweet_feats.to(device)
    if topology_feats is not None:
        topology_feats = topology_feats.to(device)
    if neighbour_feats is not None:
        neighbour_feats = neighbour_feats.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    y_train_labels = data.y[train_mask]
    n_pos = (y_train_labels == 1).sum()
    n_neg = (y_train_labels == 0).sum()
    pos_weight_val = n_neg / (n_pos + 1e-8)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    val_interval = 5

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        if model_type == "mlp":
            pred = model(node_features)
        elif model_type == "sage":
            pred = model(node_features, data.edge_index)
        elif model_type == "relsage":
            pred = model(node_features, data.edge_index_follow, data.edge_index_following)
        elif model_type == "domain_relsage":
            pred = model(node_features, data.edge_index_follow, data.edge_index_following, data.domain)
        elif model_type == "rgcn":
            pred = model(profile_feats, tweet_feats, topology_feats, neighbour_feats,
                         data.edge_index_rgcn, data.edge_type)

        train_pred = pred[train_mask]
        train_y = data.y[train_mask]
        weights = torch.where(train_y == 1, pos_weight_val, 1.0)
        loss = F.binary_cross_entropy(train_pred, train_y, weight=weights)
        loss.backward()
        optimizer.step()

        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                if model_type == "mlp":
                    val_pred = model(node_features)
                elif model_type == "sage":
                    val_pred = model(node_features, data.edge_index)
                elif model_type == "relsage":
                    val_pred = model(node_features, data.edge_index_follow, data.edge_index_following)
                elif model_type == "domain_relsage":
                    val_pred = model(node_features, data.edge_index_follow, data.edge_index_following, data.domain)
                elif model_type == "rgcn":
                    val_pred = model(profile_feats, tweet_feats, topology_feats, neighbour_feats,
                                     data.edge_index_rgcn, data.edge_type)

                val_pred_masked = val_pred[val_mask]
                val_y = data.y[val_mask]
                val_weights = torch.where(val_y == 1, pos_weight_val, 1.0)
                val_loss = F.binary_cross_entropy(val_pred_masked, val_y, weight=val_weights)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += val_interval
                if patience_counter >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, data, test_mask, model_type, node_features=None,
             profile_feats=None, tweet_feats=None, topology_feats=None, neighbour_feats=None,
             device="cpu"):
    model.eval()
    data = data.to(device)
    if node_features is not None:
        node_features = node_features.to(device)
    if profile_feats is not None:
        profile_feats = profile_feats.to(device)
    if tweet_feats is not None:
        tweet_feats = tweet_feats.to(device)
    if topology_feats is not None:
        topology_feats = topology_feats.to(device)
    if neighbour_feats is not None:
        neighbour_feats = neighbour_feats.to(device)
    test_mask = test_mask.to(device)
    with torch.no_grad():
        if model_type == "mlp":
            pred = model(node_features)
        elif model_type == "sage":
            pred = model(node_features, data.edge_index)
        elif model_type == "relsage":
            pred = model(node_features, data.edge_index_follow, data.edge_index_following)
        elif model_type == "domain_relsage":
            pred = model(node_features, data.edge_index_follow, data.edge_index_following, data.domain)
        elif model_type == "rgcn":
            pred = model(profile_feats, tweet_feats, topology_feats, neighbour_feats,
                         data.edge_index_rgcn, data.edge_type)

        y_true = data.y[test_mask].cpu().numpy()
        y_prob = pred[test_mask].cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_binary = f1_score(y_true, y_pred, average="binary")
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return y_true, y_pred, y_prob, {
        "f1_macro": round(f1_macro, 4),
        "f1_binary": round(f1_binary, 4),
        "auc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def compute_per_node_homophily(y, edge_index):
    labeled = y >= 0
    src, dst = edge_index
    both = labeled[src] & labeled[dst]
    src_l, dst_l = src[both], dst[both]
    same = (y[src_l] == y[dst_l]).float()
    N = y.size(0)
    deg = torch.zeros(N, device=y.device, dtype=torch.float)
    same_count = torch.zeros(N, device=y.device, dtype=torch.float)
    deg.index_add_(0, dst_l, torch.ones_like(same))
    same_count.index_add_(0, dst_l, same)
    homo = torch.where(deg > 0, same_count / deg, torch.tensor(-1.0, device=y.device))
    return homo, deg.long()


def mcnemar_pvalue(y_true, y_pred1, y_pred2):
    c01 = ((y_pred1 != y_true) & (y_pred2 == y_true)).sum()
    c10 = ((y_pred1 == y_true) & (y_pred2 != y_true)).sum()
    n = c01 + c10
    if n == 0:
        return 1.0
    chi2_stat = (abs(c01 - c10) - 1) ** 2 / n
    return float(chi2_dist.sf(chi2_stat, 1))


def ensemble_preds(y_prob_list, threshold=0.5):
    y_prob = np.mean(y_prob_list, axis=0)
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


def bucket_eval(y_true_list, y_pred_list, y_prob_list, bucket_idx, label):
    n = len(bucket_idx)
    if n == 0:
        return None
    f1s, aucs = [], []
    for s in range(len(y_true_list)):
        yt = y_true_list[s][bucket_idx]
        yp = y_pred_list[s][bucket_idx]
        ypr = y_prob_list[s][bucket_idx]
        f1s.append(f1_score(yt, yp, average="macro"))
        if len(set(yt.tolist())) > 1:
            aucs.append(roc_auc_score(yt, ypr))
        else:
            aucs.append(0.0)
    return {
        "bucket": label,
        "n": n,
        "f1_macro_mean": round(float(np.mean(f1s)), 4),
        "f1_macro_std": round(float(np.std(f1s)), 4),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    data = torch.load("data/twibot-20/twibot_graph.pt", weights_only=False)

    with open(os.path.join(FEATURE_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    feats = {}
    for group in ["profile", "tweet", "topology", "neighbour_attr"]:
        feats[group] = np.load(os.path.join(FEATURE_DIR, f"features_{group}.npy"))

    # Standardise features using training-set statistics (critical for neural nets
    # — raw community_id ranges 0–24296 and destroys linear-layer gradients)
    def standardise(arr):
        mu = np.mean(arr[data.train_mask.cpu().numpy()], axis=0, keepdims=True)
        sd = np.std(arr[data.train_mask.cpu().numpy()], axis=0, keepdims=True)
        sd = np.where(sd < 1e-10, 1.0, sd)
        return (arr - mu) / sd

    feats["profile"] = standardise(feats["profile"])
    feats["tweet"] = standardise(feats["tweet"])
    feats["topology"] = standardise(feats["topology"])
    feats["neighbour_attr"] = standardise(feats["neighbour_attr"])

    x_profile = torch.tensor(feats["profile"], dtype=torch.float)
    profile_dim = x_profile.size(1)
    x_tweet = torch.tensor(feats["tweet"], dtype=torch.float)
    tweet_dim = x_tweet.size(1)
    x_topology = torch.tensor(feats["topology"], dtype=torch.float)
    topology_dim = x_topology.size(1)
    x_neighbour = torch.tensor(feats["neighbour_attr"], dtype=torch.float)
    neighbour_dim = x_neighbour.size(1)

    x_all = torch.tensor(np.concatenate([
        feats["profile"], feats["tweet"],
        feats["topology"], feats["neighbour_attr"],
    ], axis=1), dtype=torch.float)
    all_dim = x_all.size(1)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # Per-node homophily on the merged undirected graph (same edge_index HeteroSAGE will use)
    node_homophily, node_deg = compute_per_node_homophily(data.y, data.edge_index)
    test_homo = node_homophily[test_mask]
    test_deg = node_deg[test_mask]
    test_valid = test_homo >= 0
    print(f"\nHomophily diagnostic (on edge_index used by SAGE/HeteroSAGE):")
    print(f"  Test nodes with degree>0: N={test_valid.sum().item()}, "
          f"mean homophily={test_homo[test_valid].mean().item():.4f}, "
          f"%<0.5={(test_homo[test_valid] < 0.5).float().mean().item()*100:.1f}%")
    print(f"  Test node degree: mean={test_deg[test_valid].float().mean().item():.2f}, "
          f"% deg=1={(test_deg[test_valid] == 1).float().mean().item()*100:.1f}%")

    pairwise_store = {}

    # Config definitions
    configs = [
        {
            "name": "MLP-Profile",
            "model_class": BotMLP,
            "model_type": "mlp",
            "in_dim": profile_dim,
            "features": x_profile,
        },
        {
            "name": "SAGE-Profile",
            "model_class": BotSAGE,
            "model_type": "sage",
            "in_dim": profile_dim,
            "features": x_profile,
        },
        {
            "name": "MLP-All",
            "model_class": BotMLP,
            "model_type": "mlp",
            "in_dim": all_dim,
            "features": x_all,
        },
        {
            "name": "SAGE-All",
            "model_class": BotSAGE,
            "model_type": "sage",
            "in_dim": all_dim,
            "features": x_all,
        },
        {
            "name": "RelSAGE-All",
            "model_class": RelBotSAGE,
            "model_type": "relsage",
            "in_dim": all_dim,
            "features": x_all,
        },
        {
            "name": "DomainRelSAGE-All",
            "model_class": DomainRelBotSAGE,
            "model_type": "domain_relsage",
            "in_dim": all_dim,
            "features": x_all,
        },
        {
            "name": "HeteroSAGE-Profile",
            "model_class": BotHeteroSAGE,
            "model_type": "sage",
            "in_dim": profile_dim,
            "features": x_profile,
        },
        {
            "name": "HeteroSAGE-All",
            "model_class": BotHeteroSAGE,
            "model_type": "sage",
            "in_dim": all_dim,
            "features": x_all,
        },
        {
            "name": "RGCN-All",
            "model_class": BotRGCN,
            "model_type": "rgcn",
            "model_kwargs": {"profile_dim": profile_dim, "tweet_dim": tweet_dim,
                              "topology_dim": topology_dim, "neighbour_dim": neighbour_dim},
            "profile_feats": x_profile,
            "tweet_feats": x_tweet,
            "topology_feats": x_topology,
            "neighbour_feats": x_neighbour,
        },
        {
            "name": "RGCN-Profile",
            "model_class": BotRGCNProfile,
            "model_type": "rgcn",
            "model_kwargs": {"profile_dim": profile_dim},
            "profile_feats": x_profile,
            "tweet_feats": x_tweet,
            "topology_feats": x_topology,
            "neighbour_feats": x_neighbour,
        },
        {
            "name": "RGCN-Profile+Tweet",
            "model_class": BotRGCNProfileTweet,
            "model_type": "rgcn",
            "model_kwargs": {"profile_dim": profile_dim, "tweet_dim": tweet_dim},
            "profile_feats": x_profile,
            "tweet_feats": x_tweet,
            "topology_feats": x_topology,
            "neighbour_feats": x_neighbour,
        },
        {
            "name": "RGCN-Topo+Neighbour",
            "model_class": BotRGCNTopoNeighbour,
            "model_type": "rgcn",
            "model_kwargs": {"topology_dim": topology_dim, "neighbour_dim": neighbour_dim},
            "profile_feats": x_profile,
            "tweet_feats": x_tweet,
            "topology_feats": x_topology,
            "neighbour_feats": x_neighbour,
        },
    ]

    all_results = []
    domain_test = pd.read_parquet("data/twibot-20/twibot_df.parquet")["domain"].values[data.test_mask.cpu().numpy()]

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Config: {name}")

        seed_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        for seed in SEEDS:
            print(f"  Seed {seed}...")
            if cfg["model_type"] == "rgcn":
                model = cfg["model_class"](**cfg.get("model_kwargs", {}))
            else:
                model = cfg["model_class"](cfg["in_dim"])
            trained = train_model(
                model, data, train_mask, val_mask, seed,
                cfg["model_type"], node_features=cfg.get("features"),
                profile_feats=cfg.get("profile_feats"),
                tweet_feats=cfg.get("tweet_feats"),
                topology_feats=cfg.get("topology_feats"),
                neighbour_feats=cfg.get("neighbour_feats"),
                device=device
            )
            y_true, y_pred, y_prob, metrics = evaluate(
                trained, data, test_mask,
                cfg["model_type"], node_features=cfg.get("features"),
                profile_feats=cfg.get("profile_feats"),
                tweet_feats=cfg.get("tweet_feats"),
                topology_feats=cfg.get("topology_feats"),
                neighbour_feats=cfg.get("neighbour_feats"),
                device=device
            )
            seed_metrics.append(metrics)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_prob.append(y_prob)
            print(f"    F1 macro: {metrics['f1_macro']:.4f}, AUC: {metrics['auc']:.4f}")

        # Aggregate across seeds
        f1_list = [m["f1_macro"] for m in seed_metrics]
        auc_list = [m["auc"] for m in seed_metrics]
        prec_list = [m["precision"] for m in seed_metrics]
        rec_list = [m["recall"] for m in seed_metrics]

        mean_row = {
            "config": name,
            "f1_macro_mean": round(np.mean(f1_list), 4),
            "f1_macro_std": round(np.std(f1_list), 4),
            "auc_mean": round(np.mean(auc_list), 4),
            "auc_std": round(np.std(auc_list), 4),
            "precision_mean": round(np.mean(prec_list), 4),
            "recall_mean": round(np.mean(rec_list), 4),
        }
        all_results.append(mean_row)
        print(f"  Mean: F1={mean_row['f1_macro_mean']:.4f}±{mean_row['f1_macro_std']:.4f}, AUC={mean_row['auc_mean']:.4f}±{mean_row['auc_std']:.4f}")

        if name in ("SAGE-All", "HeteroSAGE-All", "MLP-All"):
            pairwise_store[name] = {
                "y_true": deepcopy(all_y_true),
                "y_pred": deepcopy(all_y_pred),
                "y_prob": deepcopy(all_y_prob),
                "metrics": deepcopy(seed_metrics),
            }

        # Confusion matrix (first seed)
        cm = confusion_matrix(all_y_true[0], all_y_pred[0])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Human", "Bot"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f"cm_{name.lower().replace('-', '_')}.png"), dpi=150)
        plt.close()

        # Per-domain for DomainRelSAGE-All (and RelSAGE for comparison)
        if name in ("RelSAGE-All", "DomainRelSAGE-All"):
            print(f"  Per-domain breakdown for {name}:")
            for domain in sorted(set(domain_test)):
                mask = domain_test == domain
                if mask.sum() == 0:
                    continue
                d_y_true = all_y_true[0][mask]
                d_y_pred = all_y_pred[0][mask]
                d_y_prob = all_y_prob[0][mask]
                f1_d = f1_score(d_y_true, d_y_pred, average="macro")
                auc_d = roc_auc_score(d_y_true, d_y_prob)
                prec_d, rec_d, _, _ = precision_recall_fscore_support(d_y_true, d_y_pred, average="binary")
                base_rate = d_y_true.mean()
                print(f"    {domain}: F1={f1_d:.4f}, AUC={auc_d:.4f}, P={prec_d:.4f}, R={rec_d:.4f}, bot_rate={base_rate:.4f}")
                all_results.append({
                    "config": f"{name}_{domain}",
                    "f1_macro_mean": round(f1_d, 4),
                    "f1_macro_std": 0.0,
                    "auc_mean": round(auc_d, 4),
                    "auc_std": 0.0,
                    "precision_mean": round(prec_d, 4),
                    "recall_mean": round(rec_d, 4),
                })
                cm_d = confusion_matrix(d_y_true, d_y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                ConfusionMatrixDisplay(cm_d, display_labels=["Human", "Bot"]).plot(ax=ax, cmap="Blues")
                ax.set_title(f"CM - {name} ({domain})")
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURE_DIR, f"cm_{name.lower().replace('-', '_')}_{domain}.png"), dpi=150)
                plt.close()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "gnn_results.csv"), index=False)
    print(f"\nSaved {os.path.join(OUTPUT_DIR, 'gnn_results.csv')}")
    print(results_df.to_string(index=False))

    # ---- Conditional bucket evaluation: SAGE-All vs HeteroSAGE-All ----
    if "SAGE-All" in pairwise_store and "HeteroSAGE-All" in pairwise_store:
        print(f"\n{'='*60}")
        print("Conditional evaluation: SAGE-All vs HeteroSAGE-All")
        print("Pre-registered split threshold: per-node homophily < 0.5 (low) vs >= 0.5 (high)")
        print()

        sage = pairwise_store["SAGE-All"]
        hetero = pairwise_store["HeteroSAGE-All"]

        test_idx_np = torch.where(test_mask.cpu())[0].numpy()
        test_homo_np = node_homophily[test_mask].cpu().numpy()
        test_deg_np = node_deg[test_mask].cpu().numpy()
        valid = test_homo_np >= 0

        for min_deg, suffix in [(0, ""), (3, "_deg3plus")]:
            deg_ok = test_deg_np >= min_deg
            bucket_valid = valid & deg_ok

            low = np.where((test_homo_np < 0.5) & bucket_valid)[0]
            high = np.where((test_homo_np >= 0.5) & bucket_valid)[0]
            overall = np.where(bucket_valid)[0]

            label = f"  Bucket comparison (deg>={min_deg}):"
            print(label)
            header = f"  {'Bucket':<22} {'N':>6} {'SAGE F1':>14} {'Hetero F1':>14} {'ΔF1':>8} {'McNemar p':>10}"
            print(header)
            print("  " + "-" * len(header))

            rows = []
            for b_idx, b_name in [(low, "Low homophily (<0.5)"),
                                   (high, "High homophily (>=0.5)"),
                                   (overall, "Overall")]:
                if len(b_idx) == 0:
                    continue

                s_eval = bucket_eval(sage["y_true"], sage["y_pred"], sage["y_prob"], b_idx, b_name)
                h_eval = bucket_eval(hetero["y_true"], hetero["y_pred"], hetero["y_prob"], b_idx, b_name)
                if s_eval is None or h_eval is None:
                    continue

                delta = round(h_eval["f1_macro_mean"] - s_eval["f1_macro_mean"], 4)

                sage_prob_e, sage_pred_e = ensemble_preds(sage["y_prob"])
                hetero_prob_e, hetero_pred_e = ensemble_preds(hetero["y_prob"])
                p_val = mcnemar_pvalue(
                    sage["y_true"][0][b_idx],
                    sage_pred_e[b_idx],
                    hetero_pred_e[b_idx],
                )

                print(f"  {b_name:<22} {s_eval['n']:>6}  "
                      f"{s_eval['f1_macro_mean']:.4f}±{s_eval['f1_macro_std']:.4f}  "
                      f"{h_eval['f1_macro_mean']:.4f}±{h_eval['f1_macro_std']:.4f}  "
                      f"{delta:+8.4f}  {p_val:.4f}")

                rows.append({
                    "comparison": f"HeteroSAGE_vs_SAGE{suffix}",
                    "bucket": b_name,
                    "n": s_eval["n"],
                    "sage_f1": f"{s_eval['f1_macro_mean']:.4f}±{s_eval['f1_macro_std']:.4f}",
                    "hetero_f1": f"{h_eval['f1_macro_mean']:.4f}±{h_eval['f1_macro_std']:.4f}",
                    "delta_f1": delta,
                    "delta_auc": round(h_eval["auc_mean"] - s_eval["auc_mean"], 4),
                    "mcnemar_p": p_val,
                })

            print(f"\n  Degree stats (deg>={min_deg}):")
            for b_idx, b_name in [(low, "Low homophily"), (high, "High homophily"), (overall, "Overall")]:
                if len(b_idx) == 0:
                    continue
                degs = test_deg_np[b_idx]
                print(f"  {b_name:<22}: N={len(b_idx):>5}, "
                      f"mean_deg={degs.mean():.2f}, "
                      f"deg1={((degs==1).sum())/len(b_idx)*100:.0f}%, "
                      f"deg>=3={((degs>=3).sum())/len(b_idx)*100:.0f}%")

            bucket_df = pd.DataFrame(rows)
            bucket_df.to_csv(os.path.join(OUTPUT_DIR, f"hetero_bucket_comparison{suffix}.csv"), index=False)

        print(f"\nSaved bucket comparison tables.")

    # ---- Headline comparison: HeteroSAGE-All vs MLP-All ----
    if "MLP-All" in pairwise_store and "HeteroSAGE-All" in pairwise_store:
        print(f"\n{'='*60}")
        print("Headline comparison: HeteroSAGE-All vs MLP-All")
        print()

        mlp = pairwise_store["MLP-All"]
        hetero = pairwise_store["HeteroSAGE-All"]

        for min_deg, suffix in [(0, ""), (3, "_deg3plus")]:
            deg_ok = test_deg_np >= min_deg
            bucket_valid = valid & deg_ok

            low = np.where((test_homo_np < 0.5) & bucket_valid)[0]
            high = np.where((test_homo_np >= 0.5) & bucket_valid)[0]
            overall = np.where(bucket_valid)[0]

            print(f"\n  HeteroSAGE-All vs MLP-All (deg>={min_deg}):")
            header = f"  {'Bucket':<22} {'N':>6} {'MLP F1':>14} {'Hetero F1':>14} {'ΔF1':>8} {'McNemar p':>10}"
            print(header)
            print("  " + "-" * len(header))

            mlp_rows = []
            for b_idx, b_name in [(low, "Low homophily (<0.5)"),
                                   (high, "High homophily (>=0.5)"),
                                   (overall, "Overall")]:
                if len(b_idx) == 0:
                    continue

                m_eval = bucket_eval(mlp["y_true"], mlp["y_pred"], mlp["y_prob"], b_idx, b_name)
                h_eval = bucket_eval(hetero["y_true"], hetero["y_pred"], hetero["y_prob"], b_idx, b_name)
                if m_eval is None or h_eval is None:
                    continue

                delta = round(h_eval["f1_macro_mean"] - m_eval["f1_macro_mean"], 4)

                mlp_prob_e, mlp_pred_e = ensemble_preds(mlp["y_prob"])
                hetero_prob_e, hetero_pred_e = ensemble_preds(hetero["y_prob"])
                p_val = mcnemar_pvalue(
                    mlp["y_true"][0][b_idx],
                    mlp_pred_e[b_idx],
                    hetero_pred_e[b_idx],
                )

                print(f"  {b_name:<22} {m_eval['n']:>6}  "
                      f"{m_eval['f1_macro_mean']:.4f}±{m_eval['f1_macro_std']:.4f}  "
                      f"{h_eval['f1_macro_mean']:.4f}±{h_eval['f1_macro_std']:.4f}  "
                      f"{delta:+8.4f}  {p_val:.4f}")

                mlp_rows.append({
                    "comparison": f"HeteroSAGE_vs_MLP{suffix}",
                    "bucket": b_name,
                    "n": m_eval["n"],
                    "mlp_f1": f"{m_eval['f1_macro_mean']:.4f}±{m_eval['f1_macro_std']:.4f}",
                    "hetero_f1": f"{h_eval['f1_macro_mean']:.4f}±{h_eval['f1_macro_std']:.4f}",
                    "delta_f1": delta,
                    "delta_auc": round(h_eval["auc_mean"] - m_eval["auc_mean"], 4),
                    "mcnemar_p": p_val,
                })

            mlp_bucket_df = pd.DataFrame(mlp_rows)
            mlp_bucket_df.to_csv(os.path.join(OUTPUT_DIR, f"mlp_vs_hetero_bucket_comparison{suffix}.csv"), index=False)

        print(f"\nSaved MLP-All vs HeteroSAGE-All comparison tables.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in gnn_models.py: {e}", file=sys.stderr)
        raise
