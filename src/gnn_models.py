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
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_fscore_support, ConfusionMatrixDisplay
from torch.optim import Adam
from tqdm import tqdm
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


def train_model(model, data, train_mask, val_mask, seed, model_type, node_features=None, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WD)

    data = data.to(device)
    if node_features is not None:
        node_features = node_features.to(device)
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


def evaluate(model, data, test_mask, model_type, node_features=None, device="cpu"):
    model.eval()
    data = data.to(device)
    if node_features is not None:
        node_features = node_features.to(device)
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

    x_all = torch.tensor(np.concatenate([
        feats["profile"], feats["tweet"],
        feats["topology"], feats["neighbour_attr"],
    ], axis=1), dtype=torch.float)
    all_dim = x_all.size(1)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

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
            model = cfg["model_class"](cfg["in_dim"])
            trained = train_model(
                model, data, train_mask, val_mask, seed,
                cfg["model_type"], node_features=cfg["features"], device=device
            )
            y_true, y_pred, y_prob, metrics = evaluate(
                trained, data, test_mask,
                cfg["model_type"], node_features=cfg["features"], device=device
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in gnn_models.py: {e}", file=sys.stderr)
        raise
