#!/usr/bin/env python3
"""Train AdaRelBot on TwiBot-20.

Config:
  - Edge-attributed TransformerConv (cos-des, cos-tweet, rel-type)
  - Prototype + MLP dual-head with learned gate
  - Focal Loss for class imbalance
"""

import os
import warnings
import copy

import numpy as np
import torch
from torch.optim import AdamW
import torch.nn.functional as F

from model import AdaRelBot, compute_edge_attr, focal_loss

warnings.filterwarnings("ignore")

DATA_DIR = "data/twibot-20"
GRAPH_PATH = os.path.join(DATA_DIR, "twibot_graph.pt")
OUTPUT_DIR = "results/tables"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "adarel_clean_5seeds.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 2024, 9999]
EPOCHS = 50
LR = 5e-3
WD = 5e-4
DROPOUT = 0.3
EMBEDDING_DIM = 128
NUM_HEADS = 8
GAMMA_FOCAL = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.05), 2)

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
        logits, _ = model(x_des, x_tweet, x_num, x_cat, edge_index, edge_attr)
        loss = focal_loss(logits[train_mask], y[train_mask], alpha=alpha, gamma=GAMMA_FOCAL)
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

        test_acc = get_metrics(probs[test_mask], y[test_mask])
        test_f1 = F1_score(probs[test_mask], y[test_mask])
        test_mcc = MCC_score(probs[test_mask], y[test_mask])

    return test_acc, test_f1, test_mcc, probs

print("=" * 76)
print("AdaRelBot (proto+ea+focal)")
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
print(f"  Edges (follow+only): {edge_index.size(1):,}")

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

y_val_np = y[val_mask].cpu().numpy()
y_test_np = y[test_mask].cpu().numpy()

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
    test_acc, test_f1, test_mcc, probs = train_one_seed(model, seed)
    print(f"  → test: Acc={test_acc:.4f} F1={test_f1:.4f} MCC={test_mcc:.4f}")
    all_metrics.append({"seed": seed, "acc": test_acc, "f1": test_f1, "mcc": test_mcc})
    all_probs.append(probs)

accs = [m["acc"] for m in all_metrics]
f1s = [m["f1"] for m in all_metrics]
mccs = [m["mcc"] for m in all_metrics]

print("\n" + "=" * 76)
print("Summary (5 seeds)")
print("=" * 76)
print("  Mean ± Std:")
print(f"    Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"    F1:   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"    MCC:  {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")

ens_probs = torch.mean(torch.stack(all_probs), dim=0)
ens_acc = get_metrics(ens_probs[test_mask], y[test_mask])
ens_f1 = F1_score(ens_probs[test_mask], y[test_mask])
ens_mcc = MCC_score(ens_probs[test_mask], y[test_mask])

print(f"\n  Ensemble: Acc={ens_acc:.4f}  F1={ens_f1:.4f}  MCC={ens_mcc:.4f}")
