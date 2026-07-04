import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, LayerNorm


def compute_edge_attr(x_des, x_tweet, edge_index, rel_type, batch_size=50000):
    src, dst = edge_index
    E = src.size(0)
    cd, ct = [], []
    for start in range(0, E, batch_size):
        end = min(start + batch_size, E)
        s, d = src[start:end], dst[start:end]
        cd.append(F.cosine_similarity(x_des[s], x_des[d], dim=1).cpu())
        ct.append(F.cosine_similarity(x_tweet[s], x_tweet[d], dim=1).cpu())
    cos_des = torch.cat(cd).unsqueeze(1)
    cos_tweet = torch.cat(ct).unsqueeze(1)
    flw = torch.full((E, 1), 1.0 if rel_type == 1 else 0.0)

    return torch.cat([cos_des, cos_tweet, flw], dim=1)

def focal_loss(logits, targets, alpha=None, gamma=2.0, smooth=0.0):
    n_classes = logits.size(-1)
    if smooth > 0:
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, smooth / (n_classes - 1),
                                             device=logits.device)
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - smooth)
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(smooth_targets * log_probs).sum(dim=-1)
    else:
        ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    if alpha is not None:
        focal = alpha[targets] * focal
    return focal.mean()


class AdaRelBot(nn.Module):
    """Relational Graph Transformer with heterophily-adaptive edge features
    + prototype-calibrated dual-head classifier + focal loss.
    """
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5,
                 cat_prop_size=3, embedding_dim=128, num_heads=8,
                 dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        h = embedding_dim // 4
        self.enc_des = nn.Sequential(nn.Linear(des_size, h), nn.LeakyReLU())
        self.enc_tweet = nn.Sequential(nn.Linear(tweet_size, h), nn.LeakyReLU())
        self.enc_num = nn.Sequential(nn.BatchNorm1d(num_prop_size), nn.Linear(num_prop_size, h), nn.LeakyReLU())
        self.enc_cat = nn.Sequential(nn.Linear(cat_prop_size, h), nn.LeakyReLU())
        self.enc_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), 
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(), 
            nn.Dropout(dropout)
        )
        self.conv1 = TransformerConv(
            embedding_dim, embedding_dim // num_heads,
            heads=num_heads, edge_dim=3, dropout=dropout, beta=True)
        self.norm1 = LayerNorm(embedding_dim)
        self.conv2 = TransformerConv(
            embedding_dim, embedding_dim // num_heads,
            heads=num_heads, edge_dim=3, dropout=dropout, beta=True)
        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 2)
        )
        self.prototypes = nn.Parameter(torch.randn(2, embedding_dim))
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nl = "leaky_relu" if "LeakyReLU" in str(type(m).__name__) else "relu"
                nn.init.kaiming_normal_(m.weight, nonlinearity=nl)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x_des, x_tweet, x_num, x_cat):
        x = torch.cat([
            self.enc_des(x_des), self.enc_tweet(x_tweet),
            self.enc_num(x_num), self.enc_cat(x_cat),
        ], dim=1)
        return self.enc_proj(x)

    def forward(self, x_des, x_tweet, x_num, x_cat, edge_index, edge_attr, return_heads=False):
        x = self.encode(x_des, x_tweet, x_num, x_cat)
        x_res = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x + x_res)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_res = x
        x = self.conv2(x, edge_index, edge_attr)
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

    def reset_parameters(self):
        self._init_weights()
