"""Graph neural network models for TwiBot-20 bot detection.

Includes the standard BotRGCN baseline and a gated RGCN variant that mixes
low-pass (mean-neighbour) and high-pass (ego-minus-neighbour) aggregation per
relation, controlled by a learned per-node, per-relation gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class BotRGCN(nn.Module):
    """Standard two-layer RGCN baseline with four feature-group encoders."""

    def __init__(self, profile_dim=22, tweet_dim=12, topology_dim=8,
                 neighbour_dim=6, embedding_dim=128, dropout=0.3, num_relations=2):
        super().__init__()
        self.dropout = dropout
        h = embedding_dim // 4
        self.enc_profile = nn.Sequential(nn.Linear(profile_dim, h), nn.LeakyReLU())
        self.enc_tweet = nn.Sequential(nn.Linear(tweet_dim, h), nn.LeakyReLU())
        self.enc_topology = nn.Sequential(nn.Linear(topology_dim, h), nn.LeakyReLU())
        self.enc_neighbour = nn.Sequential(nn.Linear(neighbour_dim, h), nn.LeakyReLU())
        self.proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())

        from torch_geometric.nn import RGCNConv
        self.rgcn1 = RGCNConv(embedding_dim, embedding_dim, num_relations=num_relations)
        self.rgcn2 = RGCNConv(embedding_dim, embedding_dim, num_relations=num_relations)

        self.out1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())
        self.out2 = nn.Linear(embedding_dim, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        p = self.enc_profile(profile)
        t = self.enc_tweet(tweet)
        to = self.enc_topology(topology)
        n = self.enc_neighbour(neighbour)
        x = torch.cat([p, t, to, n], dim=1)
        x = self.proj(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.out1(x)
        x = self.out2(x)
        return torch.sigmoid(x).squeeze(-1)


class GatedRGCNConv(MessagePassing):
    """RGCN-style convolution with a per-node, per-relation low/high gate.

    For relation r and node v:
        low_r(v)  = mean_{u in N_r(v)} W_r h_u
        high_r(v) = W_r h_v - low_r(v)
        alpha_r(v) = sigmoid(a_r^T [h_v || low_r(v)])
        m_r(v)    = alpha_r(v) * low_r(v) + (1 - alpha_r(v)) * high_r(v)

    Setting relation_specific=False shares the attention vector a across all
    relations (global gate ablation).
    """

    def __init__(self, in_channels, out_channels, num_relations=2,
                 relation_specific=True, dropout=0.3, att_bias_init=0.0):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.relation_specific = relation_specific
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        if relation_specific:
            self.att = nn.Parameter(torch.Tensor(num_relations, 2 * out_channels))
            self.att_bias = nn.Parameter(torch.Tensor(num_relations))
        else:
            self.att = nn.Parameter(torch.Tensor(2 * out_channels))
            self.att_bias = nn.Parameter(torch.Tensor(1))

        self.att_bias_init = att_bias_init
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.att)
        nn.init.constant_(self.att_bias, self.att_bias_init)

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)

        for r in range(self.num_relations):
            mask = edge_type == r
            n_r = int(mask.sum().item())
            if n_r == 0:
                # No edges of this relation: still apply self-transform via high-pass.
                self_trans = x @ self.weight[r]
                out += self_trans
                continue

            edge_r = edge_index[:, mask]
            src, dst = edge_r[0], edge_r[1]

            # low-pass: mean of relation-r neighbours transformed by W_r.
            messages = x[src] @ self.weight[r]  # [E_r, out]
            low = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
            low.index_add_(0, dst, messages)
            deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1).unsqueeze(1)
            low = low / deg

            # high-pass: self-transform minus low-pass.
            self_trans = x @ self.weight[r]
            high = self_trans - low

            # Gate alpha(v) from concatenation of ego and low-pass term.
            concat = torch.cat([x, low], dim=1)  # [N, 2*out]
            concat = F.dropout(concat, p=self.dropout, training=self.training)
            if self.relation_specific:
                logits = concat @ self.att[r] + self.att_bias[r]  # [N]
            else:
                logits = concat @ self.att + self.att_bias          # [N]
            alpha = torch.sigmoid(logits).unsqueeze(1)  # [N, 1]

            out += alpha * low + (1.0 - alpha) * high

        return F.dropout(out, p=self.dropout, training=self.training)


class GatedBotRGCN(nn.Module):
    """BotRGCN where both RGCN layers use relation-specific low/high gates."""

    def __init__(self, profile_dim=22, tweet_dim=12, topology_dim=8,
                 neighbour_dim=6, embedding_dim=128, dropout=0.3, num_relations=2,
                 relation_specific=True, att_bias_init=0.0):
        super().__init__()
        self.dropout = dropout
        h = embedding_dim // 4
        self.enc_profile = nn.Sequential(nn.Linear(profile_dim, h), nn.LeakyReLU())
        self.enc_tweet = nn.Sequential(nn.Linear(tweet_dim, h), nn.LeakyReLU())
        self.enc_topology = nn.Sequential(nn.Linear(topology_dim, h), nn.LeakyReLU())
        self.enc_neighbour = nn.Sequential(nn.Linear(neighbour_dim, h), nn.LeakyReLU())
        self.proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())

        self.rgcn1 = GatedRGCNConv(embedding_dim, embedding_dim, num_relations,
                                   relation_specific=relation_specific, dropout=dropout,
                                   att_bias_init=att_bias_init)
        self.rgcn2 = GatedRGCNConv(embedding_dim, embedding_dim, num_relations,
                                   relation_specific=relation_specific, dropout=dropout,
                                   att_bias_init=att_bias_init)

        self.out1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())
        self.out2 = nn.Linear(embedding_dim, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        p = self.enc_profile(profile)
        t = self.enc_tweet(tweet)
        to = self.enc_topology(topology)
        n = self.enc_neighbour(neighbour)
        x = torch.cat([p, t, to, n], dim=1)
        x = self.proj(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.out1(x)
        x = self.out2(x)
        return torch.sigmoid(x).squeeze(-1)


class SoftContrastRGCNConv(MessagePassing):
    """RGCN-style convolution with a soft residual contrast gate.

    For relation r and node v:
        self_r(v) = W_r h_v
        low_r(v)  = mean_{u in N_r(v)} W_r h_u
        beta_r(v) = MLP_r( [self_r(v) || low_r(v)] )  in [0,1]
        m_r(v)    = low_r(v) + beta_r(v) * (self_r(v) - low_r(v))

    beta_r(v) = 0 recovers standard RGCN mean aggregation.
    beta_r(v) = 1 ignores neighbours and uses only the self-transform.

    The final MLP bias is initialized so beta starts near 0, i.e. the model
    starts from the baseline and learns to add contrast where useful.
    """

    def __init__(self, in_channels, out_channels, num_relations=2,
                 relation_specific=True, dropout=0.3, use_raw_ego=False):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.relation_specific = relation_specific
        self.dropout = dropout
        self.use_raw_ego = use_raw_ego

        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))

        mlp_hidden = out_channels
        if relation_specific:
            self.gate_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * out_channels, mlp_hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden, 1),
                    nn.Sigmoid(),
                ) for _ in range(num_relations)
            ])
        else:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * out_channels, mlp_hidden),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
                nn.Sigmoid(),
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        mlps = self.gate_mlps if self.relation_specific else [self.gate_mlp]
        for mlp in mlps:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            # Final bias -> beta starts near 0.05 (mostly low-pass / baseline).
            final = mlp[-2]
            nn.init.constant_(final.bias, -3.0)

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)

        for r in range(self.num_relations):
            mask = edge_type == r
            n_r = int(mask.sum().item())
            self_trans = x @ self.weight[r]

            if n_r == 0:
                out += self_trans
                continue

            edge_r = edge_index[:, mask]
            src, dst = edge_r[0], edge_r[1]

            messages = x[src] @ self.weight[r]
            low = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
            low.index_add_(0, dst, messages)
            deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1).unsqueeze(1)
            low = low / deg

            # beta from concatenation of ego and low-pass term.
            ego = x if self.use_raw_ego else self_trans
            concat = torch.cat([ego, low], dim=1)
            if self.relation_specific:
                beta = self.gate_mlps[r](concat)
            else:
                beta = self.gate_mlp(concat)

            out += low + beta * (ego - low)

        return F.dropout(out, p=self.dropout, training=self.training)


class SoftContrastBotRGCN(nn.Module):
    """BotRGCN with soft-contrast relation-specific/global gates."""

    def __init__(self, profile_dim=22, tweet_dim=12, topology_dim=8,
                 neighbour_dim=6, embedding_dim=128, dropout=0.3, num_relations=2,
                 relation_specific=True, use_raw_ego=False):
        super().__init__()
        self.dropout = dropout
        h = embedding_dim // 4
        self.enc_profile = nn.Sequential(nn.Linear(profile_dim, h), nn.LeakyReLU())
        self.enc_tweet = nn.Sequential(nn.Linear(tweet_dim, h), nn.LeakyReLU())
        self.enc_topology = nn.Sequential(nn.Linear(topology_dim, h), nn.LeakyReLU())
        self.enc_neighbour = nn.Sequential(nn.Linear(neighbour_dim, h), nn.LeakyReLU())
        self.proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())

        self.rgcn1 = SoftContrastRGCNConv(embedding_dim, embedding_dim, num_relations,
                                          relation_specific=relation_specific, dropout=dropout,
                                          use_raw_ego=use_raw_ego)
        self.rgcn2 = SoftContrastRGCNConv(embedding_dim, embedding_dim, num_relations,
                                          relation_specific=relation_specific, dropout=dropout,
                                          use_raw_ego=use_raw_ego)

        self.out1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())
        self.out2 = nn.Linear(embedding_dim, 1)

    def forward(self, profile, tweet, topology, neighbour, edge_index, edge_type):
        p = self.enc_profile(profile)
        t = self.enc_tweet(tweet)
        to = self.enc_topology(topology)
        n = self.enc_neighbour(neighbour)
        x = torch.cat([p, t, to, n], dim=1)
        x = self.proj(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.out1(x)
        x = self.out2(x)
        return torch.sigmoid(x).squeeze(-1)
