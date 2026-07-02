"""Graph neural network models for TwiBot-20 bot detection.

Includes the standard BotRGCN baseline and gated RGCN variants that mix
low-pass (mean-neighbour) and ego-contrast aggregation per relation. Feature
dimensions and architecture follow the original BotRGCN paper:

    des  (768-dim)  -> h
    tweet(768-dim)  -> h
    num_prop (5-dim)-> h
    cat_prop (3-dim)-> h
    concat -> 4h = embedding_dim (=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class BotRGCN(nn.Module):
    """Standard two-layer RGCN baseline matching the original BotRGCN paper.

    Inputs (paper names):
        des      -- RoBERTa description embeddings,         shape [N, 768]
        tweet    -- RoBERTa tweet embeddings,               shape [N, 768]
        num_prop -- standardized numerical properties,      shape [N, 5]
        cat_prop -- categorical properties,                 shape [N, 3]

    The original code uses a single RGCNConv module applied twice (shared
    weights). Output is 2-class logits for CrossEntropyLoss.
    """

    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5,
                 cat_prop_size=3, embedding_dimension=32, dropout=0.1,
                 num_relations=2):
        super().__init__()
        self.dropout = dropout
        h = embedding_dimension // 4
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, h), nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, h), nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, h), nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, h), nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        from torch_geometric.nn import RGCNConv
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension,
                             num_relations=num_relations)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


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
                 relation_specific=True, dropout=0.1, att_bias_init=0.0):
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
    """BotRGCN where the RGCN layer uses relation-specific low/high gates."""

    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5,
                 cat_prop_size=3, embedding_dimension=32, dropout=0.1,
                 num_relations=2, relation_specific=True, att_bias_init=0.0):
        super().__init__()
        self.dropout = dropout
        h = embedding_dimension // 4
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, h), nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, h), nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, h), nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, h), nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )

        self.rgcn = GatedRGCNConv(embedding_dimension, embedding_dimension,
                                  num_relations, relation_specific=relation_specific,
                                  dropout=dropout, att_bias_init=att_bias_init)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


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
                 relation_specific=True, dropout=0.1, use_raw_ego=False):
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

    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5,
                 cat_prop_size=3, embedding_dimension=32, dropout=0.1,
                 num_relations=2, relation_specific=True, use_raw_ego=False):
        super().__init__()
        self.dropout = dropout
        h = embedding_dimension // 4
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, h), nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, h), nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, h), nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, h), nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )

        self.rgcn = SoftContrastRGCNConv(embedding_dimension, embedding_dimension,
                                         num_relations, relation_specific=relation_specific,
                                         dropout=dropout, use_raw_ego=use_raw_ego)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x
