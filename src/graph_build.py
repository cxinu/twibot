import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

INPUT_PARQUET = "data/twibot-20/twibot_df.parquet"
OUTPUT_GRAPH = "data/twibot-20/twibot_graph.pt"
STATS_CSV = "results/tables/graph_stats.csv"


def main():
    os.makedirs("data/twibot-20", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("Loading DataFrame...")
    df = pd.read_parquet(INPUT_PARQUET)
    all_ids = df["id"].astype(str).values
    id_to_idx = {uid: i for i, uid in enumerate(all_ids)}
    n_nodes = len(all_ids)
    print(f"Total nodes: {n_nodes}")

    # Build edge_index_follow (u→v: v follows u → edge from followed to follower)
    # Build edge_index_following (u→v: u follows v → edge from follower to followed)
    follow_edges = []    # followed → follower
    following_edges = []  # follower → followed

    follower_ids_col = df["follower_ids"].values
    following_ids_col = df["following_ids"].values

    for i in tqdm(range(n_nodes), desc="Building edges"):
        uid = all_ids[i]
        # Follower: v follows uid, edge from uid (followed) → v (follower)
        followers = follower_ids_col[i]
        if followers is not None and len(followers) > 0:
            for fid in followers:
                fid_str = str(fid)
                if fid_str in id_to_idx and fid_str != uid:
                    follow_edges.append((i, id_to_idx[fid_str]))
        following = following_ids_col[i]
        if following is not None and len(following) > 0:
            for fid in following:
                fid_str = str(fid)
                if fid_str in id_to_idx and fid_str != uid:
                    following_edges.append((i, id_to_idx[fid_str]))

    edge_index_follow = torch.tensor(follow_edges, dtype=torch.long).t().contiguous()
    edge_index_following = torch.tensor(following_edges, dtype=torch.long).t().contiguous()

    print(f"Follow edges: {edge_index_follow.size(1)}")
    print(f"Following edges: {edge_index_following.size(1)}")

    # Combined undirected edge_index (for SAGE)
    combined_src = torch.cat([edge_index_follow[0], edge_index_following[0],
                              edge_index_follow[1], edge_index_following[1]])
    combined_dst = torch.cat([edge_index_follow[1], edge_index_following[1],
                              edge_index_follow[0], edge_index_following[0]])
    edge_index = torch.stack([combined_src, combined_dst], dim=0)
    edge_index = torch.unique(edge_index, dim=1)
    print(f"Combined undirected edges: {edge_index.size(1)}")

    # RGCN edge_index: concatenation of follow and following edges (aligned with edge_type)
    edge_index_rgcn = torch.cat([edge_index_follow, edge_index_following], dim=1)

    # Edge type tensor: 0=follow, 1=following
    edge_type_follow = torch.zeros(edge_index_follow.size(1), dtype=torch.long)
    edge_type_following = torch.ones(edge_index_following.size(1), dtype=torch.long)

    # Labels: -1 for unlabeled (support), 0/1 for labeled
    y = torch.tensor(df["label"].values, dtype=torch.float)

    # Masks
    n_train = (df["split"] == "train").sum()
    n_dev = (df["split"] == "dev").sum()
    n_test = (df["split"] == "test").sum()
    n_support = (df["split"] == "support").sum()

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_dev] = True
    test_mask[n_train + n_dev:n_train + n_dev + n_test] = True

    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}, Support: {n_support}")

    # Domain encoding
    domain_map = {"politics": 0, "business": 1, "entertainment": 2, "sports": 3}
    domain_arr = np.array([domain_map.get(d.lower(), -1) for d in df["domain"]], dtype=np.int64)
    domain_tensor = torch.tensor(domain_arr, dtype=torch.long)

    data = Data(
        x=None,
        edge_index=edge_index,
        edge_index_rgcn=edge_index_rgcn,
        edge_index_follow=edge_index_follow,
        edge_index_following=edge_index_following,
        edge_type=torch.cat([edge_type_follow, edge_type_following]),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        domain=domain_tensor,
    )
    data.num_nodes = n_nodes

    torch.save(data, OUTPUT_GRAPH)
    print(f"Saved {OUTPUT_GRAPH}")

    # Graph stats
    n_edges_total = edge_index.size(1) // 2  # undirected edges counted once
    connected_edges = edge_index.size(1) // 2
    avg_degree = 2 * connected_edges / n_nodes
    max_degree = int(torch.bincount(edge_index[0]).max().item())

    source_counts = torch.bincount(edge_index[0], minlength=n_nodes)
    isolated = (source_counts == 0).sum().item()

    stats = {
        "n_nodes": n_nodes,
        "n_edges": connected_edges,
        "n_follow_edges": edge_index_follow.size(1),
        "n_following_edges": edge_index_following.size(1),
        "density": 2 * connected_edges / (n_nodes * (n_nodes - 1)),
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "n_isolated": isolated,
        "p_isolated": isolated / n_nodes,
    }
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"Saved {STATS_CSV}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in graph_build.py: {e}", file=sys.stderr)
        raise
