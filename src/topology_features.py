import os
import sys
import json
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

INPUT_PARQUET = "dataset/twibot_df.parquet"
OUTPUT_FILE = "dataset/features_topology.npy"
FEATURE_NAMES_FILE = "dataset/feature_names.json"

TOPOLOGY_FEATURES = [
    "degree", "in_degree", "out_degree",
    "clustering_coefficient",
    "pagerank",
    "k_core_number",
    "community_id",
    "in_out_ratio",
]


def main():
    os.makedirs("dataset", exist_ok=True)
    print("Loading DataFrame...")
    df = pd.read_parquet(INPUT_PARQUET)
    all_ids = df["id"].astype(str).values
    id_set = set(all_ids)
    print(f"Total nodes: {len(all_ids)}")

    print("Building directed graph...")
    G = nx.DiGraph()
    G.add_nodes_from(all_ids)
    edge_count = 0
    follower_ids_col = df["follower_ids"].values
    following_ids_col = df["following_ids"].values

    for i in tqdm(range(len(df)), desc="Adding edges"):
        uid = all_ids[i]
        # Following edges: user follows X -> uid -> X
        following = following_ids_col[i]
        if following is not None and len(following) > 0:
            for fid in following:
                fid = str(fid)
                if fid in id_set and fid != uid:
                    G.add_edge(uid, fid)
                    edge_count += 1
        # Follower edges: X follows user -> X -> uid
        followers = follower_ids_col[i]
        if followers is not None and len(followers) > 0:
            for fid in followers:
                fid = str(fid)
                if fid in id_set and fid != uid:
                    G.add_edge(fid, uid)
                    edge_count += 1

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (added {edge_count})")

    print("Computing degree features...")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    degrees = dict(G.degree())
    n_nodes = len(all_ids)
    feat_matrix = np.zeros((n_nodes, len(TOPOLOGY_FEATURES)), dtype=np.float32)

    for i, uid in enumerate(tqdm(all_ids, desc="Degree")):
        feat_matrix[i, 0] = degrees.get(uid, 0)
        feat_matrix[i, 1] = in_degrees.get(uid, 0)
        feat_matrix[i, 2] = out_degrees.get(uid, 0)
        in_d = in_degrees.get(uid, 0)
        out_d = out_degrees.get(uid, 0)
        feat_matrix[i, 7] = in_d / (out_d + 1)

    deg_cols = [0, 1, 2, 7]
    for c in deg_cols:
        feat_matrix[:, c] = np.log1p(feat_matrix[:, c])

    print("Computing clustering coefficient...")
    G_undirected = G.to_undirected()
    clust = nx.clustering(G_undirected, nodes=all_ids)
    for i, uid in enumerate(tqdm(all_ids, desc="Clustering")):
        feat_matrix[i, 3] = clust.get(uid, 0.0)

    print("Computing PageRank...")
    pr = nx.pagerank(G, alpha=0.85, tol=1e-4)
    for i, uid in enumerate(tqdm(all_ids, desc="PageRank")):
        feat_matrix[i, 4] = pr.get(uid, 0.0)

    print("Computing k-core number...")
    kcore = nx.core_number(G_undirected)
    for i, uid in enumerate(tqdm(all_ids, desc="K-core")):
        feat_matrix[i, 5] = kcore.get(uid, 0)

    print("Computing community detection (Louvain)...")
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G_undirected, seed=42)
        node_to_community = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = cid
        for i, uid in enumerate(tqdm(all_ids, desc="Community")):
            feat_matrix[i, 6] = node_to_community.get(uid, -1)
    except Exception as e:
        print(f"Louvain failed ({e}), using connected components instead")
        comp = nx.connected_components(G_undirected)
        node_to_comp = {}
        for cid, comp_set in enumerate(comp):
            for node in comp_set:
                node_to_comp[node] = cid
        for i, uid in enumerate(all_ids):
            feat_matrix[i, 6] = node_to_comp.get(uid, -1)

    feat_matrix = np.nan_to_num(feat_matrix, 0.0)
    np.save(OUTPUT_FILE, feat_matrix)
    print(f"Saved {OUTPUT_FILE} with shape {feat_matrix.shape}")

    with open(FEATURE_NAMES_FILE, "r") as f:
        names = json.load(f)
    names["topology"] = TOPOLOGY_FEATURES
    with open(FEATURE_NAMES_FILE, "w") as f:
        json.dump(names, f, indent=2)
    print(f"Updated {FEATURE_NAMES_FILE} with topology features")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in topology_features.py: {e}", file=sys.stderr)
        raise
