import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

PROFILE_FEATURES = [
    "followers_count", "friends_count", "listed_count",
    "favourites_count", "statuses_count", "verified",
    "protected", "geo_enabled", "default_profile",
    "default_profile_image", "has_extended_profile",
    "profile_use_background_image", "contributors_enabled",
    "is_translator", "is_translation_enabled",
    "profile_background_tile", "has_description", "has_url",
    "screen_name_length", "name_length", "description_length",
    "account_age_days",
]

TWEET_FEATURES = [
    "tweet_count", "avg_tweet_length", "hashtag_count",
    "url_count", "mention_count", "retweet_count",
    "avg_retweet_count", "avg_favorite_count",
    "num_numeric", "num_special_chars",
    "tweet_url_ratio", "tweet_hashtag_ratio",
]

NEIGHBOUR_ATTR_SOURCE = [
    "followers_count", "friends_count", "statuses_count",
    "favourites_count", "account_age_days",
]

NEIGHBOUR_ATTR_FEATURES = [
    "mean_neighbour_followers", "mean_neighbour_friends",
    "mean_neighbour_statuses", "mean_neighbour_favourites",
    "mean_neighbour_account_age_days", "std_neighbour_followers",
]

LABEL_PROP_FEATURES = ["neighbour_bot_rate"]

INPUT_PARQUET = "dataset/twibot_df.parquet"
OUTPUT_DIR = "dataset"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading DataFrame...")
    df = pd.read_parquet(INPUT_PARQUET)
    # Expand list neighbor_ids for fast lookup
    print("Building ID→index map...")
    id_to_idx = {str(uid): i for i, uid in enumerate(df["id"])}
    labeled_mask = df["label"] >= 0
    labeled_idx = np.where(labeled_mask)[0]
    print(f"Labeled nodes: {len(labeled_idx)}")

    # Build neighbour features
    print("Computing neighbour-attribute features...")
    n_total = len(df)
    neighbour_attr_array = np.zeros((n_total, len(NEIGHBOUR_ATTR_FEATURES)), dtype=np.float32)

    source_cols = NEIGHBOUR_ATTR_SOURCE
    source_data = df[source_cols].values.astype(np.float32)

    nbr_ids_col = df["neighbor_ids"].values
    for i in tqdm(range(n_total), desc="Neighbour attr"):
        nbr_ids = nbr_ids_col[i]
        if nbr_ids is None or (isinstance(nbr_ids, (list, np.ndarray)) and len(nbr_ids) == 0):
            continue
        nbr_indices = [id_to_idx.get(nid) for nid in nbr_ids if nid in id_to_idx]
        nbr_indices = [idx for idx in nbr_indices if idx is not None]
        if not nbr_indices:
            continue
        nbr_vals = source_data[nbr_indices]
        means = np.log1p(np.nanmean(nbr_vals, axis=0))
        neighbour_attr_array[i, :5] = means
        follower_vals = nbr_vals[:, 0]
        std_val = np.log1p(np.nanstd(follower_vals)) if len(follower_vals) > 1 else 0.0
        neighbour_attr_array[i, 5] = std_val

    # Fill NaN/inf
    neighbour_attr_array = np.nan_to_num(neighbour_attr_array, 0.0)

    # Build label propagation feature
    print("Computing label-propagation feature (neighbour_bot_rate)...")
    train_mask = df["split"] == "train"
    train_label = df["label"].values
    label_prop_array = np.zeros((n_total, 1), dtype=np.float32)
    for i in tqdm(range(n_total), desc="Label prop"):
        nbr_ids = nbr_ids_col[i]
        if nbr_ids is None or (isinstance(nbr_ids, (list, np.ndarray)) and len(nbr_ids) == 0):
            continue
        nbr_indices = [id_to_idx.get(nid) for nid in nbr_ids if nid in id_to_idx]
        nbr_indices = [idx for idx in nbr_indices if idx is not None]
        if not nbr_indices:
            continue
        nbr_labels = train_label[nbr_indices]
        train_nbr_mask = train_mask.values[nbr_indices]
        valid = train_nbr_mask & (nbr_labels >= 0)
        if valid.sum() > 0:
            bot_count = (nbr_labels[valid] == 1).sum()
            label_prop_array[i, 0] = bot_count / valid.sum()

    print("Preparing profile features (log1p on count-based)...")
    count_cols = ["followers_count", "friends_count", "listed_count",
                  "favourites_count", "statuses_count", "account_age_days"]
    profile_array = df[PROFILE_FEATURES].values.astype(np.float32)
    for ci, col in enumerate(PROFILE_FEATURES):
        if col in count_cols:
            profile_array[:, ci] = np.log1p(np.clip(profile_array[:, ci], 0, None))

    print("Preparing tweet features...")
    tweet_array = df[TWEET_FEATURES].values.astype(np.float32)
    tweet_count_cols = ["hashtag_count", "url_count", "mention_count",
                        "retweet_count", "num_numeric", "num_special_chars"]
    for ci, col in enumerate(TWEET_FEATURES):
        if col in tweet_count_cols:
            tweet_array[:, ci] = np.log1p(np.clip(tweet_array[:, ci], 0, None))

    # Save feature arrays
    np.save(os.path.join(OUTPUT_DIR, "features_profile.npy"), profile_array)
    np.save(os.path.join(OUTPUT_DIR, "features_tweet.npy"), tweet_array)
    np.save(os.path.join(OUTPUT_DIR, "features_neighbour_attr.npy"), neighbour_attr_array)
    np.save(os.path.join(OUTPUT_DIR, "features_label_prop.npy"), label_prop_array)

    # Save feature names
    feature_names = {
        "profile": PROFILE_FEATURES,
        "tweet": TWEET_FEATURES,
        "neighbour_attr": NEIGHBOUR_ATTR_FEATURES,
        "label_prop": LABEL_PROP_FEATURES,
    }
    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Profile features: {profile_array.shape} -> {OUTPUT_DIR}/features_profile.npy")
    print(f"Tweet features: {tweet_array.shape} -> {OUTPUT_DIR}/features_tweet.npy")
    print(f"Neighbour-attr features: {neighbour_attr_array.shape} -> {OUTPUT_DIR}/features_neighbour_attr.npy")
    print(f"Label-prop features: {label_prop_array.shape} -> {OUTPUT_DIR}/features_label_prop.npy")
    print(f"Feature names saved to {OUTPUT_DIR}/feature_names.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in features.py: {e}", file=sys.stderr)
        raise
