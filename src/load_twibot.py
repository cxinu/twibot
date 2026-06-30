import json
import os
import sys
from collections import Counter
from datetime import datetime

import ijson
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "dataset/Twibot-20"
OUTPUT_PARQUET = "dataset/twibot_df.parquet"
STATS_CSV = "results/tables/dataset_stats.csv"
FIG_OVERVIEW = "results/figures/fig_dataset_overview.png"


def parse_created_at(date_str):
    if not date_str:
        return None
    date_str = str(date_str).strip()
    try:
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_str, errors="coerce")
        except Exception:
            return None


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0.0):
    if val is None:
        return default
    try:
        return float(str(val).strip()) if "." in str(val).strip() else int(str(val).strip())
    except (ValueError, TypeError):
        return default


def safe_str(val):
    if val is None:
        return ""
    return str(val).strip()


def extract_profile_features(profile):
    if not profile or not isinstance(profile, dict):
        return {}
    return {
        "followers_count": safe_int(profile.get("followers_count")),
        "friends_count": safe_int(profile.get("friends_count")),
        "listed_count": safe_int(profile.get("listed_count")),
        "favourites_count": safe_int(profile.get("favourites_count")),
        "statuses_count": safe_int(profile.get("statuses_count")),
        "verified": 1 if safe_str(profile.get("verified")).lower() == "true" else 0,
        "protected": 1 if safe_str(profile.get("protected")).lower() == "true" else 0,
        "geo_enabled": 1 if safe_str(profile.get("geo_enabled")).lower() == "true" else 0,
        "default_profile": 1 if safe_str(profile.get("default_profile")).lower() == "true" else 0,
        "default_profile_image": 1 if safe_str(profile.get("default_profile_image")).lower() == "true" else 0,
        "has_extended_profile": 1 if safe_str(profile.get("has_extended_profile")).lower() == "true" else 0,
        "profile_use_background_image": 1 if safe_str(profile.get("profile_use_background_image")).lower() == "true" else 0,
        "contributors_enabled": 1 if safe_str(profile.get("contributors_enabled")).lower() == "true" else 0,
        "is_translator": 1 if safe_str(profile.get("is_translator")).lower() == "true" else 0,
        "is_translation_enabled": 1 if safe_str(profile.get("is_translation_enabled")).lower() == "true" else 0,
        "profile_background_tile": 1 if safe_str(profile.get("profile_background_tile")).lower() == "true" else 0,
        "has_description": 1 if len(safe_str(profile.get("description", ""))) > 0 else 0,
        "has_url": 1 if len(safe_str(profile.get("url", ""))) > 0 else 0,
        "screen_name_length": len(safe_str(profile.get("screen_name", ""))),
        "name_length": len(safe_str(profile.get("name", ""))),
        "description_length": len(safe_str(profile.get("description", ""))),
        "created_at": safe_str(profile.get("created_at", "")),
    }


def extract_tweet_features(tweets):
    if not tweets:
        return {k: 0.0 for k in [
            "tweet_count", "avg_tweet_length", "hashtag_count", "url_count",
            "mention_count", "retweet_count", "avg_retweet_count",
            "avg_favorite_count", "num_numeric", "num_special_chars",
            "tweet_url_ratio", "tweet_hashtag_ratio",
        ]}
    lengths = []
    total_hashtags = 0
    total_urls = 0
    total_mentions = 0
    total_retweets = 0
    total_retweet_count = 0
    total_favorite_count = 0
    total_numeric = 0
    total_special = 0
    for t in tweets:
        text = safe_str(t.get("text", "") if isinstance(t, dict) else t)
        if isinstance(t, dict):
            total_retweet_count += safe_int(t.get("retweet_count", 0))
            total_favorite_count += safe_int(t.get("favorite_count", 0))
        lengths.append(len(text))
        total_hashtags += text.count("#")
        total_urls += text.count("http")
        total_mentions += text.count("@")
        total_retweets += 1 if text.startswith("RT") else 0
        total_numeric += sum(c.isdigit() for c in text)
        total_special += sum(not c.isalnum() and not c.isspace() for c in text)
    n = len(lengths)
    return {
        "tweet_count": n,
        "avg_tweet_length": float(np.mean(lengths)) if lengths else 0.0,
        "hashtag_count": total_hashtags,
        "url_count": total_urls,
        "mention_count": total_mentions,
        "retweet_count": total_retweets,
        "avg_retweet_count": total_retweet_count / n if n else 0.0,
        "avg_favorite_count": total_favorite_count / n if n else 0.0,
        "num_numeric": total_numeric,
        "num_special_chars": total_special,
        "tweet_url_ratio": total_urls / n if n else 0.0,
        "tweet_hashtag_ratio": total_hashtags / n if n else 0.0,
    }


def extract_neighbor_ids(neighbor):
    following_ids = []
    follower_ids = []
    if neighbor and isinstance(neighbor, dict):
        following_ids = [str(x) for x in neighbor.get("following", [])]
        follower_ids = [str(x) for x in neighbor.get("follower", [])]
    return following_ids, follower_ids


def process_user_record(record, split, default_label=-1):
    record_id = safe_str(record.get("ID", ""))
    if not record_id:
        return None
    profile = record.get("profile") or {}
    profile_feats = extract_profile_features(profile)
    created_at_str = profile_feats.pop("created_at", "")
    parsed_date = parse_created_at(created_at_str)
    account_age_days = (pd.Timestamp.now() - pd.Timestamp(parsed_date)).days if parsed_date else 0
    following_ids, follower_ids = extract_neighbor_ids(record.get("neighbor"))
    all_neighbor_ids = list(set(following_ids + follower_ids))
    label_raw = record.get("label")
    if label_raw is not None:
        label = int(str(label_raw).strip())
    else:
        label = default_label
    domain_raw = record.get("domain")
    if isinstance(domain_raw, list):
        domain = domain_raw[0] if domain_raw else "unknown"
    else:
        domain = safe_str(domain_raw) if domain_raw else "unknown"
    domain = domain.lower()

    result = {
        "id": record_id,
        "split": split,
        "label": label,
        "domain": domain,
        "account_age_days": account_age_days,
        "neighbor_ids": all_neighbor_ids,
        "follower_ids": follower_ids,
        "following_ids": following_ids,
    }
    result.update(profile_feats)
    tweet_feats = extract_tweet_features(record.get("tweet"))
    result.update(tweet_feats)
    return result


def main():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_records = []
    for split_name in ["train", "dev", "test"]:
        fp = os.path.join(DATA_DIR, f"{split_name}.json")
        print(f"Loading {split_name} from {fp}...")
        split_data = json.load(open(fp, "r"))
        for rec in tqdm(split_data, desc=f"Processing {split_name}"):
            processed = process_user_record(rec, split_name)
            if processed:
                all_records.append(processed)
        del split_data
    print(f"Loaded {len(all_records)} records from train/dev/test")

    print("Streaming support.json...")
    support_path = os.path.join(DATA_DIR, "support.json")
    with open(support_path, "rb") as f:
        parser = ijson.items(f, "item")
        for rec in tqdm(parser, desc="Processing support", total=217754):
            processed = process_user_record(rec, "support")
            if processed:
                all_records.append(processed)
    print(f"Total records after support: {len(all_records)}")

    df = pd.DataFrame(all_records)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    df["id"] = df["id"].astype(str)

    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Split distribution:\n{df['split'].value_counts()}")

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved {OUTPUT_PARQUET}")

    split_order = ["train", "dev", "test", "support"]
    stats_rows = []
    for s in split_order:
        sub = df[df["split"] == s]
        labeled = sub[sub["label"] >= 0]
        n_bots = int((labeled["label"] == 1).sum())
        n_humans = int((labeled["label"] == 0).sum())
        stats_rows.append({
            "split": s,
            "n_users": len(sub),
            "n_labeled": len(labeled),
            "n_bots": n_bots,
            "n_humans": n_humans,
            "bot_ratio": round(n_bots / len(labeled), 4) if len(labeled) > 0 else 0,
        })
    pd.DataFrame(stats_rows).to_csv(STATS_CSV, index=False)
    print(f"Saved {STATS_CSV}")

    per_domain = df[df["label"] >= 0].groupby(["split", "domain"])["label"].agg(["count", "mean"]).reset_index()
    per_domain.columns = ["split", "domain", "n_users", "bot_rate"]
    per_domain["bot_rate"] = per_domain["bot_rate"].round(4)
    per_domain.to_csv(STATS_CSV.replace(".csv", "_per_domain.csv"), index=False)
    print(f"Saved {STATS_CSV.replace('.csv', '_per_domain.csv')}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#457B9D", "#E63946"]
    for idx, s in enumerate(["train", "dev", "test"]):
        sub = df[(df["split"] == s) & (df["label"] >= 0)]
        counts = sub["label"].value_counts()
        ax = axes[idx]
        bars = ax.bar(["Human (0)", "Bot (1)"], [counts.get(0, 0), counts.get(1, 0)], color=colors)
        ax.set_title(f"{s.capitalize()} (n={len(sub)})")
        ax.set_ylabel("Count")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{int(h)}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_OVERVIEW, dpi=150)
    plt.close()
    print(f"Saved {FIG_OVERVIEW}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in load_twibot.py: {e}", file=sys.stderr)
        raise
