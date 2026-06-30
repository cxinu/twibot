import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")

OUTPUT_CSV = "results/tables/baselines.csv"
CONFUSION_DIR = "results/figures"


def compute_metrics(y_true, y_pred, y_prob):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_binary = f1_score(y_true, y_pred, average="binary")
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return {
        "f1_macro": round(f1_macro, 4),
        "f1_binary": round(f1_binary, 4),
        "auc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def save_confusion(y_true, y_pred, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Human", "Bot"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    path = os.path.join(CONFUSION_DIR, f"cm_{name.lower().replace('-', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    os.makedirs(CONFUSION_DIR, exist_ok=True)
    print("Loading data...")
    df = pd.read_parquet("dataset/twibot_df.parquet")
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    profile_cols = [
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

    X_train_profile = train[profile_cols].values.astype(np.float32)
    y_train = train["label"].values.astype(int)
    X_test_profile = test[profile_cols].values.astype(np.float32)
    y_test = test["label"].values.astype(int)

    count_cols = ["followers_count", "friends_count", "listed_count",
                  "favourites_count", "statuses_count", "account_age_days"]
    for ci, col in enumerate(profile_cols):
        if col in count_cols:
            X_train_profile[:, ci] = np.log1p(np.clip(X_train_profile[:, ci], 0, None))
            X_test_profile[:, ci] = np.log1p(np.clip(X_test_profile[:, ci], 0, None))

    # Baseline-Majority: predict majority class
    majority_class = int(y_train.mean() >= 0.5)
    y_pred_majority = np.full_like(y_test, majority_class)
    y_prob_majority = np.full_like(y_test, y_train.mean())
    metrics_majority = compute_metrics(y_test, y_pred_majority, y_prob_majority)
    metrics_majority["config"] = "Baseline-Majority"
    print(f"Baseline-Majority: {metrics_majority}")
    save_confusion(y_test, y_pred_majority, "Baseline-Majority")

    # Baseline-LogReg
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    lr.fit(X_train_profile, y_train)
    y_pred_lr = lr.predict(X_test_profile)
    y_prob_lr = lr.predict_proba(X_test_profile)[:, 1]
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr)
    metrics_lr["config"] = "Baseline-LogReg"
    print(f"Baseline-LogReg: {metrics_lr}")
    save_confusion(y_test, y_pred_lr, "Baseline-LogReg")

    results = pd.DataFrame([metrics_majority, metrics_lr])
    results = results[["config", "f1_macro", "f1_binary", "auc", "precision", "recall"]]
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in baselines.py: {e}", file=sys.stderr)
        raise
