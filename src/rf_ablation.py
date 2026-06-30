import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score,
    precision_recall_fscore_support, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUT_DIR = "results/tables"
FIGURE_DIR = "results/figures"
FEATURE_DIR = "dataset"

CONFIGS = [
    "RF-Profile",
    "RF-Profile+Tweet",
    "RF-Profile+Tweet+Topology",
    "RF-Profile+Tweet+Topology+NeighbourAttr",
    "RF-All",
    "RF-All-minus-LabelProp",
]

FEATURE_GROUPS = ["profile", "tweet", "topology", "neighbour_attr", "label_prop"]
COLORS = {
    "profile": "#2A9D8F",
    "tweet": "#E9C46A",
    "topology": "#6A994E",
    "neighbour_attr": "#F4A261",
    "label_prop": "#9D4EDD",
    "all": "#264653",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("Loading features...")
    with open(os.path.join(FEATURE_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    feats = {}
    for group in FEATURE_GROUPS:
        path = os.path.join(FEATURE_DIR, f"features_{group}.npy")
        feats[group] = np.load(path)
        print(f"  {group}: {feats[group].shape}")

    df = pd.read_parquet("dataset/twibot_df.parquet")
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    dev_mask = df["split"] == "dev"

    y_train = df["label"].values[train_mask].astype(int)
    y_test = df["label"].values[test_mask].astype(int)
    y_dev = df["label"].values[dev_mask].astype(int)
    print(f"Train: {len(y_train)}, Dev: {len(y_dev)}, Test: {len(y_test)}")

    domain_test = df["domain"].values[test_mask]

    def build_X(groups):
        parts = [feats[g][train_mask] for g in groups]
        return np.concatenate(parts, axis=1)

    def build_X_test(groups):
        parts = [feats[g][test_mask] for g in groups]
        return np.concatenate(parts, axis=1)

    config_groups = {
        "RF-Profile": ["profile"],
        "RF-Profile+Tweet": ["profile", "tweet"],
        "RF-Profile+Tweet+Topology": ["profile", "tweet", "topology"],
        "RF-Profile+Tweet+Topology+NeighbourAttr": ["profile", "tweet", "topology", "neighbour_attr"],
        "RF-All": ["profile", "tweet", "topology", "neighbour_attr", "label_prop"],
        "RF-All-minus-LabelProp": ["profile", "tweet", "topology", "neighbour_attr"],
    }

    results = []

    for cfg_name in CONFIGS:
        groups = config_groups[cfg_name]
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name} (groups: {groups})")
        X_train = build_X(groups)
        X_test = build_X_test(groups)

        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        clf = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_binary = f1_score(y_test, y_pred, average="binary")
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0

        row = {
            "config": cfg_name,
            "f1_macro": round(f1_macro, 4),
            "f1_binary": round(f1_binary, 4),
            "auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }
        results.append(row)
        print(f"  F1 macro: {f1_macro:.4f}, AUC: {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Human", "Bot"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {cfg_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f"cm_{cfg_name.lower().replace('+', '_').replace('-', '_')}.png"), dpi=150)
        plt.close()

        # Per-domain breakdown for RF-All and RF-All-minus-LabelProp
        if cfg_name in ("RF-All", "RF-All-minus-LabelProp"):
            print(f"  Per-domain breakdown for {cfg_name}:")
            for domain in sorted(set(domain_test)):
                mask = domain_test == domain
                if mask.sum() == 0:
                    continue
                y_domain = y_test[mask]
                y_pred_domain = clf.predict(X_test[mask])
                y_prob_domain = clf.predict_proba(X_test[mask])[:, 1]
                f1_d = f1_score(y_domain, y_pred_domain, average="macro")
                auc_d = roc_auc_score(y_domain, y_prob_domain)
                prec_d, rec_d, _, _ = precision_recall_fscore_support(y_domain, y_pred_domain, average="binary")
                base_rate = y_domain.mean()
                print(f"    {domain}: F1={f1_d:.4f}, AUC={auc_d:.4f}, P={prec_d:.4f}, R={rec_d:.4f}, bot_rate={base_rate:.4f}")
                results.append({
                    "config": f"{cfg_name}_{domain}",
                    "f1_macro": round(f1_d, 4),
                    "f1_binary": round(f1_score(y_domain, y_pred_domain, average="binary"), 4),
                    "auc": round(auc_d, 4),
                    "precision": round(prec_d, 4),
                    "recall": round(rec_d, 4),
                    "bot_rate": round(base_rate, 4),
                })
                # Per-domain CM
                cm_d = confusion_matrix(y_domain, y_pred_domain)
                fig, ax = plt.subplots(figsize=(5, 4))
                ConfusionMatrixDisplay(cm_d, display_labels=["Human", "Bot"]).plot(ax=ax, cmap="Blues")
                ax.set_title(f"CM - {cfg_name} ({domain})")
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURE_DIR, f"cm_{cfg_name.lower().replace('+', '_').replace('-', '_')}_{domain}.png"), dpi=150)
                plt.close()

        # Save feature importances for RF-All
        if cfg_name == "RF-All":
            importances = clf.feature_importances_
            all_feature_names = []
            for g in groups:
                all_feature_names.extend(feature_names[g])
            # Top 20
            top_k = min(20, len(all_feature_names))
            top_idx = np.argsort(importances)[::-1][:top_k]
            top_names = [all_feature_names[i] for i in top_idx]
            top_imps = importances[top_idx]

            # Map feature to group
            feat_to_group = {}
            offset = 0
            for g in groups:
                for fn in feature_names[g]:
                    feat_to_group[fn] = g
                offset += len(feature_names[g])

            group_colors = [COLORS.get(feat_to_group.get(n, "all"), "#264653") for n in top_names]

            fig, ax = plt.subplots(figsize=(10, 7))
            bars = ax.barh(range(top_k), top_imps, color=group_colors[::-1])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(top_names[::-1])
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top-20 Feature Importances - RF-All")
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLORS[g], label=g) for g in groups
            ]
            ax.legend(handles=legend_elements, loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURE_DIR, "fig_rf_feature_importance.png"), dpi=150)
            plt.close()
            print("  Saved top-20 feature importance chart")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "rf_ablation.csv"), index=False)
    print(f"\nSaved {os.path.join(OUTPUT_DIR, 'rf_ablation.csv')}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in rf_ablation.py: {e}", file=sys.stderr)
        raise
