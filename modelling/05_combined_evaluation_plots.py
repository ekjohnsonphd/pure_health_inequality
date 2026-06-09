#!/usr/bin/env python3
"""
Combined evaluation of all XGBoost models.
Creates ROC and PR curves aggregated across cohorts.
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

BASE_DIR = Path("/XBoost_results")
PLOTS_DIR = BASE_DIR / "evaluation" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

all_roc = []   # list of dicts with keys: fpr, tpr, auc, cohort
all_pr = []    # list of dicts with keys: recall, precision, ap, cohort

for cohort_dir in sorted([p for p in BASE_DIR.iterdir() if p.is_dir() and p.name.startswith(("female_", "male_"))]):
    cohort_name = cohort_dir.name

    # find X_test CSV
    possible_csv = list(cohort_dir.glob("X_test_*.csv"))
    if not possible_csv:
        print(f"⚠️ No X_test CSV found in {cohort_dir}, skipping")
        continue
    test_csv = possible_csv[0]

    # find y_test CSV
    y_test_candidates = list(cohort_dir.glob("y_test_*.csv"))
    if not y_test_candidates:
        print(f"⚠️ No y_test CSV found in {cohort_dir}, skipping")
        continue
    y_test_file = y_test_candidates[0]

    # load model
    model_json_candidates = list(cohort_dir.glob("model5_best_model_*json"))
    metrics_json_candidates = list(cohort_dir.glob("Model_5_*.json"))
    if not model_json_candidates:
        print(f"⚠️ No model JSON found in {cohort_dir}, skipping")
        continue
    model_json_path = model_json_candidates[0]
    metrics_json_path = metrics_json_candidates[0] if metrics_json_candidates else None

    model = xgb.XGBClassifier()
    model.load_model(model_json_path)

    X_test = pd.read_csv(test_csv)
    y_test = pd.read_csv(y_test_file)["y_test"]

    thr = 0.5
    if metrics_json_path is not None:
        with open(metrics_json_path) as fp:
            metrics_json = json.load(fp)
        thr = metrics_json.get("test_metrics", {}).get("threshold_used", 0.5)
        if not isinstance(thr, (float, int)):
            thr = 0.5

    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    all_roc.append({"fpr": fpr, "tpr": tpr, "auc": auc, "cohort": cohort_name})
    all_pr.append({"recall": recall, "precision": precision, "ap": pr_auc, "cohort": cohort_name})

    print(f"Cohort {cohort_name}: ROC AUC={auc:.4f}, PR AUC={pr_auc:.4f}, Threshold={thr:.3f}")

# Plot ROC
plt.rcParams["figure.figsize"] = (12, 8)

plt.figure()
for entry in all_roc:
    plt.plot(entry["fpr"], entry["tpr"], label=f"{entry['cohort']} (AUC={entry['auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(PLOTS_DIR / "combined_roc.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot PR
plt.figure()
for entry in all_pr:
    plt.plot(entry["recall"], entry["precision"], label=f"{entry['cohort']} (AP={entry['ap']:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Combined Precision‑Recall Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(PLOTS_DIR / "combined_pr.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Combined evaluation plots saved to {PLOTS_DIR}")
