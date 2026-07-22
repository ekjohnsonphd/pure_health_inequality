
"""
Combined evaluation of all XGBoost models.
Creates ROC and PR curves aggregated across cohorts.
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


# ─────────────────────────────────────────────────────────────────────────────
# Define paths
# BASE_DIR contains all cohort-specific model result folders.
# PLOTS_DIR is where the combined evaluation plots will be saved.
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/XBoost_results")
PLOTS_DIR = BASE_DIR / "evaluation" / "plots"

# Create plots folder if it does not already exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Empty lists for storing ROC and PR results from each cohort
# ─────────────────────────────────────────────────────────────────────────────

all_roc = []   # stores false positive rate, true positive rate, AUC, cohort
all_pr = []    # stores recall, precision, average precision, cohort


# ─────────────────────────────────────────────────────────────────────────────
# Loop through all cohort folders
#
# Only folders starting with "female_" or "male_" are included.
# Example:
# female_50-54
# male_65-69
# ─────────────────────────────────────────────────────────────────────────────

for cohort_dir in sorted([
    p for p in BASE_DIR.iterdir()
    if p.is_dir() and p.name.startswith(("female_", "male_"))
]):
    cohort_name = cohort_dir.name


    # ─────────────────────────────────────────────────────────────────────────
    # Find X_test file for this cohort
    # ─────────────────────────────────────────────────────────────────────────

    possible_csv = list(cohort_dir.glob("X_test_*.csv"))
    
    if not possible_csv:
        print(f" No X_test CSV found in {cohort_dir}, skipping")
        continue
    
    test_csv = possible_csv[0]


    # ─────────────────────────────────────────────────────────────────────────
    # Find y_test file for this cohort
    # ─────────────────────────────────────────────────────────────────────────

    y_test_candidates = list(cohort_dir.glob("y_test_*.csv"))
    
    if not y_test_candidates:
        print(f" No y_test CSV found in {cohort_dir}, skipping")
        continue
    
    y_test_file = y_test_candidates[0]


    # ─────────────────────────────────────────────────────────────────────────
    # Find model file and metrics file
    #
    # model5_best_model_*.json contains the trained XGBoost model.
    # Model_5_*.json contain saved evaluation metrics and threshold.
    # ─────────────────────────────────────────────────────────────────────────

    model_json_candidates = list(cohort_dir.glob("model5_best_model_*json"))
    metrics_json_candidates = list(cohort_dir.glob("Model_5_*.json"))

    if not model_json_candidates:
        print(f" No model JSON found in {cohort_dir}, skipping")
        continue

    model_json_path = model_json_candidates[0]
    metrics_json_path = metrics_json_candidates[0] if metrics_json_candidates else None


    # ─────────────────────────────────────────────────────────────────────────
    # Load XGBoost model
    # ─────────────────────────────────────────────────────────────────────────

    model = xgb.XGBClassifier()
    model.load_model(model_json_path)


    # ─────────────────────────────────────────────────────────────────────────
    # Load test data
    #
    # X_test contains features.
    # y_test contains true outcome labels.
    # ─────────────────────────────────────────────────────────────────────────

    X_test = pd.read_csv(test_csv)
    y_test = pd.read_csv(y_test_file)["y_test"]


    # ─────────────────────────────────────────────────────────────────────────
    # Load threshold from metrics file if available
    #
    # Default threshold is 0.5.
    # The threshold is printed but not used for ROC/PR curves,
    # because ROC and PR curves use predicted probabilities.
    # ─────────────────────────────────────────────────────────────────────────

    thr = 0.5

    if metrics_json_path is not None:
        with open(metrics_json_path) as fp:
            metrics_json = json.load(fp)
        
        thr = metrics_json.get("test_metrics", {}).get("threshold_used", 0.5)
        
        if not isinstance(thr, (float, int)):
            thr = 0.5


    # ─────────────────────────────────────────────────────────────────────────
    # Predict probability of early death
    #
    # predict_proba() returns probabilities for class 0 and class 1.
    # [:, 1] selects the probability of class 1 = early death.
    # ─────────────────────────────────────────────────────────────────────────

    y_prob = model.predict_proba(X_test)[:, 1]


    # ─────────────────────────────────────────────────────────────────────────
    # Calculate ROC curve and ROC AUC
    #
    # ROC compares true positive rate and false positive rate
    # across all possible thresholds.
    # ─────────────────────────────────────────────────────────────────────────

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)


    # ─────────────────────────────────────────────────────────────────────────
    # Calculate Precision-Recall curve and average precision
    #
    # PR curves are useful when the positive class is rare.
    # ─────────────────────────────────────────────────────────────────────────

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)


    # ─────────────────────────────────────────────────────────────────────────
    # Store results for this cohort
    # These are later used to create combined plots.
    # ─────────────────────────────────────────────────────────────────────────

    all_roc.append({
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "cohort": cohort_name
    })

    all_pr.append({
        "recall": recall,
        "precision": precision,
        "ap": pr_auc,
        "cohort": cohort_name
    })


    # ─────────────────────────────────────────────────────────────────────────
    # Print cohort-specific evaluation summary
    # ─────────────────────────────────────────────────────────────────────────

    print(
        f"Cohort {cohort_name}: "
        f"ROC AUC={auc:.4f}, "
        f"PR AUC={pr_auc:.4f}, "
        f"Threshold={thr:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot combined ROC curves
# One line per cohort.
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams["figure.figsize"] = (12, 8)

plt.figure()

for entry in all_roc:
    plt.plot(
        entry["fpr"],
        entry["tpr"],
        label=f"{entry['cohort']} (AUC={entry['auc']:.3f})"
    )

# Diagonal reference line: random classifier
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Save ROC plot
plt.savefig(PLOTS_DIR / "combined_roc.png", dpi=300, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot combined Precision-Recall curves
# One line per cohort.
# ─────────────────────────────────────────────────────────────────────────────

plt.figure()

for entry in all_pr:
    plt.plot(
        entry["recall"],
        entry["precision"],
        label=f"{entry['cohort']} (AP={entry['ap']:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Combined Precision-Recall Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Save PR plot
plt.savefig(PLOTS_DIR / "combined_pr.png", dpi=300, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Final message
# ─────────────────────────────────────────────────────────────────────────────

print(f"Combined evaluation plots saved to {PLOTS_DIR}")

