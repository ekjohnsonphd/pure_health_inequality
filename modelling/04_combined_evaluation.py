
"""
Combined evaluation of all XGBoost models.
Creates ROC and PR curves aggregated across cohorts, using the calibrated model.
"""

import json
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


# ─────────────────────────────────────────────────────────────────────────────
# Define paths
# DATA_DIR contains one folder per cohort (plus _combined/ for cross-cohort output).
# Set DATA_DIR to your server data path when replicating.
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("../data")
PLOTS_DIR = DATA_DIR / "_combined"

# Create the combined-output folder if it does not already exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Empty lists for storing ROC and PR results from each cohort
# ─────────────────────────────────────────────────────────────────────────────

all_roc = []   # stores false positive rate, true positive rate, AUC, cohort
all_pr = []    # stores recall, precision, average precision, cohort


# ─────────────────────────────────────────────────────────────────────────────
# Loop through all cohort folders
#
# Only folders starting with "female_" or "male_" are included
# (this also excludes the _combined/ output folder).
# ─────────────────────────────────────────────────────────────────────────────

for cohort_dir in sorted([
    p for p in DATA_DIR.iterdir()
    if p.is_dir() and p.name.startswith(("female_", "male_"))
]):
    cohort_name = cohort_dir.name

    model_path = cohort_dir / "calibrated_model.joblib"
    x_test_path = cohort_dir / "X_test_raw.parquet"
    y_test_path = cohort_dir / "y_test_raw.parquet"
    metrics_path = cohort_dir / "metrics.json"


    # ─────────────────────────────────────────────────────────────────────────
    # Skip cohorts that are missing the calibrated model or the raw test set
    # ─────────────────────────────────────────────────────────────────────────

    if not model_path.exists():
        print(f" No calibrated model in {cohort_dir}, skipping")
        continue

    if not (x_test_path.exists() and y_test_path.exists()):
        print(f" No raw test parquet in {cohort_dir}, skipping")
        continue


    # ─────────────────────────────────────────────────────────────────────────
    # Load the calibrated pipeline
    #
    # This is the full estimator saved by 03 (preprocessing + XGBoost +
    # isotonic calibrator). It takes RAW features and returns calibrated
    # probabilities — the same inputs the SHAP step uses.
    # ─────────────────────────────────────────────────────────────────────────

    calibrated_model = joblib.load(model_path)

    X_test = pd.read_parquet(x_test_path)
    y_test_df = pd.read_parquet(y_test_path)
    y_test = y_test_df["y_test"] if "y_test" in y_test_df.columns else y_test_df.iloc[:, 0]


    # ─────────────────────────────────────────────────────────────────────────
    # Load threshold from metrics file if available
    #
    # Default threshold is 0.5.
    # The threshold is printed but not used for ROC/PR curves,
    # because ROC and PR curves use predicted probabilities.
    # ─────────────────────────────────────────────────────────────────────────

    thr = 0.5

    if metrics_path.exists():
        with open(metrics_path) as fp:
            metrics_json = json.load(fp)

        thr = metrics_json.get("metrics", {}).get("threshold_used", 0.5)

        if not isinstance(thr, (float, int)):
            thr = 0.5


    # ─────────────────────────────────────────────────────────────────────────
    # Predict calibrated probability of early death
    #
    # predict_proba() returns probabilities for class 0 and class 1.
    # [:, 1] selects the probability of class 1 = early death.
    # ─────────────────────────────────────────────────────────────────────────

    y_prob = calibrated_model.predict_proba(X_test)[:, 1]


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
