# ============================================================================
# DEPRECATED (archived 2026-07-21). Was shap_analysis/06a_compute_shap_values_raw_model.py
# Raw (uncalibrated) model SHAP is no longer part of the analysis. The primary
# SHAP decomposition uses the calibrated model — see shap_analysis/05_shap_calibrated.py.
# Kept for reference only; not run in the pipeline.
# ============================================================================

from pathlib import Path
import pandas as pd
import xgboost as xgb
import numpy as np
import shap


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
#
# This section defines:
# - which cohort we are working with
# - the random seed for reproducible sampling
# - where the model, test data, and output files are stored
# ─────────────────────────────────────────────────────────────────────────────

cohort      = "female_50-54"
RANDOM_SEED = 42

RESULTS_DIR = Path(f"/XBoost_results/{cohort}")

MODEL_PATH  = RESULTS_DIR / f"model5_best_model_{cohort}.json"
X_PATH      = RESULTS_DIR / f"X_test_{cohort}.csv"
Y_PATH      = RESULTS_DIR / f"y_test_{cohort}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Load model
#
# The model was saved as an XGBoost JSON file.
# ─────────────────────────────────────────────────────────────────────────────

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Load test data
#
# X_test contains the feature values.
# y_test contains the true outcome:
# - 1 = early death
# - 0 = survivor
#
# The rows in X_test and y_test must be in the same order.
# ─────────────────────────────────────────────────────────────────────────────

X_test = pd.read_csv(X_PATH)
y_test = pd.read_csv(Y_PATH)["y_test"]


# ─────────────────────────────────────────────────────────────────────────────
# Split the test data by outcome
#
# early_deaths contains the rows where y_test == 1.
# survivors contains the rows where y_test == 0.
#
# This makes it possible to compare predictions and SHAP values
# between people who died and people who survived.
# ─────────────────────────────────────────────────────────────────────────────

early_deaths = X_test[y_test == 1]
survivors    = X_test[y_test == 0]


# ─────────────────────────────────────────────────────────────────────────────
# Predict mortality probabilities
#
# predict_proba() returns two probabilities:
# - probability of class 0
# - probability of class 1
#
# [:, 1] selects the probability of class 1, which is early death.
# ─────────────────────────────────────────────────────────────────────────────

p_early_deaths = model.predict_proba(early_deaths)[:, 1]
p_survivors    = model.predict_proba(survivors)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Print average predicted probabilities
#
# This is a quick diagnostic check.
# We expect the average predicted risk to be higher among early deaths
# than among survivors.
# ─────────────────────────────────────────────────────────────────────────────

print(f"Early deaths prediction: {p_early_deaths.mean():.4f}")
print(f"Survivors prediction:    {p_survivors.mean():.4f}")
print(f"Gap:                     {p_early_deaths.mean() - p_survivors.mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Create SHAP baseline/background data
#
# SHAP needs a background dataset to define the reference prediction.
# Here we use a random sample of 5000 survivors as the baseline.
#
# The random seed makes sure we get the same sample every time the script runs.
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)

idx = rng.choice(
    survivors.shape[0],
    size=5000,
    replace=False
)

baseline = survivors.iloc[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Create SHAP explainer
#
# TreeExplainer is efficient for tree-based models like XGBoost.
#
# data = baseline:
#   SHAP values are calculated relative to the survivor baseline sample.
#
# feature_perturbation = "interventional":
#   SHAP estimates feature effects using the background data distribution.
#
# model_output = "probability":
#   SHAP values explain predicted probabilities, not log-odds.
# ─────────────────────────────────────────────────────────────────────────────

explainer = shap.TreeExplainer(
    model,
    data=baseline,
    feature_perturbation="interventional",
    model_output="probability"
)


# ─────────────────────────────────────────────────────────────────────────────
# Compute SHAP values
#
# shap_early_deaths contains one SHAP value per feature per early-death person.
# shap_survivors contains one SHAP value per feature per survivor.
#
# Each SHAP value tells how much that feature pushes the predicted probability
# up or down compared with the baseline prediction.
# ─────────────────────────────────────────────────────────────────────────────

shap_early_deaths = explainer.shap_values(early_deaths)
shap_survivors    = explainer.shap_values(survivors)


# ─────────────────────────────────────────────────────────────────────────────
# Get baseline prediction
#
# expected_value is the SHAP baseline value.
# Since model_output = "probability", this is on the probability scale.
# ─────────────────────────────────────────────────────────────────────────────

base_value = explainer.expected_value


# ─────────────────────────────────────────────────────────────────────────────
# Build output data frame for early deaths
#
# Each row is one person.
# Each feature column contains the SHAP value for that feature.
#
# Extra columns:
# - y: true outcome, here 1
# - pred: predicted probability of early death
# - baseline: SHAP baseline prediction
# ─────────────────────────────────────────────────────────────────────────────

feature_names = list(X_test.columns)

df_early_deaths = pd.DataFrame(
    shap_early_deaths,
    columns=feature_names
)

df_early_deaths["y"]        = 1
df_early_deaths["pred"]     = p_early_deaths
df_early_deaths["baseline"] = base_value


# ─────────────────────────────────────────────────────────────────────────────
# Build output data frame for survivors
#
# Same structure as above, but y = 0.
# ─────────────────────────────────────────────────────────────────────────────

df_survivors = pd.DataFrame(
    shap_survivors,
    columns=feature_names
)

df_survivors["y"]        = 0
df_survivors["pred"]     = p_survivors
df_survivors["baseline"] = base_value


# ─────────────────────────────────────────────────────────────────────────────
# Combine early deaths and survivors into one SHAP dataset
#
# The final dataset contains:
# - SHAP values for all features
# - true outcome y
# - predicted probability pred
# - baseline value
# ─────────────────────────────────────────────────────────────────────────────

df_shap = pd.concat(
    [df_early_deaths, df_survivors],
    ignore_index=True
)


# ─────────────────────────────────────────────────────────────────────────────
# Save SHAP values
#
# The output is saved as shap_values.csv in the cohort results folder.
# This file can later be used for SHAP plots and decomposition analyses.
# ─────────────────────────────────────────────────────────────────────────────

df_shap.to_csv(RESULTS_DIR / "shap_values.csv", index=False)

print("Saved:", RESULTS_DIR / "shap_values.csv")

