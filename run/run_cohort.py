"""
Run the cohort analysis pipeline.

This script orchestrates the full modelling workflow for the cohort data:
1. Load and split the raw data.
2. Resample the training set to handle class imbalance.
3. Train and evaluate an XGBoost model (logistic regression parts are kept for reference).
4. Perform SHAP analysis and Shapley-Owens-Shorrocks (SOS) decomposition for model interpretation.
5. Save outputs to disk.

All files are written under ``results/cohort``.
"""

import joblib
import os
import sys
import numpy as np

# ----------------------------------------------------------------------
# 1️⃣  Project setup: add the root `code/` folder to ``sys.path`` so we can
#     import modules using absolute imports.
# ----------------------------------------------------------------------
project_root = os.path.abspath("code/")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ----------------------------------------------------------------------
# 2️⃣  Import pipeline components
# ----------------------------------------------------------------------
from preprocessing.split_format import split_and_format_data
from preprocessing.resample import upsample_minority, downsample_majority
from models.logistic_regression import (
    train_logistic_model,
    extract_coefficients,
)
from models.xgboost_model import train_xgboost_model
from explain.evaluate_model import evaluate_model
from explain.feature_importance import run_shap_analysis
from explain.decompose import shapley_owens_decomp

# ----------------------------------------------------------------------
# 3️⃣  Global settings
# ----------------------------------------------------------------------
DATA_PATH = "data/ZZ_cohort_data/"
OUTPUT_DIR = "results/cohort"
RANDOM_STATE = 42
MAXIMIZE_METRIC = "f1"          # Metric to optimise during hyper‑parameter search
MIN_PRECISION = 0.15            # Minimum precision threshold for early stopping

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 4️⃣  Load & split the raw dataset
# ----------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test, id_train, id_test, preprocessor = split_and_format_data(
    data_path=DATA_PATH,
    stratify_on_year=True,  # preserve year distribution across train/test splits
    drop_cols=[
        "pnr",
        "family_id",
        "in_dx",
        "de_age",
        "alive",
        "de_parish",
        "de_region",
        "de_municipality",
        "de_time_to_death",
        "de_age_at_death",
        "se_educ_date",
    ],  # columns that are not needed for modelling
)

# Persist the fitted pre‑processor for future inference or debugging
joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))

# ----------------------------------------------------------------------
# 5️⃣  Address class imbalance with oversampling + undersampling
# ----------------------------------------------------------------------
# 5a. Oversample the minority class five times
X_up, y_up = upsample_minority(X_train_raw, y_train, factor=5)

# 5b. Downsample the majority class to halve its size
X_train_resampled, y_train_resampled = downsample_majority(X_up, y_up, factor=2)

# ----------------------------------------------------------------------
# 6️⃣  Logistic regression for comparison
# ----------------------------------------------------------------------
# This code runs a logistic regression with regularization to estimate early death
# as a function of all possible variables. 
#
# # Transform raw data to one‑hot / numeric format
# X_train_lr = preprocessor.transform(X_train_resampled)
# X_test_lr = preprocessor.transform(X_test_raw)
#
# # Convert sparse matrices to dense if necessary
# if hasattr(X_train_lr, "toarray"):
#     X_train_lr = X_train_lr.toarray()
#     X_test_lr = X_test_lr.toarray()
#
# # Train the model and retrieve the best hyper‑parameters
# print("\n--- Logistic regression ---")
# model_lr, best_params_lr = train_logistic_model(
#     X_train_lr,
#     y_train_resampled,
#     maximize=MAXIMIZE_METRIC,
#     min_precision=MIN_PRECISION,
# )
#
# # Evaluate on the held‑out test set
# metrics_lr = evaluate_model(
#     model_lr,
#     X_test_lr,
#     y_test,
#     output_dir=os.path.join(OUTPUT_DIR, "logistic", f"{MAXIMIZE_METRIC}_optimization"),
#     metadata={"model_type": "logistic", **best_params_lr},
# )
#
# # SHAP explanation for the logistic model
# run_shap_analysis(
#     model=model_lr,
#     X_test=X_test_lr,
#     y_test=y_test,
#     output_dir=f"results/cohort/logistic/{MAXIMIZE_METRIC}_optimization/shap",
#     threshold=0.5,
#     preprocessor=preprocessor,
#     sample_size=50000
# )
#
# # Extract and store coefficient table
# feature_names = preprocessor.get_feature_names_out()
# coefs = extract_coefficients(model_lr, feature_names)
# coefs.to_csv(f"results/cohort/logistic/{MAXIMIZE_METRIC}_optimization/coefficients.csv", index=False)

# ----------------------------------------------------------------------
# 7️⃣  XGBoost – the primary model of interest
# ----------------------------------------------------------------------
print("\n--- XGBoost ---")

# Hyper‑parameter grid to explore during grid‑search
param_grid = {
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# Train the model (this function internally performs grid‑search and
# returns the best model along with the selected hyper‑parameters)
model_xgb, best_params_xgb = train_xgboost_model(
    X_train_resampled,
    y_train_resampled,
    param_grid=param_grid,
    output_dir=os.path.join(OUTPUT_DIR, "xgb", f"{MAXIMIZE_METRIC}_optimization"),
    random_state=RANDOM_STATE,
    maximize=MAXIMIZE_METRIC,
    min_precision=MIN_PRECISION,
)

# XGBoost expects categorical columns to be of type ``category``.
# Ensure all object dtype columns in the test set are converted.
for col in X_test_raw.select_dtypes(include="object").columns:
    X_test_raw[col] = X_test_raw[col].astype("category")

# Model evaluation - F1 score, recall and precision, etc.
metrics_xgb = evaluate_model(
    model_xgb,
    X_test_raw,
    y_test,
    id_test,
    output_dir=os.path.join(OUTPUT_DIR, "xgb", f"{MAXIMIZE_METRIC}_optimization"),
    metadata={"model_type": "xgboost", **best_params_xgb},
)

# SHAP analysis for feature importance
# Computes the contribution of each individual feature in predicting an individual's risk of mortality
shap = run_shap_analysis(
    model_xgb,
    X_test_raw,
    y_test,
    output_dir=os.path.join(OUTPUT_DIR, "xgb", f"{MAXIMIZE_METRIC}_optimization"),
)
shap.to_csv("results/cohort/xgb/f1_optimization/all_shap.csv")

# Shapley-Owens-Shorrocks (SOS) decomposition
# This estimates the amount each variable explains the difference in predicted 
# mortality risk between the survivors and the early death group

# Top features can be a subset of total features based on SHAP feature importance. Not applied now,
# so 1852 is the total number of features in the model.
top_features = shap[shap["version"] == "deaths"].nlargest(1852, "mean_abs_shap")["feature"]

decomp = shapley_owens_decomp(
    model_xgb,
    X=X_test_raw,
    y=y_test,
    top_features=top_features,
    outdir=os.path.join(OUTPUT_DIR, "xgb", f"{MAXIMIZE_METRIC}_optimization"),
)