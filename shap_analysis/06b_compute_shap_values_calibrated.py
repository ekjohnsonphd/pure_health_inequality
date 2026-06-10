
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# Define cohort, random seed, input paths, and output path.
# ─────────────────────────────────────────────────────────────────────────────

cohort      = "female_50-54"
RANDOM_SEED = 42

RESULTS_DIR           = Path(f"/XBoost_results/{cohort}")
CALIBRATED_MODEL_PATH = RESULTS_DIR / f"model5_calibrated_model_{cohort}.joblib"
X_CAL_PATH            = RESULTS_DIR / f"X_cal_raw_{cohort}.parquet"
Y_CAL_PATH            = RESULTS_DIR / f"y_cal_raw_{cohort}.parquet"
OUT_PATH              = RESULTS_DIR / "calibrated_shap_values.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Load calibrated model and calibration data
# The calibrated model is saved as a joblib file.
# X_cal contains features, and y_cal contains the true outcome.
# ─────────────────────────────────────────────────────────────────────────────

calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)

X_cal    = pd.read_parquet(X_CAL_PATH)
y_cal_df = pd.read_parquet(Y_CAL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Extract outcome column
# y_cal = 1 means early death.
# y_cal = 0 means survivor.
# ─────────────────────────────────────────────────────────────────────────────

if "y_cal" in y_cal_df.columns:
    y_cal = y_cal_df["y_cal"].to_numpy().astype(int)
else:
    y_cal = y_cal_df.iloc[:, 0].to_numpy().astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Identify categorical columns
# Model-agnostic SHAP needs numeric input, so categorical variables are encoded.
# The original categories are saved so we can decode them again before prediction.
# ─────────────────────────────────────────────────────────────────────────────

cat_cols     = X_cal.select_dtypes(include=["object", "category"]).columns.tolist()
cat_mappings = {}

for col in cat_cols:
    X_cal[col]        = X_cal[col].astype("category")
    cat_mappings[col] = X_cal[col].cat.categories


# ─────────────────────────────────────────────────────────────────────────────
# Encode categorical columns as numeric codes for SHAP.
# Example: Married, Divorced, Widow become 0, 1, 2.
# ─────────────────────────────────────────────────────────────────────────────

def encode(df):
    df = df.copy()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col], categories=cat_mappings[col]).codes
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Decode numeric category codes back to the original categories.
# This is needed because the calibrated model may expect categorical variables.
# ─────────────────────────────────────────────────────────────────────────────

def decode(df):
    df = df.copy()
    for col in cat_cols:
        df[col] = pd.Categorical.from_codes(
            df[col].astype(int),
            categories=cat_mappings[col]
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Split calibration data by outcome
# This separates deaths and survivors.
# ─────────────────────────────────────────────────────────────────────────────

X_deaths    = X_cal[y_cal == 1].copy()
X_survivors = X_cal[y_cal == 0].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducible sampling
# The random seed makes sure the same people are sampled every time.
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Create SHAP baseline sample
# The baseline is sampled from survivors and used as the SHAP reference group.
# ─────────────────────────────────────────────────────────────────────────────

baseline_n   = min(500, X_survivors.shape[0])
baseline_idx = rng.choice(X_survivors.index, size=baseline_n, replace=False)
baseline     = X_survivors.loc[baseline_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Sample people to explain
# Up to 100 deaths and 100 survivors are selected.
# ─────────────────────────────────────────────────────────────────────────────

n_deaths     = min(100, X_deaths.shape[0])
n_survivors  = min(100, X_survivors.shape[0])

death_idx    = rng.choice(X_deaths.index, size=n_deaths, replace=False)
survivor_idx = rng.choice(X_survivors.index, size=n_survivors, replace=False)


# ─────────────────────────────────────────────────────────────────────────────
# Combine sampled deaths and survivors into one dataset.
# ─────────────────────────────────────────────────────────────────────────────

X_explain = pd.concat([
    X_deaths.loc[death_idx],
    X_survivors.loc[survivor_idx],
], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Create outcome labels for the explained sample.
# First deaths are labelled 1, then survivors are labelled 0.
# ─────────────────────────────────────────────────────────────────────────────

y_explain = np.concatenate([
    np.ones(n_deaths, dtype=int),
    np.zeros(n_survivors, dtype=int),
])


# ─────────────────────────────────────────────────────────────────────────────
# Encode baseline and explanation data for SHAP.
# ─────────────────────────────────────────────────────────────────────────────

baseline_enc  = encode(baseline)
X_explain_enc = encode(X_explain)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction function for the calibrated model
#
# SHAP gives encoded numeric data to this function.
# The function decodes categorical variables back to their original form,
# then returns predicted probability of early death.
# ─────────────────────────────────────────────────────────────────────────────

def predict_death_probability(X):
    X_df = pd.DataFrame(X, columns=X_cal.columns)
    X_df = decode(X_df)
    
    for col in cat_cols:
        X_df[col] = X_df[col].astype("category")
    
    return calibrated_model.predict_proba(X_df)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Create SHAP explainer
# This is model-agnostic SHAP because the calibrated model is wrapped
# in a custom prediction function.
# ─────────────────────────────────────────────────────────────────────────────

explainer = shap.Explainer(
    predict_death_probability,
    baseline_enc
)


# ─────────────────────────────────────────────────────────────────────────────
# Compute SHAP values
# max_evals controls how many model evaluations SHAP can use.
# ─────────────────────────────────────────────────────────────────────────────

max_evals = 2 * X_explain_enc.shape[1] + 1

shap_exp = explainer(
    X_explain_enc,
    max_evals=max_evals
)


# ─────────────────────────────────────────────────────────────────────────────
# Create output dataframe
# Each row is one sampled person.
# Each feature column contains that feature's SHAP value.
# ─────────────────────────────────────────────────────────────────────────────

df_shap = pd.DataFrame(
    shap_exp.values,
    columns=X_cal.columns
)


# ─────────────────────────────────────────────────────────────────────────────
# Add extra columns:
# y        = true outcome
# pred     = calibrated predicted probability of early death
# baseline = SHAP baseline value
# ─────────────────────────────────────────────────────────────────────────────

df_shap["y"]        = y_explain
df_shap["pred"]     = predict_death_probability(X_explain_enc.to_numpy())
df_shap["baseline"] = shap_exp.base_values


# ─────────────────────────────────────────────────────────────────────────────
# Save calibrated SHAP values
# ─────────────────────────────────────────────────────────────────────────────

df_shap.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Quick check of saved file
# Prints the first rows and the shape of the output.
# ─────────────────────────────────────────────────────────────────────────────

df_check = pd.read_csv(OUT_PATH)

print(df_check[["y", "pred", "baseline"]].head())
print(df_check.shape)

