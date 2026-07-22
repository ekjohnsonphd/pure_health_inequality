
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# Define cohort, random seed, input paths, output path, and sampling sizes.
#
# SHAP is computed on the TEST set (held out from both training and calibration),
# using the calibrated model.
# ─────────────────────────────────────────────────────────────────────────────

cohort      = "female_50-54"
RANDOM_SEED = 42

# Project data root — set to your server data path when replicating.
DATA_DIR              = Path("../data")
RESULTS_DIR           = DATA_DIR / cohort
CALIBRATED_MODEL_PATH = RESULTS_DIR / "calibrated_model.joblib"
X_TEST_PATH           = RESULTS_DIR / "X_test_raw.parquet"
Y_TEST_PATH           = RESULTS_DIR / "y_test_raw.parquet"
OUT_PATH              = RESULTS_DIR / "calibrated_shap_values.csv"

# Sampling for the explained set.
# The calibrated model uses model-agnostic (permutation) SHAP, which is too
# expensive to run on the whole test set (~200k+ rows). The decomposition only
# needs stable group means (deaths vs survivors), and deaths are rare, so we
# explain ALL deaths plus a random survivor sample of equal size.
# N_SURVIVORS = None  -> match the number of deaths.
N_SURVIVORS   = None   # or an int to fix the survivor sample size
BACKGROUND_N  = 100    # survivor reference (background) size for SHAP


# ─────────────────────────────────────────────────────────────────────────────
# Load calibrated model and test data
# The calibrated model is saved as a joblib file.
# X_test contains features, and y_test contains the true outcome.
# ─────────────────────────────────────────────────────────────────────────────

calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)

X_test    = pd.read_parquet(X_TEST_PATH)
y_test_df = pd.read_parquet(Y_TEST_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Extract outcome column
# y_test = 1 means early death.
# y_test = 0 means survivor.
# ─────────────────────────────────────────────────────────────────────────────

if "y_test" in y_test_df.columns:
    y_test = y_test_df["y_test"].to_numpy().astype(int)
else:
    y_test = y_test_df.iloc[:, 0].to_numpy().astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Identify categorical columns
# Model-agnostic SHAP needs numeric input, so categorical variables are encoded.
# The original categories are saved so we can decode them again before prediction.
# ─────────────────────────────────────────────────────────────────────────────

cat_cols     = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
cat_mappings = {}

for col in cat_cols:
    X_test[col]       = X_test[col].astype("category")
    cat_mappings[col] = X_test[col].cat.categories


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
# This is needed because the calibrated model expects the original values.
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
# Split test data by outcome
# This separates deaths and survivors.
# ─────────────────────────────────────────────────────────────────────────────

X_deaths    = X_test[y_test == 1].copy()
X_survivors = X_test[y_test == 0].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducible sampling
# The random seed makes sure the same people are sampled every time.
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Create SHAP baseline sample
# The baseline is sampled from survivors and used as the SHAP reference group.
# ─────────────────────────────────────────────────────────────────────────────

baseline_n   = min(BACKGROUND_N, X_survivors.shape[0])
baseline_idx = rng.choice(X_survivors.index, size=baseline_n, replace=False)
baseline     = X_survivors.loc[baseline_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Select people to explain
# ALL deaths are explained (they are rare). Survivors are randomly sampled to a
# matched size so the death and survivor group means are both stable.
# ─────────────────────────────────────────────────────────────────────────────

n_deaths    = X_deaths.shape[0]
n_survivors = n_deaths if N_SURVIVORS is None else N_SURVIVORS
n_survivors = min(n_survivors, X_survivors.shape[0])

survivor_idx = rng.choice(X_survivors.index, size=n_survivors, replace=False)


# ─────────────────────────────────────────────────────────────────────────────
# Combine all deaths and the sampled survivors into one dataset.
# ─────────────────────────────────────────────────────────────────────────────

X_explain = pd.concat([
    X_deaths,
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

print(f"Explaining {n_deaths} deaths + {n_survivors} survivors "
      f"(background {baseline_n}) for cohort {cohort}")


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
    X_df = pd.DataFrame(X, columns=X_test.columns)
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
# max_evals controls how many model evaluations SHAP can use per person
# (one permutation). Runtime scales with the number of people explained, so the
# sampling above keeps this tractable on the full test set.
# ─────────────────────────────────────────────────────────────────────────────

max_evals = 2 * X_explain_enc.shape[1] + 1

shap_exp = explainer(
    X_explain_enc,
    max_evals=max_evals
)


# ─────────────────────────────────────────────────────────────────────────────
# Create output dataframe
# Each row is one explained person.
# Each feature column contains that feature's SHAP value.
# ─────────────────────────────────────────────────────────────────────────────

df_shap = pd.DataFrame(
    shap_exp.values,
    columns=X_test.columns
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
