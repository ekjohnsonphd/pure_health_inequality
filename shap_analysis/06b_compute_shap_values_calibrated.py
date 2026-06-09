from pathlib import Path
import numpy as np
import pandas as pd
import shap
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
cohort      = "female_50-54"
RANDOM_SEED = 42

RESULTS_DIR           = Path(f"/XBoost_results/{cohort}")
CALIBRATED_MODEL_PATH = RESULTS_DIR / f"model5_calibrated_model_{cohort}.joblib"
X_CAL_PATH            = RESULTS_DIR / f"X_cal_raw_{cohort}.parquet"
Y_CAL_PATH            = RESULTS_DIR / f"y_cal_raw_{cohort}.parquet"
OUT_PATH              = RESULTS_DIR / "calibrated_shap_values.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Load model and data
calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)

X_cal    = pd.read_parquet(X_CAL_PATH)
y_cal_df = pd.read_parquet(Y_CAL_PATH)

if "y_cal" in y_cal_df.columns:
    y_cal = y_cal_df["y_cal"].to_numpy().astype(int)
else:
    y_cal = y_cal_df.iloc[:, 0].to_numpy().astype(int)

# Encode categorical variables to numeric (needed for model-agnostic SHAP)
cat_cols     = X_cal.select_dtypes(include=["object", "category"]).columns.tolist()
cat_mappings = {}
for col in cat_cols:
    X_cal[col]        = X_cal[col].astype("category")
    cat_mappings[col] = X_cal[col].cat.categories

def encode(df):
    df = df.copy()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col], categories=cat_mappings[col]).codes
    return df

def decode(df):
    df = df.copy()
    for col in cat_cols:
        df[col] = pd.Categorical.from_codes(df[col].astype(int), categories=cat_mappings[col])
    return df

# Split by outcome
X_deaths    = X_cal[y_cal == 1].copy()
X_survivors = X_cal[y_cal == 0].copy()

# Reproducible sampling
rng          = np.random.default_rng(RANDOM_SEED)
baseline_n   = min(500, X_survivors.shape[0])
baseline_idx = rng.choice(X_survivors.index, size=baseline_n, replace=False)
baseline     = X_survivors.loc[baseline_idx]

n_deaths     = min(100, X_deaths.shape[0])
n_survivors  = min(100, X_survivors.shape[0])
death_idx    = rng.choice(X_deaths.index,    size=n_deaths,    replace=False)
survivor_idx = rng.choice(X_survivors.index, size=n_survivors, replace=False)

X_explain = pd.concat([
    X_deaths.loc[death_idx],
    X_survivors.loc[survivor_idx],
], axis=0)

y_explain = np.concatenate([
    np.ones(n_deaths,    dtype=int),
    np.zeros(n_survivors, dtype=int),
])

baseline_enc  = encode(baseline)
X_explain_enc = encode(X_explain)

# Prediction function for calibrated model
def predict_death_probability(X):
    X_df = pd.DataFrame(X, columns=X_cal.columns)
    X_df = decode(X_df)
    for col in cat_cols:
        X_df[col] = X_df[col].astype("category")
    return calibrated_model.predict_proba(X_df)[:, 1]

# Compute SHAP values
explainer  = shap.Explainer(predict_death_probability, baseline_enc)
max_evals  = 2 * X_explain_enc.shape[1] + 1
shap_exp   = explainer(X_explain_enc, max_evals=max_evals)

df_shap = pd.DataFrame(shap_exp.values, columns=X_cal.columns)
df_shap["y"]        = y_explain
df_shap["pred"]     = predict_death_probability(X_explain_enc.to_numpy())
df_shap["baseline"] = shap_exp.base_values

df_shap.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)

df_check = pd.read_csv(OUT_PATH)
print(df_check[["y", "pred", "baseline"]].head())
print(df_check.shape)
