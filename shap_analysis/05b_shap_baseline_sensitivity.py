
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary analysis: sensitivity of calibrated SHAP to the baseline size
#
# This is a variant of 05_shap_calibrated.py. It answers a single question:
# how much do the SHAP contributions move when the reference (background)
# sample used by permutation SHAP is made larger?
#
# Everything except the background size is held fixed:
#   - one cohort (female 65-69)
#   - the same explained people in every run (same deaths, same survivors)
#   - the same seed, model, and max_evals
#
# Only BACKGROUND_N varies, over 100, 500, 1000, 5000, 10000.
#
# Output is one file per baseline size, read by block 3 of
# 06_shap_decomposition.qmd.
# ─────────────────────────────────────────────────────────────────────────────

# Baseline sizes to run. Pass one as a CLI arg to run a single size, e.g.
#   python 05b_shap_baseline_sensitivity.py 5000
# so the sizes can be run as parallel processes. With no arg, all sizes are
# run in sequence (the largest is slow — prefer the parallel form).
BASELINE_SIZES = [100, 500, 1000, 5000, 10000]

# Cohort to explain. Fixed to female 65-69 for the supplement; the second CLI
# arg overrides it if the sensitivity is ever repeated for another cohort.
COHORT      = sys.argv[2] if len(sys.argv) > 2 else "female_65-69"
RANDOM_SEED = 42

# Project data root — set to your server data path when replicating.
DATA_DIR              = Path("../data")
RESULTS_DIR           = DATA_DIR / COHORT
CALIBRATED_MODEL_PATH = RESULTS_DIR / "calibrated_model.joblib"
X_TEST_PATH           = RESULTS_DIR / "X_test.parquet"
Y_TEST_PATH           = RESULTS_DIR / "y_test.parquet"

# Size of the explained set, held constant across all baseline sizes.
#
# The main analysis (05) explains ALL deaths plus a matched survivor sample.
# That is too expensive here: permutation SHAP cost is roughly linear in the
# background size, so the 10,000 arm alone would cost ~100x the main run. The
# sensitivity only needs to show whether the *pattern* of contributions is
# stable as the background grows, so a small balanced sample is enough.
N_EXPLAIN_DEATHS    = 100
N_EXPLAIN_SURVIVORS = 100

# Memory guard. Permutation SHAP evaluates a batch of masks at a time, and each
# mask expands to one copy of the whole background sample. With a 10,000-row
# background and ~1,600 features a single mask is already ~130 MB, so the
# default batching can exhaust memory on the large arms. This caps the rows
# handed to predict_proba in one call; lower it if the 10,000 arm still runs
# out of memory, raise it if the small arms are dominated by call overhead.
MAX_MASKED_ROWS = 20_000


# ─────────────────────────────────────────────────────────────────────────────
# Resolve which baseline sizes this process should run
# ─────────────────────────────────────────────────────────────────────────────

if len(sys.argv) > 1:
    requested = int(sys.argv[1])
    if requested not in BASELINE_SIZES:
        print(f"Warning: {requested} is not one of {BASELINE_SIZES}; running it anyway")
    sizes_to_run = [requested]
else:
    sizes_to_run = BASELINE_SIZES


# ─────────────────────────────────────────────────────────────────────────────
# Load calibrated model and test data
# Identical to 05_shap_calibrated.py: SHAP is computed on the TEST set (held
# out from both training and calibration), using the calibrated model.
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
# ─────────────────────────────────────────────────────────────────────────────

X_deaths    = X_test[y_test == 1].copy()
X_survivors = X_test[y_test == 0].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Two independent random streams
#
# This is the one structural change from 05. There, the baseline is drawn from
# the same generator as the explained sample, so changing the baseline size
# also changes WHICH people get explained — baseline size and sample
# composition would be confounded, which is exactly what this analysis is
# trying to measure. Separate seeded streams keep the explained set byte-for-
# byte identical across all baseline sizes.
# ─────────────────────────────────────────────────────────────────────────────

rng_explain  = np.random.default_rng(RANDOM_SEED)
rng_baseline = np.random.default_rng(RANDOM_SEED + 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Select the people to explain — drawn once, used by every baseline size
#
# A balanced sample of deaths and survivors. Note this differs from the main
# analysis, where all deaths are explained: absolute SHAP levels here are not
# comparable to the headline numbers, only the relative pattern across feature
# groups is.
# ─────────────────────────────────────────────────────────────────────────────

n_deaths    = min(N_EXPLAIN_DEATHS, X_deaths.shape[0])
n_survivors = min(N_EXPLAIN_SURVIVORS, X_survivors.shape[0])

death_idx    = rng_explain.choice(X_deaths.index, size=n_deaths, replace=False)
survivor_idx = rng_explain.choice(X_survivors.index, size=n_survivors, replace=False)

X_explain = pd.concat([
    X_deaths.loc[death_idx],
    X_survivors.loc[survivor_idx],
], axis=0)

# First deaths are labelled 1, then survivors are labelled 0.
y_explain = np.concatenate([
    np.ones(n_deaths, dtype=int),
    np.zeros(n_survivors, dtype=int),
])


# ─────────────────────────────────────────────────────────────────────────────
# Nested baseline draw
#
# One shuffle of the survivor pool, then take the first N rows for each
# baseline size. The 500-row baseline therefore CONTAINS the 100-row baseline,
# and so on, so any movement between arms comes from adding reference people
# rather than from swapping them. Drawn from the full survivor pool, as in 05,
# so the baseline may overlap the explained survivors — with a pool of ~200k
# the expected overlap is a handful of rows.
# ─────────────────────────────────────────────────────────────────────────────

survivor_pool = rng_baseline.permutation(X_survivors.index.to_numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Prediction function for the calibrated model
#
# SHAP gives encoded numeric data to this function. The function decodes
# categorical variables back to their original form, then returns predicted
# probability of early death.
# ─────────────────────────────────────────────────────────────────────────────

def predict_death_probability(X):
    X_df = pd.DataFrame(X, columns=X_test.columns)
    X_df = decode(X_df)

    for col in cat_cols:
        X_df[col] = X_df[col].astype("category")

    return calibrated_model.predict_proba(X_df)[:, 1]


X_explain_enc = encode(X_explain)
max_evals     = 2 * X_explain_enc.shape[1] + 1

print(f"Cohort {COHORT}: explaining {n_deaths} deaths + {n_survivors} survivors, "
      f"{X_explain_enc.shape[1]} features, max_evals={max_evals}")
print(f"Baseline sizes to run: {sizes_to_run}")


# ─────────────────────────────────────────────────────────────────────────────
# Run one baseline size
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_size(background_n):

    baseline_n = min(background_n, X_survivors.shape[0])
    if baseline_n < background_n:
        print(f"Warning: only {baseline_n} survivors available, "
              f"requested baseline {background_n}")

    baseline     = X_survivors.loc[survivor_pool[:baseline_n]]
    baseline_enc = encode(baseline)

    # ── The load-bearing line ────────────────────────────────────────────────
    # Passing a raw DataFrame to shap.Explainer wraps it in an Independent
    # masker whose default is max_samples=100. A 10,000-row background would be
    # silently subsampled back to 100 and every arm of this sensitivity would
    # return the same answer. Build the masker explicitly and assert the size
    # that SHAP actually held on to.
    masker = shap.maskers.Independent(baseline_enc, max_samples=baseline_n)

    realised_n = masker.data.shape[0]
    if realised_n != baseline_n:
        raise RuntimeError(
            f"SHAP masker kept {realised_n} background rows, expected {baseline_n}. "
            "The sensitivity is meaningless unless these match."
        )

    explainer = shap.Explainer(predict_death_probability, masker)

    # Masks per model call, so that masks x background stays under the cap.
    batch_size = max(1, MAX_MASKED_ROWS // baseline_n)

    t0 = time.time()
    shap_exp = explainer(X_explain_enc, max_evals=max_evals, batch_size=batch_size)
    elapsed  = time.time() - t0

    # ── Output frame ────────────────────────────────────────────────────────
    # Column names follow block 3 of 06_shap_decomposition.qmd, which melts on
    # id.vars = c("pred", "y", "base_value", "baseline_size"). Note the main
    # analysis calls the base value "baseline"; this file uses "base_value".
    df_shap = pd.DataFrame(shap_exp.values, columns=X_test.columns)

    df_shap["y"]             = y_explain
    df_shap["pred"]          = predict_death_probability(X_explain_enc.to_numpy())
    df_shap["base_value"]    = shap_exp.base_values
    df_shap["baseline_size"] = realised_n

    out_path = RESULTS_DIR / f"{background_n}_shap_values.csv"
    df_shap.to_csv(out_path, index=False)

    print(f"baseline {background_n}: {realised_n} background rows, "
          f"{elapsed / 60:.1f} min, base_value mean "
          f"{df_shap['base_value'].mean():.6f} -> {out_path}")

    return df_shap


# ─────────────────────────────────────────────────────────────────────────────
# Run the requested baseline sizes
# ─────────────────────────────────────────────────────────────────────────────

summaries = []

for background_n in sizes_to_run:
    df_shap = run_baseline_size(background_n)

    summaries.append({
        "baseline_size": background_n,
        "base_value":    df_shap["base_value"].mean(),
        "mean_pred_0":   df_shap.loc[df_shap["y"] == 0, "pred"].mean(),
        "mean_pred_1":   df_shap.loc[df_shap["y"] == 1, "pred"].mean(),
        "count_0":       int((df_shap["y"] == 0).sum()),
        "count_1":       int((df_shap["y"] == 1).sum()),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Quick check
#
# base_value is the model's expected prediction over the background sample, so
# it should settle down as the baseline grows. mean_pred_0 and mean_pred_1 are
# properties of the explained set and must be IDENTICAL across every baseline
# size — if they are not, the explained sample moved and the comparison is not
# clean.
# ─────────────────────────────────────────────────────────────────────────────

print(pd.DataFrame(summaries).to_string(index=False))
